//! One authenticated websocket connection.
//!
//! Use one connection per feed: Kalshi serves channels on a socket in order,
//! so a busy channel head-of-line blocks the rest (measured in monymaker: BRTI
//! went from 0.07s to 11s behind when sharing with the book). Kalshi does not
//! answer pings, so liveness is judged by silence instead.

use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use crate::Signer;

pub const WS_URL: &str = "wss://external-api-ws.kalshi.com/trade-api/ws/v2";
pub const WS_PATH: &str = "/trade-api/ws/v2";

pub struct WsMsg {
    pub recv_ns: i64,
    pub raw: String,
    pub value: Value,
}

impl WsMsg {
    pub fn kind(&self) -> &str {
        self.value["type"].as_str().unwrap_or("")
    }
}

pub struct WsConn {
    ws: WebSocketStream<MaybeTlsStream<TcpStream>>,
    next_id: u64,
}

impl WsConn {
    pub async fn connect(signer: &Signer) -> Result<Self> {
        let mut req = WS_URL.into_client_request()?;
        for (k, v) in signer.headers("GET", WS_PATH) {
            req.headers_mut().insert(k, v.parse()?);
        }
        let (ws, _) = tokio_tungstenite::connect_async(req).await?;
        Ok(Self { ws, next_id: 1 })
    }

    async fn command(&mut self, cmd: &str, params: Value) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;
        let body = json!({ "id": id, "cmd": cmd, "params": params });
        self.ws.send(Message::text(body.to_string())).await?;
        Ok(id)
    }

    /// Returns the command id; the `subscribed` reply carries it with the sid.
    pub async fn subscribe(&mut self, params: Value) -> Result<u64> {
        self.command("subscribe", params).await
    }

    pub async fn unsubscribe(&mut self, sids: &[u64]) -> Result<u64> {
        self.command("unsubscribe", json!({ "sids": sids })).await
    }

    /// Adds or removes markets on an existing subscription. Kalshi keeps one
    /// subscription per channel per connection -- a second `subscribe` is
    /// merged into the first sid -- so this is how the market set changes.
    /// `action` is `add_markets` or `delete_markets`.
    pub async fn update_markets(&mut self, sid: u64, tickers: &[String], action: &str) -> Result<u64> {
        self.command("update_subscription", json!({ "sids": [sid], "market_tickers": tickers, "action": action })).await
    }

    /// Next text message. Errors when no text arrives within `timeout` (pings
    /// don't count), on close, and on Kalshi `error` messages.
    pub async fn recv(&mut self, timeout: Duration) -> Result<WsMsg> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let frame = tokio::time::timeout_at(deadline, self.ws.next())
                .await
                .map_err(|_| anyhow!("silent for {}s", timeout.as_secs()))?
                .ok_or_else(|| anyhow!("connection closed"))??;
            let raw = match frame {
                Message::Text(t) => t.to_string(),
                Message::Close(c) => bail!("closed by server: {c:?}"),
                _ => continue,
            };
            let value: Value = serde_json::from_str(&raw)?;
            if value["type"] == "error" {
                bail!("kalshi error: {}", value["msg"]);
            }
            return Ok(WsMsg { recv_ns: crate::now_ns(), raw, value });
        }
    }
}

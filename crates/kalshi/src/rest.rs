//! Signed REST calls: reads, plus the few writes live trading needs.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Result};
use serde_json::Value;

use crate::Signer;

pub const BASE: &str = "https://api.elections.kalshi.com";
pub const API: &str = "/trade-api/v2";

#[derive(Clone)]
pub struct Rest {
    http: reqwest::Client,
    signer: Arc<Signer>,
}

impl Rest {
    pub fn new(signer: Arc<Signer>) -> Result<Self> {
        let http = reqwest::Client::builder().timeout(Duration::from_secs(15)).build()?;
        Ok(Self { http, signer })
    }

    pub async fn get(&self, path: &str, query: &[(&str, &str)]) -> Result<Value> {
        let mut req = self.http.get(format!("{BASE}{path}")).query(query);
        for (k, v) in self.signer.headers("GET", path) {
            req = req.header(k, v);
        }
        let resp = req.send().await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("GET {path}: HTTP {status}: {text}");
        }
        Ok(serde_json::from_str(&text)?)
    }

    pub async fn post(&self, path: &str, body: &Value) -> Result<Value> {
        let mut req = self.http.post(format!("{BASE}{path}")).json(body);
        for (k, v) in self.signer.headers("POST", path) {
            req = req.header(k, v);
        }
        let resp = req.send().await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("POST {path}: HTTP {status}: {text}");
        }
        Ok(serde_json::from_str(&text)?)
    }

    /// Markets of a series in a given status (`open`, `settled`, ...), first page.
    pub async fn markets(&self, series: &str, status: &str, limit: u32) -> Result<Vec<Value>> {
        let limit = limit.to_string();
        let v = self
            .get(&format!("{API}/markets"), &[("series_ticker", series), ("status", status), ("limit", &limit)])
            .await?;
        Ok(v["markets"].as_array().cloned().unwrap_or_default())
    }

    /// The current window and the next one, by close time. The next window is
    /// listed as `initialized` (no strike yet) about 15 minutes ahead, so the
    /// recorder can be subscribed before it opens. A closed window can stay
    /// listed as `active` with an empty book, which the close filter drops.
    pub async fn upcoming_markets(&self, series: &str, horizon_s: i64) -> Result<Vec<Value>> {
        let now = crate::now_ms();
        let (lo, hi) = ((now / 1000).to_string(), (now / 1000 + horizon_s).to_string());
        let v = self
            .get(
                &format!("{API}/markets"),
                &[("series_ticker", series), ("min_close_ts", &lo), ("max_close_ts", &hi), ("limit", "20")],
            )
            .await?;
        Ok(v["markets"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|m| crate::parse_time_ms(&m["close_time"]).is_some_and(|c| c > now))
            .collect())
    }

    pub async fn market(&self, ticker: &str) -> Result<Value> {
        let v = self.get(&format!("{API}/markets/{ticker}"), &[]).await?;
        Ok(v.get("market").cloned().unwrap_or(v))
    }

    /// Bid ladders for both sides: `orderbook_fp.yes_dollars` and
    /// `orderbook_fp.no_dollars`, each `[[price, size], ...]` ascending.
    pub async fn orderbook(&self, ticker: &str, depth: u32) -> Result<Value> {
        let depth = depth.to_string();
        self.get(&format!("{API}/markets/{ticker}/orderbook"), &[("depth", &depth)]).await
    }

    pub async fn series(&self, series: &str) -> Result<Value> {
        let v = self.get(&format!("{API}/series/{series}"), &[]).await?;
        Ok(v.get("series").cloned().unwrap_or(v))
    }

    pub async fn balance(&self) -> Result<Value> {
        self.get(&format!("{API}/portfolio/balance"), &[]).await
    }

    /// Cash per exchange shard, in dollars. Collateral doesn't cross shards
    /// on its own: deposits land on 0, crypto markets trade on 2.
    pub async fn shard_balances(&self) -> Result<Vec<(i64, f64)>> {
        let v = self.balance().await?;
        Ok(v["balance_breakdown"]
            .as_array()
            .map(|rows| {
                rows.iter().filter_map(|r| Some((r["exchange_index"].as_i64()?, crate::parse_num(&r["balance"])?))).collect()
            })
            .unwrap_or_default())
    }

    /// Moves `dollars` of collateral between shards (amounts are in
    /// centicents on the wire).
    pub async fn transfer(&self, dollars: f64, from_shard: i64, to_shard: i64) -> Result<Value> {
        let body = serde_json::json!({
            "source": "event_contract",
            "destination": "event_contract",
            "amount": (dollars * 10_000.0).round() as i64,
            "source_exchange_shard": from_shard,
            "destination_exchange_shard": to_shard,
        });
        self.post(&format!("{API}/portfolio/intra_exchange_instance_transfer"), &body).await
    }

    /// A marketable limit order that never rests (immediate-or-cancel).
    /// `book_side` is `bid` (buy YES / close NO) or `ask` (buy NO / close
    /// YES); `yes_price` is always on the YES axis, as Kalshi's fixed-point
    /// string. `reduce_only` can only shrink a position, never open or flip one.
    pub async fn place_ioc(
        &self,
        ticker: &str,
        book_side: &str,
        count: u32,
        yes_price: &str,
        exchange_index: Option<i64>,
        reduce_only: bool,
        client_order_id: &str,
    ) -> Result<Value> {
        let mut body = serde_json::json!({
            "ticker": ticker,
            "client_order_id": client_order_id,
            "side": book_side,
            "count": format!("{count}.00"),
            "price": yes_price,
            "time_in_force": "immediate_or_cancel",
            "self_trade_prevention_type": "taker_at_cross",
        });
        if let Some(x) = exchange_index {
            body["exchange_index"] = x.into();
        }
        if reduce_only {
            body["reduce_only"] = true.into();
        }
        self.post(&format!("{API}/portfolio/events/orders"), &body).await
    }

    /// Fills of one order: `count_fp`, `outcome_side`, `yes_price_dollars`,
    /// `no_price_dollars`, `fee_cost`, ...
    pub async fn fills_for_order(&self, order_id: &str) -> Result<Vec<Value>> {
        let v = self.get(&format!("{API}/portfolio/fills"), &[("order_id", order_id), ("limit", "200")]).await?;
        Ok(v["fills"].as_array().cloned().unwrap_or_default())
    }

    /// One order by id: `fill_count_fp`, `taker_fill_cost_dollars`,
    /// `taker_fees_dollars`, `status`, ...
    pub async fn order(&self, order_id: &str) -> Result<Value> {
        let v = self.get(&format!("{API}/portfolio/orders/{order_id}"), &[]).await?;
        Ok(v.get("order").cloned().unwrap_or(v))
    }
}

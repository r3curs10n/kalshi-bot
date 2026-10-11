//! Live Kalshi feeds shared by the recorder and the paper trader.
//!
//! Three websockets (BRTI; quotes; trades + lifecycle), one per feed so a busy
//! channel can't delay the others, plus a REST poller for market metadata,
//! orderbook snapshots and settlement results. Everything is delivered as
//! `Feed` messages on one channel.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Result};
use kalshi::{now_ms, now_ns, Rest, Signer, WsConn, SERIES};
use serde_json::{json, Value};
use tokio::sync::{mpsc, watch};
use tracing::{info, warn};


/// Kalshi pushes BRTI once a second, so 15s of silence means the feed is dead.
pub const PRICE_SILENCE: Duration = Duration::from_secs(15);
/// `ticker` sends about once a second on a live window.
pub const QUOTE_SILENCE: Duration = Duration::from_secs(90);
/// Trades can legitimately pause; the lifecycle firehose keeps this one busy.
pub const FLOW_SILENCE: Duration = Duration::from_secs(300);
/// BRTI silent for longer than this is a gap. Quotes and trades can pause on
/// their own, so for them only a reconnect counts.
pub const GAP_MS: i64 = 3_000;
/// How far ahead to look for the next window.
pub const HORIZON_S: i64 = 30 * 60;
/// Give up waiting for a result this long after close.
pub const SETTLE_DEADLINE_MS: i64 = 2 * 3600 * 1000;
/// BRTI kept in memory to backfill a session's lookback when it first appears.
pub const RING_MS: i64 = 20 * 60 * 1000;

pub enum Feed {
    Btc { recv_ns: i64, t_ms: i64, raw: String },
    Market { stream: &'static str, ticker: String, recv_ns: i64, raw: String },
    /// A REST market object for a live window.
    Upsert(Value),
    /// A REST market object carrying the result.
    Settled(Value),
    Gap { streams: &'static [&'static str], from_ms: i64, to_ms: i64, reason: String },
    /// Finish every file and exit.
    Shutdown,
}


/// Starts every feed. `pending` are windows from earlier runs still awaiting
/// a result. Abort the returned tasks to stop them.
pub fn spawn_all(
    signer: Arc<Signer>,
    rest: Rest,
    tx: mpsc::UnboundedSender<Feed>,
    pending: HashMap<String, i64>,
) -> Vec<tokio::task::JoinHandle<()>> {
    let (tickers_tx, tickers_rx) = watch::channel(Vec::<String>::new());
    let price = tokio::spawn(price_feed(signer.clone(), tx.clone()));
    let quotes = FeedSpec { name: "quotes", channels: &["ticker"], global: None, silence: QUOTE_SILENCE, streams: &["quotes"] };
    let flow = FeedSpec {
        name: "flow",
        channels: &["trade"],
        global: Some(json!({ "channels": ["market_lifecycle_v2"] })),
        silence: FLOW_SILENCE,
        streams: &["trades", "lifecycle"],
    };
    vec![
        price,
        tokio::spawn(market_feed(quotes, signer.clone(), tickers_rx.clone(), tx.clone())),
        tokio::spawn(market_feed(flow, signer, tickers_rx, tx.clone())),
        tokio::spawn(poller(rest, tickers_tx, tx, pending)),
    ]
}


/// Exponential reconnect backoff, 1s doubling to 30s, reset after a
/// connection that lasted a minute.
struct Backoff {
    delay: Duration,
    started: std::time::Instant,
}

impl Backoff {
    fn new() -> Self {
        Self { delay: Duration::from_secs(1), started: std::time::Instant::now() }
    }

    async fn after(&mut self, name: &str, result: Result<()>) {
        if let Err(e) = result {
            warn!("{name} feed: {e:#}");
        }
        if self.started.elapsed() > Duration::from_secs(60) {
            self.delay = Duration::from_secs(1);
        }
        tokio::time::sleep(self.delay).await;
        self.delay = (self.delay * 2).min(Duration::from_secs(30));
        self.started = std::time::Instant::now();
    }
}

async fn price_feed(signer: Arc<Signer>, tx: mpsc::UnboundedSender<Feed>) {
    let mut last_ms: Option<i64> = None;
    let mut backoff = Backoff::new();
    loop {
        let r = price_conn(&signer, &tx, &mut last_ms).await;
        backoff.after("price", r).await;
    }
}

async fn price_conn(signer: &Signer, tx: &mpsc::UnboundedSender<Feed>, last_ms: &mut Option<i64>) -> Result<()> {
    let mut ws = WsConn::connect(signer).await?;
    ws.subscribe(json!({ "channels": ["cfbenchmarks_value"], "index_ids": ["BRTI"] })).await?;
    loop {
        let msg = ws.recv(PRICE_SILENCE).await?;
        if msg.kind() != "cfbenchmarks_value" {
            continue;
        }
        // `data` is the upstream CF frame, still a JSON string.
        let inner: Value = serde_json::from_str(msg.value["msg"]["data"].as_str().unwrap_or("{}"))?;
        let t_ms = inner["time"].as_i64().unwrap_or(msg.recv_ns / 1_000_000);
        note_gap(tx, last_ms, msg.recv_ns / 1_000_000, &["btc"], "feed");
        tx.send(Feed::Btc { recv_ns: msg.recv_ns, t_ms, raw: msg.raw })?;
    }
}

fn note_gap(tx: &mpsc::UnboundedSender<Feed>, last: &mut Option<i64>, now: i64, streams: &'static [&'static str], reason: &str) {
    if let Some(prev) = *last
        && now - prev > GAP_MS {
            let _ = tx.send(Feed::Gap { streams, from_ms: prev, to_ms: now, reason: reason.into() });
        }
    *last = Some(now);
}

fn stream_for(kind: &str) -> Option<&'static str> {
    match kind {
        "ticker" => Some("quotes"),
        "trade" => Some("trades"),
        "market_lifecycle_v2" => Some("lifecycle"),
        _ => None,
    }
}

struct FeedSpec {
    name: &'static str,
    channels: &'static [&'static str],
    /// Subscription made once per connection, not per ticker.
    global: Option<Value>,
    silence: Duration,
    /// Streams a dead connection leaves gaps in.
    streams: &'static [&'static str],
}

/// A socket subscribed per-ticker to `spec.channels`, following the live
/// windows in `tickers`.
async fn market_feed(spec: FeedSpec, signer: Arc<Signer>, tickers: watch::Receiver<Vec<String>>, tx: mpsc::UnboundedSender<Feed>) {
    let mut last_ms: Option<i64> = None;
    let mut backoff = Backoff::new();
    loop {
        let r = market_conn(&spec, &signer, tickers.clone(), &tx, &mut last_ms).await;
        backoff.after(spec.name, r).await;
    }
}

async fn market_conn(
    spec: &FeedSpec,
    signer: &Signer,
    mut tickers: watch::Receiver<Vec<String>>,
    tx: &mpsc::UnboundedSender<Feed>,
    last_ms: &mut Option<i64>,
) -> Result<()> {
    let mut ws = WsConn::connect(signer).await?;
    if let Some(g) = &spec.global {
        ws.subscribe(g.clone()).await?;
    }
    // Kalshi keeps one subscription per channel per connection: a second
    // `subscribe` is merged into the first sid, so unsubscribing "one
    // ticker's sid" would drop every market. Instead there is exactly one
    // subscription, and markets are added and removed on it.
    let mut sub = Subscription::default();
    let mut fresh = true;
    // Pings keep a socket open even when it has stopped delivering, so
    // liveness is judged on market data alone.
    let mut last_data = std::time::Instant::now();
    let mut watchdog = tokio::time::interval(Duration::from_secs(5));
    tickers.mark_changed();
    loop {
        tokio::select! {
            msg = ws.recv(spec.silence) => {
                let msg = msg?;
                let v = &msg.value;
                if msg.kind() == "subscribed" {
                    if v["id"].as_u64() == sub.cmd_id {
                        sub.sid = v["msg"]["sid"].as_u64();
                        sub.reconcile(&mut ws, spec).await?;
                    }
                    continue;
                }
                let Some(stream) = stream_for(msg.kind()) else { continue };
                let Some(ticker) = v["msg"]["market_ticker"].as_str() else { continue };
                if !ticker.starts_with(SERIES) {
                    continue;
                }
                let now = msg.recv_ns / 1_000_000;
                if fresh {
                    note_gap(tx, last_ms, now, spec.streams, "reconnect");
                    fresh = false;
                }
                *last_ms = Some(now);
                if stream != "lifecycle" {
                    last_data = std::time::Instant::now();
                }
                tx.send(Feed::Market { stream, ticker: ticker.to_string(), recv_ns: msg.recv_ns, raw: msg.raw })?;
            }
            changed = tickers.changed() => {
                changed?;
                sub.want = tickers.borrow_and_update().iter().cloned().collect();
                sub.reconcile(&mut ws, spec).await?;
            }
            _ = watchdog.tick() => {
                if !sub.have.is_empty() && last_data.elapsed() > spec.silence {
                    bail!("no market data for {}s", last_data.elapsed().as_secs());
                }
            }
        }
    }
}

/// The single per-channel subscription of a market feed.
#[derive(Default)]
struct Subscription {
    want: BTreeSet<String>,
    /// Markets the server has (or will have, once confirmed).
    have: BTreeSet<String>,
    cmd_id: Option<u64>,
    sid: Option<u64>,
}

impl Subscription {
    async fn reconcile(&mut self, ws: &mut WsConn, spec: &FeedSpec) -> Result<()> {
        if self.cmd_id.is_none() {
            if self.want.is_empty() {
                return Ok(());
            }
            let tickers: Vec<&String> = self.want.iter().collect();
            self.cmd_id = Some(ws.subscribe(json!({ "channels": spec.channels, "market_tickers": tickers })).await?);
            self.have = self.want.clone();
            info!("{}: following {:?}", spec.name, self.have);
            return Ok(());
        }
        // Edits wait for the sid; `subscribed` triggers another reconcile.
        let Some(sid) = self.sid else { return Ok(()) };
        let add: Vec<String> = self.want.difference(&self.have).cloned().collect();
        let del: Vec<String> = self.have.difference(&self.want).cloned().collect();
        if !add.is_empty() {
            ws.update_markets(sid, &add, "add_markets").await?;
            info!("{}: following {add:?}", spec.name);
        }
        if !del.is_empty() {
            ws.update_markets(sid, &del, "delete_markets").await?;
        }
        self.have = self.want.clone();
        Ok(())
    }
}

/// REST: current + next window every 5s, orderbooks every 2s, results after close.
pub async fn poller(
    rest: Rest,
    tickers_tx: watch::Sender<Vec<String>>,
    tx: mpsc::UnboundedSender<Feed>,
    mut pending: HashMap<String, i64>,
) {
    // (ticker, open_ms, close_ms) of the current and next windows.
    let mut live: Vec<(String, i64, i64)> = vec![];
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    for n in 0u64.. {
        tick.tick().await;
        let now = now_ms();

        if n % 5 == 0 {
            match rest.upcoming_markets(SERIES, HORIZON_S).await {
                Ok(markets) => {
                    let mut next = vec![];
                    for m in markets {
                        let (Some(t), Some(open), Some(close)) = (
                            m["ticker"].as_str(),
                            kalshi::parse_time_ms(&m["open_time"]),
                            kalshi::parse_time_ms(&m["close_time"]),
                        ) else {
                            continue;
                        };
                        next.push((t.to_string(), open, close));
                        let _ = tx.send(Feed::Upsert(m));
                    }
                    for (t, _, close) in &live {
                        if !next.iter().any(|(n, _, _)| n == t) {
                            pending.insert(t.clone(), *close);
                        }
                    }
                    live = next;
                    let names: Vec<String> = live.iter().map(|(t, _, _)| t.clone()).collect();
                    tickers_tx.send_if_modified(|cur| {
                        let changed = *cur != names;
                        *cur = names.clone();
                        changed
                    });
                }
                Err(e) => warn!("live markets: {e:#}"),
            }
        }

        if n % 2 == 0 {
            for (t, open, close) in &live {
                if now < *open || *close <= now {
                    continue;
                }
                match rest.orderbook(t, 10).await {
                    Ok(book) => {
                        let _ = tx.send(Feed::Market { stream: "book", ticker: t.clone(), recv_ns: now_ns(), raw: book.to_string() });
                    }
                    Err(e) => warn!("orderbook {t}: {e:#}"),
                }
            }
        }

        if n % 5 == 2 {
            let due: Vec<(String, i64)> = pending.iter().filter(|(_, c)| **c <= now).map(|(t, c)| (t.clone(), *c)).collect();
            for (t, close) in due {
                match rest.market(&t).await {
                    Ok(m) if m["result"].as_str().is_some_and(|r| !r.is_empty()) => {
                        info!("{t} settled: {}", m["result"]);
                        pending.remove(&t);
                        let _ = tx.send(Feed::Settled(m));
                    }
                    Ok(_) if now > close + SETTLE_DEADLINE_MS => {
                        warn!("{t}: no result {}h after close; giving up", SETTLE_DEADLINE_MS / 3_600_000);
                        pending.remove(&t);
                    }
                    Ok(_) => {}
                    Err(e) => warn!("market {t}: {e:#}"),
                }
            }
        }
    }
}


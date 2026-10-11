//! Live trading from the viz page: one strategy on the person's own Kalshi
//! account with real orders, from the session it was started on through
//! every later one, until it is stopped.
//!
//! Each session gets its own account: it starts with `budget` in cash and
//! every real fill (count, cost and fee as Kalshi reports them) moves it, so
//! the strategy never spends more than the budget in one session. Before a
//! session trades, the market's shard is topped up to the budget from shard
//! 0. Orders are immediate-or-cancel limits one cent through the touch, so
//! nothing is left resting on the book. The key lives in memory for the run
//! only and is never written anywhere.
//!
//! Snapshots are assembled exactly as in `kbot paper` and replay. Unlike the
//! paper trader, a decision goes out at once rather than a second later.

use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use kalshi::{now_ms, parse_num, Rest, Signer};
use market::replay::{brti_tick, parse_quotes, Assembler};
use market::Store;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use strategy::fees::{fee_per_contract, TAKER_RATE};
use strategy::{apply_fill, Action, Decision, Position, Snapshot, Strategy, TradeLimits, Window};
use tokio::sync::{mpsc, watch};
use tracing::{info, warn};

use crate::backtest::make_strategy;
use crate::feeds::{self, Feed};

/// How far through the touch an order reaches, in dollars: the worst price
/// accepted, not the price paid.
const TOLERANCE: f64 = 0.01;
/// Quotes older than this don't price an order.
const QUOTE_STALE_MS: i64 = 10_000;
/// After a failed order, wait this long before trying again.
const RETRY_MS: i64 = 3_000;

#[derive(Deserialize)]
pub struct LiveRequest {
    pub ticker: String,
    pub strategy: String,
    pub key_id: String,
    /// RSA private key, PEM.
    pub private_key: String,
    /// Dollars each session may use.
    pub budget: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveOrder {
    pub t_ms: i64,
    pub action: Action,
    pub wanted: u32,
    /// The YES-axis limit sent.
    pub yes_limit: String,
    pub filled: u32,
    /// Average price per contract on the side bought or sold.
    pub price: Option<f64>,
    pub fee: f64,
    pub status: String,
    pub error: Option<String>,
    pub order_id: Option<String>,
    /// False when Kalshi took the order but what filled couldn't be read
    /// back: the account is then unknown, so trading stops.
    pub confirmed: bool,
}

/// One session of a run, with its own account.
#[derive(Debug, Clone, Serialize)]
pub struct LiveSession {
    pub ticker: String,
    pub strike: f64,
    pub open_ms: i64,
    pub close_ms: i64,
    /// trading, closed, settled, skipped
    pub state: String,
    pub position: Position,
    pub fees: f64,
    pub orders: Vec<LiveOrder>,
    pub result: Option<String>,
    /// Settled P&L against the budget.
    pub pnl: Option<f64>,
}

/// What the page shows; one per run.
#[derive(Debug, Clone, Serialize)]
pub struct LiveStatus {
    pub strategy: String,
    /// Per session.
    pub budget: f64,
    /// starting, running, stopped, failed
    pub state: String,
    pub started_ms: i64,
    /// The session being traded now, if any.
    pub current: Option<String>,
    pub sessions: Vec<LiveSession>,
    pub last_decision: Option<Decision>,
    /// Sum of settled sessions' P&L.
    pub total_pnl: f64,
    pub log: Vec<String>,
}

impl LiveStatus {
    fn note(&mut self, msg: impl Into<String>) {
        let msg = msg.into();
        info!("live: {msg}");
        self.log.push(format!("{} {msg}", chrono::Local::now().format("%H:%M:%S")));
        if self.log.len() > 300 {
            self.log.remove(0);
        }
    }

    fn done(&self) -> bool {
        matches!(self.state.as_str(), "stopped" | "failed")
    }

    fn session(&mut self, ticker: &str) -> Option<&mut LiveSession> {
        self.sessions.iter_mut().find(|s| s.ticker == ticker)
    }
}

/// A running (or finished) live run.
pub struct LiveRun {
    pub status: Arc<Mutex<LiveStatus>>,
    stop: watch::Sender<bool>,
}

impl LiveRun {
    pub fn is_done(&self) -> bool {
        self.status.lock().unwrap().done()
    }

    pub fn stop(&self) {
        let _ = self.stop.send(true);
    }
}

/// Validates the request, then runs it in the background.
pub fn start(store: Store, req: LiveRequest) -> Result<LiveRun> {
    if !(req.budget.is_finite() && req.budget >= 1.0) {
        bail!("budget must be at least $1");
    }
    let signer = Arc::new(Signer::from_pem(req.key_id.trim(), &req.private_key).context("reading the private key")?);
    let mut strategy = make_strategy(&req.strategy, TradeLimits::default(), 1, 0.03)?;
    let warmed = crate::paper::warm_up(&store, strategy.as_mut());
    let mut status = LiveStatus {
        strategy: req.strategy.clone(),
        budget: req.budget,
        state: "starting".into(),
        started_ms: now_ms(),
        current: None,
        sessions: vec![],
        last_decision: None,
        total_pnl: 0.0,
        log: vec![],
    };
    status.note(format!("strategy {} warmed up with {warmed}s of BRTI", req.strategy));
    let status = Arc::new(Mutex::new(status));
    let (stop, stop_rx) = watch::channel(false);
    let st = status.clone();
    tokio::spawn(async move {
        let r = run(store, signer, req, strategy, st.clone(), stop_rx).await;
        let mut s = st.lock().unwrap();
        if let Err(e) = r {
            s.note(format!("stopped on error: {e:#}"));
            s.state = "failed".into();
        }
    });
    Ok(LiveRun { status, stop })
}

/// Kalshi's tapered tick ladder from a market's `price_ranges`.
fn tick_size(price: f64, ranges: &Value) -> f64 {
    ranges
        .as_array()
        .into_iter()
        .flatten()
        .find_map(|r| {
            let (lo, hi, step) = (parse_num(&r["start"])?, parse_num(&r["end"])?, parse_num(&r["step"])?);
            (lo <= price && price < hi).then_some(step)
        })
        .unwrap_or(0.01)
}

/// A YES-axis limit `tol` through `touch` (up when `up`), snapped away from
/// the touch onto the ladder and kept inside (0, 1).
fn yes_limit(touch: f64, up: bool, ranges: &Value) -> f64 {
    let target = (touch + if up { TOLERANCE } else { -TOLERANCE }).clamp(0.001, 0.999);
    let step = tick_size(target, ranges);
    let n = target / step;
    let n = if up { (n - 1e-9).ceil() } else { (n + 1e-9).floor() };
    (n * step).clamp(0.001, 0.999)
}

/// How a decision goes on the wire: (book side, YES-axis limit, reduce only,
/// worst price per contract on the side traded).
fn order_terms(action: Action, s: &Snapshot, ranges: &Value) -> Option<(&'static str, f64, bool, f64)> {
    let ok = |p: Option<f64>| p.filter(|p| *p > 0.0 && *p < 1.0);
    Some(match action {
        Action::BuyYes => {
            let y = yes_limit(ok(s.yes.ask)?, true, ranges);
            ("bid", y, false, y)
        }
        // Buying NO at q is selling YES at 1 - q.
        Action::BuyNo => {
            let y = yes_limit(1.0 - ok(s.no.ask)?, false, ranges);
            ("ask", y, false, 1.0 - y)
        }
        Action::SellYes => {
            let y = yes_limit(ok(s.yes.bid)?, false, ranges);
            ("ask", y, true, y)
        }
        Action::SellNo => {
            let y = yes_limit(1.0 - ok(s.no.bid)?, true, ranges);
            ("bid", y, true, 1.0 - y)
        }
        Action::Nothing => return None,
    })
}

struct Ctx {
    rest: Rest,
    status: Arc<Mutex<LiveStatus>>,
    log: Option<std::fs::File>,
}

impl Ctx {
    fn record(&mut self, v: Value) {
        if let Some(f) = self.log.as_mut() {
            let _ = writeln!(f, "{v}");
        }
    }

    fn note(&self, msg: impl Into<String>) {
        self.status.lock().unwrap().note(msg);
    }
}

/// Makes sure the market's shard holds at least `budget`, moving the
/// shortfall from shard 0 if that covers it.
async fn fund(cx: &Ctx, shard: i64, budget: f64) -> Result<()> {
    let shards = cx.rest.shard_balances().await.context("reading balance (is the key right?)")?;
    let have = |i: i64| shards.iter().find(|(s, _)| *s == i).map(|(_, b)| *b).unwrap_or(0.0);
    let on_shard = have(shard);
    let total: f64 = shards.iter().map(|(_, b)| b).sum();
    cx.note(format!("balance ${total:.2}, ${on_shard:.2} on shard {shard} where these markets trade"));
    if on_shard + 1e-9 >= budget {
        return Ok(());
    }
    let short = ((budget - on_shard) * 100.0).ceil() / 100.0;
    if shard == 0 || have(0) + 1e-9 < short {
        bail!("budget ${budget:.2} needs ${short:.2} more on shard {shard}, and shard 0 has ${:.2}", have(0));
    }
    cx.rest.transfer(short, 0, shard).await.context("moving cash to the market's shard")?;
    cx.note(format!("moved ${short:.2} from shard 0 to shard {shard}"));
    Ok(())
}

/// The session being traded: its window plus what orders need.
struct Active {
    window: Window,
    shard: i64,
    ranges: Value,
}

async fn run(
    store: Store,
    signer: Arc<Signer>,
    req: LiveRequest,
    mut strategy: Box<dyn Strategy>,
    status: Arc<Mutex<LiveStatus>>,
    mut stop: watch::Receiver<bool>,
) -> Result<()> {
    let rest = Rest::new(signer.clone())?;
    let log_path = store.root.join("live").join(format!("{}.ndjson", chrono::Utc::now().format("%Y-%m-%d")));
    let log = std::fs::create_dir_all(log_path.parent().unwrap())
        .and_then(|_| std::fs::OpenOptions::new().create(true).append(true).open(&log_path))
        .ok();
    let mut cx = Ctx { rest: rest.clone(), status: status.clone(), log };

    // Fail fast on a bad key, and never trade windows before the one picked.
    rest.shard_balances().await.context("reading balance (is the key right?)")?;
    let m = rest.market(&req.ticker).await.context("looking up the market")?;
    let first_close = kalshi::parse_time_ms(&m["close_time"]).context("market has no close time")?;
    if first_close <= now_ms() {
        bail!("{} has already closed", req.ticker);
    }
    cx.record(json!({ "type": "start", "ticker": req.ticker, "strategy": req.strategy, "budget": req.budget, "t_ms": now_ms() }));

    let (tx, mut rx) = mpsc::unbounded_channel();
    let tasks = feeds::spawn_all(signer, rest.clone(), tx, HashMap::new());
    let (fill_tx, mut fill_rx) = mpsc::unbounded_channel::<(String, LiveOrder)>();

    let mut asm = Assembler::default();
    asm.clear_quotes();
    let mut active: Option<Active> = None;
    // Windows already begun (or skipped), so a re-listed one isn't begun twice.
    let mut seen: Vec<String> = vec![];
    let mut quote_ms = i64::MIN;
    let mut in_flight = false;
    let mut retry_after = i64::MIN;
    {
        let mut s = status.lock().unwrap();
        s.state = "running".into();
        s.note(format!("running from {} on; ${:.2} per session", req.ticker, req.budget));
    }

    let start = tokio::time::Instant::now() + Duration::from_millis((1000 - now_ms().rem_euclid(1000)) as u64);
    let mut tick = tokio::time::interval_at(start, Duration::from_secs(1));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let outcome = loop {
        tokio::select! {
            _ = stop.changed() => break "stopped",
            msg = rx.recv() => {
                let Some(msg) = msg else { break "failed" };
                match msg {
                    Feed::Btc { raw, .. } => {
                        if let Some((_, v)) = serde_json::from_str(&raw).ok().as_ref().and_then(brti_tick) {
                            asm.btc(v);
                        }
                    }
                    Feed::Market { stream, ticker, recv_ns, raw } if active.as_ref().is_some_and(|a| a.window.ticker == ticker) => {
                        let v: Value = serde_json::from_str(&raw).unwrap_or(Value::Null);
                        if let Some((yes, no)) = parse_quotes(stream, &v) {
                            asm.quotes(recv_ns / 1_000_000, yes, no);
                            quote_ms = recv_ns / 1_000_000;
                        }
                    }
                    Feed::Upsert(m) => {
                        let Some(ticker) = m["ticker"].as_str().map(String::from) else { continue };
                        if active.as_ref().is_some_and(|a| a.window.ticker == ticker) {
                            if let Some((yes, no)) = parse_quotes("market", &m) {
                                let t = now_ms();
                                asm.quotes(t, yes, no);
                                quote_ms = t;
                            }
                            continue;
                        }
                        let now = now_ms();
                        let (Some(open), Some(close)) = (kalshi::parse_time_ms(&m["open_time"]), kalshi::parse_time_ms(&m["close_time"])) else { continue };
                        // A new window is open and has its strike: it becomes the session.
                        let Some(strike) = m["floor_strike"].as_f64() else { continue };
                        if !(open <= now && now < close) || close < first_close || seen.contains(&ticker) || active.is_some() {
                            continue;
                        }
                        seen.push(ticker.clone());
                        let shard = m["exchange_index"].as_i64().unwrap_or(0);
                        let mut session = LiveSession {
                            ticker: ticker.clone(),
                            strike,
                            open_ms: open,
                            close_ms: close,
                            state: "trading".into(),
                            position: Position::with_cash(req.budget),
                            fees: 0.0,
                            orders: vec![],
                            result: None,
                            pnl: None,
                        };
                        if let Err(e) = fund(&cx, shard, req.budget).await {
                            session.state = "skipped".into();
                            let mut s = status.lock().unwrap();
                            s.note(format!("skipping {ticker}: {e:#}"));
                            s.sessions.push(session);
                            continue;
                        }
                        let w = Window { ticker: ticker.clone(), strike, open_ms: open, close_ms: close };
                        strategy.begin_window(&w);
                        asm.clear_quotes();
                        quote_ms = i64::MIN;
                        retry_after = i64::MIN;
                        active = Some(Active { window: w, shard, ranges: m["price_ranges"].clone() });
                        let mut s = status.lock().unwrap();
                        s.note(format!("trading {ticker}: strike {strike:.2}, ${:.2} budget", req.budget));
                        s.current = Some(ticker.clone());
                        s.sessions.push(session);
                        drop(s);
                        cx.record(json!({ "type": "session", "ticker": ticker, "budget": req.budget, "t_ms": now }));
                    }
                    Feed::Settled(m) => {
                        let (Some(ticker), Some(result)) = (m["ticker"].as_str(), m["result"].as_str()) else { continue };
                        let mut s = status.lock().unwrap();
                        let budget = s.budget;
                        let Some(sess) = s.session(ticker).filter(|x| x.result.is_none() && x.state != "skipped") else { continue };
                        let payout = match result { "yes" => sess.position.yes, "no" => sess.position.no, _ => 0 } as f64;
                        let pnl = sess.position.cash + payout - budget;
                        sess.result = Some(result.into());
                        sess.pnl = Some(pnl);
                        sess.state = "settled".into();
                        let fees = sess.fees;
                        s.total_pnl += pnl;
                        let total = s.total_pnl;
                        s.note(format!("{ticker} settled {}: P&L ${pnl:+.2} (fees ${fees:.2}); run total ${total:+.2}", result.to_uppercase()));
                        drop(s);
                        cx.record(json!({ "type": "settle", "ticker": ticker, "result": result, "pnl": pnl, "running_total": total }));
                    }
                    Feed::Gap { streams, from_ms, to_ms, .. } => cx.note(format!("feed gap in {streams:?}: {}s", (to_ms - from_ms) / 1000)),
                    _ => {}
                }
            }
            order = fill_rx.recv() => {
                let Some((ticker, o)) = order else { continue };
                in_flight = false;
                let mut s = status.lock().unwrap();
                let Some(sess) = s.session(&ticker) else { continue };
                let msg = if let Some(px) = o.price.filter(|_| o.filled > 0) {
                    apply_fill(&mut sess.position, o.action, o.filled, px, o.fee);
                    sess.fees += o.fee;
                    format!("{ticker} {:?}: filled {}/{} at {px:.3}, fee ${:.2}", o.action, o.filled, o.wanted, o.fee)
                } else {
                    retry_after = now_ms() + RETRY_MS;
                    format!("{ticker} {:?} x{}: {}", o.action, o.wanted, o.error.clone().unwrap_or_else(|| format!("nothing filled ({})", o.status)))
                };
                let pos = sess.position;
                sess.orders.push(o.clone());
                s.note(msg);
                if !o.confirmed {
                    s.note(format!(
                        "stopping: order {} may have filled, so the account is unknown. Check the position on Kalshi",
                        o.order_id.as_deref().unwrap_or("?")
                    ));
                }
                drop(s);
                cx.record(json!({ "type": "order", "ticker": ticker, "order": o, "position": pos }));
                if !o.confirmed {
                    break "stopped";
                }
            }
            _ = tick.tick() => {
                let t = now_ms();
                let t = t - t.rem_euclid(1000);
                if let Some(a) = active.as_ref().filter(|a| t >= a.window.close_ms) {
                    let ticker = a.window.ticker.clone();
                    active = None;
                    let mut s = status.lock().unwrap();
                    if let Some(sess) = s.session(&ticker) {
                        sess.state = "closed".into();
                    }
                    s.current = None;
                    s.note(format!("{ticker} closed; waiting for its result and the next session"));
                }
                let Some(snap) = asm.snapshot(t) else { continue };
                let Some(a) = active.as_ref() else {
                    // Keeps BRTI history going between sessions.
                    strategy.on_tick(&snap, &Position::default());
                    continue;
                };
                let ticker = a.window.ticker.clone();
                let pos = status.lock().unwrap().session(&ticker).map(|x| x.position).unwrap_or_default();
                let d = strategy.on_tick(&snap, &pos);
                status.lock().unwrap().last_decision = Some(d);
                if d.action == Action::Nothing || d.size == 0 || in_flight || t < retry_after {
                    continue;
                }
                if t - quote_ms > QUOTE_STALE_MS {
                    cx.note(format!("skipped {:?}: quotes are {}s old", d.action, (t - quote_ms).min(99_000) / 1000));
                    retry_after = t + RETRY_MS;
                    continue;
                }
                let Some((side, yes_px, reduce, worst)) = order_terms(d.action, &snap, &a.ranges) else { continue };
                // Price the budget check at the worst fill the limit allows.
                let count = if d.action.is_buy() {
                    d.size.min((pos.cash / (worst + fee_per_contract(worst, TAKER_RATE))).floor().max(0.0) as u32)
                } else {
                    d.size
                };
                if count == 0 {
                    continue;
                }
                in_flight = true;
                let (rest, tx, action, shard) = (rest.clone(), fill_tx.clone(), d.action, a.shard);
                tokio::spawn(async move {
                    let o = send(&rest, &ticker, action, count, side, yes_px, reduce, shard).await;
                    let _ = tx.send((ticker, o));
                });
            }
        }
    };
    for t in tasks {
        t.abort();
    }
    let mut s = status.lock().unwrap();
    let open: Vec<String> = s
        .sessions
        .iter()
        .filter(|x| x.result.is_none() && x.position.held() > 0)
        .map(|x| format!("{} (YES {} / NO {})", x.ticker, x.position.yes, x.position.no))
        .collect();
    if open.is_empty() {
        s.note("stopped");
    } else {
        s.note(format!("stopped holding {}; Kalshi settles it", open.join(", ")));
    }
    s.state = outcome.into();
    s.current = None;
    drop(s);
    cx.record(json!({ "type": "end", "state": outcome, "t_ms": now_ms() }));
    Ok(())
}

/// Places one order and reads back what filled. Never fails: errors come
/// back in the order record. An order Kalshi rejected is confirmed with
/// nothing filled; one it accepted is confirmed only once its fills are read.
#[allow(clippy::too_many_arguments)]
async fn send(rest: &Rest, ticker: &str, action: Action, count: u32, side: &str, yes_px: f64, reduce: bool, shard: i64) -> LiveOrder {
    let yes_limit = format!("{yes_px:.4}");
    let mut o = LiveOrder {
        t_ms: now_ms(),
        action,
        wanted: count,
        yes_limit: yes_limit.clone(),
        filled: 0,
        price: None,
        fee: 0.0,
        status: "error".into(),
        error: None,
        order_id: None,
        confirmed: true,
    };
    let placed = match rest.place_ioc(ticker, side, count, &yes_limit, Some(shard), reduce, &uuid_v4()).await {
        Ok(v) => v,
        Err(e) => {
            o.error = Some(format!("{e:#}"));
            return o;
        }
    };
    info!("order placed: {placed}");
    let placed = placed.get("order").cloned().unwrap_or(placed);
    o.order_id = placed["order_id"].as_str().map(String::from);
    o.status = placed["status"].as_str().unwrap_or("placed").into();
    match read_fills(rest, &placed, o.order_id.as_deref()).await {
        Some((filled, cost, fee)) => {
            o.filled = filled;
            o.fee = fee;
            if filled > 0 {
                let px = cost / filled as f64;
                if px > 0.0 && px < 1.0 {
                    o.price = Some(px);
                } else {
                    o.confirmed = false;
                    o.error = Some(format!("filled {filled} for ${cost:.4}, which doesn't make sense"));
                }
            }
        }
        None => {
            o.confirmed = false;
            o.error = Some("Kalshi accepted the order but its fills couldn't be read back".into());
        }
    }
    o
}

/// (contracts, cost on the side traded, fees) of a placed order: from the
/// create response when it says, else the order record (which can 404 for a
/// moment after creation), else the fills list. `None` if nothing answers.
async fn read_fills(rest: &Rest, placed: &Value, order_id: Option<&str>) -> Option<(u32, f64, f64)> {
    let from_order = |v: &Value| -> Option<(u32, f64, f64)> {
        let n = |k: &str| parse_num(&v[k]);
        let filled = n("fill_count_fp")?;
        // A terminal IOC order has no remainder resting; anything else may still change.
        if n("remaining_count_fp").unwrap_or(0.0) > 0.0 && v["status"] == "resting" {
            return None;
        }
        let cost = n("taker_fill_cost_dollars").unwrap_or(0.0) + n("maker_fill_cost_dollars").unwrap_or(0.0);
        let fee = n("taker_fees_dollars").unwrap_or(0.0) + n("maker_fees_dollars").unwrap_or(0.0);
        // Fill counts without the cost fields aren't enough to account for.
        if filled > 0.0 && cost == 0.0 {
            return None;
        }
        Some((filled.round() as u32, cost, fee))
    };
    if let Some(r) = from_order(placed) {
        return Some(r);
    }
    let id = order_id?;
    for attempt in 0..8 {
        tokio::time::sleep(Duration::from_millis(250 * (attempt + 1))).await;
        match rest.order(id).await {
            Ok(v) => {
                if let Some(r) = from_order(&v) {
                    return Some(r);
                }
            }
            Err(e) => warn!("reading order {id} (attempt {}): {e:#}", attempt + 1),
        }
        match rest.fills_for_order(id).await {
            Ok(fills) if !fills.is_empty() => return Some(sum_fills(&fills)),
            Ok(_) => {}
            Err(e) => warn!("reading fills of {id}: {e:#}"),
        }
    }
    None
}

/// Totals of fills, priced on the side each one bought or sold.
fn sum_fills(fills: &[Value]) -> (u32, f64, f64) {
    let (mut n, mut cost, mut fee) = (0.0, 0.0, 0.0);
    for f in fills {
        let count = parse_num(&f["count_fp"]).unwrap_or(0.0);
        let px = if f["outcome_side"] == "no" { &f["no_price_dollars"] } else { &f["yes_price_dollars"] };
        n += count;
        cost += count * parse_num(px).unwrap_or(0.0);
        fee += parse_num(&f["fee_cost"]).unwrap_or(0.0);
    }
    (n.round() as u32, cost, fee)
}

/// A random UUID, as Kalshi's `client_order_id` expects.
fn uuid_v4() -> String {
    let b = (rand::random::<u128>() & !(0xf << 76) & !(0x3 << 62)) | (0x4 << 76) | (0x2 << 62);
    let h = format!("{b:032x}");
    format!("{}-{}-{}-{}-{}", &h[..8], &h[8..12], &h[12..16], &h[16..20], &h[20..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use strategy::Quote;

    #[test]
    fn limits_cross_by_a_cent_on_the_right_axis_and_snap_away_from_the_touch() {
        let ranges = json!([
            { "start": "0.0000", "end": "0.1000", "step": "0.0010" },
            { "start": "0.1000", "end": "0.9000", "step": "0.0100" },
            { "start": "0.9000", "end": "1.0000", "step": "0.0010" },
        ]);
        let s = Snapshot {
            t_ms: 0,
            btc: 1.0,
            yes: Quote { bid: Some(0.60), ask: Some(0.62), ..Default::default() },
            no: Quote { bid: Some(0.38), ask: Some(0.40), ..Default::default() },
        };
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        let (side, y, reduce, worst) = order_terms(Action::BuyYes, &s, &ranges).unwrap();
        assert!(side == "bid" && !reduce && close(y, 0.63) && close(worst, 0.63));
        // NO ask 0.40 is YES bid 0.60; a cent worse for us is selling YES at 0.59.
        let (side, y, reduce, worst) = order_terms(Action::BuyNo, &s, &ranges).unwrap();
        assert!(side == "ask" && !reduce && close(y, 0.59) && close(worst, 0.41));
        let (side, y, reduce, _) = order_terms(Action::SellYes, &s, &ranges).unwrap();
        assert!(side == "ask" && reduce && close(y, 0.59));
        let (side, y, reduce, worst) = order_terms(Action::SellNo, &s, &ranges).unwrap();
        assert!(side == "bid" && reduce && close(y, 0.63) && close(worst, 0.37));
        // Deci-cent ticks near the edges; never snapped back toward the touch.
        assert!(close(yes_limit(0.955, true, &ranges), 0.965));
        assert!(close(yes_limit(0.0951, false, &ranges), 0.085));
        assert!(close(yes_limit(0.995, true, &ranges), 0.999));
    }

    #[test]
    fn fills_are_priced_on_the_side_traded() {
        // Kalshi's record of a real sell-NO order: 30 at NO 0.81 cost $24.30.
        let fills = [json!({ "count_fp": "30.00", "outcome_side": "no", "no_price_dollars": "0.8100", "yes_price_dollars": "0.1900", "fee_cost": "0.323200" })];
        let (n, cost, fee) = sum_fills(&fills);
        assert!(n == 30 && (cost - 24.3).abs() < 1e-9 && (fee - 0.3232).abs() < 1e-9);
    }
}

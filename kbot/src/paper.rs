//! `kbot paper`: the strategy on live Kalshi data, with simulated fills.
//!
//! Snapshots are assembled once a second exactly as replay builds them, fed
//! to the `Strategy`, and any decision is filled one second later at the
//! touch (the backtest's default latency) with Kalshi's fees. Each window
//! starts with `--balance` dollars and is settled from Kalshi's own result.
//! No orders are ever sent.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::Args;
use kalshi::{now_ms, Rest};
use market::replay::{brti_tick, parse_quotes, Assembler};
use market::{Store, DEFAULT_BALANCE};
use serde_json::{json, Value};
use strategy::fees::{taker_fee, TAKER_RATE};
use strategy::{apply_fill, fill_size, Action, Position, Snapshot, Strategy, TradeLimits, Window};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::backtest::make_strategy;
use crate::feeds::{self, Feed};

#[derive(Args, Clone)]
pub struct PaperArgs {
    /// `fair`, `rule`, `rule-swing:<trunk dir>[@<bps>]`, `nothing`, or a model directory.
    #[arg(long)]
    pub strategy: String,
    /// fair: minimum edge beyond fees.
    #[arg(long, default_value_t = 0.03)]
    pub edge: f64,
    /// fair: seconds between decisions.
    #[arg(long, default_value_t = 1)]
    pub every: u32,
    /// Starting cash for every window, in dollars.
    #[arg(long, default_value_t = DEFAULT_BALANCE)]
    pub balance: f64,
}

/// One window's simulated account.
struct Book {
    /// Contracts and cash.
    pos: Position,
    fees: f64,
    fills: u32,
}

struct Paper {
    strategy: Box<dyn Strategy>,
    limits: TradeLimits,
    balance: f64,
    asm: Assembler,
    window: Option<Window>,
    books: HashMap<String, Book>,
    /// Decided last second, filled at this second's touch.
    pending: Option<(String, Action, u32)>,
    total: f64,
    settled: u32,
    log: std::fs::File,
}

impl Paper {
    fn record(&mut self, v: Value) {
        let _ = writeln!(self.log, "{v}");
    }

    fn on_market(&mut self, m: &Value) {
        let now = now_ms();
        let (Some(ticker), Some(open), Some(close)) =
            (m["ticker"].as_str(), kalshi::parse_time_ms(&m["open_time"]), kalshi::parse_time_ms(&m["close_time"]))
        else {
            return;
        };
        if !(open..close).contains(&now) {
            return;
        }
        if self.window.as_ref().is_none_or(|w| w.ticker != ticker) {
            // The strike is published at open; wait for it.
            let Some(strike) = m["floor_strike"].as_f64() else { return };
            let w = Window { ticker: ticker.into(), strike, open_ms: open, close_ms: close };
            info!("window {ticker}: strike {strike:.2}, closes {}", m["close_time"]);
            self.strategy.begin_window(&w);
            self.asm.clear_quotes();
            self.pending = None;
            let balance = self.balance;
            self.books.entry(ticker.into()).or_insert_with(|| Book { pos: Position::with_cash(balance), fees: 0.0, fills: 0 });
            self.window = Some(w);
        }
        if let Some((yes, no)) = parse_quotes("market", m) {
            self.asm.quotes(now, yes, no);
        }
    }

    fn on_tick(&mut self, t_ms: i64) {
        let Some(snap) = self.asm.snapshot(t_ms) else { return };
        let Some(w) = self.window.clone() else {
            // Still builds BRTI history for when a window starts.
            self.strategy.on_tick(&snap, &Position::default());
            return;
        };
        if t_ms >= w.close_ms {
            self.window = None;
            return;
        }
        if let Some((ticker, action, size)) = self.pending.take()
            && ticker == w.ticker {
                self.fill(&ticker, action, size, &snap);
            }
        let pos = self.books.get(&w.ticker).map(|b| b.pos).unwrap_or_default();
        let d = self.strategy.on_tick(&snap, &pos);
        if d.action != Action::Nothing {
            info!(
                "{} {:?} x{} @ {:?}  fair {:.3}  yes {:?}/{:?}  btc {:.2}",
                w.ticker, d.action, d.size, d.limit, d.fair_value, snap.yes.bid, snap.yes.ask, snap.btc
            );
            self.record(json!({ "type": "decision", "t_ms": t_ms, "ticker": w.ticker, "decision": d, "snapshot": snap }));
            self.pending = Some((w.ticker.clone(), d.action, d.size));
        }
    }

    fn fill(&mut self, ticker: &str, action: Action, wanted: u32, at: &Snapshot) {
        let Some(book) = self.books.get_mut(ticker) else { return };
        let (px, touch) = match action {
            Action::BuyYes => (at.yes.ask, at.yes.ask_size),
            Action::SellYes => (at.yes.bid, at.yes.bid_size),
            Action::BuyNo => (at.no.ask, at.no.ask_size),
            Action::SellNo => (at.no.bid, at.no.bid_size),
            Action::Nothing => return,
        };
        let Some(px) = px.filter(|p| *p > 0.0 && *p < 1.0) else { return };
        let n = fill_size(&book.pos, action, wanted, px, touch, &self.limits, TAKER_RATE);
        if n == 0 {
            return;
        }
        let fee = taker_fee(n, px, TAKER_RATE);
        book.fees += fee;
        book.fills += 1;
        apply_fill(&mut book.pos, action, n, px, fee);
        let pos = book.pos;
        self.record(json!({ "type": "fill", "t_ms": at.t_ms, "ticker": ticker, "action": action, "size": n, "price": px, "fee": fee, "position": pos }));
    }

    fn on_settled(&mut self, m: &Value) {
        let (Some(ticker), Some(result)) = (m["ticker"].as_str(), m["result"].as_str()) else { return };
        let Some(book) = self.books.remove(ticker) else { return };
        let payout = if result == "yes" { book.pos.yes } else { book.pos.no } as f64;
        let pnl = book.pos.cash + payout - self.balance;
        self.total += pnl;
        self.settled += 1;
        info!(
            "SETTLED {ticker} -> {result}: pnl ${pnl:+.2} (fees ${:.2}, {} fills)  running ${:+.2} over {} windows",
            book.fees, book.fills, self.total, self.settled
        );
        self.record(json!({ "type": "settle", "ticker": ticker, "result": result, "pnl": pnl, "fees": book.fees, "fills": book.fills, "running_total": self.total }));
    }
}

/// Recent BRTI from the recorder's tape, so volatility features are warm.
pub(crate) fn warm_up(store: &Store, strategy: &mut dyn Strategy) -> usize {
    let now = now_ms();
    let mut ticks: Vec<(i64, f64)> = vec![];
    let mut paths = vec![store.brti_path(now - feeds::RING_MS), store.brti_path(now)];
    paths.dedup();
    for p in paths {
        for line in market::store::read_lines(&p).unwrap_or_default() {
            if let Some((_, v)) = brti_tick(&line.value()) {
                ticks.push((line.recv_ms(), v));
            }
        }
    }
    ticks.sort_by_key(|(t, _)| *t);
    let Some(&(first, _)) = ticks.first() else { return 0 };
    let (mut k, mut btc, mut n) = (0, 0.0, 0);
    let mut t = first - first.rem_euclid(1000) + 1000;
    while t < now - 1000 {
        while k < ticks.len() && ticks[k].0 <= t {
            btc = ticks[k].1;
            k += 1;
        }
        strategy.on_tick(&Snapshot { t_ms: t, btc, ..Default::default() }, &Position::default());
        n += 1;
        t += 1000;
    }
    n
}

pub async fn run(config: kalshi::Config, args: PaperArgs) -> Result<()> {
    let signer = Arc::new(config.signer()?);
    let rest = Rest::new(signer.clone())?;
    let store = Store::new(&config.data_dir);
    let mut strategy = make_strategy(&args.strategy, TradeLimits::default(), args.every, args.edge)?;
    let warmed = warm_up(&store, strategy.as_mut());
    info!("paper trading {:?}; warmed up with {warmed}s of BRTI from the recorder's tape", args.strategy);

    let log_path: PathBuf = config.data_dir.join("paper").join(format!("{}.ndjson", chrono::Utc::now().format("%Y-%m-%d")));
    std::fs::create_dir_all(log_path.parent().unwrap())?;
    let log = std::fs::OpenOptions::new().create(true).append(true).open(&log_path)?;
    info!("logging decisions, fills and settlements to {}", log_path.display());

    let limits = strategy.limits();
    let mut p = Paper {
        strategy,
        limits,
        balance: args.balance,
        asm: Assembler::default(),
        window: None,
        books: HashMap::new(),
        pending: None,
        total: 0.0,
        settled: 0,
        log,
    };
    p.asm.clear_quotes();
    p.record(json!({ "type": "start", "strategy": args.strategy, "balance": args.balance, "t_ms": now_ms() }));

    let (tx, mut rx) = mpsc::unbounded_channel();
    feeds::spawn_all(signer, rest, tx, HashMap::new());

    // Ticks on whole seconds, matching the replay grid.
    let start = tokio::time::Instant::now() + Duration::from_millis((1000 - now_ms().rem_euclid(1000)) as u64);
    let mut tick = tokio::time::interval_at(start, Duration::from_secs(1));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            msg = rx.recv() => {
                let Some(msg) = msg else { break };
                match msg {
                    Feed::Btc { raw, .. } => {
                        if let Some((_, v)) = serde_json::from_str(&raw).ok().as_ref().and_then(brti_tick) {
                            p.asm.btc(v);
                        }
                    }
                    Feed::Market { stream, ticker, recv_ns, raw } => {
                        if p.window.as_ref().is_some_and(|w| w.ticker == ticker) {
                            let v: Value = serde_json::from_str(&raw).unwrap_or(Value::Null);
                            if let Some((yes, no)) = parse_quotes(stream, &v) {
                                p.asm.quotes(recv_ns / 1_000_000, yes, no);
                            }
                        }
                    }
                    Feed::Upsert(m) => p.on_market(&m),
                    Feed::Settled(m) => p.on_settled(&m),
                    Feed::Gap { streams, from_ms, to_ms, .. } => warn!("feed gap in {streams:?}: {}s", (to_ms - from_ms) / 1000),
                    Feed::Shutdown => break,
                }
            }
            _ = tick.tick() => {
                let t = now_ms();
                p.on_tick(t - t.rem_euclid(1000));
            }
            _ = tokio::signal::ctrl_c() => break,
        }
    }
    info!("stopped; running pnl ${:+.2} over {} settled windows", p.total, p.settled);
    Ok(())
}

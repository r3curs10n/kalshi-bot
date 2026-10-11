//! `kbot calibrate`: is the favorite priced right when the rule buys it, does
//! the fair-value cost check pick the better bets, would a stop-loss have
//! paid, and which stop / entry cap backtests best.
//!
//! Everything is per contract and before slippage, so the first two tables
//! answer "is there an edge here at all" independently of sizing.

use anyhow::{bail, Result};
use clap::Args;
use market::replay::episode_dirs;
use market::{run_strategy, Episode, SimConfig, Store, DEFAULT_BALANCE};
use strategy::fees::{fee_per_contract, TAKER_RATE};
use strategy::fair::fair_value;
use strategy::rules::{favorite, ENTRY_SLIP};
use strategy::{Action, History, Snapshot, TradeLimits, RULE_LAST_S};

use crate::backtest::{make_strategy, print_summary, summarize};

#[derive(Args, Clone)]
pub struct CalibrateArgs {
    /// First UTC date (YYYY-MM-DD) of episodes to use.
    #[arg(long)]
    pub from: Option<String>,
    /// Last UTC date of episodes to use.
    #[arg(long)]
    pub to: Option<String>,
    /// Strategy the sweep adds stop/cap options to: `rule` or `rule-swing:<trunk dir>[@<bps>]`,
    /// optionally with options already, e.g. `rule,edge=0.02`.
    #[arg(long, default_value = "rule")]
    pub base: String,
    /// Stop levels (held side's bid, dollars) for the drop table and the sweep.
    #[arg(long, value_delimiter = ',', default_values_t = [0.70, 0.60, 0.50, 0.40, 0.30, 0.20])]
    pub stops: Vec<f64>,
    /// Entry caps (favorite's ask, dollars) for the sweep.
    #[arg(long, value_delimiter = ',', default_values_t = [0.95, 0.90, 0.85, 0.80])]
    pub caps: Vec<f64>,
    #[arg(long, default_value_t = 1)]
    pub latency: u32,
    #[arg(long, default_value_t = 0.0)]
    pub slip: f64,
    /// Starting cash for every session, in dollars.
    #[arg(long, default_value_t = DEFAULT_BALANCE)]
    pub balance: f64,
}

/// Where the rule would have bought in one episode.
#[derive(Clone, Copy)]
struct Entry {
    /// Index of the snapshot the rule decides on.
    idx: usize,
    side: Action,
    ask: f64,
    won: bool,
    /// The fair value's chance of `side` winning, if it had enough data.
    p_model: Option<f64>,
}

fn bid(s: &Snapshot, side: Action) -> Option<f64> {
    if side == Action::BuyYes { s.yes.bid } else { s.no.bid }
}

fn ask(s: &Snapshot, side: Action) -> Option<f64> {
    if side == Action::BuyYes { s.yes.ask } else { s.no.ask }
}

/// The first snapshot inside the last `RULE_LAST_S` seconds with a favorite
/// and an ask to buy it at.
fn entry(ep: &Episode) -> Option<Entry> {
    let from = ep.window.close_ms - (RULE_LAST_S * 1000.0) as i64;
    let (idx, side, ask) =
        ep.snapshots.iter().enumerate().filter(|(_, s)| s.t_ms >= from && s.t_ms < ep.window.close_ms).find_map(|(idx, s)| {
            let side = favorite(s)?;
            Some((idx, side, ask(s, side).filter(|a| *a > 0.0 && *a < 1.0)?))
        })?;
    let mut hist = History::default();
    for s in &ep.snapshots[..=idx] {
        hist.push(*s);
    }
    let p_model = fair_value(&ep.window, &hist).map(|f| if side == Action::BuyYes { f.p } else { 1.0 - f.p });
    Some(Entry { idx, side, ask, won: (side == Action::BuyYes) == ep.result_yes, p_model })
}

/// Profit per contract held to settlement after the taker fee.
fn realized(e: &Entry) -> f64 {
    e.won as u8 as f64 - e.ask - fee_per_contract(e.ask, TAKER_RATE)
}

/// What the cost check expected per contract: the fair value's win chance
/// against the worst fill plus its fee, as `RuleOpts::covers_cost` prices it.
fn expected(e: &Entry) -> Option<f64> {
    let worst = (e.ask + ENTRY_SLIP).min(0.999);
    Some(e.p_model? - worst - fee_per_contract(worst, TAKER_RATE))
}

/// Entries the cost check would take against the ones it would skip, and how
/// each did: the check only helps if the model beats the market.
fn cost_table(entries: &[Entry]) {
    println!("\n== Cost check: does the fair value's expected edge show up in results? ==");
    println!("{:<20} {:>4} {:>9} {:>8} {:>8} {:>12} {:>10}", "model expects", "n", "model p", "won", "avg ask", "realized/c", "± se");
    let rows: [(&str, fn(f64) -> bool); 4] = [
        ("> +5¢ (takes)", |x| x > 0.05),
        ("0 to +5¢ (takes)", |x| x > 0.0 && x <= 0.05),
        ("<= 0 (skips)", |x| x <= 0.0),
        ("all", |_| true),
    ];
    for (label, keep) in rows {
        let es: Vec<&Entry> = entries.iter().filter(|e| expected(e).is_some_and(keep)).collect();
        if es.is_empty() {
            continue;
        }
        let n = es.len() as f64;
        let p = es.iter().filter_map(|e| e.p_model).sum::<f64>() / n;
        let win = es.iter().filter(|e| e.won).count() as f64 / n;
        let avg_ask = es.iter().map(|e| e.ask).sum::<f64>() / n;
        let (m, se) = mean_se(&es.iter().map(|e| realized(e)).collect::<Vec<_>>());
        println!("{label:<20} {:>4} {:>8.1}% {:>7.1}% {:>7.1}¢ {:>12} {:>10}", es.len(), p * 100.0, win * 100.0, avg_ask * 100.0, cents(m), cents(se));
    }
    let blind = entries.iter().filter(|e| e.p_model.is_none()).count();
    if blind > 0 {
        println!("({blind} entries had too little BRTI history for a fair value; the check skips those)");
    }
    println!("At the first chance to enter only; the rule with the check may still enter later in the window.");
    println!("Worth trusting once the takes' realized edge is positive and the skips' is not.");
}

/// Mean and standard error.
fn mean_se(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    if n == 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let m = xs.iter().sum::<f64>() / n;
    let var = if n > 1.0 { xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0) } else { f64::NAN };
    (m, (var / n).sqrt())
}

fn cents(x: f64) -> String {
    if x.is_nan() { "-".into() } else { format!("{:+.1}¢", x * 100.0) }
}

/// Win rate against price paid, by entry-price bucket.
fn entry_table(entries: &[Entry]) {
    println!("\n== Entry calibration: the favorite at {RULE_LAST_S:.0}s before close, held to settlement ==");
    println!("{:<11} {:>4} {:>8} {:>8} {:>12} {:>10}", "ask", "n", "avg ask", "win", "edge/contr", "± se");
    const EDGES: [f64; 8] = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00];
    let rows = EDGES.windows(2).map(|w| (format!("{:.2}-{:.2}", w[0], w[1]), w[0], w[1])).chain([("all".to_string(), 0.0, 1.0)]);
    for (label, lo, hi) in rows {
        let es: Vec<&Entry> = entries.iter().filter(|e| e.ask >= lo && e.ask < hi).collect();
        if es.is_empty() {
            continue;
        }
        let n = es.len() as f64;
        let avg_ask = es.iter().map(|e| e.ask).sum::<f64>() / n;
        let win = es.iter().filter(|e| e.won).count() as f64 / n;
        let edge: Vec<f64> = es.iter().map(|e| realized(e)).collect();
        let (m, se) = mean_se(&edge);
        println!("{label:<11} {:>4} {:>7.1}¢ {:>7.1}% {:>12} {:>10}", es.len(), avg_ask * 100.0, win * 100.0, cents(m), cents(se));
    }
    println!("edge > 0 by more than ~2 se: the favorite is underpriced there. Near 0: no edge to harvest.");
}

/// For each stop level: of the entries whose held bid later fell to it, how
/// often they still won, against what selling there would have fetched.
fn drop_table(eps: &[Episode], entries: &[Option<Entry>], stops: &[f64]) {
    println!("\n== After a drop: entries whose held side's bid later fell to the stop ==");
    println!("{:<6} {:>4} {:>9} {:>8} {:>13} {:>10}", "stop", "n", "avg bid", "won", "stop gain/c", "± se");
    for &stop in stops {
        let (mut bids, mut gains, mut wins) = (vec![], vec![], 0);
        for (ep, e) in eps.iter().zip(entries) {
            let Some(e) = e.as_ref().filter(|e| e.ask > stop) else { continue };
            let close = ep.window.close_ms;
            let hit = ep.snapshots[e.idx + 1..].iter().take_while(|s| s.t_ms < close).find_map(|s| bid(s, e.side).filter(|b| *b <= stop));
            if let Some(b) = hit {
                // Selling at the bid now, less the fee, against holding to the payout.
                gains.push(b - fee_per_contract(b, TAKER_RATE) - e.won as u8 as f64);
                bids.push(b);
                wins += e.won as usize;
            }
        }
        if bids.is_empty() {
            println!("{:<6.2} {:>4}", stop, 0);
            continue;
        }
        let n = bids.len();
        let (g, se) = mean_se(&gains);
        let avg_bid = bids.iter().sum::<f64>() / n as f64;
        println!("{:<6.2} {:>4} {:>8.1}¢ {:>7.1}% {:>13} {:>10}", stop, n, avg_bid * 100.0, wins as f64 / n as f64 * 100.0, cents(g), cents(se));
    }
    println!("stop gain > 0: the market under-reacts, so cutting there pays. Near or below 0: the bid is a fair price and a stop only costs fees.");
}

pub fn run(store: Store, args: CalibrateArgs) -> Result<()> {
    let dirs = episode_dirs(&store, args.from.as_deref(), args.to.as_deref())?;
    if dirs.is_empty() {
        bail!("no episodes in range; run `kbot dataset` first");
    }
    let eps: Vec<Episode> = dirs.iter().map(|d| Episode::load(d)).collect::<Result<_>>()?;
    let entries: Vec<Option<Entry>> = eps.iter().map(entry).collect();
    let taken: Vec<Entry> = entries.iter().flatten().copied().collect();
    println!("{} episodes, {} with an entry", eps.len(), taken.len());
    if eps.len() < 200 {
        println!("note: few episodes, so expect wide error bars; don't pick thresholds off this alone");
    }
    entry_table(&taken);
    cost_table(&taken);
    drop_table(&eps, &entries, &args.stops);

    println!("\n== Sweep: {} with stop / cap, through the simulator ==", args.base);
    // The rules check cost by default; the unchecked rule is the reference.
    let mut specs = vec![format!("{},edge=off", args.base), args.base.clone()];
    specs.extend(args.stops.iter().map(|s| format!("{},stop={s:.2}", args.base)));
    specs.extend(args.caps.iter().map(|c| format!("{},cap={c:.2}", args.base)));
    for s in &args.stops {
        specs.extend(args.caps.iter().map(|c| format!("{},stop={s:.2},cap={c:.2}", args.base)));
    }
    let mut rows = vec![];
    for spec in specs {
        let mut strategy = make_strategy(&spec, TradeLimits::default(), 1, 0.0)?;
        let cfg = SimConfig {
            latency_s: (args.latency, args.latency),
            slip_prob: args.slip,
            limits: strategy.limits(),
            balance: args.balance,
            ..Default::default()
        };
        let results: Vec<_> = eps.iter().enumerate().map(|(i, ep)| run_strategy(strategy.as_mut(), ep, cfg, i as u64)).collect();
        rows.push((spec, summarize(&results)));
    }
    rows.sort_by(|a, b| b.1.total_pnl.total_cmp(&a.1.total_pnl));
    for (spec, s) in &rows {
        print_summary(spec, s);
    }
    println!("sorted by total PnL. Pick thresholds on one date range and confirm them on a later one (--from/--to).");
    Ok(())
}

//! `kbot backtest`: any strategy over recorded episodes, through the same
//! simulator used for training.

use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::Args;
use market::replay::episode_dirs;
use market::{run_strategy, Episode, EpisodeResult, SimConfig, Store, DEFAULT_BALANCE};
use serde::Serialize;
use strategy::{RuleOpts, FairValueStrategy, NothingStrategy, RlStrategy, RuleStrategy, RuleSwingStrategy, Strategy, TradeLimits, MAX_SWING_BPS, RULE_LAST_S};

#[derive(Args, Clone)]
pub struct BacktestArgs {
    /// `nothing`, `fair`, `rule`, `rule-swing:<trunk dir>[@<max swing bps>]`, or a model directory.
    /// The rules take `,stop=<price>` (sell out once the held side's bid falls
    /// there) and `,cap=<price>` (don't buy above it), e.g. `rule,stop=0.55,cap=0.90`.
    #[arg(long, default_value = "fair")]
    pub strategy: String,
    /// First UTC date (YYYY-MM-DD) of episodes to use.
    #[arg(long)]
    pub from: Option<String>,
    /// Last UTC date of episodes to use.
    #[arg(long)]
    pub to: Option<String>,
    /// fair: minimum edge per contract beyond fees, in dollars.
    #[arg(long, default_value_t = 0.03)]
    pub edge: f64,
    /// fair: seconds between decisions.
    #[arg(long, default_value_t = 1)]
    pub every: u32,
    #[arg(long, default_value_t = 1)]
    pub latency: u32,
    #[arg(long, default_value_t = 0.0)]
    pub slip: f64,
    /// Starting cash for every session, in dollars.
    #[arg(long, default_value_t = DEFAULT_BALANCE)]
    pub balance: f64,
    /// Write the full report (per-episode results and fills) here.
    #[arg(long)]
    pub report: Option<PathBuf>,
}

/// Prefix of the rule strategy gated by a forecast trunk: `rule-swing:<trunk dir>[@<bps>]`.
pub const RULE_SWING: &str = "rule-swing:";

pub fn make_strategy(name: &str, limits: TradeLimits, every: u32, edge: f64) -> Result<Box<dyn Strategy>> {
    // Rule options trail the spec after the first comma.
    let rule_opts = |name: &str| -> Option<(String, String)> {
        let (base, opts) = name.split_once(',')?;
        (base == "rule" || base.starts_with(RULE_SWING)).then(|| (base.to_string(), opts.to_string()))
    };
    let (name, opts) = match rule_opts(name) {
        Some((base, opts)) => (base, RuleOpts::parse(&opts)?),
        None => (name.to_string(), RuleOpts::default()),
    };
    Ok(match name.as_str() {
        "nothing" => Box::new(NothingStrategy::new(limits)),
        "fair" => Box::new(FairValueStrategy::new(limits, every, edge)),
        "rule" => Box::new(RuleStrategy::new(limits, every, RULE_LAST_S).with_opts(opts)),
        spec if spec.starts_with(RULE_SWING) => {
            let spec = &spec[RULE_SWING.len()..];
            let (dir, max_swing) = match spec.rsplit_once('@') {
                Some((dir, bps)) => (dir, bps.parse().map_err(|_| anyhow::anyhow!("bad swing threshold {bps:?}"))?),
                None => (spec, MAX_SWING_BPS),
            };
            Box::new(RuleSwingStrategy::load(dir.as_ref(), limits, RULE_LAST_S, max_swing)?.with_opts(opts))
        }
        dir if std::path::Path::new(dir).is_dir() => Box::new(RlStrategy::load(dir.as_ref())?),
        other => bail!("unknown strategy {other:?}: use nothing, fair, rule, rule-swing:<trunk dir>, or a model directory"),
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub episodes: usize,
    pub total_pnl: f64,
    pub mean_pnl: f64,
    pub std_pnl: f64,
    /// Mean / std of per-episode PnL.
    pub sharpe_per_episode: f64,
    pub max_drawdown: f64,
    pub fees: f64,
    pub trades: usize,
    pub traded_episodes: usize,
    pub win_rate_traded: f64,
}

pub fn summarize(results: &[EpisodeResult]) -> Summary {
    let n = results.len().max(1) as f64;
    let pnls: Vec<f64> = results.iter().map(|r| r.pnl).collect();
    let total: f64 = pnls.iter().sum();
    let mean = total / n;
    let std = (pnls.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / n).sqrt();
    let (mut peak, mut cum, mut dd) = (0f64, 0f64, 0f64);
    for p in &pnls {
        cum += p;
        peak = peak.max(cum);
        dd = dd.max(peak - cum);
    }
    let traded: Vec<&EpisodeResult> = results.iter().filter(|r| r.trades > 0).collect();
    Summary {
        episodes: results.len(),
        total_pnl: total,
        mean_pnl: mean,
        std_pnl: std,
        sharpe_per_episode: if std > 0.0 { mean / std } else { 0.0 },
        max_drawdown: dd,
        fees: results.iter().map(|r| r.fees).sum(),
        trades: results.iter().map(|r| r.trades).sum(),
        traded_episodes: traded.len(),
        win_rate_traded: traded.iter().filter(|r| r.pnl > 0.0).count() as f64 / traded.len().max(1) as f64,
    }
}

pub fn print_summary(name: &str, s: &Summary) {
    println!(
        "{name:<24} eps {:>5}  pnl ${:>9.2}  mean ${:>7.3} ± {:>6.3}  sharpe/ep {:>6.3}  maxDD ${:>8.2}  fees ${:>8.2}  trades {:>6}  traded {:>5}  win {:>5.1}%",
        s.episodes,
        s.total_pnl,
        s.mean_pnl,
        s.std_pnl,
        s.sharpe_per_episode,
        s.max_drawdown,
        s.fees,
        s.trades,
        s.traded_episodes,
        s.win_rate_traded * 100.0
    );
}

pub fn run(store: Store, args: BacktestArgs) -> Result<()> {
    let dirs = episode_dirs(&store, args.from.as_deref(), args.to.as_deref())?;
    if dirs.is_empty() {
        bail!("no episodes in range; run `kbot dataset` first");
    }
    let mut strategy = make_strategy(&args.strategy, TradeLimits::default(), args.every, args.edge)?;
    let cfg = SimConfig {
        latency_s: (args.latency, args.latency),
        slip_prob: args.slip,
        limits: strategy.limits(),
        balance: args.balance,
        ..Default::default()
    };
    let mut results = vec![];
    for (i, dir) in dirs.iter().enumerate() {
        let ep = Episode::load(dir)?;
        results.push(run_strategy(strategy.as_mut(), &ep, cfg, i as u64));
    }
    let s = summarize(&results);
    print_summary(&args.strategy, &s);
    if let Some(path) = args.report {
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        std::fs::write(&path, serde_json::to_vec_pretty(&serde_json::json!({ "summary": s, "episodes": results }))?)?;
        println!("report: {}", path.display());
    }
    Ok(())
}

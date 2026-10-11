//! `kbot pretrain` and `kbot train`: the two training stages, on one
//! chronological train / validation / test split, then a held-out comparison
//! against the baselines through the same `Strategy` interface used live.

use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::Args;
use market::replay::episode_dirs;
use market::{run_strategy, Episode, Store};
use rl::{Encoded, PpoConfig, PretrainConfig, Prepared};
use strategy::rl::Model;
use strategy::{FairValueStrategy, NothingStrategy, RlStrategy, RuleStrategy, Strategy, TradeLimits, RULE_LAST_S};

use crate::backtest::{print_summary, summarize};

/// Which episodes, and how they split. Both stages must use the same split.
#[derive(Args, Clone)]
pub struct SplitArgs {
    #[arg(long)]
    pub from: Option<String>,
    #[arg(long)]
    pub to: Option<String>,
    /// Fraction of the (chronologically last) episodes held out for validation.
    #[arg(long, default_value_t = 0.15)]
    pub val_frac: f64,
    /// Fraction of the last episodes held out for the final test.
    #[arg(long, default_value_t = 0.15)]
    pub test_frac: f64,
    /// Train and validate on the first N episodes only (overfit check).
    #[arg(long)]
    pub overfit: Option<usize>,
}

struct Split {
    train: Vec<Episode>,
    val: Vec<Episode>,
    test: Vec<Episode>,
}

fn split(store: &Store, args: &SplitArgs) -> Result<Split> {
    let dirs = episode_dirs(store, args.from.as_deref(), args.to.as_deref())?;
    let mut eps = dirs.iter().map(|d| Episode::load(d)).collect::<Result<Vec<_>>>()?;
    eps.sort_by_key(|e| e.window.open_ms);
    if eps.len() < 3 {
        bail!("need at least 3 episodes, have {}", eps.len());
    }
    let s = if let Some(n) = args.overfit {
        let few: Vec<Episode> = eps.into_iter().take(n).collect();
        Split { train: few.clone(), val: few.clone(), test: few }
    } else {
        let n = eps.len();
        let n_test = ((n as f64 * args.test_frac).round() as usize).max(1);
        let n_val = ((n as f64 * args.val_frac).round() as usize).max(1);
        let test = eps.split_off(n - n_test);
        let val = eps.split_off(n - n_test - n_val);
        Split { train: eps, val, test }
    };
    let span = |v: &[Episode]| format!("{} .. {}", v[0].window.ticker, v[v.len() - 1].window.ticker);
    println!("train {} eps ({})", s.train.len(), span(&s.train));
    println!("val   {} eps ({})", s.val.len(), span(&s.val));
    println!("test  {} eps ({})", s.test.len(), span(&s.test));
    Ok(s)
}

fn prepare(eps: &[Episode], every_s: u32) -> Vec<Prepared> {
    eps.iter().map(|e| Prepared::new(e.clone(), TradeLimits::default(), every_s)).collect()
}

#[derive(Args, Clone)]
pub struct PretrainArgs {
    /// Output directory for the trunk.
    #[arg(long)]
    pub out: PathBuf,
    #[command(flatten)]
    pub split: SplitArgs,
    /// LSTM width (parameters grow with its square).
    #[arg(long, default_value_t = 32)]
    pub hidden: usize,
    /// Forecast horizons in seconds, comma-separated.
    #[arg(long, value_delimiter = ',', default_value = "1")]
    pub horizons: Vec<usize>,
    /// Seconds between steps; the trading head will decide at this cadence.
    #[arg(long, default_value_t = 1)]
    pub every: u32,
    #[arg(long, default_value_t = 30)]
    pub epochs: usize,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

pub fn pretrain(store: Store, args: PretrainArgs) -> Result<()> {
    let s = split(&store, &args.split)?;
    let cfg = PretrainConfig {
        hidden: args.hidden,
        horizons: args.horizons,
        every_s: args.every,
        epochs: args.epochs,
        seed: args.seed,
        ..Default::default()
    };
    let outcome = rl::pretrain(&cfg, &prepare(&s.train, cfg.every_s), &prepare(&s.val, cfg.every_s), &args.out)?;
    let b = &outcome.best_val;
    println!(
        "best validation forecast: nll {:.4} vs {:.4} for a constant Gaussian, mean R2 {:.4} over {} targets -> {}",
        b.nll,
        b.base_nll,
        b.r2(),
        b.targets,
        outcome.dir.display()
    );
    Ok(())
}

#[derive(Args, Clone)]
pub struct TrainArgs {
    /// Pretrained trunk (from `kbot pretrain`); frozen while the head trains.
    #[arg(long)]
    pub trunk: PathBuf,
    /// Output directory for the run (model in <out>/best).
    #[arg(long)]
    pub out: PathBuf,
    #[command(flatten)]
    pub split: SplitArgs,
    #[arg(long, default_value_t = 1000)]
    pub iterations: usize,
    /// Width of the policy and value heads.
    #[arg(long, default_value_t = 32)]
    pub head_hidden: usize,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

pub fn run(store: Store, args: TrainArgs) -> Result<()> {
    let mut cfg = PpoConfig { iterations: args.iterations, head_hidden: args.head_hidden, seed: args.seed, ..Default::default() };
    cfg.sim.limits = cfg.limits;
    cfg.eval_sim.limits = cfg.limits;
    let base = Model::from_trunk(&args.trunk, cfg.head_hidden, cfg.limits)?;
    let every_s = base.meta.trunk.every_s;

    let sp = split(&store, &args.split)?;
    let seen = base.meta.trunk.trained_until_ms;
    if args.split.overfit.is_none() && sp.val.iter().chain(&sp.test).any(|e| e.window.open_ms <= seen) {
        bail!("the trunk was pretrained on sessions this split uses for validation or test; pretrain it on this split");
    }
    let encode = |eps: &[Episode]| Encoded::all(prepare(eps, every_s), &base);
    let (train_e, val_e, test_e) = (encode(&sp.train), encode(&sp.val), encode(&sp.test));
    let outcome = rl::train(&cfg, &base, &train_e, &val_e, &args.out)?;
    println!("best validation mean pnl ${:.3}/episode -> {}", outcome.best_val.mean_pnl, outcome.best_dir.display());

    // Held-out test, through the live Strategy interface.
    let mut strategies: Vec<(String, Box<dyn Strategy>)> = vec![
        ("nothing".into(), Box::new(NothingStrategy::new(cfg.limits))),
        ("fair edge=0.03".into(), Box::new(FairValueStrategy::new(cfg.limits, every_s, 0.03))),
        ("fair edge=0.01".into(), Box::new(FairValueStrategy::new(cfg.limits, every_s, 0.01))),
        ("rule last 3m".into(), Box::new(RuleStrategy::new(cfg.limits, every_s, RULE_LAST_S))),
        ("rl (best)".into(), Box::new(RlStrategy::load(&outcome.best_dir)?)),
    ];
    println!("\ntest set:");
    let mut rl_total = 0.0;
    for (name, s) in strategies.iter_mut() {
        let results: Vec<_> =
            sp.test.iter().enumerate().map(|(i, e)| run_strategy(s.as_mut(), e, cfg.eval_sim, i as u64)).collect();
        let summary = summarize(&results);
        if name.starts_with("rl") {
            rl_total = summary.total_pnl;
        }
        print_summary(name, &summary);
    }
    // The trainer's own evaluation and the Strategy path must agree exactly.
    let best = Model::load(&outcome.best_dir)?;
    let direct = rl::ppo::evaluate(&best.head, &test_e, &cfg);
    let parity = (direct.total_pnl - rl_total).abs() < 1e-6;
    println!(
        "\nparity (trainer eval vs Strategy interface): ${:.4} vs ${:.4} {}",
        direct.total_pnl,
        rl_total,
        if parity { "OK" } else { "MISMATCH" }
    );
    if !parity {
        bail!("training and inference paths disagree");
    }
    Ok(())
}

//! kbot: record Kalshi KXBTC15M sessions, build episodes, train, backtest, paper trade.

mod backtest;
mod calibrate;
mod dataset;
mod feeds;
mod live;
mod paper;
mod record;
mod sessions;
mod train;
mod viz;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(about = "Kalshi 15-minute BTC quant bot")]
struct Cli {
    /// Credentials and data dir; env vars KALSHI_KEY_ID / KALSHI_KEY_PATH / KBOT_DATA_DIR override.
    #[arg(long, global = true, default_value = "config.toml")]
    config: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the recorder daemon: live Kalshi feeds -> per-session files.
    Record,
    /// Inspect recorded sessions.
    Sessions {
        #[command(subcommand)]
        cmd: sessions::Cmd,
    },
    /// Build an episode file in every settled session directory.
    Dataset {
        /// Rebuild episodes that already exist.
        #[arg(long)]
        rebuild: bool,
        /// Longest tolerated BRTI silence inside a session, seconds.
        #[arg(long, default_value_t = 10)]
        max_btc_gap: i64,
    },
    /// Run a strategy over recorded episodes.
    Backtest(backtest::BacktestArgs),
    /// Is the favorite priced right near close, does a stop-loss pay, and which stop / entry cap backtests best.
    Calibrate(calibrate::CalibrateArgs),
    /// Stage 1: pretrain the LSTM trunk to forecast BRTI from market data alone.
    Pretrain(train::PretrainArgs),
    /// Stage 2: train policy heads on a frozen trunk with PPO, then compare to the baselines on held-out sessions.
    Train(train::TrainArgs),
    /// Run a strategy on live data with simulated fills (never sends orders).
    Paper(paper::PaperArgs),
    /// Browse recorded sessions in a local web page.
    Viz {
        #[arg(long, default_value_t = 8787)]
        port: u16,
        /// Where to look for trained models to backtest from the page.
        #[arg(long, default_value = "models")]
        models: PathBuf,
    },
    /// Write synthetic sessions (for pipeline tests) into --data-dir.
    Synth {
        #[arg(long)]
        data_dir: PathBuf,
        #[arg(long, default_value_t = 500)]
        episodes: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,burn_core=error".into()),
        )
        .init();
    let cli = Cli::parse();
    let config = kalshi::Config::load(&cli.config);
    match cli.cmd {
        Cmd::Record => record::run(config?).await,
        Cmd::Sessions { cmd } => sessions::run(data_dir(config), cmd),
        Cmd::Dataset { rebuild, max_btc_gap } => dataset::run(data_dir(config), rebuild, max_btc_gap),
        Cmd::Backtest(args) => backtest::run(data_dir(config), args),
        Cmd::Calibrate(args) => calibrate::run(data_dir(config), args),
        Cmd::Pretrain(args) => train::pretrain(data_dir(config), args),
        Cmd::Train(args) => train::run(data_dir(config), args),
        Cmd::Paper(args) => paper::run(config?, args).await,
        Cmd::Viz { port, models } => viz::run(data_dir(config), models, port).await,
        Cmd::Synth { data_dir, episodes, seed } => {
            market::synth::generate(&market::Store::new(&data_dir), episodes, &Default::default(), seed)?;
            println!("wrote {episodes} synthetic sessions to {}", data_dir.display());
            Ok(())
        }
    }
}

/// Commands that only read local data don't need credentials.
fn data_dir(config: Result<kalshi::Config>) -> market::Store {
    let root = match config {
        Ok(c) => c.data_dir,
        Err(_) => std::env::var("KBOT_DATA_DIR").map(PathBuf::from).unwrap_or_else(|_| "data".into()),
    };
    market::Store::new(root)
}

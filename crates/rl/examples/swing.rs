//! Calibration aid for `RuleSwingStrategy`: per episode, the forecast swing
//! when the rule first may bet, the realized move to close, and whether the
//! favorite won. `cargo run --release -p rl --example swing -- <trunk dir> <session dir>...`
use market::Episode;
use strategy::{Position, RuleSwingStrategy, Strategy, TradeLimits, RULE_LAST_S};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    println!("ticker,swing_bps,realized_bps,yes_mid,favorite_won");
    for dir in &args[2..] {
        let ep = Episode::load(dir.as_ref())?;
        let mut s = RuleSwingStrategy::load(args[1].as_ref(), TradeLimits::default(), RULE_LAST_S, f64::INFINITY)?;
        s.begin_window(&ep.window);
        let at = ep.window.close_ms - (RULE_LAST_S * 1000.0) as i64;
        let pos = Position::with_cash(0.0);
        let mut row = None;
        for snap in &ep.snapshots {
            s.on_tick(snap, &pos);
            if snap.t_ms == at {
                row = Some((s.swing_bps(), snap.btc, snap.yes.mid()));
            }
        }
        let Some((Some(swing), btc, Some(mid))) = row else { continue };
        let close = ep.snapshots.iter().rev().find(|x| x.t_ms < ep.window.close_ms && x.btc > 0.0).map(|x| x.btc).unwrap_or(btc);
        let won = (mid > 0.5) == ep.result_yes;
        println!("{},{swing:.1},{:.1},{mid:.2},{won}", ep.window.ticker, (close / btc).ln() * 1e4);
    }
    Ok(())
}

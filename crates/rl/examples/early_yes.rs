//! Debug tool: does the policy buy YES below the strike, and does it "know" the result?
//! `cargo run --release -p rl --example early_yes -- <model dir> <data dir> <first ep> <last ep>`
use market::{run_strategy, Episode, SimConfig};
use strategy::{Action, RlStrategy};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let (model, data) = (&a[1], &a[2]);
    let (lo, hi): (usize, usize) = (a[3].parse()?, a[4].parse()?);
    let mut dirs: Vec<_> = glob(data);
    dirs.sort();
    let mut s = RlStrategy::load(model.as_ref())?;
    let cfg = SimConfig { latency_s: (1, 1), limits: strategy::Strategy::limits(&s), ..Default::default() };
    let (mut n_ep, mut below, mut below_win, mut above, mut above_win, mut first120) = (0, 0, 0, 0, 0, 0);
    let (mut fv_below_sum, mut fv_n) = (0.0, 0);
    for d in &dirs[lo..=hi] {
        let ep = Episode::load(d)?;
        let r = run_strategy(&mut s, &ep, cfg, 0);
        n_ep += 1;
        // First YES buy in the session.
        if let Some(f) = r.fills.iter().find(|f| f.action == Action::BuyYes) {
            let snap = &ep.snapshots[f.idx];
            let secs = (snap.t_ms - ep.window.open_ms) / 1000;
            if secs < 120 { first120 += 1; }
            if snap.btc < ep.window.strike {
                below += 1;
                below_win += ep.result_yes as u32;
                fv_below_sum += f.price; fv_n += 1;
            } else {
                above += 1;
                above_win += ep.result_yes as u32;
            }
        }
    }
    println!("{n_ep} eps; first YES buy below strike: {below} (settled YES {below_win}), above: {above} (settled YES {above_win}); within 120s of open: {first120}; avg price paid below strike {:.3}", fv_below_sum / fv_n.max(1) as f64);
    Ok(())
}

fn glob(root: &str) -> Vec<std::path::PathBuf> {
    let mut out = vec![];
    for day in std::fs::read_dir(format!("{root}/sessions")).unwrap().flatten() {
        for s in std::fs::read_dir(day.path()).unwrap().flatten() {
            if s.path().join("episode.json.zst").exists() { out.push(s.path()); }
        }
    }
    out
}

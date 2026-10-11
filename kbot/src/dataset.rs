//! `kbot dataset`: settled sessions -> episode files in their own directories.

use anyhow::Result;
use market::replay::{build_episode, Requirements, EPISODE_FILE};
use market::{SessionStatus, Store};

pub fn run(store: Store, rebuild: bool, max_btc_gap_s: i64) -> Result<()> {
    let req = Requirements { max_btc_gap_ms: max_btc_gap_s * 1000, ..Default::default() };
    let (mut built, mut kept, mut dropped) = (0, 0, std::collections::BTreeMap::<String, u32>::new());
    let (mut strike_diffs, mut settle_diffs, mut disagree) = (vec![], vec![], vec![]);
    for dir in store.list_sessions()? {
        let meta = Store::read_meta(&dir)?;
        if meta.status != SessionStatus::Settled {
            continue;
        }
        if !rebuild && dir.join(EPISODE_FILE).exists() {
            kept += 1;
            continue;
        }
        match build_episode(&dir, &req)? {
            Ok((ep, checks)) => {
                ep.save(&dir)?;
                built += 1;
                strike_diffs.extend(checks.strike_diff);
                settle_diffs.extend(checks.settle_diff);
                if checks.result_agrees == Some(false) {
                    disagree.push(meta.ticker.clone());
                }
            }
            Err(reason) => {
                let _ = std::fs::remove_file(dir.join(EPISODE_FILE));
                // Group "BRTI gap 12s" style reasons by their kind.
                let kind = reason.trim_end_matches(|c: char| c.is_ascii_digit() || c == 's').trim().to_string();
                *dropped.entry(kind).or_default() += 1;
                tracing::debug!("{}: {reason}", meta.ticker);
            }
        }
    }
    println!("episodes built: {built}, already built: {kept}");
    for (reason, n) in &dropped {
        println!("  dropped {n:>4}: {reason}");
    }
    let stat = |name: &str, xs: &[f64]| {
        if xs.is_empty() {
            return;
        }
        let max = xs.iter().fold(0f64, |m, x| m.max(x.abs()));
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        println!("  {name}: mean {mean:+.2}, max |diff| {max:.2} over {} sessions", xs.len());
    };
    println!("checks against Kalshi (our BRTI averages minus theirs, $):");
    stat("strike (60s before open)", &strike_diffs);
    stat("settlement (60s before close)", &settle_diffs);
    if disagree.is_empty() {
        println!("  result: matches expiration_value >= strike everywhere");
    } else {
        println!("  result DISAGREES for: {disagree:?}");
    }
    Ok(())
}

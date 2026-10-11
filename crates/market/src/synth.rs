//! Synthetic sessions for pipeline testing -- never mixed with recorded data.
//!
//! BRTI is a random walk; the market quotes the analytic fair value of a
//! *stale* price (`lag_s` behind) plus noise, rounded to Kalshi's ticks. That
//! staleness is a known, exploitable edge: a working learner must find it.

use std::collections::BTreeMap;

use anyhow::Result;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use strategy::fair::normal_cdf;
use strategy::fees::tick_size;
use strategy::{Quote, Snapshot, Window};

use crate::replay::{complement, Episode};
use crate::store::{SessionMeta, SessionStatus, Store, BTC_LOOKBACK_MS};

pub struct SynthConfig {
    pub sigma: f64,
    pub lag_s: usize,
    pub noise: f64,
    pub start_ms: i64,
}

impl Default for SynthConfig {
    fn default() -> Self {
        Self { sigma: 1e-4, lag_s: 10, noise: 0.01, start_ms: 1_767_225_600_000 } // 2026-01-01
    }
}

fn round_tick(p: f64, up: bool) -> f64 {
    let t = tick_size(p);
    let n = p / t;
    (if up { n.ceil() } else { n.floor() } * t).clamp(0.001, 0.999)
}

pub fn generate(store: &Store, n: usize, cfg: &SynthConfig, seed: u64) -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let normal = |rng: &mut SmallRng| -> f64 {
        // Box-Muller
        let (u1, u2): (f64, f64) = (rng.random::<f64>().max(1e-12), rng.random());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let mut price = 100_000.0;
    for e in 0..n {
        let open = cfg.start_ms + e as i64 * 900_000;
        let close = open + 900_000;
        let start = open - BTC_LOOKBACK_MS;
        let secs = ((close - start) / 1000) as usize;
        let mut path = Vec::with_capacity(secs);
        for _ in 0..secs {
            price *= (cfg.sigma * normal(&mut rng)).exp();
            path.push(price);
        }
        let lb = (BTC_LOOKBACK_MS / 1000) as usize;
        let strike = path[lb - 60..lb].iter().sum::<f64>() / 60.0;
        let settle = path[secs - 60..].iter().sum::<f64>() / 60.0;
        let ticker = format!("SYNTH-{e:06}");
        let window = Window { ticker: ticker.clone(), strike, open_ms: open, close_ms: close };

        let mut snaps = Vec::with_capacity(secs);
        for (i, &btc) in path.iter().enumerate() {
            let t = start + i as i64 * 1000;
            let mut yes = Quote::default();
            if t >= open {
                let stale = path[i.saturating_sub(cfg.lag_s)];
                let tau = ((close - t) as f64 / 1000.0).max(1.0);
                let sd = cfg.sigma * stale * (tau - 40.0).max(1.0).sqrt();
                let p = (normal_cdf((stale - strike) / sd) + cfg.noise * normal(&mut rng)).clamp(0.02, 0.98);
                let half = if rng.random_bool(0.7) { 0.005 } else { 0.01 };
                yes = Quote {
                    bid: Some(round_tick(p - half, false)),
                    ask: Some(round_tick(p + half, true)),
                    bid_size: Some(500.0),
                    ask_size: Some(500.0),
                };
            }
            snaps.push(Snapshot { t_ms: t, btc, yes, no: complement(&yes) });
        }
        let ep = Episode { window, result_yes: settle >= strike, expiration_value: Some(settle), snapshots: snaps };

        let meta = SessionMeta {
            ticker,
            event_ticker: "SYNTH".into(),
            strike: Some(strike),
            strike_type: "greater_or_equal".into(),
            open_ms: open,
            close_ms: close,
            price_ranges: serde_json::Value::Null,
            exchange_index: None,
            status: SessionStatus::Settled,
            result: Some(if ep.result_yes { "yes" } else { "no" }.into()),
            expiration_value: Some(settle),
            settled_ms: Some(close),
            first_seen_ms: start,
            last_write_ms: close,
            gaps: vec![],
            counts: BTreeMap::new(),
            recorder_version: "synth".into(),
        };
        let dir = store.session_dir(&meta);
        Store::write_meta(&dir, &meta)?;
        ep.save(&dir)?;
    }
    Ok(())
}

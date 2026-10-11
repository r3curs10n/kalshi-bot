//! Analytic fair value of YES under a driftless log-normal BRTI, accounting
//! for settlement on the 60-second average before close.
//!
//! YES pays if `mean(BRTI over the 60 seconds before close) >= strike`. With
//! `n` of those seconds still in the future and spot `S`, the average has
//! mean `(known_sum + n*S)/60` and, treating price moves as a random walk with
//! per-second vol `sigma*S`, variance
//! `(sigma*S)^2 * [ (n/60)^2 * T0 + n(n+1)(2n+1)/6/3600 ]`,
//! where `T0` is the time until the averaging window starts.

use crate::history::History;
use crate::Window;

/// Settlement average length.
pub const AVG_SECONDS: i64 = 60;
/// Used until enough history exists to measure volatility (about 0.5%/hour).
pub const DEFAULT_SIGMA: f64 = 8e-5;

pub fn normal_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Complementary error function (Numerical Recipes erfcc, |err| < 1.2e-7).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * (-z * z - 1.26551223
        + t * (1.00002368
            + t * (0.37409196
                + t * (0.09678418
                    + t * (-0.18628806
                        + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
        .exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FairValue {
    /// P(YES).
    pub p: f64,
    /// (expected settlement average - strike) / its std dev.
    pub z: f64,
    /// Per-second log vol used.
    pub sigma: f64,
}

pub fn sigma(hist: &History) -> f64 {
    hist.vol(300_000).or_else(|| hist.vol(900_000)).unwrap_or(DEFAULT_SIGMA).max(1e-6)
}

pub fn fair_value(w: &Window, hist: &History) -> Option<FairValue> {
    let now = hist.last()?;
    let s = now.btc;
    let sigma = sigma(hist);
    let avg_start = w.close_ms - AVG_SECONDS * 1000;

    // Seconds of the averaging window already observed.
    let (mut known_sum, mut known) = (0.0, 0i64);
    if now.t_ms >= avg_start {
        let mut last_sec = i64::MIN;
        for snap in hist.range(avg_start, now.t_ms.min(w.close_ms - 1)) {
            let sec = snap.t_ms.div_euclid(1000);
            if sec != last_sec {
                known_sum += snap.btc;
                known += 1;
                last_sec = sec;
            }
        }
        // Seconds missed by gaps are filled with spot.
        let elapsed = ((now.t_ms.min(w.close_ms - 1) - avg_start) / 1000 + 1).clamp(0, AVG_SECONDS);
        known_sum += (elapsed - known).max(0) as f64 * s;
        known = elapsed;
    }
    let n = (AVG_SECONDS - known) as f64;
    let t0 = ((avg_start - now.t_ms) as f64 / 1000.0).max(0.0);
    let mean = (known_sum + n * s) / AVG_SECONDS as f64;
    let var = (sigma * s).powi(2) * ((n / 60.0).powi(2) * t0 + n * (n + 1.0) * (2.0 * n + 1.0) / 6.0 / 3600.0);
    let diff = mean - w.strike;
    if var <= 1e-18 {
        let p = if diff >= 0.0 { 1.0 } else { 0.0 };
        return Some(FairValue { p, z: if diff >= 0.0 { 10.0 } else { -10.0 }, sigma });
    }
    let z = diff / var.sqrt();
    Some(FairValue { p: normal_cdf(z), z, sigma })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Snapshot;

    fn hist_flat(price: f64, from_ms: i64, to_ms: i64) -> History {
        let mut h = History::default();
        let mut t = from_ms;
        while t <= to_ms {
            // tiny alternating moves so vol is measurable
            let p = price * (1.0 + if (t / 1000) % 2 == 0 { 5e-5 } else { -5e-5 });
            h.push(Snapshot { t_ms: t, btc: p, ..Default::default() });
            t += 1000;
        }
        h
    }

    #[test]
    fn cdf() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-4);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 1e-4);
    }

    #[test]
    fn at_the_money_is_half_and_moves_with_spot() {
        let w = Window { ticker: "T".into(), strike: 100_000.0, open_ms: 0, close_ms: 900_000 };
        let h = hist_flat(100_000.0, -600_000, 300_000);
        let fv = fair_value(&w, &h).unwrap();
        assert!((fv.p - 0.5).abs() < 0.05, "{fv:?}");
        let h = hist_flat(100_100.0, -600_000, 300_000);
        assert!(fair_value(&w, &h).unwrap().p > 0.6);
    }

    #[test]
    fn last_seconds_are_nearly_decided_by_known_average() {
        let w = Window { ticker: "T".into(), strike: 100_000.0, open_ms: 0, close_ms: 900_000 };
        // Spot sits $50 above strike for the whole averaging window.
        let h = hist_flat(100_050.0, -600_000, 899_000);
        let fv = fair_value(&w, &h).unwrap();
        assert!(fv.p > 0.999, "{fv:?}");
    }
}

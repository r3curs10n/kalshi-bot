use market::Episode;
use strategy::features::MARKET_FEATURES;
use strategy::{Position, Runner, TradeLimits};

/// An episode with its decision times and market features precomputed.
pub struct Prepared {
    pub ep: Episode,
    /// (snapshot index, market features, fair value) per decision time.
    pub points: Vec<(usize, [f32; MARKET_FEATURES], f64)>,
}

impl Prepared {
    pub fn new(ep: Episode, limits: TradeLimits, every_s: u32) -> Self {
        let mut runner = Runner::new(limits, every_s);
        runner.begin_window(&ep.window);
        let flat = Position::default();
        let mut points = vec![];
        for (i, s) in ep.snapshots.iter().enumerate() {
            if let Some(ctx) = runner.tick(s, &flat) {
                points.push((i, ctx.market, ctx.fv.p));
            }
        }
        Self { ep, points }
    }
}

/// The trunk's pretraining targets at snapshot `i`: for each horizon h, the
/// BRTI log return to `t + h` in bps. Labels only, never inputs. NaN where
/// `t + h` is past the episode or BRTI is missing.
pub fn forecast_targets(ep: &Episode, i: usize, horizons: &[usize]) -> Vec<f32> {
    let now = &ep.snapshots[i];
    horizons
        .iter()
        .map(|h| {
            let Some(later) = ep.snapshots.get(i + h) else { return f32::NAN };
            // The grid is 1s, but don't trust that blindly.
            if later.t_ms - now.t_ms != *h as i64 * 1000 || now.btc <= 0.0 || later.btc <= 0.0 {
                return f32::NAN;
            }
            ((later.btc / now.btc).ln() * 1e4) as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use strategy::{Quote, Snapshot, Window};

    #[test]
    fn forecast_targets_look_ahead_on_the_grid_and_stop_at_the_end() {
        // BRTI rises 1 bp per second for 100s.
        let snapshots = (0..100)
            .map(|i| Snapshot { t_ms: i * 1000, btc: 100.0 * (1e-4 * i as f64).exp(), yes: Quote::default(), no: Quote::default() })
            .collect();
        let ep = Episode {
            window: Window { ticker: "T".into(), strike: 100.0, open_ms: 0, close_ms: 100_000 },
            result_yes: true,
            expiration_value: None,
            snapshots,
        };
        let t = forecast_targets(&ep, 50, &[1, 10, 60]);
        assert!((t[0] - 1.0).abs() < 1e-3 && (t[1] - 10.0).abs() < 1e-3, "{t:?}");
        assert!(t[2].is_nan(), "50 + 60 runs past the episode");
        assert!(forecast_targets(&ep, 99, &[1])[0].is_nan());
    }
}

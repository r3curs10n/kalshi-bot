//! Rolling buffer of recent snapshots, kept across windows so volatility and
//! returns have lookback from the first second of a new window.

use std::collections::VecDeque;

use crate::Snapshot;

/// Longest lookback any feature uses, plus slack.
pub const KEEP_MS: i64 = 20 * 60 * 1000;

#[derive(Debug, Clone, Default)]
pub struct History {
    snaps: VecDeque<Snapshot>,
}

impl History {
    pub fn push(&mut self, s: Snapshot) {
        if let Some(l) = self.snaps.back() {
            if s.t_ms == l.t_ms {
                return;
            }
            // Time only goes backwards when a backtest starts replaying
            // another session; that session brings its own lookback.
            if s.t_ms < l.t_ms {
                self.snaps.clear();
            }
        }
        self.snaps.push_back(s);
        while self.snaps.front().is_some_and(|f| f.t_ms < s.t_ms - KEEP_MS) {
            self.snaps.pop_front();
        }
    }

    /// Forget quotes but keep BRTI: a new window is a different contract, and
    /// its features must not compare against the previous one's prices.
    pub fn clear_quotes(&mut self) {
        for s in self.snaps.iter_mut() {
            s.yes = Default::default();
            s.no = Default::default();
        }
    }

    pub fn last(&self) -> Option<&Snapshot> {
        self.snaps.back()
    }

    pub fn len(&self) -> usize {
        self.snaps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.snaps.is_empty()
    }

    /// Latest snapshot at or before `t_ms`.
    pub fn at(&self, t_ms: i64) -> Option<&Snapshot> {
        let i = self.snaps.partition_point(|s| s.t_ms <= t_ms);
        if i == 0 { None } else { self.snaps.get(i - 1) }
    }

    /// Snapshots with `from_ms <= t <= to_ms`.
    pub fn range(&self, from_ms: i64, to_ms: i64) -> impl Iterator<Item = &Snapshot> {
        let a = self.snaps.partition_point(|s| s.t_ms < from_ms);
        let b = self.snaps.partition_point(|s| s.t_ms <= to_ms);
        self.snaps.range(a..b.max(a))
    }

    /// Realized per-second log volatility of BRTI over the last `window_ms`.
    pub fn vol(&self, window_ms: i64) -> Option<f64> {
        let now = self.last()?.t_ms;
        let mut prev: Option<&Snapshot> = None;
        let (mut sum_sq, mut secs, mut n) = (0.0, 0.0, 0);
        for s in self.range(now - window_ms, now) {
            if let Some(p) = prev {
                let dt = (s.t_ms - p.t_ms) as f64 / 1000.0;
                if dt > 0.0 && p.btc > 0.0 && s.btc > 0.0 {
                    let r = (s.btc / p.btc).ln();
                    sum_sq += r * r;
                    secs += dt;
                    n += 1;
                }
            }
            prev = Some(s);
        }
        (n >= 20 && secs > 0.0).then(|| (sum_sq / secs).sqrt())
    }

    /// Log return of BRTI over the last `lag_ms`.
    pub fn ret(&self, lag_ms: i64) -> f64 {
        let (Some(now), Some(then)) = (self.last(), self.last().and_then(|l| self.at(l.t_ms - lag_ms))) else {
            return 0.0;
        };
        if now.btc > 0.0 && then.btc > 0.0 { (now.btc / then.btc).ln() } else { 0.0 }
    }
}

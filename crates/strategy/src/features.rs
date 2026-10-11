//! Observation features. The same functions serve training (replayed
//! snapshots) and live inference, so the two cannot drift apart.

use crate::fair::{fair_value, FairValue};
use crate::history::History;
use crate::{Position, Snapshot, Window};

/// Bumped whenever the feature set changes; models record the version they
/// were trained with.
pub const FEATURE_VERSION: u32 = 2;

pub const MARKET_FEATURES: usize = 29;
pub const POSITION_FEATURES: usize = 5;

pub const RETURN_LAGS_S: [i64; 6] = [1, 5, 15, 30, 60, 300];
pub const VOL_WINDOWS_S: [i64; 3] = [60, 300, 900];

/// Market-only features at the latest snapshot in `hist`, plus the fair value
/// they were built around.
pub fn market_features(w: &Window, hist: &History) -> Option<([f32; MARKET_FEATURES], FairValue)> {
    let now = *hist.last()?;
    let fv = fair_value(w, hist)?;
    let dur = w.duration_s().max(1.0);
    let tau = ((w.close_ms - now.t_ms) as f64 / 1000.0).clamp(0.0, dur);
    let mut f = [0f32; MARKET_FEATURES];
    let mut i = 0;
    let mut put = |x: f64| {
        f[i] = if x.is_finite() { x as f32 } else { 0.0 };
        i += 1;
    };

    // time
    put(tau / dur);
    put((tau / dur).sqrt());
    put(if tau <= 60.0 { 1.0 } else { 0.0 });
    // price vs strike
    put((now.btc / w.strike).ln() * 1000.0);
    put(fv.z.clamp(-6.0, 6.0));
    put(fv.p);
    // BRTI dynamics
    for lag in RETURN_LAGS_S {
        put(hist.ret(lag * 1000) * 1000.0);
    }
    for win in VOL_WINDOWS_S {
        put(hist.vol(win * 1000).unwrap_or(fv.sigma) * 1e4);
    }
    // quotes
    let missing = now.yes.bid.is_none() || now.yes.ask.is_none() || now.no.bid.is_none() || now.no.ask.is_none();
    let yes_bid = now.yes.bid.unwrap_or(0.0);
    let yes_ask = now.yes.ask.unwrap_or(1.0);
    let no_bid = now.no.bid.unwrap_or(0.0);
    let no_ask = now.no.ask.unwrap_or(1.0);
    put(yes_bid);
    put(yes_ask);
    put(no_bid);
    put(no_ask);
    put(yes_ask - yes_bid);
    let mid = |t: i64| hist.at(t).and_then(|s| s.yes.mid());
    let mid_now = now.yes.mid();
    for lag in [5_000, 30_000] {
        put(match (mid_now, mid(now.t_ms - lag)) {
            (Some(a), Some(b)) => a - b,
            _ => 0.0,
        });
    }
    put(now.yes.bid_size.unwrap_or(0.0).ln_1p() / 10.0);
    put(now.yes.ask_size.unwrap_or(0.0).ln_1p() / 10.0);
    // edges vs the model's fair value
    put(fv.p - yes_ask);
    put((1.0 - fv.p) - no_ask);
    put(yes_bid - fv.p);
    put(no_bid - (1.0 - fv.p));
    put(if missing { 1.0 } else { 0.0 });
    debug_assert_eq!(i, MARKET_FEATURES);
    Some((f, fv))
}

/// The account: how much of it sits in each side (valued at that side's
/// mid), and how the held contracts are doing. Scale-free, so it doesn't
/// depend on the balance.
pub fn position_features(pos: &Position, now: &Snapshot) -> [f32; POSITION_FEATURES] {
    let held = pos.held();
    let avg = if held > 0 { pos.cost / held as f64 } else { 0.0 };
    let mark = if pos.yes > 0 { now.yes.mid() } else if pos.no > 0 { now.no.mid() } else { None };
    let value = held as f64 * mark.unwrap_or(avg);
    let equity = (pos.cash + value).max(1e-9);
    [
        if pos.yes > 0 { (value / equity) as f32 } else { 0.0 },
        if pos.no > 0 { (value / equity) as f32 } else { 0.0 },
        avg as f32,
        mark.map(|m| m - avg).filter(|_| held > 0).unwrap_or(0.0) as f32,
        if held > 0 { 1.0 } else { 0.0 },
    ]
}

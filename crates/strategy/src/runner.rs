//! The `Strategy` trait and the bookkeeping every implementation shares:
//! history, the active window, decision cadence, and action masking.

use crate::features::{market_features, MARKET_FEATURES};
use crate::fees::{fee_per_contract, max_sell_without_debt, taker_fee, TAKER_RATE};
use crate::history::History;
use crate::{choice, Action, Decision, FairValue, Position, Snapshot, TradeLimits, Window, N_CHOICES};

/// Feed market state at every instant; get back what to do now.
pub trait Strategy: Send {
    /// A new 15-minute session starts. History of BRTI is kept.
    fn begin_window(&mut self, w: &Window);

    /// Called for every new snapshot (about once a second). Returns
    /// `Action::Nothing` between decision times, outside the window, and when
    /// inputs are unusable.
    fn on_tick(&mut self, s: &Snapshot, pos: &Position) -> Decision;

    fn limits(&self) -> TradeLimits;

    /// Run a whole series through `on_tick` from `cash`, assuming every
    /// decision fills at its limit price immediately (as far as cash allows).
    fn decide_series(&mut self, w: &Window, series: &[Snapshot], cash: f64) -> Vec<Decision> {
        self.begin_window(w);
        let mut pos = Position::with_cash(cash);
        series
            .iter()
            .map(|s| {
                let d = self.on_tick(s, &pos);
                if let Some(px) = d.limit.filter(|_| d.size > 0) {
                    let n = fill_size(&pos, d.action, d.size, px, None, &self.limits(), TAKER_RATE);
                    apply_fill(&mut pos, d.action, n, px, taker_fee(n, px, TAKER_RATE));
                }
                d
            })
            .collect()
    }
}

/// How many of `wanted` contracts fill at `px` given the account at fill
/// time: sells are capped by what is held, buys by the cap, the other side
/// being flat, and cash for the contracts plus the fee; both by `touch`, the
/// size shown at that price (unknown means unlimited). No fill ever takes
/// cash below zero.
pub fn fill_size(pos: &Position, action: Action, wanted: u32, px: f64, touch: Option<f64>, limits: &TradeLimits, fee_rate: f64) -> u32 {
    let mut n = wanted;
    if let Some(t) = touch {
        n = n.min(t.max(0.0).floor() as u32);
    }
    match action {
        Action::Nothing => return 0,
        Action::SellYes => return max_sell_without_debt(pos.cash, n.min(pos.yes), px, fee_rate),
        Action::SellNo => return max_sell_without_debt(pos.cash, n.min(pos.no), px, fee_rate),
        Action::BuyYes if pos.no > 0 => return 0,
        Action::BuyNo if pos.yes > 0 => return 0,
        Action::BuyYes => n = n.min(limits.max_contracts.saturating_sub(pos.yes)),
        Action::BuyNo => n = n.min(limits.max_contracts.saturating_sub(pos.no)),
    }
    // The fee rounds up to the cent, so step down until it fits.
    n = n.min((pos.cash.max(0.0) / (px + fee_per_contract(px, fee_rate))).floor() as u32);
    while n > 0 && n as f64 * px + taker_fee(n, px, fee_rate) > pos.cash + 1e-9 {
        n -= 1;
    }
    n
}

/// Account bookkeeping for a fill of `size` contracts at `price`, paying `fee`.
pub fn apply_fill(pos: &mut Position, action: Action, size: u32, price: f64, fee: f64) {
    if action == Action::Nothing || size == 0 {
        return;
    }
    let notional = size as f64 * price;
    pos.cash += if action.is_buy() { -notional } else { notional } - fee;
    let reduce = |held: &mut u32, cost: &mut f64| {
        let avg = if *held > 0 { *cost / *held as f64 } else { 0.0 };
        let n = size.min(*held);
        *held -= n;
        *cost = if *held == 0 { 0.0 } else { *cost - avg * n as f64 };
    };
    match action {
        Action::Nothing => {}
        Action::BuyYes => {
            pos.yes += size;
            pos.cost += size as f64 * price;
        }
        Action::BuyNo => {
            pos.no += size;
            pos.cost += size as f64 * price;
        }
        Action::SellYes => reduce(&mut pos.yes, &mut pos.cost),
        Action::SellNo => reduce(&mut pos.no, &mut pos.cost),
    }
    debug_assert!(pos.cash >= -1e-9, "cash went negative: {}", pos.cash);
}

/// What a strategy sees at a decision time.
pub struct Ctx<'a> {
    pub window: &'a Window,
    pub hist: &'a History,
    pub snap: Snapshot,
    pub market: [f32; MARKET_FEATURES],
    pub fv: FairValue,
    /// Allowed choices (see `choice`) for the account passed to `tick`.
    pub mask: [bool; N_CHOICES],
    pub pos: Position,
}

#[derive(Debug, Clone)]
pub struct Runner {
    pub hist: History,
    pub window: Option<Window>,
    pub limits: TradeLimits,
    pub every_s: u32,
    last_slot: Option<i64>,
}

impl Runner {
    pub fn new(limits: TradeLimits, every_s: u32) -> Self {
        Self { hist: History::default(), window: None, limits, every_s: every_s.max(1), last_slot: None }
    }

    pub fn begin_window(&mut self, w: &Window) {
        self.window = Some(w.clone());
        self.last_slot = None;
        self.hist.clear_quotes();
    }

    /// Records `s` and, at a decision time inside the window, returns the
    /// context to decide on.
    pub fn tick(&mut self, s: &Snapshot, pos: &Position) -> Option<Ctx<'_>> {
        self.hist.push(*s);
        let w = self.window.as_ref()?;
        if s.t_ms < w.open_ms || s.t_ms >= w.close_ms || s.btc <= 0.0 {
            return None;
        }
        let slot = (s.t_ms - w.open_ms) / (self.every_s as i64 * 1000);
        if self.last_slot == Some(slot) {
            return None;
        }
        self.last_slot = Some(slot);
        let (market, fv) = market_features(w, &self.hist)?;
        Some(Ctx { window: w, hist: &self.hist, snap: *s, market, fv, mask: self.limits.mask(s, pos), pos: *pos })
    }

    /// The fair value at the latest snapshot, for logging outside decisions.
    pub fn fair_value(&self) -> f64 {
        self.window.as_ref().and_then(|w| crate::fair::fair_value(w, &self.hist)).map(|f| f.p).unwrap_or(f64::NAN)
    }
}

/// A decision for choice `k` (see `choice`), assuming the mask allowed it.
/// `probs` is the policy over choices; the decision reports it per action.
pub fn decide(k: usize, ctx: &Ctx, limits: &TradeLimits, probs: &[f32; N_CHOICES], fair_value: f64) -> Decision {
    let mut by_action = [0f32; 5];
    for (j, p) in probs.iter().enumerate() {
        by_action[choice(j).0.index()] += p;
    }
    let (action, fraction) = choice(k);
    let size = limits.order_size(action, fraction, &ctx.snap, &ctx.pos);
    if action == Action::Nothing || size == 0 {
        return Decision { probs: by_action, ..Decision::nothing(fair_value) };
    }
    Decision { action, size, limit: TradeLimits::limit(action, &ctx.snap), probs: by_action, fair_value }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Quote;

    /// Strategies must decide the same on a session whether or not another
    /// session was replayed through them first.
    #[test]
    fn decisions_do_not_depend_on_previous_sessions() {
        let q = Quote { bid: Some(0.30), ask: Some(0.31), bid_size: Some(100.0), ask_size: Some(100.0) };
        let mk = |open: i64, btc: f64| -> (Window, Vec<Snapshot>) {
            let w = Window { ticker: format!("T{open}"), strike: 100_000.0, open_ms: open, close_ms: open + 900_000 };
            let snaps = (-900..900)
                .map(|k| {
                    let t = open + k * 1000;
                    let wiggle = if k % 2 == 0 { 3.0 } else { -3.0 };
                    let quotes = if k >= 0 { q } else { Quote::default() };
                    Snapshot { t_ms: t, btc: btc + wiggle + k as f64 * 0.01, yes: quotes, no: quotes }
                })
                .collect();
            (w, snaps)
        };
        let (w1, s1) = mk(10_000_000, 100_010.0);
        let (w2, s2) = mk(10_900_000, 100_020.0);

        let alone = crate::FairValueStrategy::new(TradeLimits::default(), 5, 0.0).decide_series(&w2, &s2, 1000.0);
        let mut after = crate::FairValueStrategy::new(TradeLimits::default(), 5, 0.0);
        after.decide_series(&w1, &s1, 1000.0);
        assert_eq!(alone, after.decide_series(&w2, &s2, 1000.0));
    }
}

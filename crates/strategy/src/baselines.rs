//! Non-learned strategies the RL policy has to beat.

use crate::fees::{fee_per_contract, TAKER_RATE};
use crate::runner::{decide, Runner, Strategy};
use crate::{choice_index, Action, Decision, Position, Snapshot, TradeLimits, Window, N_CHOICES, SIZE_FRACTIONS};

/// Never trades. Its PnL is exactly zero; anything else is a simulator bug.
pub struct NothingStrategy {
    runner: Runner,
}

impl NothingStrategy {
    pub fn new(limits: TradeLimits) -> Self {
        Self { runner: Runner::new(limits, 1) }
    }
}

impl Strategy for NothingStrategy {
    fn begin_window(&mut self, w: &Window) {
        self.runner.begin_window(w);
    }

    fn on_tick(&mut self, s: &Snapshot, _pos: &Position) -> Decision {
        self.runner.hist.push(*s);
        Decision::nothing(self.runner.fair_value())
    }

    fn limits(&self) -> TradeLimits {
        self.runner.limits
    }
}

/// Trades when the analytic fair value disagrees with the book by more than
/// `edge` after fees: buys the cheap side with `BUY_FRACTION` of cash, exits
/// the whole position when the bid overshoots fair.
pub struct FairValueStrategy {
    runner: Runner,
    /// Minimum edge per contract, in dollars, beyond fees.
    pub edge: f64,
}

/// Index into `SIZE_FRACTIONS`: 10% of cash per buy.
const BUY_FRACTION: usize = 1;

impl FairValueStrategy {
    pub fn new(limits: TradeLimits, every_s: u32, edge: f64) -> Self {
        Self { runner: Runner::new(limits, every_s), edge }
    }
}

impl Strategy for FairValueStrategy {
    fn begin_window(&mut self, w: &Window) {
        self.runner.begin_window(w);
    }

    fn on_tick(&mut self, s: &Snapshot, pos: &Position) -> Decision {
        let limits = self.runner.limits;
        let edge = self.edge;
        let Some(ctx) = self.runner.tick(s, pos) else {
            return Decision::nothing(self.runner.fair_value());
        };
        let p = ctx.fv.p;
        let gain = |action: Action| -> f64 {
            let Some(px) = TradeLimits::limit(action, &ctx.snap) else { return f64::NEG_INFINITY };
            let fee = fee_per_contract(px, TAKER_RATE);
            match action {
                Action::BuyYes => p - px - fee,
                Action::BuyNo => (1.0 - p) - px - fee,
                Action::SellYes => px - p - fee,
                Action::SellNo => px - (1.0 - p) - fee,
                Action::Nothing => 0.0,
            }
        };
        let size_of = |a: Action| if a.is_buy() { BUY_FRACTION } else { SIZE_FRACTIONS.len() - 1 };
        let best = [Action::SellYes, Action::SellNo, Action::BuyYes, Action::BuyNo]
            .into_iter()
            .map(|a| (choice_index(a, size_of(a)), gain(a)))
            .filter(|(k, g)| ctx.mask[*k] && *g > edge)
            .max_by(|a, b| a.1.total_cmp(&b.1));
        let k = best.map(|(k, _)| k).unwrap_or(0);
        let mut probs = [0.0; N_CHOICES];
        probs[k] = 1.0;
        decide(k, &ctx, &limits, &probs, p)
    }

    fn limits(&self) -> TradeLimits {
        self.runner.limits
    }
}

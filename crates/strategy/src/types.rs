//! The inference interface's data types: raw market state in, decisions out.

use serde::{Deserialize, Serialize};

use crate::fees::{fee_per_contract, max_sell_without_debt, TAKER_RATE};

/// One 15-minute session: the market that pays out at `close_ms`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Window {
    pub ticker: String,
    /// Average BRTI over the 60s before open; YES pays if the closing
    /// 60s average is at least this.
    pub strike: f64,
    pub open_ms: i64,
    pub close_ms: i64,
}

impl Window {
    pub fn duration_s(&self) -> f64 {
        (self.close_ms - self.open_ms) as f64 / 1000.0
    }
}

/// Top of book for one side (YES or NO), in dollars per contract.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Quote {
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    pub bid_size: Option<f64>,
    pub ask_size: Option<f64>,
}

impl Quote {
    pub fn mid(&self) -> Option<f64> {
        Some((self.bid? + self.ask?) / 2.0)
    }

    pub fn spread(&self) -> Option<f64> {
        Some(self.ask? - self.bid?)
    }
}

/// One instant of market state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    pub t_ms: i64,
    /// BRTI spot, the settlement index.
    pub btc: f64,
    pub yes: Quote,
    pub no: Quote,
}

/// The account: cash plus contracts held. At most one side is non-zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub yes: u32,
    pub no: u32,
    /// Dollars paid for the contracts currently held (fees excluded).
    pub cost: f64,
    /// Dollars available to buy with, after every fill and fee so far.
    pub cash: f64,
}

impl Position {
    pub fn with_cash(cash: f64) -> Self {
        Self { cash, ..Default::default() }
    }

    pub fn held(&self) -> u32 {
        self.yes.max(self.no)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Action {
    Nothing,
    BuyYes,
    SellYes,
    BuyNo,
    SellNo,
}

impl Action {
    pub const ALL: [Action; 5] = [Action::Nothing, Action::BuyYes, Action::SellYes, Action::BuyNo, Action::SellNo];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn from_index(i: usize) -> Action {
        Action::ALL[i]
    }

    pub fn is_buy(self) -> bool {
        matches!(self, Action::BuyYes | Action::BuyNo)
    }
}

/// Order sizes a strategy picks from: buys spend this fraction of cash,
/// sells close this fraction of the contracts held.
pub const SIZE_FRACTIONS: [f64; 5] = [0.05, 0.10, 0.25, 0.50, 1.0];

/// Policy choices: `Nothing`, then every (trading action, size fraction).
pub const N_CHOICES: usize = 1 + 4 * SIZE_FRACTIONS.len();

/// Choice index `k` as (action, fraction); `Nothing` has fraction 0.
pub fn choice(k: usize) -> (Action, f64) {
    if k == 0 {
        return (Action::Nothing, 0.0);
    }
    let n = SIZE_FRACTIONS.len();
    (Action::ALL[1 + (k - 1) / n], SIZE_FRACTIONS[(k - 1) % n])
}

/// Index of `action` with `SIZE_FRACTIONS[frac_idx]`.
pub fn choice_index(action: Action, frac_idx: usize) -> usize {
    if action == Action::Nothing { 0 } else { 1 + (action.index() - 1) * SIZE_FRACTIONS.len() + frac_idx }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    pub action: Action,
    /// Contracts.
    pub size: u32,
    /// Price to cross at: the ask for buys, the bid for sells.
    pub limit: Option<f64>,
    /// Policy probability of each of `Action::ALL`, summed over sizes, for inspection.
    pub probs: [f32; 5],
    /// The strategy's estimate of P(YES).
    pub fair_value: f64,
}

impl Decision {
    pub fn nothing(fair_value: f64) -> Self {
        Decision { action: Action::Nothing, size: 0, limit: None, probs: [1.0, 0.0, 0.0, 0.0, 0.0], fair_value }
    }
}

/// Sizing and risk limits shared by every strategy and the simulator.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TradeLimits {
    /// Most contracts held on one side, whatever cash allows.
    pub max_contracts: u32,
}

impl Default for TradeLimits {
    fn default() -> Self {
        TradeLimits { max_contracts: 5000 }
    }
}

fn tradable(p: Option<f64>) -> Option<f64> {
    p.filter(|p| *p > 0.0 && *p < 1.0)
}

impl TradeLimits {
    /// Contracts an order of `fraction` would be for at `s`, or 0 if it isn't
    /// allowed: sell only what is held, never hold both sides, stay under the
    /// cap, spend only cash on hand (fees included), and need a quote to cross.
    pub fn order_size(&self, action: Action, fraction: f64, s: &Snapshot, pos: &Position) -> u32 {
        let buy = |px: Option<f64>, held: u32, other: u32| -> u32 {
            let Some(px) = tradable(px) else { return 0 };
            if other > 0 {
                return 0;
            }
            let afford = (fraction * pos.cash.max(0.0) / (px + fee_per_contract(px, TAKER_RATE))).floor() as u32;
            afford.min(self.max_contracts.saturating_sub(held))
        };
        let sell = |px: Option<f64>, held: u32| -> u32 {
            let Some(px) = tradable(px) else { return 0 };
            if held == 0 {
                return 0;
            }
            let n = ((fraction * held as f64).ceil() as u32).clamp(1, held);
            max_sell_without_debt(pos.cash, n, px, TAKER_RATE)
        };
        match action {
            Action::Nothing => 0,
            Action::BuyYes => buy(s.yes.ask, pos.yes, pos.no),
            Action::SellYes => sell(s.yes.bid, pos.yes),
            Action::BuyNo => buy(s.no.ask, pos.no, pos.yes),
            Action::SellNo => sell(s.no.bid, pos.no),
        }
    }

    /// Which choices (see `choice`) are executable now.
    pub fn mask(&self, s: &Snapshot, pos: &Position) -> [bool; N_CHOICES] {
        let mut m = [false; N_CHOICES];
        m[0] = true;
        for (k, slot) in m.iter_mut().enumerate().skip(1) {
            let (a, f) = choice(k);
            *slot = self.order_size(a, f, s, pos) > 0;
        }
        m
    }

    /// Price an action would cross at.
    pub fn limit(action: Action, s: &Snapshot) -> Option<f64> {
        match action {
            Action::Nothing => None,
            Action::BuyYes => s.yes.ask,
            Action::SellYes => s.yes.bid,
            Action::BuyNo => s.no.ask,
            Action::SellNo => s.no.bid,
        }
    }
}

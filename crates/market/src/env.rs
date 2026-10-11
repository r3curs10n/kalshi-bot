//! Trading simulator over a recorded episode.
//!
//! The account starts with `balance` dollars. Orders are takers for a
//! strategy-chosen number of contracts: they fill at the touch observed
//! `latency` seconds after the decision (the market may have moved),
//! optionally one tick worse, for as many contracts as the touch shows and
//! the account can pay for (partial fills happen). Fees follow Kalshi's
//! quadratic schedule. Positions are marked at mid, and at close at the
//! $1/$0 payout, so the sum of step rewards equals realized PnL exactly.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use strategy::fees::{taker_fee, tick_size, TAKER_RATE};
use strategy::{apply_fill, fill_size, Action, Position, Snapshot, Strategy, TradeLimits};

use crate::replay::Episode;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SimConfig {
    pub fee_rate: f64,
    /// Fill delay in seconds, drawn uniformly from this inclusive range.
    pub latency_s: (u32, u32),
    /// Chance a fill lands one tick worse than the observed touch.
    pub slip_prob: f64,
    pub limits: TradeLimits,
    /// Starting cash for every episode, in dollars.
    pub balance: f64,
}

pub const DEFAULT_BALANCE: f64 = 1000.0;

impl Default for SimConfig {
    fn default() -> Self {
        Self { fee_rate: TAKER_RATE, latency_s: (1, 1), slip_prob: 0.0, limits: TradeLimits::default(), balance: DEFAULT_BALANCE }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Fill {
    pub idx: usize,
    pub action: Action,
    pub size: u32,
    pub price: f64,
    pub fee: f64,
}

pub struct Sim<'a> {
    pub ep: &'a Episode,
    pub cfg: SimConfig,
    /// Contracts and cash.
    pub pos: Position,
    pub fees: f64,
    pub fills: Vec<Fill>,
    /// Orders that found no price or size.
    pub missed: u32,
    last_mark: [f64; 2],
    rng: SmallRng,
}

impl<'a> Sim<'a> {
    pub fn new(ep: &'a Episode, cfg: SimConfig, seed: u64) -> Self {
        Self {
            ep,
            cfg,
            pos: Position::with_cash(cfg.balance),
            fees: 0.0,
            fills: vec![],
            missed: 0,
            last_mark: [0.5, 0.5],
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Sends an order for `size` contracts, decided at snapshot `idx`.
    /// Returns the fill, if any; it may be for fewer contracts.
    pub fn act(&mut self, idx: usize, action: Action, size: u32) -> Option<Fill> {
        if action == Action::Nothing || size == 0 {
            return None;
        }
        let (lo, hi) = self.cfg.latency_s;
        let lat = if hi > lo { self.rng.random_range(lo..=hi) } else { lo } as usize;
        let Some(at) = self.ep.snapshots.get(idx + lat) else {
            self.missed += 1;
            return None;
        };
        let (px, touch) = match action {
            Action::BuyYes => (at.yes.ask, at.yes.ask_size),
            Action::SellYes => (at.yes.bid, at.yes.bid_size),
            Action::BuyNo => (at.no.ask, at.no.ask_size),
            Action::SellNo => (at.no.bid, at.no.bid_size),
            Action::Nothing => unreachable!(),
        };
        let Some(mut px) = px else {
            self.missed += 1;
            return None;
        };
        if self.cfg.slip_prob > 0.0 && self.rng.random_bool(self.cfg.slip_prob) {
            let t = tick_size(px);
            px = if action.is_buy() { px + t } else { px - t };
        }
        px = px.clamp(0.001, 0.999);
        let n = fill_size(&self.pos, action, size, px, touch, &self.cfg.limits, self.cfg.fee_rate);
        if n == 0 {
            self.missed += 1;
            return None;
        }
        let fee = taker_fee(n, px, self.cfg.fee_rate);
        self.fees += fee;
        apply_fill(&mut self.pos, action, n, px, fee);
        let f = Fill { idx: idx + lat, action, size: n, price: px, fee };
        self.fills.push(f);
        Some(f)
    }

    /// Profit so far: cash plus holdings at mid as of snapshot `idx`, less
    /// the starting balance.
    pub fn equity(&mut self, idx: usize) -> f64 {
        let s: &Snapshot = &self.ep.snapshots[idx.min(self.ep.snapshots.len() - 1)];
        if let Some(m) = s.yes.mid() {
            self.last_mark[0] = m;
        }
        if let Some(m) = s.no.mid() {
            self.last_mark[1] = m;
        }
        self.pos.cash + self.pos.yes as f64 * self.last_mark[0] + self.pos.no as f64 * self.last_mark[1] - self.cfg.balance
    }

    /// Cash after the $1/$0 payout, less the starting balance: the episode's
    /// realized PnL.
    pub fn settle(&self) -> f64 {
        let payout = if self.ep.result_yes { self.pos.yes } else { self.pos.no };
        self.pos.cash + payout as f64 - self.cfg.balance
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeResult {
    pub ticker: String,
    pub pnl: f64,
    pub fees: f64,
    pub trades: usize,
    pub missed: u32,
    pub result_yes: bool,
    pub fills: Vec<Fill>,
}

/// Plays a `Strategy` through an episode, tick by tick, as it would run live.
pub fn run_strategy(strategy: &mut dyn Strategy, ep: &Episode, cfg: SimConfig, seed: u64) -> EpisodeResult {
    run_strategy_traced(strategy, ep, cfg, seed, &mut |_, _| {})
}

/// `run_strategy`, also reporting equity (cash plus holdings at mid) after
/// every in-window snapshot as `(t_ms, equity)`.
pub fn run_strategy_traced(
    strategy: &mut dyn Strategy,
    ep: &Episode,
    cfg: SimConfig,
    seed: u64,
    on_equity: &mut dyn FnMut(i64, f64),
) -> EpisodeResult {
    strategy.begin_window(&ep.window);
    let mut sim = Sim::new(ep, cfg, seed);
    let open = ep.open_idx();
    for (i, s) in ep.snapshots.iter().enumerate() {
        let d = strategy.on_tick(s, &sim.pos);
        if i >= open && d.action != Action::Nothing {
            sim.act(i, d.action, d.size);
        }
        if i >= open {
            on_equity(s.t_ms, sim.equity(i));
        }
    }
    EpisodeResult {
        ticker: ep.window.ticker.clone(),
        pnl: sim.settle(),
        fees: sim.fees,
        trades: sim.fills.len(),
        missed: sim.missed,
        result_yes: ep.result_yes,
        fills: sim.fills,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use strategy::fees::tick_size;
    use strategy::{Quote, Window};

    fn ep(result_yes: bool) -> Episode {
        let q = |b: f64, a: f64| Quote { bid: Some(b), ask: Some(a), bid_size: Some(100.0), ask_size: Some(100.0) };
        let snaps = (0..10)
            .map(|i| {
                let yes = q(0.40 + 0.01 * i as f64, 0.42 + 0.01 * i as f64);
                Snapshot { t_ms: i * 1000, btc: 1.0, yes, no: crate::replay::complement(&yes) }
            })
            .collect();
        Episode {
            window: Window { ticker: "T".into(), strike: 1.0, open_ms: 0, close_ms: 10_000 },
            result_yes,
            expiration_value: None,
            snapshots: snaps,
        }
    }

    #[test]
    fn rewards_telescope_to_realized_pnl() {
        let e = ep(true);
        let cfg = SimConfig { latency_s: (0, 2), slip_prob: 0.5, ..Default::default() };
        let mut sim = Sim::new(&e, cfg, 7);
        let plan = [(Action::BuyYes, 10), (Action::Nothing, 0), (Action::BuyYes, 25), (Action::SellYes, 20), (Action::Nothing, 0)];
        let mut total = 0.0;
        let mut prev = sim.equity(0);
        for (k, a) in plan.iter().enumerate() {
            let idx = k * 2;
            let before = sim.equity(idx);
            assert!((before - prev).abs() < 1e-12 || k == 0);
            sim.act(idx, a.0, a.1);
            let after = if k + 1 == plan.len() { sim.settle() } else { sim.equity(idx + 2) };
            total += after - before;
            prev = after;
        }
        assert!((total - sim.settle()).abs() < 1e-9, "{total} vs {}", sim.settle());
        // Long YES into a YES result: paid ~0.42-0.47, received $1.
        assert!(sim.settle() > 0.0);
    }

    #[test]
    fn buys_never_spend_more_than_cash() {
        let e = ep(false);
        let cfg = SimConfig { balance: 20.0, ..Default::default() };
        let mut sim = Sim::new(&e, cfg, 0);
        let f = sim.act(0, Action::BuyYes, 10_000).expect("partial fill");
        assert!(f.size < 10_000 && f.size > 0);
        assert!(sim.pos.cash >= 0.0, "cash {}", sim.pos.cash);
        // Not enough left for even one more contract.
        assert!(sim.pos.cash < f.price + 0.02);
        assert!(sim.act(2, Action::BuyYes, 1).is_none());
    }

    #[test]
    fn cash_never_goes_negative() {
        // Prices from the extremes of the ladder to the middle, with slippage
        // and a balance small enough that rounding and fees bite.
        let mut rng = SmallRng::seed_from_u64(3);
        let prices = [0.002, 0.01, 0.05, 0.3, 0.5, 0.7, 0.95, 0.99, 0.998];
        let snaps = (0..400)
            .map(|i| {
                let p = prices[rng.random_range(0..prices.len())];
                let t = tick_size(p);
                let yes = Quote {
                    bid: Some((p - t).max(0.001)),
                    ask: Some((p + t).min(0.999)),
                    bid_size: Some(rng.random_range(1..50) as f64),
                    ask_size: Some(rng.random_range(1..50) as f64),
                };
                Snapshot { t_ms: i * 1000, btc: 1.0, yes, no: crate::replay::complement(&yes) }
            })
            .collect();
        let e = Episode {
            window: Window { ticker: "T".into(), strike: 1.0, open_ms: 0, close_ms: 400_000 },
            result_yes: true,
            expiration_value: None,
            snapshots: snaps,
        };
        let actions = [Action::BuyYes, Action::SellYes, Action::BuyNo, Action::SellNo];
        for seed in 0..50 {
            let cfg = SimConfig { latency_s: (0, 2), slip_prob: 0.5, balance: 0.01 + seed as f64 * 0.03, ..Default::default() };
            let mut sim = Sim::new(&e, cfg, seed);
            for idx in 0..e.snapshots.len() {
                let a = actions[rng.random_range(0..actions.len())];
                // Sized the way strategies size orders, including the all-in buy.
                let frac = [0.05, 0.1, 0.25, 0.5, 1.0][rng.random_range(0..5)];
                let size = cfg.limits.order_size(a, frac, &e.snapshots[idx], &sim.pos).max(rng.random_range(0..3));
                sim.act(idx, a, size);
                assert!(sim.pos.cash >= -1e-9, "seed {seed} idx {idx}: cash {}", sim.pos.cash);
            }
        }
    }

    #[test]
    fn a_sale_whose_fee_exceeds_its_proceeds_is_refused_when_cash_is_gone() {
        // YES at 0.001 / 0.002. With 1.2 cents, buying one contract (0.2 cents
        // plus a 1-cent fee) leaves nothing; selling it back would bring in
        // 0.1 cents and owe another cent.
        let yes = Quote { bid: Some(0.001), ask: Some(0.002), bid_size: Some(10.0), ask_size: Some(10.0) };
        let snaps = (0..4).map(|i| Snapshot { t_ms: i * 1000, btc: 1.0, yes, no: crate::replay::complement(&yes) }).collect();
        let e = Episode {
            window: Window { ticker: "T".into(), strike: 1.0, open_ms: 0, close_ms: 4_000 },
            result_yes: false,
            expiration_value: None,
            snapshots: snaps,
        };
        let cfg = SimConfig { latency_s: (0, 0), balance: 0.012, ..Default::default() };
        let mut sim = Sim::new(&e, cfg, 0);
        assert_eq!(sim.act(0, Action::BuyYes, 1).unwrap().size, 1);
        assert!(sim.pos.cash.abs() < 1e-9);
        assert!(sim.act(1, Action::SellYes, 1).is_none());
        assert_eq!(cfg.limits.order_size(Action::SellYes, 1.0, &e.snapshots[1], &sim.pos), 0, "masked too");
        assert!(sim.pos.cash >= 0.0);
    }

    #[test]
    fn fills_are_capped_by_the_touch() {
        let e = ep(true);
        let mut sim = Sim::new(&e, SimConfig::default(), 0);
        // The test book shows 100 contracts at the ask.
        assert_eq!(sim.act(0, Action::BuyYes, 500).unwrap().size, 100);
    }

    #[test]
    fn masks_block_selling_unheld_and_both_sides() {
        let e = ep(false);
        let cfg = SimConfig { limits: TradeLimits { max_contracts: 50 }, ..Default::default() };
        let mut sim = Sim::new(&e, cfg, 1);
        assert!(sim.act(0, Action::SellYes, 10).is_none());
        assert!(sim.act(0, Action::BuyNo, 10).is_some());
        assert!(sim.act(2, Action::BuyYes, 10).is_none(), "can't hold both sides");
        assert_eq!(sim.act(0, Action::SellNo, 99).unwrap().size, 10, "sell only what is held");
        for _ in 0..7 {
            sim.act(0, Action::BuyNo, 10);
        }
        assert_eq!(sim.pos.no, 50);
    }

    #[test]
    fn nothing_strategy_makes_exactly_zero() {
        let e = ep(true);
        let mut s = strategy::NothingStrategy::new(TradeLimits::default());
        let r = run_strategy(&mut s, &e, SimConfig::default(), 0);
        assert_eq!(r.pnl, 0.0);
        assert_eq!(r.trades, 0);
    }
}

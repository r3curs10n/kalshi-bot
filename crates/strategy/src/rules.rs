//! Hand-written rules, optionally gated by the pretrained BRTI forecaster.

use std::path::Path;

use anyhow::{bail, Result};
use burn::backend::ndarray::NdArrayDevice;
use burn::tensor::{Tensor, TensorData};

use crate::features::MARKET_FEATURES;
use crate::net::{LstmState, Norm, Trunk};
use crate::rl::{load_trunk, Cpu, TrunkMeta};
use crate::runner::{decide, Ctx, Runner, Strategy};
use crate::fees::{fee_per_contract, TAKER_RATE};
use crate::{choice_index, Action, Decision, Position, Snapshot, TradeLimits, Window, N_CHOICES, SIZE_FRACTIONS};

/// Waits until the last `last_s` seconds of the window, then goes all in on
/// whichever side the market favors (YES if its mid is above 50¢, NO if
/// below) and holds to settlement. Never sells. If the book was too thin to
/// fill everything, it keeps buying the same side with the cash left over.
pub struct RuleStrategy {
    runner: Runner,
    /// Seconds before close when betting starts.
    pub last_s: f64,
    pub opts: RuleOpts,
    rule_state: RuleState,
}

/// Entry and exit conditions for the rules, all in dollars per contract of
/// the side being bought or held.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RuleOpts {
    /// Sell everything once the held side's bid is at or below this, then
    /// stay out for the rest of the window.
    pub stop: Option<f64>,
    /// Don't buy while the favorite's ask is above this.
    pub cap: Option<f64>,
    /// Buy only when the fair value's win probability beats the worst
    /// fill (ask + `ENTRY_SLIP`) plus its fee by more than this. `None`
    /// buys the favorite whatever it costs.
    pub edge: Option<f64>,
}

/// How much worse than the ask a buy may fill: live orders reach a cent
/// through the touch, so the cost check prices that in.
pub const ENTRY_SLIP: f64 = 0.01;

impl Default for RuleOpts {
    fn default() -> Self {
        Self { stop: None, cap: None, edge: Some(0.0) }
    }
}

impl RuleOpts {
    /// Parses `stop=0.55,cap=0.90,edge=0.02` (any of them, or empty);
    /// `edge=off` drops the cost check.
    pub fn parse(opts: &str) -> Result<Self> {
        let mut e = RuleOpts::default();
        for kv in opts.split(',').filter(|s| !s.is_empty()) {
            let (k, v) = kv.split_once('=').ok_or_else(|| anyhow::anyhow!("expected key=value, got {kv:?}"))?;
            if (k, v) == ("edge", "off") {
                e.edge = None;
                continue;
            }
            let v: f64 = v.parse().map_err(|_| anyhow::anyhow!("bad number in {kv:?}"))?;
            if !(0.0..=1.0).contains(&v) {
                bail!("{kv:?}: prices are dollars between 0 and 1");
            }
            match k {
                "stop" => e.stop = Some(v),
                "cap" => e.cap = Some(v),
                "edge" => e.edge = Some(v),
                _ => bail!("unknown rule option {k:?}: use stop=<price>, cap=<price> or edge=<dollars|off>"),
            }
        }
        Ok(e)
    }

    /// Whether buying `side` at `s` is expected to pay: the fair value's
    /// chance of `side` winning, times the $1 payout, against the worst fill
    /// plus its fee and the margin. Always true without a cost check.
    pub fn covers_cost(&self, side: Action, s: &Snapshot, p_yes: f64) -> bool {
        let Some(margin) = self.edge else { return true };
        let Some(ask) = TradeLimits::limit(side, s) else { return false };
        let worst = (ask + ENTRY_SLIP).min(0.999);
        let p_win = if side == Action::BuyYes { p_yes } else { 1.0 - p_yes };
        p_win - worst - fee_per_contract(worst, TAKER_RATE) > margin
    }
}

/// Per-window state behind `RuleOpts`.
#[derive(Debug, Clone, Copy, Default)]
struct RuleState {
    stopped: bool,
}

/// Seconds before close when `RuleStrategy` starts betting, by default.
pub const RULE_LAST_S: f64 = 180.0;

/// Index into `SIZE_FRACTIONS`: all of the cash.
const ALL_IN: usize = SIZE_FRACTIONS.len() - 1;

impl RuleStrategy {
    pub fn new(limits: TradeLimits, every_s: u32, last_s: f64) -> Self {
        Self { runner: Runner::new(limits, every_s), last_s, opts: RuleOpts::default(), rule_state: RuleState::default() }
    }

    pub fn with_opts(mut self, opts: RuleOpts) -> Self {
        self.opts = opts;
        self
    }
}

/// The side the market prices above 50%, from the YES mid (or the NO mid,
/// mirrored, when YES has no two-sided quote). `None` on a tie or no quotes.
pub fn favorite(s: &Snapshot) -> Option<Action> {
    let p_yes = s.yes.mid().or_else(|| s.no.mid().map(|m| 1.0 - m))?;
    if p_yes > 0.5 {
        Some(Action::BuyYes)
    } else if p_yes < 0.5 {
        Some(Action::BuyNo)
    } else {
        None
    }
}

impl Strategy for RuleStrategy {
    fn begin_window(&mut self, w: &Window) {
        self.runner.begin_window(w);
        self.rule_state = RuleState::default();
    }

    fn on_tick(&mut self, s: &Snapshot, pos: &Position) -> Decision {
        let (limits, last_s, opts) = (self.runner.limits, self.last_s, self.opts);
        let Some(ctx) = self.runner.tick(s, pos) else {
            return Decision::nothing(self.runner.fair_value());
        };
        let k = rule_choice(&ctx, last_s, &opts, &mut self.rule_state);
        one_hot(k, &ctx, &limits)
    }

    fn limits(&self) -> TradeLimits {
        self.runner.limits
    }
}

/// The rule's choice at `ctx`: all in on the favorite inside the last
/// `last_s` seconds (if its ask is within the cap and the expected payout
/// covers the cost), else `Nothing` (choice 0). Once the held side's bid falls to the stop, it sells everything and
/// stays out of the window.
fn rule_choice(ctx: &Ctx, last_s: f64, opts: &RuleOpts, st: &mut RuleState) -> usize {
    let s = &ctx.snap;
    let held = if ctx.pos.yes > 0 { Some((Action::SellYes, s.yes.bid)) } else if ctx.pos.no > 0 { Some((Action::SellNo, s.no.bid)) } else { None };
    if let (Some(stop), Some((_, Some(bid)))) = (opts.stop, held) {
        st.stopped |= bid <= stop;
    }
    if st.stopped {
        // Keep selling until the book has taken all of it.
        return held.map(|(a, _)| choice_index(a, ALL_IN)).filter(|k| ctx.mask[*k]).unwrap_or(0);
    }
    let left_s = (ctx.window.close_ms - s.t_ms) as f64 / 1000.0;
    favorite(s)
        .filter(|_| left_s <= last_s)
        .filter(|a| opts.cap.is_none_or(|cap| TradeLimits::limit(*a, s).is_some_and(|ask| ask <= cap)))
        .filter(|a| opts.covers_cost(*a, s, ctx.fv.p))
        .map(|a| choice_index(a, ALL_IN))
        .filter(|k| ctx.mask[*k])
        .unwrap_or(0)
}

fn one_hot(k: usize, ctx: &Ctx, limits: &TradeLimits) -> Decision {
    let mut probs = [0.0; N_CHOICES];
    probs[k] = 1.0;
    decide(k, ctx, limits, &probs, ctx.fv.p)
}

/// Default for `RuleSwingStrategy::max_swing_bps`.
pub const MAX_SWING_BPS: f64 = 6.0;

/// `RuleStrategy`, except it stays out while the pretrained trunk forecasts
/// a big BRTI move over the next `last_s` seconds. The swing is the RMS of
/// the forecast log return to the trunk horizon closest to `last_s`,
/// sqrt(mean² + variance), in bps: a big expected drift or high volatility
/// both count. The trunk steps once per decision from the window's open,
/// exactly as in pretraining, so it has state by the time the rule can bet.
pub struct RuleSwingStrategy {
    runner: Runner,
    trunk: Trunk<Cpu>,
    norm: Norm,
    meta: TrunkMeta,
    state: LstmState<Cpu>,
    /// Which of the trunk's horizons the swing is read from.
    horizon: usize,
    pub last_s: f64,
    /// Skip the rule's bet while the forecast swing is above this.
    pub max_swing_bps: f64,
    pub opts: RuleOpts,
    rule_state: RuleState,
    swing: Option<f64>,
}

impl RuleSwingStrategy {
    /// A trunk directory from `kbot pretrain`, which should have been
    /// pretrained with a horizon near `last_s` (e.g. `--horizons 1,60,180`).
    pub fn load(trunk_dir: &Path, limits: TradeLimits, last_s: f64, max_swing_bps: f64) -> Result<Self> {
        let (trunk, norm, meta) = load_trunk(trunk_dir)?;
        let horizon = (0..meta.horizons.len())
            .min_by_key(|&i| (meta.horizons[i] as f64 - last_s).abs() as u64)
            .expect("a trunk has horizons");
        if (meta.horizons[horizon] as f64) < last_s / 2.0 {
            bail!(
                "{} forecasts at most {}s ahead; pretrain one with a horizon near {last_s}s (--horizons 1,60,180)",
                trunk_dir.display(),
                meta.horizons[horizon]
            );
        }
        let state = LstmState::zeros(1, meta.hidden, &NdArrayDevice::Cpu);
        let runner = Runner::new(limits, meta.every_s);
        Ok(Self {
            runner,
            trunk,
            norm,
            meta,
            state,
            horizon,
            last_s,
            max_swing_bps,
            opts: RuleOpts::default(),
            rule_state: RuleState::default(),
            swing: None,
        })
    }

    pub fn with_opts(mut self, opts: RuleOpts) -> Self {
        self.opts = opts;
        self
    }

    /// The trunk horizon the swing is read from, seconds.
    pub fn horizon_s(&self) -> usize {
        self.meta.horizons[self.horizon]
    }

    /// The forecast swing (bps) at the latest decision, if any.
    pub fn swing_bps(&self) -> Option<f64> {
        self.swing
    }

}

/// Steps `trunk` on one decision's raw market features, advancing `state`,
/// and returns the forecast swing to horizon index `h` in bps.
fn forecast_swing(trunk: &Trunk<Cpu>, norm: &Norm, h: usize, state: &mut LstmState<Cpu>, market: &[f32; MARKET_FEATURES]) -> f64 {
    let mut x = *market;
    norm.apply(&mut x);
    let t = Tensor::<Cpu, 2>::from_data(TensorData::new(x.to_vec(), [1, MARKET_FEATURES]), &NdArrayDevice::Cpu);
    *state = trunk.step(t, state.clone());
    let out: Vec<f32> = trunk.forecast(state.h.clone()).into_data().to_vec().unwrap_or_default();
    let nh = out.len() / 2;
    let (mu, logvar) = (out[h] as f64, out[nh + h] as f64);
    (mu * mu + logvar.exp()).sqrt()
}

impl Strategy for RuleSwingStrategy {
    fn begin_window(&mut self, w: &Window) {
        self.runner.begin_window(w);
        self.state = LstmState::zeros(1, self.meta.hidden, &NdArrayDevice::Cpu);
        self.swing = None;
        self.rule_state = RuleState::default();
    }

    fn on_tick(&mut self, s: &Snapshot, pos: &Position) -> Decision {
        let (limits, last_s, max_swing, opts) = (self.runner.limits, self.last_s, self.max_swing_bps, self.opts);
        let Some(ctx) = self.runner.tick(s, pos) else {
            return Decision::nothing(self.runner.fair_value());
        };
        let swing = forecast_swing(&self.trunk, &self.norm, self.horizon, &mut self.state, &ctx.market);
        self.swing = Some(swing);
        // The swing filter only gates new bets; a stop still gets out.
        let k = rule_choice(&ctx, last_s, &opts, &mut self.rule_state);
        let k = if swing > max_swing && crate::choice(k).0.is_buy() { 0 } else { k };
        one_hot(k, &ctx, &limits)
    }

    fn limits(&self) -> TradeLimits {
        self.runner.limits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Quote;

    fn series(w: &Window, yes_mid: f64) -> Vec<Snapshot> {
        let yes = Quote { bid: Some(yes_mid - 0.01), ask: Some(yes_mid + 0.01), bid_size: None, ask_size: None };
        let no = Quote { bid: Some(1.0 - yes_mid - 0.01), ask: Some(1.0 - yes_mid + 0.01), bid_size: None, ask_size: None };
        (-900..900)
            .map(|k| Snapshot { t_ms: w.open_ms + k * 1000, btc: 100_000.0 + (k % 2) as f64, yes, no })
            .collect()
    }

    #[test]
    fn bets_everything_on_the_favorite_only_near_close() {
        let w = Window { ticker: "T".into(), strike: 100_000.0, open_ms: 10_000_000, close_ms: 10_900_000 };
        for (mid, side) in [(0.70, Action::BuyYes), (0.30, Action::BuyNo)] {
            let unchecked = RuleOpts { edge: None, ..Default::default() };
            let mut s = RuleStrategy::new(TradeLimits::default(), 1, RULE_LAST_S).with_opts(unchecked);
            let snaps = series(&w, mid);
            let ds = s.decide_series(&w, &snaps, 1000.0);
            let trades: Vec<_> = snaps.iter().zip(&ds).filter(|(_, d)| d.size > 0).collect();
            assert_eq!(trades.len(), 1, "one all-in buy, then no cash left");
            let (snap, d) = trades[0];
            assert_eq!(d.action, side);
            assert_eq!(w.close_ms - snap.t_ms, 180_000);
            let px = d.limit.unwrap();
            let left = 1000.0 - d.size as f64 * px - crate::fees::taker_fee(d.size, px, crate::fees::TAKER_RATE);
            assert!(left < px + crate::fees::fee_per_contract(px, crate::fees::TAKER_RATE), "left ${left}");
        }
    }

    #[test]
    fn stop_sells_out_and_stays_out_and_cap_blocks_entry() {
        let w = Window { ticker: "T".into(), strike: 100_000.0, open_ms: 10_000_000, close_ms: 10_900_000 };
        // The favorite at 80¢ until a minute before close, then 40¢.
        let mut snaps = series(&w, 0.80);
        let drop = series(&w, 0.40);
        for (s, d) in snaps.iter_mut().zip(drop) {
            if w.close_ms - s.t_ms < 60_000 {
                *s = d;
            }
        }
        let stop = RuleOpts { stop: Some(0.50), cap: None, edge: None };
        let mut s = RuleStrategy::new(TradeLimits::default(), 1, RULE_LAST_S).with_opts(stop);
        let acts: Vec<_> = s.decide_series(&w, &snaps, 1000.0).iter().filter(|d| d.size > 0).map(|d| d.action).collect();
        assert_eq!(acts, [Action::BuyYes, Action::SellYes]);

        let cap = RuleOpts { stop: None, cap: Some(0.75), edge: None };
        let mut s = RuleStrategy::new(TradeLimits::default(), 1, RULE_LAST_S).with_opts(cap);
        assert!(s.decide_series(&w, &series(&w, 0.80), 1000.0).iter().all(|d| d.size == 0));
    }

    #[test]
    fn exits_parse() {
        assert_eq!(RuleOpts::parse("stop=0.55,cap=0.9").unwrap(), RuleOpts { stop: Some(0.55), cap: Some(0.9), edge: Some(0.0) });
        assert_eq!(RuleOpts::parse("edge=off").unwrap().edge, None);
        assert_eq!(RuleOpts::parse("edge=0.02").unwrap().edge, Some(0.02));
        assert_eq!(RuleOpts::parse("").unwrap(), RuleOpts::default());
        assert!(RuleOpts::parse("stop=55").is_err());
        assert!(RuleOpts::parse("tp=0.5").is_err());
    }

    #[test]
    fn buys_only_when_the_fair_value_covers_price_and_fee() {
        let w = Window { ticker: "T".into(), strike: 100_000.0, open_ms: 10_000_000, close_ms: 10_900_000 };
        // The market favors YES at 70¢; BRTI sits 0.2% above or below strike.
        let with_btc = |offset: f64| -> Vec<Snapshot> {
            series(&w, 0.70).into_iter().map(|s| Snapshot { btc: s.btc + offset, ..s }).collect()
        };
        let buys = |snaps: &[Snapshot], opts: RuleOpts| {
            let mut s = RuleStrategy::new(TradeLimits::default(), 1, RULE_LAST_S).with_opts(opts);
            s.decide_series(&w, snaps, 1000.0).iter().filter(|d| d.size > 0).count()
        };
        let checked = RuleOpts::default();
        assert_eq!(buys(&with_btc(200.0), checked), 1, "fair ~100% against a 71¢ ask: worth it");
        assert_eq!(buys(&with_btc(-200.0), checked), 0, "fair ~0%: YES is the favorite but a bad bet");
        assert_eq!(buys(&with_btc(-200.0), RuleOpts { edge: None, ..checked }), 1);
        assert_eq!(buys(&with_btc(200.0), RuleOpts { edge: Some(0.5), ..checked }), 0, "margin above the ~28¢ edge");
    }
}

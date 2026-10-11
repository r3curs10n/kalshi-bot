//! Stage 2: PPO on the policy head over a frozen, pretrained trunk.
//!
//! The trunk sees market features only, so its outputs over a session don't
//! depend on what the policy does. They're computed once per episode
//! (`Encoded`), exactly as `RlStrategy` computes them live, and PPO trains
//! the small heads on `[trunk output, market features, position features]`.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::Autodiff;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::log_softmax;
use burn::tensor::{Int, Tensor, TensorData};
use market::{Sim, SimConfig};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use strategy::features::position_features;
use strategy::net::{greedy, head_inputs, masked_softmax, PolicyHead};
use strategy::rl::{Cpu, Model, ModelMeta, NET};
use strategy::{choice, TradeLimits, N_CHOICES};
use tracing::info;

use crate::Prepared;

type Train = Autodiff<Cpu>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpoConfig {
    /// Width of the policy and value MLPs.
    pub head_hidden: usize,
    pub limits: TradeLimits,
    /// Simulator for training: randomized latency and slippage.
    pub sim: SimConfig,
    /// Simulator for validation/test: fixed, matching `kbot backtest` defaults.
    pub eval_sim: SimConfig,
    pub iterations: usize,
    pub n_envs: usize,
    pub rollout_steps: usize,
    pub epochs: usize,
    pub minibatch: usize,
    pub lr: f64,
    pub clip: f32,
    pub ent_coef: f32,
    pub vf_coef: f32,
    pub gamma: f32,
    pub lambda: f32,
    /// Rewards are dollars divided by this.
    pub reward_scale: f32,
    pub eval_every: usize,
    pub seed: u64,
}

impl Default for PpoConfig {
    fn default() -> Self {
        let limits = TradeLimits::default();
        Self {
            head_hidden: 32,
            limits,
            sim: SimConfig { latency_s: (0, 2), slip_prob: 0.1, limits, ..Default::default() },
            eval_sim: SimConfig { latency_s: (1, 1), slip_prob: 0.0, limits, ..Default::default() },
            iterations: 1000,
            n_envs: 64,
            rollout_steps: 128,
            epochs: 4,
            minibatch: 1024,
            lr: 3e-4,
            clip: 0.2,
            ent_coef: 0.01,
            vf_coef: 0.5,
            gamma: 1.0,
            // About a 100s credit horizon at one decision per second.
            lambda: 0.99,
            // 1% of the starting balance.
            reward_scale: (market::DEFAULT_BALANCE / 100.0) as f32,
            eval_every: 10,
            seed: 0,
        }
    }
}

/// A prepared episode with the market side of the heads' input (frozen
/// trunk output, then normalized market features) at every decision.
pub struct Encoded {
    pub prep: Prepared,
    /// `points.len() * width`, row-major.
    pub market: Vec<f32>,
    pub width: usize,
}

impl Encoded {
    /// Runs the trunk over the episode one decision at a time from a zero
    /// state: the same calls, in the same order, as `RlStrategy` makes live.
    pub fn new(prep: Prepared, model: &Model) -> Self {
        let mut state = model.initial_state();
        let mut out = vec![];
        for (_, market, _) in &prep.points {
            out.extend(model.market_step(market, &mut state));
        }
        let width = out.len() / prep.points.len().max(1);
        Self { prep, market: out, width }
    }

    /// All of `preps`, in parallel.
    pub fn all(preps: Vec<Prepared>, model: &Model) -> Vec<Self> {
        preps.into_par_iter().map(|p| Self::new(p, model)).collect()
    }
}

/// One environment replaying one encoded episode.
struct Env<'a> {
    enc: &'a Encoded,
    sim: Sim<'a>,
    k: usize,
}

impl<'a> Env<'a> {
    fn new(enc: &'a Encoded, cfg: SimConfig, seed: u64) -> Self {
        Self { enc, sim: Sim::new(&enc.prep.ep, cfg, seed), k: 0 }
    }

    /// Appends the head's input at the current decision to `out`; returns
    /// the allowed choices.
    fn obs(&self, limits: &TradeLimits, out: &mut Vec<f32>) -> [bool; N_CHOICES] {
        let w = self.enc.width;
        let snap = &self.enc.prep.ep.snapshots[self.enc.prep.points[self.k].0];
        out.extend_from_slice(&self.enc.market[self.k * w..(self.k + 1) * w]);
        out.extend_from_slice(&position_features(&self.sim.pos, snap));
        limits.mask(snap, &self.sim.pos)
    }

    /// Takes choice `k` (sized exactly as `strategy::runner::decide` sizes
    /// it live). Returns (reward in dollars, done).
    fn step(&mut self, k: usize, limits: &TradeLimits) -> (f64, bool) {
        let idx = self.enc.prep.points[self.k].0;
        let before = self.sim.equity(idx);
        let (action, fraction) = choice(k);
        let size = limits.order_size(action, fraction, &self.enc.prep.ep.snapshots[idx], &self.sim.pos);
        self.sim.act(idx, action, size);
        self.k += 1;
        if self.k < self.enc.prep.points.len() {
            (self.sim.equity(self.enc.prep.points[self.k].0) - before, false)
        } else {
            (self.sim.settle() - before, true)
        }
    }
}

fn to_tensor2(data: Vec<f32>, rows: usize, cols: usize) -> Tensor<Train, 2> {
    Tensor::from_data(TensorData::new(data, [rows, cols]), &NdArrayDevice::Cpu)
}

fn to_tensor1(data: Vec<f32>) -> Tensor<Train, 1> {
    let n = data.len();
    Tensor::from_data(TensorData::new(data, [n]), &NdArrayDevice::Cpu)
}

/// Batched forward on the non-autodiff head: (logits, values).
fn infer(head: &PolicyHead<Cpu>, obs: Vec<f32>, rows: usize) -> (Vec<f32>, Vec<f32>) {
    let cols = obs.len() / rows.max(1);
    let x = Tensor::<Cpu, 2>::from_data(TensorData::new(obs, [rows, cols]), &NdArrayDevice::Cpu);
    let (l, v) = head.forward(x);
    (l.into_data().to_vec().unwrap(), v.into_data().to_vec().unwrap())
}

#[derive(Debug, Clone, Serialize)]
pub struct EvalStats {
    pub episodes: usize,
    pub mean_pnl: f64,
    pub total_pnl: f64,
    pub trades: usize,
}

/// Greedy policy over every episode, in batches.
pub fn evaluate(head: &PolicyHead<Cpu>, eps: &[Encoded], cfg: &PpoConfig) -> EvalStats {
    let mut total = 0.0;
    let mut trades = 0;
    for (b, chunk) in eps.chunks(256).enumerate() {
        let mut envs: Vec<Env> =
            chunk.iter().enumerate().map(|(i, e)| Env::new(e, cfg.eval_sim, (b * 256 + i) as u64)).collect();
        let mut done = vec![false; envs.len()];
        while done.iter().any(|d| !d) {
            let live: Vec<usize> = (0..envs.len()).filter(|i| !done[*i]).collect();
            let mut obs = vec![];
            let masks: Vec<_> = live.iter().map(|&i| envs[i].obs(&cfg.limits, &mut obs)).collect();
            let (logits, _) = infer(head, obs, live.len());
            for (j, &i) in live.iter().enumerate() {
                let p = masked_softmax(&logits[j * N_CHOICES..(j + 1) * N_CHOICES], &masks[j]);
                if envs[i].step(greedy(&p, &masks[j]), &cfg.limits).1 {
                    done[i] = true;
                    total += envs[i].sim.settle();
                    trades += envs[i].sim.fills.len();
                }
            }
        }
    }
    EvalStats { episodes: eps.len(), mean_pnl: total / eps.len().max(1) as f64, total_pnl: total, trades }
}

pub struct TrainOutcome {
    pub best_dir: PathBuf,
    pub best_val: EvalStats,
}

/// Trains a head over `base`'s frozen trunk (its head is ignored) and saves
/// complete models under `out`.
pub fn train(cfg: &PpoConfig, base: &Model, train_eps: &[Encoded], val_eps: &[Encoded], out: &Path) -> Result<TrainOutcome> {
    if train_eps.is_empty() {
        bail!("no training episodes");
    }
    std::fs::create_dir_all(out)?;
    std::fs::write(out.join("ppo.json"), serde_json::to_vec_pretty(cfg)?)?;
    let device = NdArrayDevice::Cpu;
    let mut rng = SmallRng::seed_from_u64(cfg.seed);
    let meta = ModelMeta { net: NET.into(), trunk: base.meta.trunk.clone(), head_hidden: cfg.head_hidden, limits: cfg.limits };
    let save = |dir: &Path, head: PolicyHead<Cpu>| Model { head, meta: meta.clone(), ..base.clone() }.save(dir);

    let hidden = base.trunk.hidden();
    let width = head_inputs(hidden);
    let mut model = PolicyHead::<Train>::new(hidden, cfg.head_hidden, &device);
    let mut optim = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(0.5)))
        .init::<Train, PolicyHead<Train>>();

    let mut csv = std::fs::File::create(out.join("curve.csv"))?;
    writeln!(csv, "iter,train_ep_pnl,episodes_done,pg_loss,v_loss,entropy,val_mean_pnl,val_trades")?;

    let pick = |rng: &mut SmallRng| rng.random_range(0..train_eps.len());
    let mut envs: Vec<Env> =
        (0..cfg.n_envs).map(|i| Env::new(&train_eps[pick(&mut rng)], cfg.sim, cfg.seed ^ i as u64)).collect();

    let best_dir = out.join("best");
    let mut best: Option<EvalStats> = None;
    let (n, t_len) = (cfg.n_envs, cfg.rollout_steps);

    for iter in 1..=cfg.iterations {
        // ---- rollout
        let inference = model.valid();
        let mut obs_buf = Vec::with_capacity(n * t_len * width);
        let mut mask_buf: Vec<[bool; N_CHOICES]> = Vec::with_capacity(n * t_len);
        let mut act_buf = Vec::with_capacity(n * t_len);
        let mut logp_buf = Vec::with_capacity(n * t_len);
        let mut val_buf = Vec::with_capacity(n * t_len);
        let mut rew_buf = Vec::with_capacity(n * t_len);
        let mut done_buf = Vec::with_capacity(n * t_len);
        let (mut ep_pnl_sum, mut ep_done) = (0.0, 0usize);

        for _t in 0..t_len {
            let mut obs = Vec::with_capacity(n * width);
            let masks: Vec<_> = envs.iter().map(|e| e.obs(&cfg.limits, &mut obs)).collect();
            let (logits, values) = infer(&inference, obs.clone(), n);
            for i in 0..n {
                let p = masked_softmax(&logits[i * N_CHOICES..(i + 1) * N_CHOICES], &masks[i]);
                let u: f32 = rng.random();
                let mut acc = 0.0;
                let mut a = 0;
                for k in 0..N_CHOICES {
                    if masks[i][k] {
                        a = k;
                        acc += p[k];
                        if u < acc {
                            break;
                        }
                    }
                }
                let (r, done) = envs[i].step(a, &cfg.limits);
                act_buf.push(a as i64);
                logp_buf.push(p[a].max(1e-8).ln());
                val_buf.push(values[i]);
                rew_buf.push(r as f32 / cfg.reward_scale);
                done_buf.push(done);
                if done {
                    ep_pnl_sum += envs[i].sim.settle();
                    ep_done += 1;
                    envs[i] = Env::new(&train_eps[pick(&mut rng)], cfg.sim, rng.random());
                }
            }
            obs_buf.extend_from_slice(&obs);
            mask_buf.extend(masks);
        }

        // ---- GAE, bootstrapping from the value of each env's current state
        let mut last_obs = Vec::with_capacity(n * width);
        for e in &envs {
            e.obs(&cfg.limits, &mut last_obs);
        }
        let (_, last_v) = infer(&inference, last_obs, n);
        let total = n * t_len;
        let mut adv = vec![0f32; total];
        let mut ret = vec![0f32; total];
        for i in 0..n {
            let mut gae = 0f32;
            let mut next_v = last_v[i];
            for t in (0..t_len).rev() {
                let j = t * n + i;
                let nonterminal = if done_buf[j] { 0.0 } else { 1.0 };
                let delta = rew_buf[j] + cfg.gamma * next_v * nonterminal - val_buf[j];
                gae = delta + cfg.gamma * cfg.lambda * nonterminal * gae;
                adv[j] = gae;
                ret[j] = gae + val_buf[j];
                next_v = val_buf[j];
            }
        }
        let mean = adv.iter().sum::<f32>() / total as f32;
        let std = (adv.iter().map(|a| (a - mean).powi(2)).sum::<f32>() / total as f32).sqrt().max(1e-8);
        for a in adv.iter_mut() {
            *a = (*a - mean) / std;
        }

        // ---- PPO update
        let mut order: Vec<usize> = (0..total).collect();
        let (mut pg_sum, mut v_sum, mut ent_sum, mut batches) = (0f32, 0f32, 0f32, 0f32);
        for _ in 0..cfg.epochs {
            order.shuffle(&mut rng);
            for chunk in order.chunks(cfg.minibatch) {
                let b = chunk.len();
                let mut o = Vec::with_capacity(b * width);
                let mut m = Vec::with_capacity(b * N_CHOICES);
                let (mut a, mut lp, mut ad, mut rt) = (vec![], vec![], vec![], vec![]);
                for &j in chunk {
                    o.extend_from_slice(&obs_buf[j * width..(j + 1) * width]);
                    m.extend(mask_buf[j].iter().map(|x| if *x { 0.0f32 } else { -1e9 }));
                    a.push(act_buf[j]);
                    lp.push(logp_buf[j]);
                    ad.push(adv[j]);
                    rt.push(ret[j]);
                }
                let (logits, v) = model.forward(to_tensor2(o, b, width));
                let logp_all = log_softmax(logits + to_tensor2(m, b, N_CHOICES), 1);
                let actions = Tensor::<Train, 2, Int>::from_data(TensorData::new(a, [b, 1]), &device);
                let logp = logp_all.clone().gather(1, actions).reshape([b]);
                let ratio = (logp - to_tensor1(lp)).exp();
                let adv_t = to_tensor1(ad);
                let surr1 = ratio.clone() * adv_t.clone();
                let surr2 = ratio.clamp(1.0 - cfg.clip, 1.0 + cfg.clip) * adv_t;
                let pg_loss = surr1.min_pair(surr2).mean().neg();
                let v_loss = (v - to_tensor1(rt)).powi_scalar(2).mean();
                // Masked actions have p = 0, so they add nothing to the entropy.
                let entropy = (logp_all.clone().exp() * logp_all.clamp_min(-30.0)).sum_dim(1).mean().neg();
                let loss = pg_loss.clone() + v_loss.clone() * cfg.vf_coef - entropy.clone() * cfg.ent_coef;

                pg_sum += pg_loss.into_scalar();
                v_sum += v_loss.into_scalar();
                ent_sum += entropy.into_scalar();
                batches += 1.0;

                let grads = GradientsParams::from_grads(loss.backward(), &model);
                model = optim.step(cfg.lr, model, grads);
            }
        }

        // Episodes run longer than one rollout, so some rollouts finish none.
        let train_pnl = if ep_done > 0 { ep_pnl_sum / ep_done as f64 } else { f64::NAN };
        let train_txt = if ep_done > 0 { format!("${train_pnl:.3}") } else { "-".into() };
        let mut val_cols = (String::new(), String::new());
        if iter % cfg.eval_every == 0 || iter == cfg.iterations {
            let head = model.valid();
            let stats = evaluate(&head, val_eps, cfg);
            info!(
                "iter {iter}: train ep pnl {train_txt}  val mean ${:.3} total ${:.2} trades {}",
                stats.mean_pnl, stats.total_pnl, stats.trades
            );
            val_cols = (format!("{:.4}", stats.mean_pnl), stats.trades.to_string());
            if best.as_ref().is_none_or(|b| stats.mean_pnl > b.mean_pnl) || val_eps.is_empty() {
                save(&best_dir, head.clone())?;
                best = Some(stats);
            }
            save(&out.join("last"), head)?;
        } else {
            info!(
                "iter {iter}: train ep pnl {train_txt} ({ep_done} eps)  pg {:.4} v {:.4} ent {:.3}",
                pg_sum / batches,
                v_sum / batches,
                ent_sum / batches
            );
        }
        writeln!(
            csv,
            "{iter},{train_pnl:.4},{ep_done},{:.5},{:.5},{:.4},{},{}",
            pg_sum / batches,
            v_sum / batches,
            ent_sum / batches,
            val_cols.0,
            val_cols.1
        )?;
    }
    Ok(TrainOutcome { best_dir, best_val: best.expect("evaluated at least once") })
}

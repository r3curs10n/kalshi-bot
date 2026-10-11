//! Stage 1: pretrain the trunk to forecast BRTI from market features alone.
//!
//! At every decision the trunk predicts a Gaussian over the BRTI log return
//! (bps) to each horizon: a mean and a variance. Direction a second ahead is
//! close to a coin flip, but the size of the next move (volatility) is not,
//! and the variance term makes the trunk learn it. Training minimizes the
//! negative log-likelihood with truncated backprop through time over whole
//! sessions, each starting from a zero state as at inference.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::Autodiff;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use strategy::features::{FEATURE_VERSION, MARKET_FEATURES};
use strategy::net::{LstmState, Norm, Trunk};
use strategy::rl::{save_trunk, Cpu, TrunkMeta};
use tracing::info;

use crate::prepared::forecast_targets;
use crate::Prepared;

type Train = Autodiff<Cpu>;

/// Log-variance bounds, so a confident miss can't blow up the loss.
const LOGVAR_RANGE: (f32, f32) = (-10.0, 10.0);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PretrainConfig {
    pub hidden: usize,
    /// Forecast horizons, seconds.
    pub horizons: Vec<usize>,
    /// Seconds between steps; the trading head must use the same cadence.
    pub every_s: u32,
    pub epochs: usize,
    /// Sessions per batch.
    pub batch: usize,
    /// Truncated backprop-through-time length, in steps.
    pub seq: usize,
    pub lr: f64,
    pub seed: u64,
}

impl Default for PretrainConfig {
    fn default() -> Self {
        Self { hidden: 32, horizons: vec![1], every_s: 1, epochs: 30, batch: 8, seq: 64, lr: 1e-3, seed: 0 }
    }
}

/// One session as the trunk sees it: normalized market features and
/// targets per decision.
struct Series {
    x: Vec<f32>,
    y: Vec<f32>,
    len: usize,
}

fn series(p: &Prepared, norm: &Norm, horizons: &[usize]) -> Series {
    let mut x = Vec::with_capacity(p.points.len() * MARKET_FEATURES);
    let mut y = Vec::with_capacity(p.points.len() * horizons.len());
    for (i, market, _) in &p.points {
        let mut m = *market;
        norm.apply(&mut m);
        x.extend_from_slice(&m);
        y.extend(forecast_targets(&p.ep, *i, horizons));
    }
    Series { x, y, len: p.points.len() }
}

#[derive(Debug, Clone, Serialize)]
pub struct ForecastStats {
    /// Mean negative log-likelihood per target (without the constant ½·ln 2π).
    pub nll: f64,
    /// The same for a constant Gaussian fitted to the training targets.
    pub base_nll: f64,
    /// Squared error of the predicted mean, and of the training mean.
    pub mse: f64,
    pub base_mse: f64,
    pub targets: usize,
}

impl ForecastStats {
    /// Share of the return's variance the mean forecast explains.
    pub fn r2(&self) -> f64 {
        1.0 - self.mse / self.base_mse.max(1e-12)
    }
}

fn gaussian_nll(mu: f64, logvar: f64, y: f64) -> f64 {
    let lv = logvar.clamp(LOGVAR_RANGE.0 as f64, LOGVAR_RANGE.1 as f64);
    0.5 * (lv + (y - mu).powi(2) * (-lv).exp())
}

/// Per horizon (mean, variance) of the targets: the constant baseline.
fn baseline(data: &[Series], nh: usize) -> Vec<(f64, f64)> {
    (0..nh)
        .map(|k| {
            let ys: Vec<f64> =
                data.iter().flat_map(|s| s.y.iter().skip(k).step_by(nh)).filter(|y| y.is_finite()).map(|y| *y as f64).collect();
            let n = ys.len().max(1) as f64;
            let mean = ys.iter().sum::<f64>() / n;
            let var = ys.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / n;
            (mean, var.max(1e-6))
        })
        .collect()
}

fn evaluate(trunk: &Trunk<Cpu>, data: &[Series], base: &[(f64, f64)]) -> ForecastStats {
    let nh = base.len();
    let device = NdArrayDevice::Cpu;
    let mut st = ForecastStats { nll: 0.0, base_nll: 0.0, mse: 0.0, base_mse: 0.0, targets: 0 };
    for s in data.iter().filter(|s| s.len > 0) {
        let x = Tensor::<Cpu, 3>::from_data(TensorData::new(s.x.clone(), [1, s.len, MARKET_FEATURES]), &device);
        let (h, _) = trunk.forward_seq(x, LstmState::zeros(1, trunk.hidden(), &device));
        let out: Vec<f32> = trunk.forecast(h).into_data().to_vec().unwrap();
        for t in 0..s.len {
            for (k, (bm, bv)) in base.iter().enumerate() {
                let y = s.y[t * nh + k];
                if !y.is_finite() {
                    continue;
                }
                let (y, mu, lv) = (y as f64, out[t * 2 * nh + k] as f64, out[t * 2 * nh + nh + k] as f64);
                st.nll += gaussian_nll(mu, lv, y);
                st.base_nll += gaussian_nll(*bm, bv.ln(), y);
                st.mse += (y - mu).powi(2);
                st.base_mse += (y - bm).powi(2);
                st.targets += 1;
            }
        }
    }
    let n = st.targets.max(1) as f64;
    ForecastStats { nll: st.nll / n, base_nll: st.base_nll / n, mse: st.mse / n, base_mse: st.base_mse / n, targets: st.targets }
}

pub struct PretrainOutcome {
    pub dir: PathBuf,
    pub best_val: ForecastStats,
}

pub fn pretrain(cfg: &PretrainConfig, train_eps: &[Prepared], val_eps: &[Prepared], out: &Path) -> Result<PretrainOutcome> {
    if train_eps.is_empty() || cfg.horizons.is_empty() {
        bail!("need training episodes and at least one horizon");
    }
    std::fs::create_dir_all(out)?;
    std::fs::write(out.join("pretrain.json"), serde_json::to_vec_pretty(cfg)?)?;
    let device = NdArrayDevice::Cpu;
    let mut rng = SmallRng::seed_from_u64(cfg.seed);
    let nh = cfg.horizons.len();

    let norm = Norm::fit(train_eps.iter().flat_map(|p| p.points.iter().map(|(_, m, _)| m)));
    let train: Vec<Series> = train_eps.iter().map(|p| series(p, &norm, &cfg.horizons)).collect();
    let val: Vec<Series> = val_eps.iter().map(|p| series(p, &norm, &cfg.horizons)).collect();
    let base = baseline(&train, nh);
    let meta = TrunkMeta {
        feature_version: FEATURE_VERSION,
        hidden: cfg.hidden,
        horizons: cfg.horizons.clone(),
        every_s: cfg.every_s,
        trained_until_ms: train_eps.iter().map(|p| p.ep.window.open_ms).max().unwrap_or(0),
    };

    let mut model = Trunk::<Train>::new(cfg.hidden, nh, &device);
    let mut optim = AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(1.0))).init::<Train, Trunk<Train>>();

    let mut csv = std::fs::File::create(out.join("pretrain.csv"))?;
    writeln!(csv, "epoch,train_nll,val_nll,val_base_nll,val_r2")?;
    let mut best: Option<ForecastStats> = None;
    let mut order: Vec<usize> = (0..train.len()).collect();

    for epoch in 1..=cfg.epochs {
        order.shuffle(&mut rng);
        let (mut loss_sum, mut loss_n) = (0f64, 0usize);
        for group in order.chunks(cfg.batch) {
            let b = group.len();
            let t_max = group.iter().map(|i| train[*i].len).max().unwrap_or(0);
            let mut state = LstmState::<Train>::zeros(b, cfg.hidden, &device);
            for t0 in (0..t_max).step_by(cfg.seq) {
                let l = cfg.seq.min(t_max - t0);
                let mut x = vec![0f32; b * l * MARKET_FEATURES];
                let mut y = vec![0f32; b * l * nh];
                let mut w = vec![0f32; b * l * nh];
                for (r, i) in group.iter().enumerate() {
                    let s = &train[*i];
                    for t in t0..(t0 + l).min(s.len) {
                        let row = r * l + (t - t0);
                        x[row * MARKET_FEATURES..(row + 1) * MARKET_FEATURES]
                            .copy_from_slice(&s.x[t * MARKET_FEATURES..(t + 1) * MARKET_FEATURES]);
                        for k in 0..nh {
                            let v = s.y[t * nh + k];
                            if v.is_finite() {
                                y[row * nh + k] = v;
                                w[row * nh + k] = 1.0;
                            }
                        }
                    }
                }
                let n_targets: f32 = w.iter().sum();
                let x = Tensor::<Train, 3>::from_data(TensorData::new(x, [b, l, MARKET_FEATURES]), &device);
                let (h, next) = model.forward_seq(x, state);
                // Sessions shorter than the batch's longest are padded at the end; no loss there.
                if n_targets > 0.0 {
                    let out = model.forecast(h);
                    let mu = out.clone().narrow(1, 0, nh);
                    let lv = out.narrow(1, nh, nh).clamp(LOGVAR_RANGE.0, LOGVAR_RANGE.1);
                    let y = Tensor::<Train, 2>::from_data(TensorData::new(y, [b * l, nh]), &device);
                    let w = Tensor::<Train, 2>::from_data(TensorData::new(w, [b * l, nh]), &device);
                    let per = (lv.clone() + (y - mu).powi_scalar(2) * lv.neg().exp()) * 0.5;
                    let loss = (per * w).sum() / n_targets;
                    loss_sum += loss.clone().into_scalar() as f64 * n_targets as f64;
                    loss_n += n_targets as usize;
                    let grads = GradientsParams::from_grads(loss.backward(), &model);
                    model = optim.step(cfg.lr, model, grads);
                }
                state = next.detach();
            }
        }

        let trunk = model.valid();
        let held = if val.is_empty() { &train } else { &val };
        let stats = evaluate(&trunk, held, &base);
        let train_nll = loss_sum / loss_n.max(1) as f64;
        info!(
            "epoch {epoch}: train nll {train_nll:.4}  val nll {:.4} (constant {:.4})  val R2 {:.4}",
            stats.nll,
            stats.base_nll,
            stats.r2()
        );
        writeln!(csv, "{epoch},{train_nll:.5},{:.5},{:.5},{:.5}", stats.nll, stats.base_nll, stats.r2())?;
        if best.as_ref().is_none_or(|b| stats.nll < b.nll) {
            save_trunk(out, &trunk, &norm, &meta)?;
            best = Some(stats);
        }
    }
    Ok(PretrainOutcome { dir: out.to_path_buf(), best_val: best.expect("at least one epoch") })
}

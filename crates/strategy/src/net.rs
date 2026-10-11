//! The networks and their input normalization.
//!
//! Two parts, trained separately:
//! - `Trunk`: an encoder and an LSTM over the market features only (BRTI and
//!   YES/NO quotes, never the account), pretrained to forecast BRTI
//!   (`kbot pretrain`) and frozen afterwards.
//! - `PolicyHead`: small policy and value MLPs on the trunk's output, the
//!   market features themselves, and the position features, trained with
//!   PPO (`kbot train`). The direct market features keep anything trading
//!   needs that forecasting BRTI gave the trunk no reason to retain.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{sigmoid, tanh};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use serde::{Deserialize, Serialize};

use crate::features::{MARKET_FEATURES, POSITION_FEATURES};
use crate::N_CHOICES;

/// Forecast outputs per horizon: mean and log-variance of the BRTI log
/// return (in bps) to `t + h`.
pub const FORECAST_OUTPUTS: usize = 2;

/// Encoder, one LSTM cell stepped once per decision, and the forecast head
/// it is pretrained with.
///
/// The cell is written out by hand rather than using `burn::nn::Lstm` so a
/// single step (live) and a whole sequence (pretraining) run the same code.
#[derive(Module, Debug)]
pub struct Trunk<B: Backend> {
    enc: Linear<B>,
    /// Input-to-gates; the gates are [input, forget, cell, output].
    gx: Linear<B>,
    /// Hidden-to-gates (the bias lives in `gx`).
    gh: Linear<B>,
    forecast: Linear<B>,
}

/// LSTM state, `[batch, hidden]` each.
#[derive(Debug, Clone)]
pub struct LstmState<B: Backend> {
    pub h: Tensor<B, 2>,
    pub c: Tensor<B, 2>,
}

impl<B: Backend> LstmState<B> {
    pub fn zeros(batch: usize, hidden: usize, device: &B::Device) -> Self {
        Self { h: Tensor::zeros([batch, hidden], device), c: Tensor::zeros([batch, hidden], device) }
    }

    /// Cuts the gradient path (truncated backprop through time).
    pub fn detach(self) -> Self {
        Self { h: self.h.detach(), c: self.c.detach() }
    }
}

impl<B: Backend> Trunk<B> {
    pub fn new(hidden: usize, horizons: usize, device: &B::Device) -> Self {
        Self {
            enc: LinearConfig::new(MARKET_FEATURES, hidden).init(device),
            gx: LinearConfig::new(hidden, 4 * hidden).init(device),
            gh: LinearConfig::new(hidden, 4 * hidden).with_bias(false).init(device),
            forecast: LinearConfig::new(hidden, FORECAST_OUTPUTS * horizons).init(device),
        }
    }

    pub fn hidden(&self) -> usize {
        self.gh.weight.dims()[0]
    }

    /// Input-side gate pre-activations for `x: [rows, MARKET_FEATURES]`.
    fn input_gates(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.gx.forward(tanh(self.enc.forward(x)))
    }

    fn cell(&self, gates_x: Tensor<B, 2>, s: LstmState<B>) -> LstmState<B> {
        let g = (gates_x + self.gh.forward(s.h)).chunk(4, 1);
        let (i, f, c_in, o) = (sigmoid(g[0].clone()), sigmoid(g[1].clone()), tanh(g[2].clone()), sigmoid(g[3].clone()));
        let c = f * s.c + i * c_in;
        let h = o * tanh(c.clone());
        LstmState { h, c }
    }

    /// One decision for each row of normalized market features
    /// `x: [batch, MARKET_FEATURES]`; the new state's `h` is the output.
    pub fn step(&self, x: Tensor<B, 2>, s: LstmState<B>) -> LstmState<B> {
        self.cell(self.input_gates(x), s)
    }

    /// Unrolls `x: [batch, T, MARKET_FEATURES]` from `s`. Returns every
    /// output `[batch * T, hidden]` (rows are (batch, t), row-major) and the
    /// final state.
    pub fn forward_seq(&self, x: Tensor<B, 3>, mut s: LstmState<B>) -> (Tensor<B, 2>, LstmState<B>) {
        let [b, t_len, f] = x.dims();
        let hidden = self.hidden();
        let gx = self.input_gates(x.reshape([b * t_len, f])).reshape([b, t_len, 4 * hidden]);
        let mut hs = Vec::with_capacity(t_len);
        for t in 0..t_len {
            s = self.cell(gx.clone().narrow(1, t, 1).reshape([b, 4 * hidden]), s);
            hs.push(s.h.clone());
        }
        // T x [b, H] -> [b, T, H] -> rows.
        (Tensor::stack::<3>(hs, 1).reshape([b * t_len, hidden]), s)
    }

    /// Forecasts from outputs `h: [rows, hidden]`: `[rows, 2 * horizons]`,
    /// every horizon's mean first, then every horizon's log-variance.
    pub fn forecast(&self, h: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forecast.forward(h)
    }
}

/// Width of a head's input over a trunk of `trunk_hidden` units.
pub fn head_inputs(trunk_hidden: usize) -> usize {
    trunk_hidden + MARKET_FEATURES + POSITION_FEATURES
}

/// Policy and value MLPs over `[trunk output, normalized market features,
/// position features]`.
#[derive(Module, Debug)]
pub struct PolicyHead<B: Backend> {
    pi1: Linear<B>,
    pi2: Linear<B>,
    v1: Linear<B>,
    v2: Linear<B>,
}

impl<B: Backend> PolicyHead<B> {
    pub fn new(trunk_hidden: usize, hidden: usize, device: &B::Device) -> Self {
        let inputs = head_inputs(trunk_hidden);
        Self {
            pi1: LinearConfig::new(inputs, hidden).init(device),
            pi2: LinearConfig::new(hidden, N_CHOICES).init(device),
            v1: LinearConfig::new(inputs, hidden).init(device),
            v2: LinearConfig::new(hidden, 1).init(device),
        }
    }

    /// `x: [batch, head_inputs(trunk_hidden)]` -> (logits
    /// `[batch, N_CHOICES]`, value `[batch]`).
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let n = x.dims()[0];
        let logits = self.pi2.forward(tanh(self.pi1.forward(x.clone())));
        let value = self.v2.forward(tanh(self.v1.forward(x))).reshape([n]);
        (logits, value)
    }
}

/// Per-feature standardization of the market features, fit on training data
/// and shipped with the model. Position features are on a unit scale already.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Norm {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl Norm {
    pub fn identity() -> Self {
        Self { mean: vec![0.0; MARKET_FEATURES], std: vec![1.0; MARKET_FEATURES] }
    }

    /// Fits the market features on `rows`.
    pub fn fit<'a>(rows: impl Iterator<Item = &'a [f32; MARKET_FEATURES]>) -> Self {
        let mut n = 0f64;
        let mut sum = vec![0f64; MARKET_FEATURES];
        let mut sq = vec![0f64; MARKET_FEATURES];
        for r in rows {
            n += 1.0;
            for (j, x) in r.iter().enumerate() {
                sum[j] += *x as f64;
                sq[j] += (*x as f64).powi(2);
            }
        }
        let n = n.max(1.0);
        let mean: Vec<f32> = sum.iter().map(|s| (s / n) as f32).collect();
        let std: Vec<f32> = sq
            .iter()
            .zip(&sum)
            .map(|(q, s)| ((q / n - (s / n).powi(2)).max(0.0).sqrt() as f32).max(1e-3))
            .collect();
        Self { mean, std }
    }

    pub fn apply(&self, x: &mut [f32; MARKET_FEATURES]) {
        for (j, v) in x.iter_mut().enumerate() {
            *v = ((*v - self.mean[j]) / self.std[j]).clamp(-10.0, 10.0);
        }
    }
}

/// Softmax over the allowed choices only.
pub fn masked_softmax(logits: &[f32], mask: &[bool; N_CHOICES]) -> [f32; N_CHOICES] {
    let max = logits.iter().zip(mask).filter(|(_, m)| **m).map(|(l, _)| *l).fold(f32::NEG_INFINITY, f32::max);
    let mut p = [0f32; N_CHOICES];
    let mut z = 0f32;
    for i in 0..N_CHOICES {
        if mask[i] {
            p[i] = (logits[i] - max).exp();
            z += p[i];
        }
    }
    for v in p.iter_mut() {
        *v /= z.max(1e-12);
    }
    p
}

/// The most likely allowed choice.
pub fn greedy(probs: &[f32; N_CHOICES], mask: &[bool; N_CHOICES]) -> usize {
    (0..N_CHOICES).filter(|i| mask[*i]).max_by(|a, b| probs[*a].total_cmp(&probs[*b])).unwrap_or(0)
}

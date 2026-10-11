//! Saved models, and the trained policy behind the `Strategy` interface.
//!
//! A pretrained trunk lives in its own directory (`kbot pretrain --out`):
//! `trunk.json`, `norm.json`, `trunk.mpk`. A trading model (`kbot train`)
//! copies the trunk it was trained on next to its head, so it is
//! self-contained: `model.json`, `norm.json`, `trunk.mpk`, `head.mpk`.

use std::path::Path;

use anyhow::{bail, Context, Result};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::{Tensor, TensorData};
use serde::{Deserialize, Serialize};

use crate::features::{position_features, FEATURE_VERSION, MARKET_FEATURES, POSITION_FEATURES};
use crate::net::{greedy, masked_softmax, LstmState, Norm, PolicyHead, Trunk};
use crate::runner::{decide, Runner, Strategy};
use crate::{Decision, Position, Snapshot, TradeLimits, Window, N_CHOICES};

pub type Cpu = NdArray<f32>;

/// Architecture tag; models saved by other architectures are refused.
pub const NET: &str = "lstm-trunk+market-head";

pub const TRUNK_META_FILE: &str = "trunk.json";
pub const META_FILE: &str = "model.json";
pub const NORM_FILE: &str = "norm.json";
pub const TRUNK_WEIGHTS: &str = "trunk";
pub const HEAD_WEIGHTS: &str = "head";

/// What a pretrained trunk needs besides weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrunkMeta {
    pub feature_version: u32,
    pub hidden: usize,
    /// Forecast horizons it was pretrained on, seconds.
    pub horizons: Vec<usize>,
    /// Seconds between steps; the trunk must be run at the cadence it learned.
    pub every_s: u32,
    /// Open time of the latest session it was trained on. Sessions up to
    /// here must not be used to validate or test anything built on it.
    #[serde(default)]
    pub trained_until_ms: i64,
}

/// Everything besides weights needed to run a trading model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMeta {
    #[serde(default)]
    pub net: String,
    pub trunk: TrunkMeta,
    pub head_hidden: usize,
    pub limits: TradeLimits,
}

fn recorder() -> NamedMpkFileRecorder<FullPrecisionSettings> {
    NamedMpkFileRecorder::<FullPrecisionSettings>::new()
}

fn read_json<T: for<'a> Deserialize<'a>>(path: &Path) -> Result<T> {
    serde_json::from_slice(&std::fs::read(path).with_context(|| format!("reading {}", path.display()))?)
        .with_context(|| format!("parsing {}", path.display()))
}

fn check_features(meta: &TrunkMeta) -> Result<()> {
    if meta.feature_version != FEATURE_VERSION {
        bail!("trained on features v{}, this build has v{FEATURE_VERSION}", meta.feature_version);
    }
    Ok(())
}

pub fn save_trunk(dir: &Path, trunk: &Trunk<Cpu>, norm: &Norm, meta: &TrunkMeta) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join(TRUNK_META_FILE), serde_json::to_vec_pretty(meta)?)?;
    std::fs::write(dir.join(NORM_FILE), serde_json::to_vec(norm)?)?;
    trunk.clone().save_file(dir.join(TRUNK_WEIGHTS), &recorder()).map_err(|e| anyhow::anyhow!("saving trunk: {e}"))
}

fn load_trunk_weights(dir: &Path, meta: &TrunkMeta) -> Result<Trunk<Cpu>> {
    let device = NdArrayDevice::Cpu;
    Trunk::<Cpu>::new(meta.hidden, meta.horizons.len(), &device)
        .load_file(dir.join(TRUNK_WEIGHTS), &recorder(), &device)
        .map_err(|e| anyhow::anyhow!("loading trunk from {}: {e}", dir.display()))
}

/// A pretrained trunk directory (from `kbot pretrain`).
pub fn load_trunk(dir: &Path) -> Result<(Trunk<Cpu>, Norm, TrunkMeta)> {
    let meta: TrunkMeta = read_json(&dir.join(TRUNK_META_FILE))?;
    check_features(&meta)?;
    let norm = read_json(&dir.join(NORM_FILE))?;
    Ok((load_trunk_weights(dir, &meta)?, norm, meta))
}

/// A complete trading model: frozen trunk, policy head, normalization.
#[derive(Debug, Clone)]
pub struct Model {
    pub trunk: Trunk<Cpu>,
    pub head: PolicyHead<Cpu>,
    pub norm: Norm,
    pub meta: ModelMeta,
}

impl Model {
    pub fn save(&self, dir: &Path) -> Result<()> {
        save_trunk(dir, &self.trunk, &self.norm, &self.meta.trunk)?;
        std::fs::write(dir.join(META_FILE), serde_json::to_vec_pretty(&self.meta)?)?;
        self.head.clone().save_file(dir.join(HEAD_WEIGHTS), &recorder()).map_err(|e| anyhow::anyhow!("saving head: {e}"))
    }

    pub fn load(dir: &Path) -> Result<Self> {
        let meta: ModelMeta = read_json(&dir.join(META_FILE))?;
        if meta.net != NET {
            let found = if meta.net.is_empty() { "mlp" } else { &meta.net };
            bail!("model is a {found:?} network, this build runs {NET:?}; retrain it");
        }
        check_features(&meta.trunk)?;
        let norm = read_json(&dir.join(NORM_FILE))?;
        let device = NdArrayDevice::Cpu;
        let trunk = load_trunk_weights(dir, &meta.trunk)?;
        let head = PolicyHead::<Cpu>::new(meta.trunk.hidden, meta.head_hidden, &device)
            .load_file(dir.join(HEAD_WEIGHTS), &recorder(), &device)
            .map_err(|e| anyhow::anyhow!("loading head: {e}"))?;
        Ok(Self { trunk, head, norm, meta })
    }

    /// A pretrained trunk (from `kbot pretrain`) with a freshly initialized
    /// head, ready for PPO.
    pub fn from_trunk(dir: &Path, head_hidden: usize, limits: TradeLimits) -> Result<Self> {
        let (trunk, norm, trunk_meta) = load_trunk(dir)?;
        let head = PolicyHead::new(trunk_meta.hidden, head_hidden, &NdArrayDevice::Cpu);
        Ok(Self { trunk, head, norm, meta: ModelMeta { net: NET.into(), trunk: trunk_meta, head_hidden, limits } })
    }

    /// The state a session starts from.
    pub fn initial_state(&self) -> LstmState<Cpu> {
        LstmState::zeros(1, self.trunk.hidden(), &NdArrayDevice::Cpu)
    }

    /// Steps the trunk on one decision's raw market features, advancing
    /// `state`. Returns the market side of the head's input: the trunk's
    /// output followed by the normalized market features.
    pub fn market_step(&self, market: &[f32; MARKET_FEATURES], state: &mut LstmState<Cpu>) -> Vec<f32> {
        let mut x = *market;
        self.norm.apply(&mut x);
        let t = Tensor::<Cpu, 2>::from_data(TensorData::new(x.to_vec(), [1, MARKET_FEATURES]), &NdArrayDevice::Cpu);
        *state = self.trunk.step(t, state.clone());
        let mut out: Vec<f32> = state.h.clone().into_data().to_vec().unwrap_or_default();
        out.extend_from_slice(&x);
        out
    }

    /// Policy logits and value for one decision, from `market_step`'s output
    /// and the position features.
    pub fn head_one(&self, market_side: &[f32], position: &[f32; POSITION_FEATURES]) -> ([f32; N_CHOICES], f32) {
        let mut x = market_side.to_vec();
        x.extend_from_slice(position);
        let n = x.len();
        let x = Tensor::<Cpu, 2>::from_data(TensorData::new(x, [1, n]), &NdArrayDevice::Cpu);
        let (logits, value) = self.head.forward(x);
        let l: Vec<f32> = logits.into_data().to_vec().unwrap_or_default();
        let v: Vec<f32> = value.into_data().to_vec().unwrap_or_default();
        (std::array::from_fn(|i| l[i]), v[0])
    }
}

pub struct RlStrategy {
    runner: Runner,
    model: Model,
    /// Advanced once per decision; reset at every new window, as in training.
    state: LstmState<Cpu>,
}

impl RlStrategy {
    pub fn load(dir: &Path) -> Result<Self> {
        let model = Model::load(dir)?;
        let state = model.initial_state();
        Ok(Self { runner: Runner::new(model.meta.limits, model.meta.trunk.every_s), model, state })
    }
}

impl Strategy for RlStrategy {
    fn begin_window(&mut self, w: &Window) {
        self.runner.begin_window(w);
        self.state = self.model.initial_state();
    }

    fn on_tick(&mut self, s: &Snapshot, pos: &Position) -> Decision {
        let limits = self.runner.limits;
        let Some(ctx) = self.runner.tick(s, pos) else {
            return Decision::nothing(self.runner.fair_value());
        };
        let market_side = self.model.market_step(&ctx.market, &mut self.state);
        let (logits, _) = self.model.head_one(&market_side, &position_features(pos, &ctx.snap));
        let probs = masked_softmax(&logits, &ctx.mask);
        // Greedy at inference: the most likely allowed (action, size).
        decide(greedy(&probs, &ctx.mask), &ctx, &limits, &probs, ctx.fv.p)
    }

    fn limits(&self) -> TradeLimits {
        self.runner.limits
    }
}

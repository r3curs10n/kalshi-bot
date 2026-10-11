//! Training in two stages over recorded episodes.
//!
//! 1. `pretrain`: the trunk (encoder + LSTM over market features only) learns
//!    to forecast BRTI, a dense supervised signal at every decision.
//! 2. `ppo`: with the trunk frozen, PPO with action masking trains small
//!    policy and value heads on its output plus the position features.
//!
//! Each environment replays one 15-minute session through `market::Sim`.
//! Market features at every decision time are precomputed once per episode
//! with the very `Runner` the live strategy uses, and the trunk's outputs
//! with the very calls `RlStrategy` makes, so training and inference see
//! identical inputs; position features are added at step time.

pub mod ppo;
pub mod prepared;
pub mod pretrain;

pub use ppo::{train, Encoded, PpoConfig, TrainOutcome};
pub use prepared::Prepared;
pub use pretrain::{pretrain, PretrainConfig};

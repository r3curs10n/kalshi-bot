//! Recorded-data side of the bot: session storage, replay into `Snapshot`s,
//! episodes, and the trading simulator.

pub mod env;
pub mod replay;
pub mod store;
pub mod synth;
pub mod view;

pub use env::{run_strategy, run_strategy_traced, EpisodeResult, Sim, SimConfig, DEFAULT_BALANCE};
pub use replay::Episode;
pub use store::{Line, SessionMeta, SessionStatus, Store};

//! The inference interface: feed a time series of BRTI spot and YES/NO quotes,
//! get a trade decision at every instant. Features, normalization and the
//! network stay behind the `Strategy` trait.
//!
//! ```ignore
//! let mut s = RlStrategy::load("models/run1".as_ref())?;
//! s.begin_window(&window);
//! for snap in live_snapshots {
//!     let d = s.on_tick(&snap, &position);   // Decision { action, size, limit, probs, fair_value }
//! }
//! ```

pub mod baselines;
pub mod fair;
pub mod features;
pub mod fees;
pub mod history;
pub mod net;
pub mod rl;
pub mod rules;
pub mod runner;
pub mod types;

pub use baselines::{FairValueStrategy, NothingStrategy};
pub use fair::FairValue;
pub use history::History;
pub use rl::RlStrategy;
pub use rules::{RuleOpts, RuleStrategy, RuleSwingStrategy, MAX_SWING_BPS, RULE_LAST_S};
pub use runner::{apply_fill, fill_size, Runner, Strategy};
pub use types::*;

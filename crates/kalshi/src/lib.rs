//! Kalshi trade-api v2 client: request signing, REST reads, and websocket feeds.
//!
//! Order placement exists only for `kbot viz` live trading, which a person
//! starts by hand with their own key.

pub mod auth;
pub mod config;
pub mod rest;
pub mod ws;

pub use auth::Signer;
pub use config::Config;
pub use rest::Rest;
pub use ws::{WsConn, WsMsg};

/// The 15-minute "BTC price up in next 15 mins?" series.
pub const SERIES: &str = "KXBTC15M";

/// Wall-clock now, in milliseconds and nanoseconds since the epoch.
pub fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

pub fn now_ns() -> i64 {
    chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
}

/// Kalshi's fixed-point dollar strings ("0.6400") and size strings ("122.02").
pub fn parse_num(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::String(s) if !s.is_empty() => s.parse().ok(),
        serde_json::Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

/// RFC 3339 timestamp field to epoch milliseconds.
pub fn parse_time_ms(v: &serde_json::Value) -> Option<i64> {
    let s = v.as_str()?;
    chrono::DateTime::parse_from_rfc3339(s).ok().map(|t| t.timestamp_millis())
}

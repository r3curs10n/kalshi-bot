//! Credentials and paths. Read from `config.toml` (if present), with
//! `KALSHI_KEY_ID` / `KALSHI_KEY_PATH` / `KBOT_DATA_DIR` env vars overriding it.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::Signer;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub key_id: String,
    pub key_path: PathBuf,
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,
}

fn default_data_dir() -> PathBuf {
    PathBuf::from("data")
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let mut table: toml::Table = if path.exists() {
            toml::from_str(&std::fs::read_to_string(path)?)
                .with_context(|| format!("parsing {}", path.display()))?
        } else {
            toml::Table::new()
        };
        for (env, key) in [
            ("KALSHI_KEY_ID", "key_id"),
            ("KALSHI_KEY_PATH", "key_path"),
            ("KBOT_DATA_DIR", "data_dir"),
        ] {
            if let Ok(v) = std::env::var(env) {
                table.insert(key.into(), toml::Value::String(v));
            }
        }
        toml::Value::Table(table)
            .try_into()
            .context("config needs key_id and key_path (config.toml or KALSHI_KEY_ID/KALSHI_KEY_PATH)")
    }

    pub fn signer(&self) -> Result<Signer> {
        let pem = std::fs::read_to_string(&self.key_path)
            .with_context(|| format!("reading key {}", self.key_path.display()))?;
        Signer::from_pem(&self.key_id, &pem)
    }
}

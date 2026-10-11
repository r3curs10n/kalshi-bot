//! Smoke test for signing: `cargo run -p kalshi --example whoami`.
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg = kalshi::Config::load("config.toml".as_ref())?;
    let rest = kalshi::Rest::new(Arc::new(cfg.signer()?))?;
    let bal = rest.balance().await?;
    println!("signed GET /portfolio/balance ok: keys={:?}", bal.as_object().map(|o| o.keys().collect::<Vec<_>>()));
    for m in rest.upcoming_markets(kalshi::SERIES, 1800).await? {
        println!("{} {} strike={} close={}", m["ticker"], m["status"], m["floor_strike"], m["close_time"]);
    }
    Ok(())
}

use anyhow::Result;
use clap::Subcommand;
use market::Store;

#[derive(Subcommand)]
pub enum Cmd {
    /// List sessions with status, result and per-stream message counts.
    Ls {
        /// Only sessions opened on this UTC date (YYYY-MM-DD).
        #[arg(long)]
        date: Option<String>,
        /// Only sessions in this status (recording, closed, settled, unsettled).
        #[arg(long)]
        status: Option<String>,
    },
}

pub fn run(store: Store, cmd: Cmd) -> Result<()> {
    let Cmd::Ls { date, status } = cmd;
    println!(
        "{:<26} {:<10} {:<6} {:>11} {:>6} {:>6} {:>6} {:>5} {:>5}  late_start",
        "ticker", "status", "result", "strike", "btc", "quotes", "trades", "book", "gaps"
    );
    let mut n = 0;
    for dir in store.list_sessions()? {
        let m = Store::read_meta(&dir)?;
        if date.as_ref().is_some_and(|d| *d != m.date()) {
            continue;
        }
        let st = serde_json::to_value(m.status)?.as_str().unwrap_or_default().to_string();
        if status.as_ref().is_some_and(|s| *s != st) {
            continue;
        }
        let c = |k: &str| m.counts.get(k).copied().unwrap_or(0);
        let late = (m.first_seen_ms - m.open_ms).max(0) / 1000;
        println!(
            "{:<26} {:<10} {:<6} {:>11} {:>6} {:>6} {:>6} {:>5} {:>5}  {}",
            m.ticker,
            st,
            m.result.as_deref().unwrap_or("-"),
            m.strike.map(|k| format!("{k:.2}")).unwrap_or("-".into()),
            c("btc"),
            c("quotes"),
            c("trades"),
            c("book"),
            m.gaps.len(),
            if late > 5 { format!("{late}s") } else { String::new() },
        );
        n += 1;
    }
    println!("{n} sessions");
    Ok(())
}

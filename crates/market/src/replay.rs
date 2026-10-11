//! Turns a recorded session directory into an `Episode`: the `Snapshot`
//! series a strategy would have seen live, on a 1-second grid, plus the label.
//!
//! Everything is ordered by when *we* received it (`recv_ns`), not exchange
//! timestamps, so replay sees exactly what live trading would have.

use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use strategy::{Quote, Snapshot, Window};

use crate::store::{read_lines, SessionMeta, Store, BTC_LOOKBACK_MS};

pub const EPISODE_FILE: &str = "episode.json.zst";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub window: Window,
    pub result_yes: bool,
    pub expiration_value: Option<f64>,
    /// 1s grid from `open - lookback` up to (excluding) close.
    pub snapshots: Vec<Snapshot>,
}

impl Episode {
    /// Index of the first snapshot inside the window.
    pub fn open_idx(&self) -> usize {
        self.snapshots.partition_point(|s| s.t_ms < self.window.open_ms)
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        let bytes = zstd::encode_all(serde_json::to_vec(self)?.as_slice(), 9)?;
        let tmp = dir.join(format!("{EPISODE_FILE}.tmp"));
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(tmp, dir.join(EPISODE_FILE))?;
        Ok(())
    }

    pub fn load(dir: &Path) -> Result<Self> {
        let bytes = zstd::decode_all(std::fs::File::open(dir.join(EPISODE_FILE))?)?;
        Ok(serde_json::from_slice(&bytes)?)
    }
}

/// (CF time ms, value) from a `cfbenchmarks_value` message.
pub fn brti_tick(v: &Value) -> Option<(i64, f64)> {
    let inner: Value = serde_json::from_str(v["msg"]["data"].as_str()?).ok()?;
    Some((inner["time"].as_i64()?, kalshi::parse_num(&inner["value"])?))
}

/// A price of 0 or 1 is Kalshi's way of saying there is no order there.
fn price(v: &Value) -> Option<f64> {
    kalshi::parse_num(v).filter(|p| *p > 0.0 && *p < 1.0)
}

fn quote(bid: &Value, ask: &Value, bid_size: &Value, ask_size: &Value) -> Quote {
    let (bid, ask) = (price(bid), price(ask));
    Quote {
        bid,
        ask,
        bid_size: bid.and(kalshi::parse_num(bid_size)),
        ask_size: ask.and(kalshi::parse_num(ask_size)),
    }
}

/// NO is the other side of the same book: NO bid = 1 - YES ask, and so on.
pub fn complement(yes: &Quote) -> Quote {
    Quote { bid: yes.ask.map(|a| 1.0 - a), ask: yes.bid.map(|b| 1.0 - b), bid_size: yes.ask_size, ask_size: yes.bid_size }
}

/// Best (highest) bid of an ascending `[[price, size], ...]` ladder.
fn best_bid(ladder: &Value) -> (Option<f64>, Option<f64>) {
    let Some(top) = ladder.as_array().and_then(|l| l.last()) else { return (None, None) };
    (price(&top[0]), kalshi::parse_num(&top[1]))
}

/// Top of book from a recorded/live message of `stream`: the YES quote, plus
/// the NO quote when the source carries it natively.
pub fn parse_quotes(stream: &str, v: &Value) -> Option<(Quote, Option<Quote>)> {
    match stream {
        "quotes" => {
            let m = &v["msg"];
            Some((quote(&m["yes_bid_dollars"], &m["yes_ask_dollars"], &m["yes_bid_size_fp"], &m["yes_ask_size_fp"]), None))
        }
        "book" => {
            let ob = &v["orderbook_fp"];
            let (yb, ybs) = best_bid(&ob["yes_dollars"]);
            let (nb, nbs) = best_bid(&ob["no_dollars"]);
            // A book has only bids; each side's ask is the other side's bid.
            let yes = Quote { bid: yb, ask: nb.map(|p| 1.0 - p), bid_size: ybs, ask_size: nbs };
            let no = Quote { bid: nb, ask: yb.map(|p| 1.0 - p), bid_size: nbs, ask_size: ybs };
            Some((yes, Some(no)))
        }
        "market" => {
            if v["status"] != "active" {
                return None;
            }
            let yes = quote(&v["yes_bid_dollars"], &v["yes_ask_dollars"], &v["yes_bid_size_fp"], &v["yes_ask_size_fp"]);
            let no = quote(&v["no_bid_dollars"], &v["no_ask_dollars"], &Value::Null, &Value::Null);
            Some((yes, Some(Quote { bid_size: yes.ask_size, ask_size: yes.bid_size, ..no })))
        }
        _ => None,
    }
}

/// Folds BRTI and quote messages into the current `Snapshot`. Used both to
/// replay sessions and live, so the two build snapshots identically.
#[derive(Debug, Clone, Default)]
pub struct Assembler {
    btc: f64,
    yes: Quote,
    yes_t: i64,
    no: Quote,
    no_t: i64,
}

impl Assembler {
    pub fn btc(&mut self, value: f64) {
        self.btc = value;
    }

    pub fn quotes(&mut self, recv_ms: i64, yes: Quote, no: Option<Quote>) {
        self.yes = yes;
        self.yes_t = recv_ms;
        if let Some(n) = no {
            self.no = n;
            self.no_t = recv_ms;
        }
    }

    /// A new contract: the previous one's quotes no longer apply.
    pub fn clear_quotes(&mut self) {
        self.yes = Quote::default();
        self.no = Quote::default();
        self.yes_t = i64::MIN;
        self.no_t = i64::MIN;
    }

    pub fn snapshot(&self, t_ms: i64) -> Option<Snapshot> {
        // Native NO when it is at least as fresh as YES; otherwise the same
        // book seen from the other side.
        let no = if self.no_t >= self.yes_t { self.no } else { complement(&self.yes) };
        (self.btc > 0.0).then_some(Snapshot { t_ms, btc: self.btc, yes: self.yes, no })
    }
}

#[derive(Debug, Clone)]
enum Ev {
    Btc(f64),
    Quote { yes: Quote, no: Option<Quote> },
}

/// Market events of one session, sorted by receive time.
fn session_events(dir: &Path) -> Result<(Vec<(i64, Ev)>, Vec<(i64, f64)>)> {
    let mut evs = vec![];
    let mut cf_ticks = vec![];
    for line in read_lines(&Store::stream_path(dir, "btc"))? {
        if let Some((t, v)) = brti_tick(&line.value()) {
            evs.push((line.recv_ms(), Ev::Btc(v)));
            cf_ticks.push((t, v));
        }
    }
    for stream in ["quotes", "book", "market"] {
        for line in read_lines(&Store::stream_path(dir, stream))? {
            if let Some((yes, no)) = parse_quotes(stream, &line.value()) {
                evs.push((line.recv_ms(), Ev::Quote { yes, no }));
            }
        }
    }
    evs.sort_by_key(|(t, _)| *t);
    cf_ticks.sort_by_key(|(t, _)| *t);
    Ok((evs, cf_ticks))
}

/// Coverage rules an episode must meet to be used.
pub struct Requirements {
    /// BRTI must be continuous from this long before open.
    pub btc_from_ms: i64,
    pub max_btc_gap_ms: i64,
    pub max_quote_gap_ms: i64,
    /// First quote must arrive within this long after open.
    pub max_quote_start_ms: i64,
}

impl Default for Requirements {
    fn default() -> Self {
        Self { btc_from_ms: 300_000, max_btc_gap_ms: 10_000, max_quote_gap_ms: 30_000, max_quote_start_ms: 15_000 }
    }
}

impl Requirements {
    /// Accepts any session with a result and a strike, whatever its coverage.
    /// For inspecting a single session, never for datasets.
    pub fn lenient() -> Self {
        const DAY_MS: i64 = 86_400_000;
        Self { btc_from_ms: 0, max_btc_gap_ms: DAY_MS, max_quote_gap_ms: DAY_MS, max_quote_start_ms: DAY_MS }
    }
}

/// Checks run against Kalshi's own numbers, reported by the dataset builder.
#[derive(Debug, Clone, Default)]
pub struct Checks {
    /// Our 60s-before-open BRTI average minus Kalshi's strike.
    pub strike_diff: Option<f64>,
    /// Our 60s-before-close average minus Kalshi's expiration value.
    pub settle_diff: Option<f64>,
    /// Our implied result agrees with Kalshi's.
    pub result_agrees: Option<bool>,
}

fn avg_cf(ticks: &[(i64, f64)], from_ms: i64, to_ms: i64) -> Option<f64> {
    let xs: Vec<f64> = ticks.iter().filter(|(t, _)| *t >= from_ms && *t < to_ms).map(|(_, v)| *v).collect();
    (xs.len() >= 50).then(|| xs.iter().sum::<f64>() / xs.len() as f64)
}

pub fn build_episode(dir: &Path, req: &Requirements) -> Result<std::result::Result<(Episode, Checks), String>> {
    let meta: SessionMeta = Store::read_meta(dir)?;
    let Some(result) = meta.result.as_deref() else { return Ok(Err("no result".into())) };
    let Some(strike) = meta.strike else { return Ok(Err("no strike".into())) };
    let window = Window { ticker: meta.ticker.clone(), strike, open_ms: meta.open_ms, close_ms: meta.close_ms };
    let (evs, cf) = session_events(dir)?;

    // Coverage
    let btc_times: Vec<i64> = evs.iter().filter(|(_, e)| matches!(e, Ev::Btc(_))).map(|(t, _)| *t).collect();
    let need_from = meta.open_ms - req.btc_from_ms;
    if btc_times.first().is_none_or(|t| *t > need_from + req.max_btc_gap_ms) {
        return Ok(Err("BRTI lookback missing".into()));
    }
    if let Some(g) = max_gap(&btc_times, need_from, meta.close_ms)
        && g > req.max_btc_gap_ms {
            return Ok(Err(format!("BRTI gap {}s", g / 1000)));
        }
    let q_times: Vec<i64> = evs.iter().filter(|(t, e)| matches!(e, Ev::Quote { .. }) && *t >= meta.open_ms).map(|(t, _)| *t).collect();
    if q_times.first().is_none_or(|t| *t > meta.open_ms + req.max_quote_start_ms) {
        return Ok(Err("quotes missing at open".into()));
    }
    if let Some(g) = max_gap(&q_times, meta.open_ms, meta.close_ms)
        && g > req.max_quote_gap_ms {
            return Ok(Err(format!("quote gap {}s", g / 1000)));
        }

    // Grid
    let start = meta.open_ms - BTC_LOOKBACK_MS;
    let mut snaps = Vec::with_capacity(((meta.close_ms - start) / 1000) as usize);
    let mut asm = Assembler::default();
    asm.clear_quotes();
    let mut k = 0;
    let mut t = start;
    while t < meta.close_ms {
        while k < evs.len() && evs[k].0 <= t {
            match &evs[k].1 {
                Ev::Btc(v) => asm.btc(*v),
                Ev::Quote { yes, no } => asm.quotes(evs[k].0, *yes, *no),
            }
            k += 1;
        }
        snaps.extend(asm.snapshot(t));
        t += 1000;
    }

    let checks = Checks {
        strike_diff: avg_cf(&cf, meta.open_ms - 60_000, meta.open_ms).map(|a| a - strike),
        settle_diff: avg_cf(&cf, meta.close_ms - 60_000, meta.close_ms)
            .zip(meta.expiration_value)
            .map(|(a, e)| a - e),
        result_agrees: meta.expiration_value.map(|e| (e >= strike) == (result == "yes")),
    };
    let ep = Episode { window, result_yes: result == "yes", expiration_value: meta.expiration_value, snapshots: snaps };
    Ok(Ok((ep, checks)))
}

/// Largest spacing between consecutive times within [from, to], including the edges.
fn max_gap(times: &[i64], from: i64, to: i64) -> Option<i64> {
    let inside: Vec<i64> = times.iter().copied().filter(|t| *t >= from && *t <= to).collect();
    let first = *inside.first()?;
    let last = *inside.last()?;
    let mut g = (first - from).max(to - last);
    for w in inside.windows(2) {
        g = g.max(w[1] - w[0]);
    }
    Some(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complement_swaps_sides() {
        let y = Quote { bid: Some(0.64), ask: Some(0.65), bid_size: Some(10.0), ask_size: Some(20.0) };
        let n = complement(&y);
        assert!((n.bid.unwrap() - 0.35).abs() < 1e-12);
        assert!((n.ask.unwrap() - 0.36).abs() < 1e-12);
        assert_eq!((n.bid_size, n.ask_size), (Some(20.0), Some(10.0)));
    }

    #[test]
    fn zero_and_one_mean_no_order() {
        let q = quote(&"0.0000".into(), &"1.0000".into(), &"5".into(), &"5".into());
        assert_eq!(q, Quote::default());
    }

    #[test]
    fn gaps() {
        assert_eq!(max_gap(&[0, 1000, 5000, 6000], 0, 6000), Some(4000));
        assert_eq!(max_gap(&[2000], 0, 3000), Some(2000));
    }
}

/// Session dirs that have a built episode, opened within `[from, to]`
/// (inclusive UTC dates, YYYY-MM-DD), oldest first.
pub fn episode_dirs(store: &Store, from: Option<&str>, to: Option<&str>) -> Result<Vec<std::path::PathBuf>> {
    Ok(store
        .list_sessions()?
        .into_iter()
        .filter(|d| d.join(EPISODE_FILE).exists())
        .filter(|d| {
            let date = d.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str()).unwrap_or("");
            from.is_none_or(|f| date >= f) && to.is_none_or(|t| date <= t)
        })
        .collect())
}

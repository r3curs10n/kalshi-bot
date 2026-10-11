//! A session as chart-ready columns, for the visualizer. Built from the raw
//! session files (not the episode), so it also works for sessions still being
//! recorded and for ones the dataset builder dropped.

use std::path::Path;

use anyhow::Result;
use serde::Serialize;
use serde_json::Value;
use strategy::fair::fair_value;
use strategy::{History, Window};

use crate::replay::{brti_tick, parse_quotes, Assembler};
use crate::store::{read_lines, SessionMeta, Store, BTC_LOOKBACK_MS, BTC_TAIL_MS};

/// Trade flow is bucketed to this many seconds.
pub const FLOW_BUCKET_S: i64 = 5;

/// Columnar series on a shared 1s grid (seconds since the epoch). `None` is a
/// gap in the line.
#[derive(Debug, Default, Serialize)]
pub struct Grid {
    pub t: Vec<i64>,
    pub btc: Vec<Option<f64>>,
    pub yes_bid: Vec<Option<f64>>,
    pub yes_ask: Vec<Option<f64>>,
    pub no_bid: Vec<Option<f64>>,
    pub no_ask: Vec<Option<f64>>,
    /// Analytic P(YES) from the settlement-aware fair value, inside the window.
    pub fair: Vec<Option<f64>>,
}

#[derive(Debug, Default, Serialize)]
pub struct Flow {
    pub t: Vec<i64>,
    /// Contracts bought by YES takers in the bucket (positive).
    pub yes: Vec<f64>,
    /// Contracts bought by NO takers in the bucket (negative, so it plots downward).
    pub no: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct SessionView {
    pub meta: SessionMeta,
    pub grid: Grid,
    pub flow: Flow,
    /// Our 60s BRTI averages by CF timestamp, comparable to Kalshi's
    /// strike and expiration value.
    pub strike_avg: Option<f64>,
    pub settle_avg: Option<f64>,
    /// Paper-trading log entries (decisions, fills, settlement) for this ticker.
    pub paper: Vec<Value>,
}

fn avg(ticks: &[(i64, f64)], from: i64, to: i64) -> Option<f64> {
    let xs: Vec<f64> = ticks.iter().filter(|(t, _)| *t >= from && *t < to).map(|(_, v)| *v).collect();
    (!xs.is_empty()).then(|| xs.iter().sum::<f64>() / xs.len() as f64)
}

pub fn session_view(dir: &Path, paper: Vec<Value>, now_ms: i64) -> Result<SessionView> {
    let meta = Store::read_meta(dir)?;
    enum Ev {
        Btc(f64),
        Quote(strategy::Quote, Option<strategy::Quote>),
    }
    let mut evs: Vec<(i64, Ev)> = vec![];
    let mut cf = vec![];
    for line in read_lines(&Store::stream_path(dir, "btc"))? {
        if let Some((t, v)) = brti_tick(&line.value()) {
            evs.push((line.recv_ms(), Ev::Btc(v)));
            cf.push((t, v));
        }
    }
    for stream in ["quotes", "book", "market"] {
        for line in read_lines(&Store::stream_path(dir, stream))? {
            if let Some((y, n)) = parse_quotes(stream, &line.value()) {
                evs.push((line.recv_ms(), Ev::Quote(y, n)));
            }
        }
    }
    evs.sort_by_key(|(t, _)| *t);
    cf.sort_by_key(|(t, _)| *t);

    // Flow
    let mut flow = Flow::default();
    let bucket = FLOW_BUCKET_S * 1000;
    let mut buckets = std::collections::BTreeMap::<i64, (f64, f64)>::new();
    for line in read_lines(&Store::stream_path(dir, "trades"))? {
        let v = line.value();
        let m = &v["msg"];
        let t = m["ts_ms"].as_i64().unwrap_or(line.recv_ms());
        let n = kalshi::parse_num(&m["count_fp"]).unwrap_or(0.0);
        let b = buckets.entry(t - t.rem_euclid(bucket)).or_default();
        match m["taker_outcome_side"].as_str().or(m["taker_side"].as_str()) {
            Some("yes") => b.0 += n,
            Some("no") => b.1 -= n,
            _ => {}
        }
    }
    for (t, (y, n)) in buckets {
        flow.t.push(t / 1000);
        flow.yes.push(y);
        flow.no.push(n);
    }

    // 1s grid, assembled exactly as replay does.
    let start = meta.open_ms - BTC_LOOKBACK_MS;
    let end = (meta.close_ms + BTC_TAIL_MS).min(now_ms - now_ms.rem_euclid(1000));
    let window = meta.strike.map(|strike| Window { ticker: meta.ticker.clone(), strike, open_ms: meta.open_ms, close_ms: meta.close_ms });
    let mut asm = Assembler::default();
    asm.clear_quotes();
    let mut hist = History::default();
    let mut grid = Grid::default();
    let mut k = 0;
    let mut t = start;
    let first_ev = evs.first().map(|e| e.0).unwrap_or(i64::MAX);
    while t <= end {
        while k < evs.len() && evs[k].0 <= t {
            match &evs[k].1 {
                Ev::Btc(v) => asm.btc(*v),
                Ev::Quote(y, n) => {
                    // Quotes stop meaning anything once the window closes.
                    if evs[k].0 < meta.close_ms {
                        asm.quotes(evs[k].0, *y, *n);
                    } else {
                        asm.clear_quotes();
                    }
                }
            }
            k += 1;
        }
        if t >= meta.close_ms {
            asm.clear_quotes();
        }
        if t >= first_ev {
            let s = asm.snapshot(t);
            if let Some(s) = s {
                hist.push(s);
            }
            let in_window = t >= meta.open_ms && t < meta.close_ms;
            let fair = match (&window, in_window, s) {
                (Some(w), true, Some(_)) => fair_value(w, &hist).map(|f| f.p),
                _ => None,
            };
            grid.t.push(t / 1000);
            grid.btc.push(s.map(|s| s.btc));
            grid.yes_bid.push(s.and_then(|s| s.yes.bid));
            grid.yes_ask.push(s.and_then(|s| s.yes.ask));
            grid.no_bid.push(s.and_then(|s| s.no.bid));
            grid.no_ask.push(s.and_then(|s| s.no.ask));
            grid.fair.push(fair);
        }
        t += 1000;
    }

    Ok(SessionView {
        strike_avg: avg(&cf, meta.open_ms - 60_000, meta.open_ms),
        settle_avg: avg(&cf, meta.close_ms - 60_000, meta.close_ms),
        meta,
        grid,
        flow,
        paper,
    })
}

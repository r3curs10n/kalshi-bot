//! Session-organized storage.
//!
//! A session is one 15-minute market, keyed by its ticker. Everything recorded
//! for it lives in one self-contained directory:
//!
//! ```text
//! data/sessions/<YYYY-MM-DD of open, UTC>/<ticker>/
//!   meta.json             session facts, status, settlement result, gaps
//!   btc.ndjson.zst        BRTI ticks over [open - lookback, close + tail]
//!   quotes.ndjson.zst     `ticker` channel messages for this market
//!   trades.ndjson.zst     public prints for this market
//!   book.ndjson.zst       REST orderbook snapshots (YES and NO bid ladders)
//!   market.ndjson.zst     REST market polls (native no_bid/no_ask, volume, status)
//!   lifecycle.ndjson.zst  market_lifecycle_v2 events for this market
//! ```
//!
//! Each line is `{recv_ns, session, stream, seq, raw}` with Kalshi's JSON kept
//! as received. A writer never appends to an existing file -- a crash leaves
//! an unfinished zstd frame, and anything appended after it would be
//! unreadable -- so a restarted recorder starts a new segment
//! (`btc.1.ndjson.zst`, `btc.2.ndjson.zst`, ...) and readers concatenate
//! segments in order, stopping cleanly at a crash-truncated tail.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use serde_json::Value;

/// BRTI history kept before a session opens, for volatility features.
pub const BTC_LOOKBACK_MS: i64 = 15 * 60 * 1000;
/// BRTI kept after close: covers the settlement average and Kalshi's publish.
pub const BTC_TAIL_MS: i64 = 60 * 1000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Line {
    pub recv_ns: i64,
    pub session: Option<String>,
    pub stream: String,
    pub seq: u64,
    pub raw: Box<RawValue>,
}

impl Line {
    pub fn value(&self) -> Value {
        serde_json::from_str(self.raw.get()).unwrap_or(Value::Null)
    }

    pub fn recv_ms(&self) -> i64 {
        self.recv_ns / 1_000_000
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    /// Market open, being recorded.
    Recording,
    /// Past close, waiting for Kalshi's result.
    Closed,
    /// Result known and all streams finished.
    Settled,
    /// Never got a result.
    Unsettled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gap {
    pub stream: String,
    pub from_ms: i64,
    pub to_ms: i64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub ticker: String,
    pub event_ticker: String,
    pub strike: Option<f64>,
    pub strike_type: String,
    pub open_ms: i64,
    pub close_ms: i64,
    pub price_ranges: Value,
    pub exchange_index: Option<i64>,
    pub status: SessionStatus,
    /// "yes" or "no" once settled.
    pub result: Option<String>,
    /// The closing 60s BRTI average Kalshi settled on.
    pub expiration_value: Option<f64>,
    pub settled_ms: Option<i64>,
    /// When this recorder first saw the market; later than `open_ms` means
    /// the start of the session was missed.
    pub first_seen_ms: i64,
    pub last_write_ms: i64,
    pub gaps: Vec<Gap>,
    pub counts: BTreeMap<String, u64>,
    pub recorder_version: String,
}

impl SessionMeta {
    /// Build from a REST market object.
    pub fn from_market(m: &Value, now_ms: i64) -> Option<Self> {
        Some(SessionMeta {
            ticker: m["ticker"].as_str()?.to_string(),
            event_ticker: m["event_ticker"].as_str().unwrap_or_default().to_string(),
            strike: m["floor_strike"].as_f64(),
            strike_type: m["strike_type"].as_str().unwrap_or_default().to_string(),
            open_ms: kalshi::parse_time_ms(&m["open_time"])?,
            close_ms: kalshi::parse_time_ms(&m["close_time"])?,
            price_ranges: m["price_ranges"].clone(),
            exchange_index: m["exchange_index"].as_i64(),
            status: SessionStatus::Recording,
            result: None,
            expiration_value: None,
            settled_ms: None,
            first_seen_ms: now_ms,
            last_write_ms: now_ms,
            gaps: vec![],
            counts: BTreeMap::new(),
            recorder_version: env!("CARGO_PKG_VERSION").to_string(),
        })
    }

    /// Fold in fresher REST data. Strike can arrive after first sight.
    pub fn refresh(&mut self, m: &Value) {
        if let Some(k) = m["floor_strike"].as_f64() {
            self.strike = Some(k);
        }
        if let Some(c) = kalshi::parse_time_ms(&m["close_time"]) {
            self.close_ms = c;
        }
        if let Some(r) = m["result"].as_str().filter(|r| !r.is_empty()) {
            self.result = Some(r.to_string());
            self.expiration_value = kalshi::parse_num(&m["expiration_value"]);
            self.settled_ms = kalshi::parse_time_ms(&m["settlement_ts"]);
        }
    }

    /// Interval of BRTI this session keeps.
    pub fn btc_span(&self) -> (i64, i64) {
        (self.open_ms - BTC_LOOKBACK_MS, self.close_ms + BTC_TAIL_MS)
    }

    pub fn date(&self) -> String {
        chrono::DateTime::from_timestamp_millis(self.open_ms)
            .map(|t| t.format("%Y-%m-%d").to_string())
            .unwrap_or_else(|| "unknown".into())
    }
}

/// Appends lines to one zstd stream file.
pub struct StreamWriter {
    enc: zstd::Encoder<'static, BufWriter<File>>,
    seq: u64,
}

impl StreamWriter {
    /// Starts a new segment of the stream whose first segment is `base`.
    pub fn open(base: &Path) -> Result<Self> {
        if let Some(dir) = base.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let path = (0..)
            .map(|k| segment_path(base, k))
            .find(|p| !p.exists())
            .expect("unbounded");
        let file = OpenOptions::new().create_new(true).write(true).open(&path)
            .with_context(|| format!("creating {}", path.display()))?;
        Ok(Self { enc: zstd::Encoder::new(BufWriter::new(file), 3)?, seq: 0 })
    }

    pub fn write(&mut self, recv_ns: i64, session: Option<&str>, stream: &str, raw: &str) -> Result<()> {
        self.seq += 1;
        // Kalshi frames end with a newline, which would split the record. A
        // JSON string can't hold a literal newline, so any newline in `raw`
        // is whitespace between tokens and is safe to drop.
        let raw = raw.trim();
        let raw: std::borrow::Cow<str> =
            if raw.contains(['\n', '\r']) { raw.replace(['\n', '\r'], " ").into() } else { raw.into() };
        let raw = raw.as_ref();
        // Built by hand so `raw` otherwise lands byte-for-byte as received.
        let session = match session {
            Some(s) => serde_json::to_string(s)?,
            None => "null".into(),
        };
        writeln!(
            self.enc,
            r#"{{"recv_ns":{recv_ns},"session":{session},"stream":{},"seq":{},"raw":{raw}}}"#,
            serde_json::to_string(stream)?,
            self.seq
        )?;
        Ok(())
    }

    /// Ends the current zstd block so everything so far survives a crash.
    pub fn flush(&mut self) -> Result<()> {
        self.enc.flush()?;
        Ok(())
    }

    pub fn finish(self) -> Result<()> {
        self.enc.finish()?.flush()?;
        Ok(())
    }
}

/// `x.ndjson.zst` -> `x.ndjson.zst`, `x.1.ndjson.zst`, `x.2.ndjson.zst`, ...
fn segment_path(base: &Path, k: usize) -> PathBuf {
    if k == 0 {
        return base.to_path_buf();
    }
    let name = base.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    let stem = name.strip_suffix(".ndjson.zst").unwrap_or(name);
    base.with_file_name(format!("{stem}.{k}.ndjson.zst"))
}

/// Every line of a stream across all its segments, in write order.
pub fn read_lines(base: &Path) -> Result<Vec<Line>> {
    let mut out = vec![];
    for k in 0.. {
        let path = segment_path(base, k);
        if !path.exists() {
            break;
        }
        out.extend(read_segment(&path)?);
    }
    Ok(out)
}

/// Reads one segment, stopping quietly at a truncated tail.
fn read_segment(path: &Path) -> Result<Vec<Line>> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
        Err(e) => return Err(e.into()),
    };
    let reader = BufReader::new(zstd::Decoder::new(file)?);
    let mut out = vec![];
    // Records written before newlines were stripped can span a few lines.
    let mut buf = String::new();
    let mut parts = 0;
    for line in reader.lines() {
        let Ok(line) = line else { break };
        buf.push_str(&line);
        parts += 1;
        match serde_json::from_str::<Line>(&buf) {
            Ok(l) => {
                out.push(l);
                buf.clear();
                parts = 0;
            }
            Err(e) if e.is_eof() && parts < 4 => buf.push('\n'),
            Err(_) => break,
        }
    }
    Ok(out)
}

/// The data directory.
#[derive(Debug, Clone)]
pub struct Store {
    pub root: PathBuf,
}

impl Store {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn sessions_dir(&self) -> PathBuf {
        self.root.join("sessions")
    }

    pub fn session_dir(&self, meta: &SessionMeta) -> PathBuf {
        self.sessions_dir().join(meta.date()).join(&meta.ticker)
    }

    pub fn stream_path(dir: &Path, stream: &str) -> PathBuf {
        dir.join(format!("{stream}.ndjson.zst"))
    }

    /// The continuous BRTI tape, rotated hourly.
    pub fn brti_path(&self, t_ms: i64) -> PathBuf {
        let t = chrono::DateTime::from_timestamp_millis(t_ms).unwrap_or_default();
        self.root
            .join("brti")
            .join(t.format("%Y-%m-%d").to_string())
            .join(format!("{}.ndjson.zst", t.format("%H")))
    }

    pub fn write_meta(dir: &Path, meta: &SessionMeta) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let tmp = dir.join("meta.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(meta)?)?;
        std::fs::rename(&tmp, dir.join("meta.json"))?;
        Ok(())
    }

    pub fn read_meta(dir: &Path) -> Result<SessionMeta> {
        let path = dir.join("meta.json");
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// Every session directory, oldest first.
    pub fn list_sessions(&self) -> Result<Vec<PathBuf>> {
        let mut out = vec![];
        let Ok(days) = std::fs::read_dir(self.sessions_dir()) else { return Ok(out) };
        for day in days.flatten() {
            for s in std::fs::read_dir(day.path())?.flatten() {
                if s.path().join("meta.json").exists() {
                    out.push(s.path());
                }
            }
        }
        out.sort();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_split_across_lines_by_old_writers_still_read() {
        let dir = std::env::temp_dir().join(format!("kbot-split-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("s.ndjson.zst");
        let text = "{\"recv_ns\":1,\"session\":null,\"stream\":\"btc\",\"seq\":1,\"raw\":{\"v\":1}\n}\n\
                    {\"recv_ns\":2,\"session\":null,\"stream\":\"btc\",\"seq\":2,\"raw\":{\"v\":2}}\n";
        std::fs::write(&path, zstd::encode_all(text.as_bytes(), 3).unwrap()).unwrap();
        assert_eq!(read_lines(&path).unwrap().len(), 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn crashed_segment_then_restart_reads_back() {
        let dir = std::env::temp_dir().join(format!("kbot-store-{}", std::process::id()));
        let base = dir.join("x.ndjson.zst");
        let _ = std::fs::remove_dir_all(&dir);

        // First run: flushed but never finished, as after a crash.
        let mut w = StreamWriter::open(&base).unwrap();
        w.write(1, Some("T1"), "quotes", r#"{"a":1}"#).unwrap();
        w.flush().unwrap();
        w.write(2, Some("T1"), "quotes", r#"{"a":2}"#).unwrap();
        w.flush().unwrap();
        std::mem::forget(w);
        // Restarted recorder: new segment. Kalshi frames end in a newline.
        let mut w = StreamWriter::open(&base).unwrap();
        w.write(3, Some("T1"), "quotes", "{\"a\": 3}\n").unwrap();
        w.write(4, Some("T1"), "quotes", "{\n \"b\": \"x y\"\n}").unwrap();
        w.finish().unwrap();
        assert!(dir.join("x.1.ndjson.zst").exists());

        let lines = read_lines(&base).unwrap();
        assert_eq!(lines.iter().map(|l| l.recv_ns).collect::<Vec<_>>(), vec![1, 2, 3, 4]);
        assert_eq!(lines[2].raw.get(), r#"{"a": 3}"#);
        assert_eq!(lines[3].value()["b"], "x y");
        assert_eq!(lines[0].session.as_deref(), Some("T1"));

        // Chop bytes off the first segment: what came before still reads.
        let bytes = std::fs::read(&base).unwrap();
        std::fs::write(&base, &bytes[..bytes.len() - 3]).unwrap();
        assert!(read_lines(&base).unwrap().len() >= 3);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

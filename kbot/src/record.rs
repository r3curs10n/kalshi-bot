//! The recorder daemon: live feeds -> per-session directories.
//!
//! One writer task owns every file and routes each message to its session.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use kalshi::{now_ms, now_ns, Rest, SERIES};
use market::store::{Gap, SessionMeta, SessionStatus, StreamWriter, BTC_TAIL_MS};
use market::Store;
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::feeds::{self, Feed};

pub async fn run(config: kalshi::Config) -> Result<()> {
    let signer = Arc::new(config.signer()?);
    let rest = Rest::new(signer.clone())?;
    rest.balance().await.context("auth check (GET /portfolio/balance) failed")?;
    let store = Store::new(&config.data_dir);
    info!("recording {SERIES} into {}", store.root.display());

    let (tx, rx) = mpsc::unbounded_channel();
    let pending = unfinished_sessions(&store)?;
    let writer = tokio::spawn(writer(store, rx));
    feeds::spawn_all(signer, rest, tx.clone(), pending);

    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    tokio::select! {
        _ = tokio::signal::ctrl_c() => info!("interrupted"),
        _ = term.recv() => info!("terminated"),
    }
    // Close zstd frames cleanly so every file is complete.
    tx.send(Feed::Shutdown)?;
    writer.await?
}

/// Sessions a previous run left waiting for a result.
fn unfinished_sessions(store: &Store) -> Result<HashMap<String, i64>> {
    let cutoff = now_ms() - feeds::SETTLE_DEADLINE_MS;
    let mut out = HashMap::new();
    for dir in store.list_sessions()? {
        let m = Store::read_meta(&dir)?;
        if matches!(m.status, SessionStatus::Recording | SessionStatus::Closed) && m.close_ms > cutoff {
            out.insert(m.ticker, m.close_ms);
        }
    }
    Ok(out)
}

// --------------------------------------------------------------- writer ---

struct Session {
    meta: SessionMeta,
    dir: PathBuf,
    writers: HashMap<&'static str, StreamWriter>,
}

impl Session {
    fn write(&mut self, stream: &'static str, recv_ns: i64, raw: &str) -> Result<()> {
        let w = match self.writers.entry(stream) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => e.insert(StreamWriter::open(&Store::stream_path(&self.dir, stream))?),
        };
        w.write(recv_ns, Some(&self.meta.ticker), stream, raw)?;
        *self.meta.counts.entry(stream.to_string()).or_default() += 1;
        Ok(())
    }

    fn sync(&mut self, now: i64) -> Result<()> {
        for w in self.writers.values_mut() {
            w.flush()?;
        }
        self.meta.last_write_ms = now;
        Store::write_meta(&self.dir, &self.meta)
    }

    fn finish(mut self, status: SessionStatus) -> Result<()> {
        for (_, w) in self.writers.drain() {
            w.finish()?;
        }
        self.meta.status = status;
        Store::write_meta(&self.dir, &self.meta)
    }
}

struct Writer {
    store: Store,
    sessions: HashMap<String, Session>,
    ring: VecDeque<(i64, i64, String)>,
    tape: Option<(PathBuf, StreamWriter)>,
    counts: HashMap<&'static str, u64>,
}

async fn writer(store: Store, mut rx: mpsc::UnboundedReceiver<Feed>) -> Result<()> {
    let ring = seed_ring(&store);
    let mut w = Writer { store, sessions: HashMap::new(), ring, tape: None, counts: HashMap::new() };
    let mut sync = tokio::time::interval(Duration::from_secs(5));
    let mut health = tokio::time::interval(Duration::from_secs(60));
    loop {
        tokio::select! {
            cmd = rx.recv() => {
                let Some(cmd) = cmd else { break };
                if matches!(cmd, Feed::Shutdown) {
                    return w.shutdown();
                }
                if let Err(e) = w.handle(cmd) {
                    warn!("writer: {e:#}");
                }
            }
            _ = sync.tick() => w.sync()?,
            _ = health.tick() => w.health(),
        }
    }
    w.sync()
}

/// Recent BRTI from the tape, so sessions created right after a restart still
/// get their lookback.
fn seed_ring(store: &Store) -> VecDeque<(i64, i64, String)> {
    let now = now_ms();
    let mut ring = VecDeque::new();
    let mut paths = vec![store.brti_path(now - feeds::RING_MS), store.brti_path(now)];
    paths.dedup();
    for path in paths {
        for line in market::store::read_lines(&path).unwrap_or_default() {
            let t_ms = market::replay::brti_tick(&line.value()).map(|(t, _)| t).unwrap_or(line.recv_ms());
            if t_ms >= now - feeds::RING_MS {
                ring.push_back((t_ms, line.recv_ns, line.raw.get().to_string()));
            }
        }
    }
    info!("seeded {} BRTI ticks from the tape", ring.len());
    ring
}

impl Writer {
    fn handle(&mut self, cmd: Feed) -> Result<()> {
        match cmd {
            Feed::Btc { recv_ns, t_ms, raw } => {
                *self.counts.entry("btc").or_default() += 1;
                let path = self.store.brti_path(t_ms);
                if self.tape.as_ref().is_none_or(|(p, _)| *p != path) {
                    if let Some((_, old)) = self.tape.take() {
                        old.finish()?;
                    }
                    self.tape = Some((path.clone(), StreamWriter::open(&path)?));
                }
                self.tape.as_mut().unwrap().1.write(recv_ns, None, "btc", &raw)?;
                for s in self.sessions.values_mut() {
                    let (a, b) = s.meta.btc_span();
                    if (a..=b).contains(&t_ms) {
                        s.write("btc", recv_ns, &raw)?;
                    }
                }
                self.ring.push_back((t_ms, recv_ns, raw));
                while self.ring.front().is_some_and(|(t, _, _)| *t < t_ms - feeds::RING_MS) {
                    self.ring.pop_front();
                }
            }
            Feed::Market { stream, ticker, recv_ns, raw } => {
                *self.counts.entry(stream).or_default() += 1;
                match self.sessions.get_mut(&ticker) {
                    Some(s) => s.write(stream, recv_ns, &raw)?,
                    None => *self.counts.entry("unrouted").or_default() += 1,
                }
            }
            Feed::Upsert(m) => {
                let ticker = m["ticker"].as_str().context("market without ticker")?.to_string();
                if !self.sessions.contains_key(&ticker) {
                    let s = self.open_session(&m)?;
                    self.sessions.insert(ticker.clone(), s);
                }
                let s = self.sessions.get_mut(&ticker).unwrap();
                s.meta.refresh(&m);
                s.write("market", now_ns(), &m.to_string())?;
            }
            Feed::Settled(m) => {
                let ticker = m["ticker"].as_str().context("market without ticker")?.to_string();
                if !self.sessions.contains_key(&ticker) {
                    // Settling a session from a previous run.
                    let s = self.open_session(&m)?;
                    self.sessions.insert(ticker.clone(), s);
                }
                let s = self.sessions.get_mut(&ticker).unwrap();
                s.meta.refresh(&m);
                s.write("market", now_ns(), &m.to_string())?;
                s.sync(now_ms())?;
            }
            Feed::Shutdown => {}
            Feed::Gap { streams, from_ms, to_ms, reason } => {
                for s in self.sessions.values_mut() {
                    for stream in streams {
                        let (a, b) = if *stream == "btc" { s.meta.btc_span() } else { (s.meta.open_ms, s.meta.close_ms) };
                        if from_ms < b && to_ms > a {
                            s.meta.gaps.push(Gap { stream: stream.to_string(), from_ms, to_ms, reason: reason.clone() });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Opens a session directory, continuing one from a previous run if it
    /// exists, and backfills BRTI lookback from memory.
    fn open_session(&mut self, m: &Value) -> Result<Session> {
        let now = now_ms();
        let fresh = SessionMeta::from_market(m, now).context("unparseable market")?;
        let dir = self.store.session_dir(&fresh);
        let (meta, since) = match Store::read_meta(&dir) {
            Ok(mut old) => {
                if old.status == SessionStatus::Recording && now < old.close_ms + BTC_TAIL_MS {
                    old.gaps.push(Gap { stream: "all".into(), from_ms: old.last_write_ms, to_ms: now, reason: "recorder_restart".into() });
                }
                let since = old.last_write_ms;
                (old, since)
            }
            Err(_) => {
                let mut meta = fresh;
                if now > meta.open_ms + feeds::GAP_MS {
                    meta.gaps.push(Gap { stream: "all".into(), from_ms: meta.open_ms, to_ms: now, reason: "joined_late".into() });
                }
                (meta, i64::MIN)
            }
        };
        info!("session {} (open {}, close {})", meta.ticker, fmt_ms(meta.open_ms), fmt_ms(meta.close_ms));
        let mut s = Session { meta, dir, writers: HashMap::new() };
        let (a, b) = s.meta.btc_span();
        let backfill: Vec<_> = self
            .ring
            .iter()
            .filter(|(t, recv_ns, _)| (a..=b).contains(t) && recv_ns / 1_000_000 > since)
            .cloned()
            .collect();
        for (_, recv_ns, raw) in backfill {
            s.write("btc", recv_ns, &raw)?;
        }
        s.sync(now)?;
        Ok(s)
    }

    fn sync(&mut self) -> Result<()> {
        let now = now_ms();
        if let Some((_, t)) = self.tape.as_mut() {
            t.flush()?;
        }
        let mut done = vec![];
        for (ticker, s) in self.sessions.iter_mut() {
            if s.meta.status == SessionStatus::Recording && now >= s.meta.close_ms {
                s.meta.status = SessionStatus::Closed;
            }
            s.sync(now)?;
            let tail_done = now > s.meta.close_ms + BTC_TAIL_MS;
            if tail_done && s.meta.result.is_some() {
                done.push((ticker.clone(), SessionStatus::Settled));
            } else if now > s.meta.close_ms + feeds::SETTLE_DEADLINE_MS {
                done.push((ticker.clone(), SessionStatus::Unsettled));
            }
        }
        for (ticker, status) in done {
            let s = self.sessions.remove(&ticker).unwrap();
            info!("session {ticker} finished: {status:?} result={:?}", s.meta.result);
            s.finish(status)?;
        }
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        if let Some((_, t)) = self.tape.take() {
            t.finish()?;
        }
        let now = now_ms();
        for (_, mut s) in self.sessions.drain() {
            s.meta.last_write_ms = now;
            let status = s.meta.status;
            s.finish(status)?;
        }
        info!("all files closed");
        Ok(())
    }

    fn health(&mut self) {
        let mut counts: Vec<_> = self.counts.drain().collect();
        counts.sort();
        let active: Vec<&str> = self.sessions.keys().map(|s| s.as_str()).collect();
        info!("last 60s: {counts:?}; sessions: {active:?}");
    }
}

fn fmt_ms(ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(ms).map(|t| t.format("%H:%M:%S").to_string()).unwrap_or_default()
}

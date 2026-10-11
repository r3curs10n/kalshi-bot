//! `kbot viz`: a local web page for browsing recorded sessions.
//!
//! Serves one embedded HTML page plus JSON endpoints, reading session
//! directories on every request so sessions still being recorded stay live.
//! It can also backtest any strategy (baseline or trained model) over one
//! session, and run one live on an ongoing session with real orders (see
//! `live`). Binds to localhost only.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use market::replay::{build_episode, Requirements, EPISODE_FILE};
use market::view::session_view;
use market::{run_strategy_traced, Episode, SimConfig, Store};
use serde::Deserialize;
use serde_json::{json, Value};
use strategy::rl::{TrunkMeta, META_FILE, NET, TRUNK_META_FILE};
use strategy::{TradeLimits, MAX_SWING_BPS, RULE_LAST_S};

use crate::backtest::{make_strategy, RULE_SWING};
use crate::live::{LiveRequest, LiveRun};

const PAGE: &str = include_str!("viz.html");

#[derive(Clone)]
struct App {
    store: Store,
    /// Searched for trained models (directories holding `model.json`).
    models: PathBuf,
    /// The one live run allowed at a time, kept after it ends so the page
    /// can show how it went.
    live: Arc<Mutex<Option<LiveRun>>>,
}

pub async fn run(store: Store, models: PathBuf, port: u16) -> Result<()> {
    let app = Router::new()
        .route("/", get(|| async { Html(PAGE) }))
        .route("/api/sessions", get(sessions))
        .route("/api/session/{date}/{ticker}", get(session))
        .route("/api/session/{date}/{ticker}/run", get(run_session))
        .route("/api/strategies", get(strategies))
        .route("/api/live", get(live_status))
        .route("/api/live/start", post(live_start))
        .route("/api/live/stop", post(live_stop))
        .with_state(App { store, models, live: Default::default() });
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("session visualizer at http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

fn err(e: impl std::fmt::Display) -> Response {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
}

/// Paper-trading log entries for one ticker, or all of them.
fn paper_entries(store: &Store, ticker: Option<&str>) -> Vec<Value> {
    let mut out = vec![];
    let Ok(files) = std::fs::read_dir(store.root.join("paper")) else { return out };
    let mut files: Vec<_> = files.flatten().map(|f| f.path()).collect();
    files.sort();
    for f in files {
        let Ok(text) = std::fs::read_to_string(&f) else { continue };
        for line in text.lines() {
            let Ok(v) = serde_json::from_str::<Value>(line) else { continue };
            if ticker.is_none_or(|t| v["ticker"] == t) {
                out.push(v);
            }
        }
    }
    out
}

async fn sessions(State(App { store, .. }): State<App>) -> Response {
    let dirs = match store.list_sessions() {
        Ok(d) => d,
        Err(e) => return err(e),
    };
    let paper = paper_entries(&store, None);
    let mut rows = vec![];
    for dir in dirs.iter().rev() {
        let Ok(m) = Store::read_meta(dir) else { continue };
        let paper_pnl = paper.iter().find(|p| p["type"] == "settle" && p["ticker"] == m.ticker.as_str()).map(|p| p["pnl"].clone());
        rows.push(json!({
            "date": m.date(),
            "ticker": m.ticker,
            "status": m.status,
            "result": m.result,
            "strike": m.strike,
            "open_ms": m.open_ms,
            "close_ms": m.close_ms,
            "gaps": m.gaps.len(),
            "counts": m.counts,
            "episode": dir.join(EPISODE_FILE).exists(),
            "paper_pnl": paper_pnl,
        }));
    }
    Json(rows).into_response()
}

/// The session directory for `date/ticker`, if the store lists it: no path tricks.
fn session_dir(store: &Store, date: &str, ticker: &str) -> Result<Option<PathBuf>> {
    let wanted = store.sessions_dir().join(date).join(ticker);
    Ok(store.list_sessions()?.into_iter().find(|d| *d == wanted))
}

async fn session(State(App { store, .. }): State<App>, Path((date, ticker)): Path<(String, String)>) -> Response {
    let dir = match session_dir(&store, &date, &ticker) {
        Ok(Some(d)) => d,
        Ok(None) => return (StatusCode::NOT_FOUND, "no such session").into_response(),
        Err(e) => return err(e),
    };
    let paper = paper_entries(&store, Some(&ticker));
    let built = tokio::task::spawn_blocking(move || session_view(&dir, paper, kalshi::now_ms())).await;
    match built {
        Ok(Ok(view)) => Json(view).into_response(),
        Ok(Err(e)) => err(e),
        Err(e) => err(e),
    }
}

/// Baselines, the rule gated by every trunk under the models root that
/// forecasts far enough ahead, and every model directory there, as
/// `{id, label}`; `id` is what `make_strategy` takes.
fn strategy_list(models: &std::path::Path) -> Vec<Value> {
    let mut out = vec![
        json!({ "id": "nothing", "label": "nothing" }),
        json!({ "id": "fair", "label": "fair value (edge 0.03)" }),
        json!({ "id": "rule", "label": "rule: all in on the favorite, last 3 min, when fair value covers cost" }),
        json!({ "id": "rule,edge=off", "label": "rule: all in on the favorite, last 3 min, unchecked" }),
    ];
    let mut found = vec![];
    let mut trunks = vec![];
    let mut stack = vec![(models.to_path_buf(), 0)];
    while let Some((dir, depth)) = stack.pop() {
        if dir.join(META_FILE).is_file() {
            found.push(dir);
            continue;
        }
        if dir.join(TRUNK_META_FILE).is_file() {
            trunks.push(dir);
            continue;
        }
        if depth >= 3 {
            continue;
        }
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        stack.extend(rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).map(|p| (p, depth + 1)));
    }
    trunks.sort();
    for d in trunks {
        let far = std::fs::read(d.join(TRUNK_META_FILE))
            .ok()
            .and_then(|b| serde_json::from_slice::<TrunkMeta>(&b).ok())
            .is_some_and(|m| m.horizons.iter().any(|h| *h as f64 >= RULE_LAST_S / 2.0));
        if far {
            let dir = d.to_string_lossy();
            out.push(json!({
                "id": format!("{RULE_SWING}{dir}"),
                "label": format!("rule + swing filter: skip if {dir} forecasts > {MAX_SWING_BPS} bps"),
            }));
        }
    }
    found.sort();
    for d in found {
        let id = d.to_string_lossy().to_string();
        // Name the network so stale (pre-LSTM) models are easy to spot.
        let net = std::fs::read(d.join(META_FILE))
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .map(|m| match m["net"].as_str().unwrap_or("mlp") {
                NET => NET.to_string(),
                old => format!("{old}, retrain"),
            })
            .unwrap_or_else(|| "unreadable".into());
        out.push(json!({ "id": id, "label": format!("model {id} ({net})") }));
    }
    out
}

async fn strategies(State(app): State<App>) -> Response {
    Json(strategy_list(&app.models)).into_response()
}

#[derive(Deserialize)]
struct RunQuery {
    strategy: String,
}

/// Runs a strategy over one session through the backtest simulator. Uses the
/// session's episode file if the dataset builder made one; otherwise builds
/// one on the spot with no coverage rules, so any settled session works.
fn backtest_session(dir: &std::path::Path, strategy: &str) -> Result<Value> {
    let (ep, source) = if dir.join(EPISODE_FILE).exists() {
        (Episode::load(dir)?, "episode")
    } else {
        match build_episode(dir, &Requirements::lenient())? {
            Ok((ep, _)) => (ep, "built on the fly (no coverage checks)"),
            Err(reason) => bail!("can't backtest this session: {reason}"),
        }
    };
    if ep.snapshots.is_empty() {
        bail!("can't backtest this session: no snapshots");
    }
    let mut s = make_strategy(strategy, TradeLimits::default(), 1, 0.03)?;
    let cfg = SimConfig { limits: s.limits(), ..Default::default() };
    let balance = cfg.balance;
    let mut equity = (vec![], vec![]);
    let r = run_strategy_traced(s.as_mut(), &ep, cfg, 0, &mut |t, e| {
        equity.0.push(t / 1000);
        equity.1.push(e);
    });
    let fills: Vec<Value> = r
        .fills
        .iter()
        .map(|f| {
            json!({
                "t_ms": ep.snapshots[f.idx].t_ms,
                "action": f.action,
                "size": f.size,
                "price": f.price,
                "fee": f.fee,
            })
        })
        .collect();
    // The final point is the settled payout, so the curve ends at realized P&L.
    equity.0.push(ep.window.close_ms / 1000);
    equity.1.push(r.pnl);
    Ok(json!({
        "strategy": strategy,
        "source": source,
        "balance": balance,
        "pnl": r.pnl,
        "fees": r.fees,
        "trades": r.trades,
        "missed": r.missed,
        "result_yes": r.result_yes,
        "fills": fills,
        "equity": { "t": equity.0, "v": equity.1 },
    }))
}

async fn run_session(
    State(app): State<App>,
    Path((date, ticker)): Path<(String, String)>,
    Query(q): Query<RunQuery>,
) -> Response {
    // Only strategies we listed: the id doubles as a filesystem path.
    if !strategy_list(&app.models).iter().any(|s| s["id"] == q.strategy.as_str()) {
        return (StatusCode::BAD_REQUEST, "unknown strategy").into_response();
    }
    let dir = match session_dir(&app.store, &date, &ticker) {
        Ok(Some(d)) => d,
        Ok(None) => return (StatusCode::NOT_FOUND, "no such session").into_response(),
        Err(e) => return err(e),
    };
    match tokio::task::spawn_blocking(move || backtest_session(&dir, &q.strategy)).await {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err(e)) => (StatusCode::UNPROCESSABLE_ENTITY, e.to_string()).into_response(),
        Err(e) => err(e),
    }
}

async fn live_status(State(app): State<App>) -> Response {
    let live = app.live.lock().unwrap();
    Json(live.as_ref().map(|r| r.status.lock().unwrap().clone())).into_response()
}

/// Starts live trading. JSON bodies only, so another site can't post here
/// from a browser without a CORS preflight this server never answers.
async fn live_start(State(app): State<App>, Json(req): Json<LiveRequest>) -> Response {
    if !strategy_list(&app.models).iter().any(|s| s["id"] == req.strategy.as_str()) {
        return (StatusCode::BAD_REQUEST, "unknown strategy").into_response();
    }
    if app.live.lock().unwrap().as_ref().is_some_and(|r| !r.is_done()) {
        return (StatusCode::CONFLICT, "a live run is already going; stop it first").into_response();
    }
    let store = app.store.clone();
    // Loading a model is blocking work.
    let started = tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        let _guard = rt.enter();
        crate::live::start(store, req)
    })
    .await;
    match started {
        Ok(Ok(run)) => {
            let status = run.status.lock().unwrap().clone();
            *app.live.lock().unwrap() = Some(run);
            Json(status).into_response()
        }
        Ok(Err(e)) => (StatusCode::UNPROCESSABLE_ENTITY, format!("{e:#}")).into_response(),
        Err(e) => err(e),
    }
}

async fn live_stop(State(app): State<App>, Json(_): Json<Value>) -> Response {
    match app.live.lock().unwrap().as_ref() {
        Some(r) => {
            r.stop();
            StatusCode::NO_CONTENT.into_response()
        }
        None => (StatusCode::NOT_FOUND, "nothing is running").into_response(),
    }
}

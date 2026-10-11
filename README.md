# kalshi-bot

Quant RL bot for Kalshi's 15-minute Bitcoin contract, `KXBTC15M` ("BTC price up in next 15 mins?").
It records live Kalshi data and trains a PPO policy on it. Paper trading uses the same interface the
backtests do: feed it BRTI spot plus YES/NO bid/ask at every instant, and get back
**Buy YES / Sell YES / Buy NO / Sell NO / Nothing**.

Real orders only go out from the viz page's live-trading card, which you start by hand (see below).

## Setup

```bash
cp config.example.toml config.toml   # key_id + key_path (RSA PEM); gitignored
cargo build --release
./target/release/kbot --help
```

`KALSHI_KEY_ID`, `KALSHI_KEY_PATH` and `KBOT_DATA_DIR` override the config file.

## Workflow

```bash
kbot record                          # daemon: live feeds -> data/sessions/<date>/<ticker>/
kbot sessions ls [--status settled]  # what has been recorded
kbot dataset                         # settled sessions -> episode.json.zst (+ checks vs Kalshi)
kbot backtest --strategy fair        # or: nothing | rule | rule-swing:models/trunk-3m | models/<run>/best
kbot calibrate                       # is the favorite priced right? does a stop-loss / entry cap pay?
kbot pretrain --out models/trunk     # stage 1: LSTM trunk forecasts BRTI (chronological train/val/test)
kbot train --trunk models/trunk --out models/run1   # stage 2: PPO heads on the frozen trunk, vs baselines
kbot paper --strategy models/run1/best   # live data, simulated fills, settles on Kalshi's result
kbot viz                             # browse sessions at http://127.0.0.1:8787
```

`kbot viz` serves a local page (localhost only) that reads session directories directly, so it also works
for sessions still being recorded; it refreshes live ones every 5 seconds.
- **Sidebar:** sessions with status, result, message counts, gaps, and paper P&L.
- **BRTI chart:** price against the strike, with the strike and settlement averaging windows shaded, Kalshi's
  settlement value, and recording gaps.
- **Contract prices:** YES and NO bid–ask bands, the fair-value P(YES), and paper fills.
- **Trade flow:** public prints split by taker side.
- **Backtest:** pick a baseline or any trained model under `models/` (`--models` changes where it looks) and
  run it on the selected session. You get its fills on the price chart, an equity curve, a fills table, and the
  settled P&L. It uses the session's episode file when there is one. Otherwise it builds an episode on the spot
  with no coverage checks, so any settled session works.

- **Live trading (real money):** on a session that hasn't closed, enter a Kalshi API key ID, paste or load its
  private key, set a budget per session, pick a strategy and press Start. The strategy trades that session and
  every one after it until you press Stop. Each session starts its own account with the budget as cash, and each
  real fill (count, cost and fee as Kalshi reports them) moves it, so no session spends more than the budget.
  Before a session trades, its shard (crypto is shard 2) is topped up to the budget from shard 0; a session that
  can't be funded is skipped. Orders are immediate-or-cancel limits 1¢ through the touch (`reduce_only` for sells),
  never sent on quotes older than 10s. If Kalshi accepts an order but its fills can't be read back, the run stops
  rather than guess. The key stays in memory and is never written anywhere. Stop leaves anything held open on
  Kalshi, and it settles there. One run at a time; every order is logged to `data/live/<date>.ndjson`.

The charts share a time axis: drag to zoom, double-click to reset. A CSV download gives the 1s series.

To run the recorder permanently, see `deploy/kalshi-recorder.service`.

### Rule strategies

- **`rule`:** in the last 3 minutes, goes all in on the side the market favors (YES mid above 50¢ → YES, below → NO)
  and holds to settlement. It only buys when the payout covers the cost: the analytic fair value's chance of that side
  winning must beat the worst live fill (ask + 1¢) plus the taker fee at that price. Following the market alone can't
  pass that check, since the market's own odds always come out a fee and a spread short. `,edge=<dollars>` demands a
  margin on top; `,edge=off` drops the check.
- **`rule-swing:<trunk dir>[@<bps>]`:** the same rule, but it stays out while a pretrained trunk forecasts a big BRTI
  move over the next 3 minutes: sqrt(mean² + variance) of the forecast log return, above `<bps>` (default 6). The trunk
  needs a horizon near 180s: `kbot pretrain --out models/trunk-3m --horizons 1,60,180`. The viz page lists one entry per
  such trunk under `models/`. To see forecast swings against outcomes, run
  `cargo run --release -p rl --example swing -- <trunk dir> <session dir>...`.

Both rules also take a stop-loss and an entry cap, in dollars of the side bought:
`rule,stop=0.40,cap=0.85` sells everything once the held side's bid falls to 40¢ and then stays out of the
session, and never buys above an 85¢ ask. The swing filter still gates only new bets, never a stop.

`kbot calibrate [--base rule-swing:<trunk>] [--stops ..] [--caps ..]` prints three tables over the episodes:
- **Entry calibration:** win rate against the ask the rule pays, by price bucket, with edge per contract after fees.
- **Cost check:** entries the fair value says cover their cost against the ones it skips, and how each actually did.
  The check is only worth keeping if its takes do better than its skips.
- **After a drop:** for entries whose bid later fell to each stop level, how often they still won, against
  what selling there fetched. A positive "stop gain" means the market under-reacts and cutting pays there.
- **Sweep:** the unchecked rule (`edge=off`) for reference, then the base rule with every stop × cap through the simulator, sorted by P&L. Choose on one date range
  and confirm on a later one (`--from/--to`).

`kbot synth --data-dir /tmp/synth` writes synthetic sessions with a planted edge (quotes lag BRTI).
Use it with `KBOT_DATA_DIR=/tmp/synth` to check that the pipeline can learn. Never mix it into `data/`.

## Inference interface (`crates/strategy`)

```rust
let mut s = RlStrategy::load("models/run1/best".as_ref())?;   // or FairValueStrategy, RuleStrategy, NothingStrategy
s.begin_window(&Window { ticker, strike, open_ms, close_ms });
loop {
    let snap = Snapshot { t_ms, btc, yes: Quote { bid, ask, bid_size, ask_size }, no: Quote { .. } };
    let d: Decision = s.on_tick(&snap, &position);   // action, size, limit, probs, fair_value
}
```

The strategy keeps its own rolling history, so the caller pushes one snapshot per second.
- **Network:** two parts, trained separately. It decides once a second (`--every 1` at pretraining).
  - **Trunk:** encoder → LSTM (32 units) over the market features only (BRTI and YES/NO quotes, never the
    account). `kbot pretrain` trains it to forecast BRTI: a mean and a variance of the log return to each
    `--horizons` (default: the next second), by Gaussian log-likelihood. It reports validation log-likelihood
    and R² against a constant Gaussian; the variance (volatility) is the learnable part.
  - **Heads:** small policy and value MLPs (32 units). Their input is the trunk's output, the 29 normalized
    market features themselves, and the 5 position features. The direct market features keep what trading needs
    but forecasting BRTI gave the trunk no reason to retain (on synthetic data, quote staleness). `kbot train`
    trains them with PPO while the trunk stays frozen.
  - **State:** the LSTM state advances once per decision and resets in `begin_window`, live and in training.
  - **Old models:** models from an older architecture won't load and need retraining (viz labels them).
- **Features:** derived from that history (returns, realized vol, spreads, edge vs. fair value). They are
  computed by the same code in training and live.
- **Sizing:** the strategy picks how much each trade is for. A buy spends 5%, 10%, 25%, 50% or 100% of the
  cash on hand, and a sell closes the same fractions of what is held. The policy chooses among all 21
  (action, size) pairs; the fair-value baseline buys with 10% and exits in full. `Position` carries the cash.
- **Masking:** applied internally, so a `Decision` is always executable. Sell only what you hold, never hold
  both sides, never let cash go negative (fees included, so a sale near 0 whose rounded-up fee exceeds its proceeds is refused when cash can't cover it), and stay under `TradeLimits::max_contracts`.
- **Balance:** backtests, training and paper trading start every session with $1,000 (`--balance`). P&L is the
  change from that. Fills take what the touch shows, so a big order on a thin book fills partially.
- **Cross-window state:** `begin_window` drops the previous contract's quotes but keeps BRTI history.

`fair_value` is P(YES) under a random walk, and it accounts for how the contract settles: the average of
the 60s of BRTI before close is compared with the 60s average before open (`floor_strike`).

## Data layout

Each 15-minute market is one **session**, keyed by its ticker. Each session directory is self-contained:

```
data/sessions/2026-10-02/KXBTC15M-26OCT021530-30/
  meta.json          strike, open/close, status (recording|closed|settled|unsettled), result,
                     expiration_value, gaps, message counts
  btc.ndjson.zst     BRTI (Kalshi `cfbenchmarks_value`) from open-15min to close+60s
  quotes.ndjson.zst  `ticker` channel: YES bid/ask + sizes
  book.ndjson.zst    REST orderbook every 2s: separate YES and NO bid ladders
  market.ndjson.zst  REST market every 5s: native yes/no bid/ask, volume, status
  trades.ndjson.zst  public prints
  lifecycle.ndjson.zst
  episode.json.zst   (built by `kbot dataset`) 1s Snapshot grid + result
data/brti/<date>/<hour>.ndjson.zst   continuous BRTI tape
data/paper/<date>.ndjson             paper decisions, fills, settlements
```

**Line format:** `{recv_ns, session, stream, seq, raw}`, where `raw` is Kalshi's JSON.

**Restarts:** a restart never appends to an old file. It starts a new segment instead
(`btc.1.ndjson.zst`, ...), so a crash can't corrupt earlier data.

**Pre-subscription:** the recorder starts a session when Kalshi lists it, about 15 minutes before open. So
quotes are subscribed and BRTI lookback is in place at the open.

### Where YES/NO prices come from

- **WS `ticker`:** only carries YES.
- **REST market and orderbook:** carry NO natively.
- **Snapshots:** use native NO whenever it is at least as fresh as the YES quote. Otherwise NO comes from
  the YES quote seen from the other side (NO bid = 1 − YES ask). Kalshi runs one book per market, so the
  two always agree.

## Kalshi API notes (learned in `../monymaker`)

- **One socket per feed.** Channels sharing a socket head-of-line block each other.
- **Avoid `orderbook_delta`.** It backlogs without bound, so we use `ticker` plus REST book snapshots
  instead.
- **No pings.** Kalshi doesn't answer them, so silence timeouts detect dead connections.
- **One subscription per channel.** A second `subscribe` on the same connection is merged into the first
  `sid` (Kalshi replies `ok`, not `subscribed`). Change the market set with `update_subscription`
  (`add_markets` / `delete_markets`); never unsubscribe "a ticker's sid".
- **Pings aren't liveness.** Kalshi pings keep a socket open even after it stops delivering data, so the
  recorder judges liveness on data alone.
- **Stale listings.** A closed window stays listed `active`/`open` for a while, so filter by `close_time`.
- **Settlement lag.** Results appear a few seconds after close, with `status: finalized`.

## Simulator and training

- **Fills:** `market::Sim` fills taker orders at the touch observed `latency` seconds after the decision.
  It fills as many contracts as the touch shows and the cash covers. Training adds randomized latency (0–2s) and 10% one-tick
  slippage.
- **Fees:** Kalshi's quadratic fee, `ceil(0.07·C·P·(1−P))`.
- **PPO on a frozen trunk:** the trunk never sees the account, so its outputs over a session don't depend on
  the policy. They're computed once per episode with the same calls `RlStrategy` makes, and PPO trains only the
  heads.
- **Split:** both stages use the same chronological split. A trunk records the last session it trained on,
  and `kbot train` refuses a split that would validate or test on sessions the trunk has seen.
- **Rewards:** marked at mid and settled at $1/$0, so episode return equals realized PnL exactly. Scaled to
  units of 1% of the starting balance.
- **Parity check:** `kbot train` replays the test split through `RlStrategy` (the live path) and fails if
  PnL differs from the trainer's own evaluation.

**The model is only worth using if it beats `fair` out of sample, after fees.**

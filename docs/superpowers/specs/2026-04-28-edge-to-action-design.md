# Edge-to-Action — Live Microstructure Paper Trader

**Date:** 2026-04-28
**Status:** Design approved; awaiting implementation plan
**Owner:** David Blau

## Goal

Wire the saved `live_home_up_5m_bootstrap` model into a real-time scoring + sizing + paper-trading loop on Kalshi NBA contracts, with a clear, gated path to flip the same engine to live execution. The trader takes 5-minute home-side YES scalp positions when the model's calibrated probability and an EV-aware trigger jointly favor it, sized by a continuous Kelly approximation, with one global concurrency cap and no take-profit / stop-loss inside the holding window.

## Non-goals (v1)

- **Pre-game / win-prob betting.** That flow already exists (`ev_analyzer.py`, `edge_detector.py`, `bet_decisions.csv`) and is untouched.
- **Hedged or cross-venue trades.** Single venue (Kalshi), single side (home YES).
- **NO-side or away-side scalps.** Excluded for v1; would require a separately trained/calibrated model.
- **Take-profit / stop-loss inside the 5-minute window.** Hard time-based exit only.
- **Order routing to live Kalshi.** Engine emits intended orders; live flip is a separately gated future step.
- **Web dashboard, alerting, or DB.** Plain CSVs + JSONL + a CLI status tool.
- **Automated A→B sizing flip.** Computed automatically, applied automatically, but only after the configured trade count + CI gates pass.
- **Automated paper→live promotion.** Always requires a manual config-flag change after the gates pass.

## Design decisions (settled during brainstorm)

| Decision | Chosen value | Rationale |
|---|---|---|
| Action type | Microstructure scalping (5-min holds) | Matches the model's target `label_home_up_5m`; pre-game flow handles win-prob bets separately. |
| Trigger | EV-aware: `p × E[Δ\|rises] + (1−p) × E[Δ\|doesn't] − fees − half_spread > 0` | Probability-only thresholds ignore that Δ shrinks at price extremes; EV form internalizes that. |
| Move-size source | Phase A: pooled historical conditional means; Phase B: per-p-decile bootstrapped means | Start with a stable estimate; only refine when sample size justifies it. |
| Sizing | Continuous-payoff Kelly: `f = μ/σ²` capped at 5% of bankroll per position | Standard for unbounded-payoff settings; 5% cap keeps any single mispricing from blowing up. |
| Concurrency | Stack freely; global cap = 20 simultaneous open positions | Independent signals across games are uncorrelated enough to compound; per-ticker cap unnecessary at this scale. |
| Exit | Hard 5-minute timer keyed off `captured_at` | Matches model horizon exactly; no exit-rule mismatch with the training target. |
| Eligibility | Live games only (`status_state=="in"`), home-YES only, liquidity floor (OI ≥ 100, depth_3 ≥ $50) | Pregame is not what the model was trained on; thin books make the realistic PnL noise dominate signal. |
| PnL accounting | Both `pnl_mid` (alpha capture) and `pnl_realistic` (after spread + fees) per trade | Mid tells you if the *signal* worked; realistic tells you if you can *trade* it. |
| Bankroll | $1,000 paper | Round number; Kelly sizing scales linearly so absolute size doesn't matter for signal validation. |
| Drawdown gate | $100 = 10% of bankroll | Paper: log + continue. Live: hard halt on new entries. |
| A→B mode flip | 200 trades + per-decile bootstrapped 95% CI on `E[Δ\|rises]` ≤ ±1¢ | Automatic; gates the per-decile estimate against being a fluke. |
| Paper→live promotion | 1000 trades + Sharpe(per-trade) > 1 + max DD < 10% + manual config flag | Engine refuses to boot in live mode unless all four hold. |
| Architecture | Separate consumer process; reads from existing `realtime_feature_store.py` outputs | Decouples scoring from feed; engine restart never disturbs ingest. |
| Polling cadence | 15 seconds, aligned to feature-store writes | Matches feed cadence exactly; finer polling adds latency, not signal. |
| Source of truth | `trader_log.jsonl` (append-only event stream) | State files reconstructable by replay; survives crashes. |

## Module layout

### New module: `paper_trader.py` (entry point)

Long-running consumer process. Reads `data/live/features/latest_features.csv` every 15 seconds, scores eligible rows, manages open positions, writes outputs to `outputs/paper_trades/`.

```
python paper_trader.py --mode paper --bankroll 1000 --interval 15 [--config <path>]
```

Refuses to start if:
- Model artifact missing or unreadable
- `mode=live` and any of (1000 trades / Sharpe > 1 / max DD < 10% / live-flag) gates fail
- Output directory not writable

### New package: `paper_trader/` (8 components)

```
paper_trader/
    __init__.py
    feed_reader.py          # reads latest_features.csv, dedupes by (ticker, captured_at)
    gate_filters.py         # eligibility checks; returns (pass: bool, reason: str)
    scorer.py               # loads .pkl, returns p_raw + p_calibrated for a row
    move_size_estimator.py  # E[Δ|rises], E[Δ|doesn't] — pooled (A) or per-decile (B)
    trigger.py              # EV computation + accept/reject decision
    position_manager.py     # open/close lifecycle, time-based exit
    cost_model.py           # Kalshi fees + half-spread cost
    trade_log.py            # atomic state writes + append-only JSONL
```

Each file is pure-functional where possible (scorer, trigger, cost_model are stateless given inputs); state lives in `position_manager.py` and `trade_log.py` only.

### Output directory: `outputs/paper_trades/`

```
outputs/paper_trades/
    trader_state.json           # atomic, single source of truth
    open_positions.csv          # mirror of state.open_positions
    closed_trades.csv           # append-only, one row per closed trade
    trader_log.jsonl            # append-only event stream (the replay source)
    rollups/
        daily_<YYYY-MM-DD>.csv
        per_game.csv
        per_p_decile.csv
        per_minute_of_game.csv
```

### `trader_state.json` schema

```json
{
  "mode": "A",
  "live_or_paper": "paper",
  "bankroll_initial": 1000.0,
  "cumulative_pnl_realistic": 0.0,
  "cumulative_pnl_mid": 0.0,
  "trades_closed": 0,
  "max_drawdown": 0.0,
  "drawdown_breached": false,
  "open_positions": [
    {
      "trade_id": "<ticker>_<entry_captured_at_iso>",
      "ticker": "...",
      "game_key": "...",
      "entry_captured_at": "2026-04-28T19:30:00Z",
      "entry_yes_ask": 0.42,
      "entry_yes_mid": 0.40,
      "contracts": 12,
      "p_raw": 0.21,
      "p_calibrated": 0.18,
      "expected_pnl_per_contract": 0.013,
      "bankroll_at_entry": 1014.20
    }
  ],
  "last_processed_captured_at_per_ticker": {"<ticker>": "<iso>"},
  "last_tick_at": "2026-04-28T19:30:15Z"
}
```

Atomic writes: write to `trader_state.json.tmp` → `os.replace()` → `trader_state.json`. The engine reads this once at startup and holds it in memory; the on-disk copy is for restart and external observers.

### `closed_trades.csv` schema

```
trade_id, ticker, game_key, home_team, away_team,
entry_captured_at, exit_captured_at, hold_seconds,
entry_yes_ask, entry_yes_mid, exit_yes_bid, exit_yes_mid,
contracts,
p_raw, p_calibrated,
trigger_mode,                   # "A" or "B"
expected_pnl_per_contract,      # what trigger.py thought at entry
realized_label_home_up_5m,      # 0 or 1
pnl_mid, pnl_realistic, fees,
exit_basis,                     # "5min_timer" | "halt_drawdown" | "halt_stale_feed"
bankroll_at_entry,
notes
```

`trade_id = f"{ticker}_{entry_captured_at.isoformat()}"` — deterministic, dedupable.

### `trader_log.jsonl` event types

```
tick           — every 15s poll, includes tick_to_decision_ms
gate_blocked   — eligibility check failed; includes reason + ticker
signal         — gates passed, model scored, trigger evaluated (accept or reject)
entry          — position opened
exit           — position closed (any reason)
mode_flip      — A→B sizing transition
halt           — engine refused to enter due to drawdown / stale feed
error          — caught exception, includes traceback
```

All events carry `event_type`, `ts_utc` (engine wall clock), `captured_at` (feed timestamp). Replaying the log from scratch must reproduce `closed_trades.csv` and `trader_state.json` exactly — this is a tested invariant (see §Testing 6b).

## Pipeline (one tick)

```
1. feed_reader.read_latest()
       → DataFrame of (ticker → latest row not yet processed)
       Dedup key: (ticker, captured_at) > last_processed_captured_at_per_ticker[ticker]

2. For each row:
       a. gate_filters.evaluate(row) — fast rejects (live? home? liquid? sane spread?)
          On reject: log "gate_blocked", continue.
       b. scorer.score(row) → (p_raw, p_calibrated)
       c. move_size_estimator.expected_moves(p_calibrated, mode)
          → (E[Δ|rises], E[Δ|doesn't])
       d. trigger.evaluate(p_calibrated, moves, row.yes_ask, row.yes_mid, contracts=1)
          → (accept: bool, expected_pnl_per_contract: float)
          On reject: log "signal" with accept=false, continue.
       e. position_manager.open(...)
          - check global concurrency cap
          - compute Kelly contracts: f = μ/σ²; cap at 5% of bankroll; floor at 1
          - write entry to log + state + open_positions.csv
          - log "entry"

3. position_manager.tick_open_positions(now_captured_at):
       For each open position with (now − entry).total_seconds() ≥ 300:
         - read latest yes_bid / yes_mid for that ticker from this tick's frame
         - compute pnl_mid, pnl_realistic via cost_model
         - append closed_trades.csv row, write log "exit", remove from state
         - update cumulative_pnl_*, trades_closed, max_drawdown

4. Drawdown gate:
       If cumulative_pnl_realistic ≤ −$100:
         - state.drawdown_breached = true
         - paper: log "halt" once, continue accepting entries (per spec)
         - live: log "halt", refuse all new entries; existing positions close normally

5. trader_state.last_tick_at = now_utc(); atomic-write state.

6. Sleep until next 15s boundary.
```

Stale-feed handling: if `now_utc − latest_row.captured_at > 5 min`, log warning. > 10 min: log halt + refuse new entries until feed recovers. Existing open positions still get their 5-min timer evaluated against `captured_at` of whatever next valid row arrives for that ticker.

Main loop wraps each tick in `try/except`: log `error`, sleep 5s, continue. State on disk is authoritative — the engine can crash arbitrarily and the next tick rebuilds in-memory state from `trader_state.json` + `closed_trades.csv`.

**Restart behavior:** on boot, the engine loads `trader_state.json` and adopts any `open_positions` as live. Their 5-min timers continue to be evaluated against `entry_captured_at`, so a restart that takes < 5 min from when a position was opened doesn't change its exit time. If the engine restarts after a position's 5-min mark has already passed, the position closes on the first tick of the new session using the next available `yes_bid` / `yes_mid` for its ticker.

## Risk controls

**Hard caps (always on):**

- Global concurrency: 20 open positions max. New entries past 20 are logged as `signal` with `accept=false, reason="concurrency_cap"`.
- Per-ticker concurrency: unlimited (stack freely).
- Liquidity floor: `open_interest ≥ 100` AND `yes_depth_notional_3 ≥ 50`.
- Side: home-YES only. `status_state == "in"` only.
- Sanity: `1 ≤ yes_bid ≤ yes_ask ≤ 99` and `(yes_ask − yes_bid) ≤ 30`. Reject otherwise.

**Soft drawdown gate:**

Threshold: cumulative realistic PnL ≤ −$100.
- Paper mode: log `halt` event (once per breach), set `drawdown_breached=true`, continue trading. Useful diagnostic without invalidating the run.
- Live mode (future): refuse all new entries; open positions close on their normal 5-min timers; manual restart required to re-enable.

**Mode transitions:**

- A→B sizing (automatic):
  - `trades_closed ≥ 200` AND
  - For every populated p-decile, bootstrapped 95% CI half-width on `E[Δ|rises]` ≤ ±1¢.
  - Computed at end of each tick after a closed trade. Logs `mode_flip` event when triggered.
- Paper→live (manual):
  - `trades_closed ≥ 1000` AND
  - Sharpe(per-trade, realistic) > 1, defined as `mean(pnl_realistic) / std(pnl_realistic)` over all closed trades — no annualization, no risk-free rate AND
  - max drawdown over full run < 10% of `bankroll_initial` AND
  - `--mode live` flag set explicitly at startup.
  - Engine performs the gate check during boot; refuses to start if any gate fails, prints which.

**Engine failsafes:**

- Refuse to start if model artifact load fails.
- Stale feed: warn at 5 min behind, halt new entries at 10 min, recover automatically when fresh data resumes.
- Always use `captured_at` (UTC) for trade timing — never engine wall clock — so a late-arriving feed doesn't expire positions early.
- Main loop: `try/except` around every tick; logged exceptions never kill the process.

**Deliberate non-controls (v1):**

- No take-profit / stop-loss inside the 5-min window.
- No daily loss limit beyond the −$100 drawdown gate.
- No per-game cap.
- No rate throttle (15s cadence is the natural throttle).

## Observability

**Live tail CLI — `tools/trader_status.py`** (read-only)

Reads `trader_state.json` + tails `trader_log.jsonl` + reads `closed_trades.csv`. Prints:

```
Mode             : A (sizing)  paper
Bankroll         : $1,000.00
Cumulative PnL   : realistic $+47.31  |  mid $+62.18  |  alpha capture $+14.87
Trades closed    : 312   wins 167 (53.5%)   sharpe(per-trade) 0.81
Open positions   : 4 / 20
Last tick        : 2026-04-28 19:42:17 UTC  (12s ago)
Drawdown         : current −$8.20   max −$31.40 (3.1%)
Recent 5 trades  : <table>
```

Designed to be safe under `watch -n 30`. Pure reads, no locks.

**Rollups — `tools/trader_rollups.py`** (idempotent batch job)

Recomputes per-day, per-game, per-p-decile, and per-minute-of-game tables from `closed_trades.csv`. Run nightly or on demand. Writes to `outputs/paper_trades/rollups/`. The `per_p_decile.csv` table is what the A→B mode flip checks against; the tool also prints whether each decile passes the ±1¢ CI gate.

**In-line diagnostics (columns on `closed_trades.csv`):**

- `expected_pnl_per_contract` — what `trigger.py` thought the trade was worth at entry. Enables `realized − expected` regression.
- `pnl_mid − pnl_realistic` per trade → alpha capture per trade.
- `bankroll_at_entry` — frozen at trade time, so PnL% reconstructs without recomputing the running balance.

**Log query patterns** (jq one-liners):

- Why didn't I enter? `jq 'select(.event_type=="gate_blocked" and .ticker=="X")' trader_log.jsonl`
- Tick latency distribution: `jq 'select(.event_type=="tick") | .tick_to_decision_ms' | awk-stats`
- Errors: `jq 'select(.event_type=="error")'`

**Health checks** (advisory, surfaced in `trader_status.py`):

- Last tick > 60s ago → red.
- Last successful model score > 90s ago → red.
- `len(state.open_positions) != row count in open_positions.csv` → red.
- `trader_state.json` mtime > 5 min while engine pid alive → red (engine hung).

The engine self-halts on stale feed; the CLI just makes it visible.

**Deliberate non-features:**

- No web dashboard.
- No real-time alerts / push notifications.
- No DB.

## Testing

**Unit tests — `tests/paper_trader/`**

- `test_scorer.py` — load `.pkl`, build a synthetic 1-row DataFrame with each feature column, assert `predict_proba` returns finite, assert calibration shrinkage matches `base_rate + alpha × (raw − base_rate)` to 1e−9.
- `test_gate_filters.py` — table-driven; one row per gate, asserts the *reason* string in addition to the boolean.
- `test_cost_model.py` — fee math against three known Kalshi scenarios (5¢, 50¢, 95¢ entry); spread-cross cost = `yes_ask − yes_mid`; loser fee = 0.
- `test_position_manager.py` — open → tick forward 5 min synthetically → assert exit triggers, `pnl_mid` and `pnl_realistic` differ correctly, one row appended to `closed_trades.csv`, state decremented.
- `test_trade_log.py` — atomic writes: 2 processes calling `record_trade()` simultaneously, assert no torn writes, line count == 2.
- `test_state_atomicity.py` — kill mid-write (raise inside the temp-file path), assert `trader_state.json` still parses as the pre-write version.

**Property / invariant tests — `tests/paper_trader/test_invariants.py`**

- For any sequence of (entry, exit) events, `cumulative_pnl_realistic == sum(closed_trades.pnl_realistic)` to 1e−6.
- For any tick, `len(open_positions) ≤ 20`.
- Over 100k synthetic trades, zero `trade_id` collisions.
- **Replay invariant:** replaying `trader_log.jsonl` reconstructs `closed_trades.csv` and `trader_state.json` byte-identically. This is the disaster-recovery story.

**Backtest harness — `tools/trader_backtest.py`**

Replays one historical day's `live_features_<date>.csv` through the full pipeline (feed_reader → gates → scorer → trigger → position_manager → cost_model → trade_log) at simulated wall-clock or as-fast-as-possible. Outputs to `outputs/paper_trades/backtest_<date>/` mirroring the live structure. Same code paths as live, different feed source.

Used for:
- Pre-merge sanity diff: `git stash` proven-good run, change code, re-run, diff `closed_trades.csv`.
- Estimating trades/day before the live paper run starts.
- Approximate trade-count read on the existing 9 days of recorded data.

**Smoke test — `tools/trader_smoke.py`**

10-minute live run with `mode=paper`, `max_positions=1`, and trigger forced to reject (`expected_pnl_per_contract=0`). Validates: feed connects, model loads, gates evaluate, log file rotates, state file writable. Run before every mode flip.

**Acceptance criteria**

Before paper goes long-running:
- All unit + invariant tests green.
- Backtest on at least 3 days runs to completion without exception.
- 60-min live smoke emits expected `tick` events and zero `error` events.

Before paper→live flip (separate from the trade-count gates):
- Smoke test in `mode=live --dry-run` (intercepts at the order placement boundary, logs intended orders only).
- Manual eyeball of one full simulated entry+exit: size sensible, fees right, exit timing right.
- Then `--dry-run=false`.

**Deliberate non-tests:**

- No mock of Kalshi's API for the live path. The smoke test is the integration test.
- No CI. Local `pytest` is enough at one developer.
- No coverage targets. The list above is the spec.

## Open questions deferred to implementation

- Exact exit-pricing rule when the 5-min mark falls *between* feed ticks. Simplest: use the next tick's `yes_bid` / `yes_mid` for that ticker. To resolve in the implementation plan.
- Whether `move_size_estimator.py` Phase A pooled means come from the existing labeled training data or get re-estimated from `closed_trades.csv` once enough exist. Likely both: bootstrap from training, then update from realized once `trades_closed ≥ 200`.
- File rotation policy for `trader_log.jsonl` if it grows past, say, 500MB. Likely daily rotation but defer until the actual size matters.

## Acceptance summary

The design is approved when this document accurately captures every brainstorm decision and reads as a self-contained handoff to whoever (David or a fresh agent) writes the implementation plan next. The `superpowers:writing-plans` skill is the next step on user approval.

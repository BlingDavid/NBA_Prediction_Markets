# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NBA betting model with two layers:

1. **Pre-game model** — `models/nba_predictor.pkl` (XGBoost + LightGBM + LogReg ensemble with isotonic calibration). Outputs `home_win_prob`, `predicted_spread`, `predicted_total`. Built/used by `model.py` (untracked); orchestrated through `run_pipeline.py`.
2. **Live in-game model** — `models/live_home_up_5m_bootstrap.pkl` (sklearn pipeline; predicts P(home YES contract rises in next 5 min)). Trained by `live_bootstrap_model.py`, consumed by the paper trader.

These feed two trading workflows: pre-game edge bets (sportsbook + Kalshi single-game) and 5-min microstructure scalps (Kalshi only, while live).

## Critical setup gotchas

### Untracked-but-load-bearing root files

These root `.py` files are present in the working tree but **not tracked in git**:

```
backtester.py  bayesian_update.py  calibration_monitor.py  data_ingest.py
features.py    kalshi_auth.py      late_lineups.py         live_adjustments.py
model.py       player_usage.py     prediction_utils.py     referee_bias.py
strength_of_schedule.py            venue_edge.py
```

Tracked modules (`realtime_feature_store.py`, `ev_analyzer.py`, `order_executor.py`, etc.) import from several of these. On a fresh clone the imports will fail. In an isolated worktree, mirror them in (symlinks + `.git/info/exclude`).

Do not commit these unless explicitly asked. Intentionally untracked by the maintainer.

### xgboost / libomp dylib path

`xgboost` in this venv resolves `libomp.dylib` against `/opt/homebrew/opt/libomp/lib/`. If brew's libomp isn't installed there, every script that imports `model.py` (`import xgboost`) crashes at startup with `Library not loaded: @rpath/libomp.dylib`. Workaround: export `DYLD_FALLBACK_LIBRARY_PATH` to any directory that contains `libomp.dylib` (e.g. an anaconda `lib` dir). Affects `run_pipeline.py`, `realtime_feature_store.py`, `ev_analyzer.py`. The paper trader itself is sklearn-only, but inherit the env var for consistency. Permanent fix: `brew install libomp`.

## Entry points

```bash
# Pre-game pipeline
python run_pipeline.py --predict --min-edge 0.05 --bankroll 2000
python run_pipeline.py --scan         # markets + matching only
python run_pipeline.py --backtest     # walk-forward
python run_pipeline.py --no-shrink    # disable extreme-edge shrinkage (raw output)

# In-game paper trader (15s polling against latest_features.csv)
python run_paper_trader.py --once     # single tick (smoke)
python run_paper_trader.py --mode paper --bankroll 1000 --interval 15
python run_paper_trader.py --mode live --once   # gated: 1000 trades + Sharpe>1 + DD<10%

# Single-ticker EV (against The Odds API)
python ev_analyzer.py KXNBAGAME-26MAY05LALOKC-LAL
python ev_analyzer.py --scan-all --series KXNBAGAME

# Live capture + observability
python realtime_feature_store.py --series KXNBAGAME --loop --interval 15
python tools/trader_status.py                      # safe under `watch -n 30`
python tools/trader_rollups.py                     # daily / per-game / per-decile / per-minute
python tools/trader_smoke.py --minutes 10          # wiring check
python tools/trader_backtest.py --features-path data/live/features/live_features_<date>.csv \
                                --output-dir outputs/paper_trades/backtest_<date>
python tools/build_pooled_means.py                 # refresh Phase A seed after retraining
```

## Tests

```bash
python -m pytest tests/                               # full suite (~233 tests)
python -m pytest tests/paper_trader/ -q               # paper-trader only
python -m pytest tests/test_edge_detector.py -v       # one file
python -m pytest tests/test_market_scanner.py::test_match_kxnbagame_uses_ticker_suffix_not_title_order -v
```

`pyproject.toml` sets `pythonpath = ["."]` and `testpaths = ["tests"]`. Tests of untracked modules guard with `pytest.importorskip(...)`.

## Architecture pointers

### Pre-game layer (see `run_pipeline.py`, `edge_detector.py`, `market_scanner.py`)

Flow: `data_ingest.fetch_current_odds()` → `data/processed/current_odds.csv` → `model.predict()` → `outputs/predictions.csv` → `edge_detector.find_edges()` → `outputs/bet_card.csv` → `market_scanner.scan_all_markets()` → `outputs/market_edges.csv`.

Three edge branches in `edge_detector.find_edges`:
- **Moneyline** — `american_to_implied_prob` + `implied_prob_to_fair` (no-vig).
- **Spread** — `_spread_cover_prob = norm.cdf((model_spread + line)/std)`. The `+ line` form is correct given the "negative-for-home-favorite" convention.
- **Total** — `_total_over_prob` against `total_fair = 0.5`.

`_shrink_edge` runs before sizing/EV: identity for `|raw_edge| ≤ 0.10`, linear shrink to factor 0.5 by `|raw_edge| ≥ 0.30`. Bet card surfaces both `model_prob/edge` (shrunk) and `model_prob_raw/edge_raw`.

`market_scanner.NBA_SERIES_TICKERS` **must** include `KXNBAGAME` (per-game H2H series). Match by ticker suffix via `_kxnbagame_team_pair` and `_kxnbagame_yes_team` — title text is unreliable (both sides share one truncated string).

### In-game layer (see `run_paper_trader.py`, `paper_trader/`)

Authoritative docs:
- `docs/superpowers/specs/2026-04-28-edge-to-action-design.md` — design with every settled decision.
- `docs/superpowers/plans/2026-05-01-edge-to-action-paper-trader.md` — TDD plan that built it.
- `paper_trader/README.md` — pointer to spec + plan + entry point.

Pipeline per 15s tick (in `run_paper_trader.process_tick`):

```
read_unprocessed (data/live/features/latest_features.csv)
  → first pass: collect quote_table + latest_capture from every row
  → detect stale (warn @ 5min, halt @ 10min)
  → second pass (skipped when stale-halt): gate_filters → scorer → trigger → kelly_contracts → open_position
  → tick_open_positions (always — exits run even on stale-halt, per spec)
  → drawdown gate ($100 cap; paper logs once + continues, live halts new entries)
  → write_state (atomic JSON) + emit tick event
```

Key invariants (enforced by tests in `tests/paper_trader/`):
- Append-only `trader_log.jsonl` replays byte-identically into `closed_trades.csv` via `replay_from_log`. Load-bearing: `tests/paper_trader/test_invariants.py::test_replay_reconstructs_closed_trades`.
- `MAX_OPEN = 20` global; no per-ticker cap.
- `kelly_contracts(mu, var, bankroll, yes_ask)`: `f = max(0, min(mu/var, 0.05))`, `dollars = f * bankroll`, `int(dollars / yes_ask)`, floor at 1. Use `int(dollars / yes_ask)` — `dollars // yes_ask` is float-imprecise (e.g. `50.0 // 0.4 = 124.0` vs `int(50.0/0.4) = 125`).
- `paper_trader/scorer.engineer_features` imports `_safe_numeric` + `_status_flag` from `live_training_matrix` — training and live feature engineering must not drift.

Live boot gates in `paper_trader/feed_reader.check_live_boot_gates`: `trades_closed ≥ 1000`, `mean(pnl_realistic)/std(pnl_realistic) > 1`, `max_dd < 10% bankroll_initial`. Exit code 2 on fail.

**sklearn pickle/runtime skew is hard-blocking, despite looking like a warning.** `live_home_up_5m_bootstrap.pkl` is a `SimpleImputer + StandardScaler + LogisticRegression` pipeline. When the venv's sklearn drifts past the version that pickled it, you'll see `InconsistentVersionWarning` at load *and* `AttributeError("'SimpleImputer' object has no attribute '_fill_dtype'")` at every `predict()` call. The trader catches these as `event_type=error` events and keeps ticking — the loop looks healthy but no row ever reaches `open_position`. Retrain via `python live_bootstrap_model.py`. **Diagnostic shortcut for "trader running, no trades":** check `grep -c '"event_type": "error"' outputs/paper_trades/trader_log.jsonl` *before* reading gate-block tallies — gate counts alone are misleading because most rows are legitimately rejected at `missing_field:status_state` (pre-game, post-game, future-series tickers) and that volume can mask scorer crashes.

**Phase B (per-decile expected-move) is seeded from historical at boot — `--mode-sizing` defaults to `B`.** `outputs/paper_trades/pooled_means.json` carries both pooled means *and* a 10-row `decile_table` built from `outputs/live_home_up_5m_bootstrap_scored_rows.csv`. Without this seed, the trigger sees a single pooled `E_rises ≈ +0.024` for every signal, which is structurally negative-EV against Kalshi's per-contract winner-fee ceiling (`ceil(0.07·p·(1−p)·100)/100`) — the system never trades, never builds the 200 closed-trades buffer needed to refresh Phase B from live data, and stays deadlocked. Per-decile means show E_rises grows from +0.012 (decile 0) to +0.067 (decile 7), which makes positive EV reachable in upper deciles. **Always re-run `python tools/build_pooled_means.py` after retraining the live model** — labels in scored_rows.csv are not model outputs, but the decile boundaries depend on `pred_bootstrap_home_up_5m`, which does change with retraining.

## Data conventions

- **Kalshi ticker**: `KXNBAGAME-YYMMMDDAWAYHOME-TEAM`. Middle 13 chars = `YY` + `MMM` + `DD` + `AWAY` (3 letters) + `HOME` (3 letters). Trailing `-TEAM` = YES side. **Same playoff series gets different tickers across dates** because home court alternates (e.g. `26MAY01DETORL` vs `26MAY03ORLDET`). Do not assume tickers within a series share home/away assignment.
- **ESPN `game_key`**: `YYYY-MM-DD_<away>_<home>` — same `<away>_<home>` order as the Kalshi ticker's middle segment.
- **Spread convention**: `spread_home` is negative when home is favored (`-4.6` = "home -4.6"). Cover prob uses `+ line` not `- line`.
- **Shrinkage**: ON by default in pre-game pipeline. `--no-shrink` for raw output (use for backtesting).

## Where state lives

| Path | Contents |
|---|---|
| `outputs/predictions.csv`, `outputs/bet_card.csv`, `outputs/all_markets.csv`, `outputs/market_edges.csv` | Pre-game artifacts |
| `outputs/watchlists/<date>_*.{csv,md}` | Pre-game briefings + BUY-YES candidate watchlists |
| `outputs/paper_trades/trader_log.jsonl` | **Source of truth** for in-game state (append-only) |
| `outputs/paper_trades/{trader_state.json, closed_trades.csv, open_positions.csv, pooled_means.json, rollups/}` | Reconstructable from `trader_log.jsonl` |
| `data/live/features/latest_features.csv` | Most recent live snapshot (one row per Kalshi market) |
| `data/live/features/live_features_<date>.csv` | Daily history |
| `data/live/{games,markets,events,labels,state}/` | Per-day captures |
| `models/*.pkl` | Trained models (gitignored) |

`data/`, `models/*.pkl`, `outputs/` are all gitignored.

## Reference files

- `docs/superpowers/specs/2026-04-28-edge-to-action-design.md` — paper-trader design.
- `docs/superpowers/plans/2026-05-01-edge-to-action-paper-trader.md` — TDD plan.
- `IMPROVEMENTS.md` — historical phase-by-phase improvements; explains why the model is structured this way.
- `paper_trader/README.md` — paper-trader entry pointer.

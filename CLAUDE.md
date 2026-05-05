# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NBA betting model with two layers:

1. **Pre-game model** (`models/nba_predictor.pkl`): ensemble of XGBoost + LightGBM + Logistic with isotonic calibration. Outputs `home_win_prob`, `predicted_spread`, `predicted_total` per upcoming game. Built/used by `model.py` (untracked) and reported through `run_pipeline.py`.

2. **Live in-game model** (`models/live_home_up_5m_bootstrap.pkl`): bootstrap-trained sklearn pipeline that predicts P(home YES contract rises in next 5 min) given current game state + Kalshi market features. Trained by `live_bootstrap_model.py`. Consumed by the paper trader.

The two layers feed two distinct trading workflows: pre-game edge bets (sportsbook + Kalshi single-game contracts) and 5-minute microstructure scalps (Kalshi only, while a game is live).

## Critical setup gotcha: untracked-but-load-bearing root files

A handful of root `.py` files are present in the working tree but **not tracked in git**:

```
backtester.py  bayesian_update.py  calibration_monitor.py  data_ingest.py
features.py    kalshi_auth.py      late_lineups.py         live_adjustments.py
model.py       player_usage.py     prediction_utils.py     referee_bias.py
strength_of_schedule.py            venue_edge.py
```

Tracked modules (`realtime_feature_store.py`, `ev_analyzer.py`, `order_executor.py`, etc.) import from several of these. On a fresh clone the imports will fail. When working in an isolated worktree, mirror these files in (the previous edge-to-action worktree did this via symlinks plus `.git/info/exclude`).

Do not commit these files unless the user explicitly asks. They've been intentionally left out of version control by the maintainer.

## Three entry points

```bash
# Pre-game pipeline: ingest odds → predict → bet card → market scan
python run_pipeline.py --predict --min-edge 0.05 --bankroll 2000
python run_pipeline.py --scan         # markets + matching only
python run_pipeline.py --backtest     # walk-forward
python run_pipeline.py --no-shrink    # disable extreme-edge shrinkage (raw model output)

# In-game paper trader (15s polling against latest_features.csv)
python run_paper_trader.py --once     # single tick (smoke)
python run_paper_trader.py --mode paper --bankroll 1000 --interval 15
python run_paper_trader.py --mode live --once  # refuses unless 1000 trades + Sharpe>1 + DD<10% gates pass

# Single-ticker EV analysis (against The Odds API)
python ev_analyzer.py KXNBAGAME-26MAY05LALOKC-LAL
python ev_analyzer.py --scan-all --series KXNBAGAME
```

Live data and observability:

```bash
# Single capture into data/live/features/latest_features.csv
python realtime_feature_store.py --series KXNBAGAME
python realtime_feature_store.py --series KXNBAGAME --loop --interval 15

# Read-only paper-trader status (safe under `watch -n 30`)
python tools/trader_status.py
python tools/trader_rollups.py        # daily / per-game / per-decile / per-minute tables
python tools/trader_smoke.py --minutes 10   # wiring check (forces every trigger to reject)
python tools/trader_backtest.py --features-path data/live/features/live_features_2026-05-02.csv \
                                --output-dir outputs/paper_trades/backtest_2026-05-02

# Refresh the Phase A pooled-means seed after retraining the in-game model
python tools/build_pooled_means.py
```

## Tests

```bash
python -m pytest tests/                            # full suite (~230 tests)
python -m pytest tests/test_edge_detector.py -v    # one file
python -m pytest tests/paper_trader/ -q            # paper-trader subsystem only
python -m pytest tests/test_market_scanner.py::test_match_kxnbagame_uses_ticker_suffix_not_title_order -v   # one test
```

`pyproject.toml` configures `pythonpath = ["."]` and `testpaths = ["tests"]`.

Tests of untracked modules (`tests/test_edge_detector.py`, `tests/test_market_scanner.py`) start with `pytest.importorskip(...)` so they cleanly skip if the source isn't present.

## Architecture: how the two trading layers fit together

### Pre-game layer (run_pipeline.py)

1. `data_ingest.fetch_current_odds()` (untracked) → The Odds API → `data/processed/current_odds.csv`. Stores `home_team`, `away_team` (using actual venue convention), `ml_home`/`ml_away`, `spread_home`/`spread_home_price`, `total_line`/`total_over_price`. Only the home side's spread/total prices are kept; the away side is assumed symmetric.
2. `model.predict()` (untracked) → `outputs/predictions.csv` with `home_win_prob`, `predicted_spread`, `predicted_total`.
3. `edge_detector.find_edges(predictions, odds, apply_shrinkage=True)` → `outputs/bet_card.csv`. Three branches:
   - **Moneyline**: `american_to_implied_prob` + `implied_prob_to_fair` for both sides → fair (no-vig) prob → edge.
   - **Spread**: `_spread_cover_prob(model_spread, line)` = `norm.cdf((model_spread + line)/std)`. The `+ line` form is correct: spreads are stored in "negative-for-home-favorite" convention so home covers iff actual margin > -line. Compares to `fair_prob = 0.5` and uses the actual `spread_home_price` for `implied_prob`.
   - **Total**: same shape, `_total_over_prob` against `total_fair = 0.5`.
4. `edge_detector._shrink_edge(model_prob, fair_prob)` is applied before sizing/EV: identity for `|raw_edge| ≤ 0.10`, linear shrinkage to factor 0.5 by `|raw_edge| ≥ 0.30`. The bet card surfaces both `model_prob`/`edge` (shrunk) and `model_prob_raw`/`edge_raw` (raw) so the reader can see where shrinkage kicks in. Backtest analysis showed the model is well-calibrated within ±0.10 edge but has per-team biases (BKN 0/13 historically) and playoff distribution shift beyond that.
5. `market_scanner.scan_all_markets(predictions)` pulls every series in `NBA_SERIES_TICKERS` (which **must** include `KXNBAGAME` — the per-game H2H series; futures-only sets like `["KXNBA", "KXNBAFINALS"]` would skip every single-game contract). The matcher uses `_kxnbagame_team_pair(ticker)` and `_kxnbagame_yes_team(ticker, ...)` to extract `(away, home)` and the YES side from the ticker — the title is unreliable because both sides share one truncated string ("Los Angeles L at Oklahoma City Winner?").

### In-game layer (run_paper_trader.py + paper_trader/)

The paper trader is documented in detail under `docs/superpowers/specs/2026-04-28-edge-to-action-design.md` and `docs/superpowers/plans/2026-05-01-edge-to-action-paper-trader.md`. Read those before making structural changes.

Pipeline per 15-second tick:

```
read_unprocessed (latest_features.csv)
  → first pass: collect quote_table + latest_capture from every row
  → detect stale (warn @ 5min, halt @ 10min)
  → second pass (only when not stale-halt): gate_filters.evaluate → scorer.score
                                              → trigger.evaluate (EV-aware) → kelly_contracts → open_position
  → tick_open_positions (always — exits run even on stale-halt, per spec)
  → drawdown gate ($100 cap, paper logs once + continues; live halts new entries)
  → write_state (atomic JSON) + emit tick event
```

Key invariants enforced by tests:

- **Append-only `trader_log.jsonl` replays byte-identically** into `closed_trades.csv` via `replay_from_log` (disaster-recovery story; `tests/paper_trader/test_invariants.py::test_replay_reconstructs_closed_trades` is the load-bearing test).
- `MAX_OPEN = 20` global concurrency cap, no per-ticker cap.
- `kelly_contracts(mu, var, bankroll, yes_ask)`: `f = max(0, min(mu/var, 0.05))`, `dollars = f * bankroll`, `int(dollars / yes_ask)`, floor at 1. Use `int(dollars / yes_ask)` not `dollars // yes_ask` — float floor-division is imprecise (e.g. `50 // 0.4 = 124.0`, `int(50/0.4) = 125`).
- `scorer.engineer_features` imports `_safe_numeric` and `_status_flag` from `live_training_matrix` rather than reimplementing them. This is intentional: training-time and live-time feature engineering must not drift.
- The shrinkage formula `(model_spread + line) / std` is derived from the docstring "Home covers if actual_spread > -line" — the previous `(model_spread - line)` form was a sign bug that produced 91% cover probabilities for moderate predicted leads.

Live mode boot gates (`check_live_boot_gates`): trades_closed ≥ 1000, `mean(pnl_realistic)/std(pnl_realistic) > 1`, max_dd < 10% of `bankroll_initial`. Engine exits with code 2 if any fail.

## Data conventions to remember

- **Kalshi ticker format**: `KXNBAGAME-YYMMMDDAWAYHOME-TEAM`. Middle 13 chars are `YY` + `MMM` + `DD` + `AWAY` (3 letters) + `HOME` (3 letters). Trailing `-TEAM` is the YES side. The same game on different dates of a playoff series gets two different tickers (e.g. `26MAY01DETORL` for DET visiting ORL, `26MAY03ORLDET` for ORL visiting DET) because home court alternates. **Do not assume tickers across a series share a home/away assignment.**
- **ESPN `game_key`**: `YYYY-MM-DD_<away>_<home>`. Same `<away>_<home>` order as the Kalshi ticker's middle segment, so the join in `realtime_feature_store` works on the parsed game_key directly.
- **Spread convention**: `spread_home` is negative when home is favored (e.g. -4.6 means "BOS -4.6"). Cover probability uses `+ line` not `- line`.
- **Shrinkage convention**: by default ON in pre-game pipeline. Pass `--no-shrink` to recover raw output for backtesting.

## Where state lives

- `outputs/predictions.csv`, `outputs/bet_card.csv`, `outputs/all_markets.csv`, `outputs/market_edges.csv` — pre-game artifacts.
- `outputs/paper_trades/{trader_state.json, trader_log.jsonl, closed_trades.csv, open_positions.csv, pooled_means.json, rollups/}` — in-game state. `trader_log.jsonl` is the source of truth; everything else is reconstructable from it.
- `data/live/features/latest_features.csv` — most recent live snapshot (one row per Kalshi market). Daily history at `data/live/features/live_features_<date>.csv`.
- `data/live/games/`, `data/live/markets/`, `data/live/events/`, `data/live/labels/`, `data/live/state/` — per-day captures.
- All of `data/`, `models/*.pkl`, and `outputs/` are gitignored.

## Specs and plans

- `docs/superpowers/specs/2026-04-28-edge-to-action-design.md` — paper-trader design, with every settled decision documented.
- `docs/superpowers/plans/2026-05-01-edge-to-action-paper-trader.md` — TDD task plan that built the paper trader.
- `IMPROVEMENTS.md` — historical phase-by-phase improvements, useful for understanding why the model is structured the way it is.
- `paper_trader/README.md` — short pointer to the spec + plan + entry point.

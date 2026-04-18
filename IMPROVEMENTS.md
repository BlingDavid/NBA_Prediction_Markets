# NBA Quant Model Improvements

## Core Pipeline (Completed Phase 1-2)
- ✅ **ELO Rating System** — Margin-of-victory multiplier, home advantage, season reversion
- ✅ **Ensemble ML Classifier** — XGBoost + LightGBM + Logistic Regression with isotonic calibration
- ✅ **Feature Engineering** — Derives ELO, rest, pace, bench depth, player availability
- ✅ **Walk-Forward Backtesting** — No look-ahead bias, walk-forward validation by season
- ✅ **Expected Value Calculation** — Kelly-based edge detection for binary Kalshi contracts

## Live Data Integration (Completed Phase 4)
- ✅ **Live Game Context** — Team stats, player stats, H2H history, real-time injury reports
- ✅ **Multi-Source Odds Comparison** — Kalshi, ESPN moneylines, multi-sportsbook spreads & O/U
- ✅ **Injury Impact Modeling** — Star player adjustments, back-to-back penalties, rest bonuses
- ✅ **Live Order Execution** — Kalshi authenticated API with RSA-PSS signing, manual placement with confirmation

## Backtester v2 Enhancements (Completed Phase 5)

### #13: Historical Odds Distribution ✅
**Previously:** Pure synthetic noise (uniform random)
**Now:** Realistic market behavior with:
- Variable noise (tight for heavy favorites, wide for close games)
- Closing line drift toward actual outcome (simulates sharp money/late info)
- Vigorish (vig) structure by sportsbook
- Bid-ask spreads (Kalshi 2%)

**Impact:** Orders generated from distribution matching real market properties, not artificial noise

---

### #14: Dynamic Bankroll-Adaptive Kelly Sizing ✅
**Previously:** Static Kelly fraction (e.g., always 25% of Kelly)
**Now:** Adaptive scaling based on bankroll state:
- **Drawdown scaling** — Reduces sizing by ~19% at 20% DD, min 25% at 50% DD (protects capital)
- **Profit boosting** — Increases sizing 1.0x to 1.3x when up 30%+ (lets winners compound)
- **Confidence scaling** — Larger edges → bet closer to full Kelly fraction (3% edge → 1.0x, 10%+ edge → 1.5x)

**Impact:** Prevents ruin during cold streaks, compounds during hot ones, scales with edge confidence

---

### #16: Closing Line Value (CLV) Based Filtering ✅
**Previously:** Bet any edge > threshold, regardless of betting history
**Now:** Filter bets by CLV expectation:
- **Warmup phase** — Allow all 50 bets to build historical CLV distribution
- **Active filtering** — Only bet if expected CLV ≥ 0.5¢ threshold
- **Self-assessment** — Track historical CLV rate; adjust filter threshold based on actual performance
- **Edge-weighted** — Larger edges expected to have higher CLV; estimate accordingly

**Impact:** Only take bets where model historically beats closing lines (strongest indicator of edge)

---

## Potential Future Improvements (Not Yet Implemented)

### #1: Opponent-Specific Strength of Schedule ✅
Adjust preseason ELO based on upcoming opponent difficulty (next 10/20 games)
- `strength_of_schedule.py`: rolling 15-game SoS (avg opponent net rating)
- Publishes `home_sos`, `away_sos`, `diff_sos`, `diff_sos_adj_net` features
- Cached at `data/processed/sos_cache.parquet`; refresh via `python strength_of_schedule.py --recompute`
- Strictly trailing windows (no leakage) — safe for walk-forward backtesting
- NOTE: needs the model to be retrained to actually consume these new features

### #2: Player Usage Rate Weighting ✅
Weight recent player minutes in injury scenarios (who replaces the injured star?)
- `player_usage.py`: Fetches USG%, minutes share, offensive load, net rating per player
- Replacement model: identifies who absorbs minutes, calculates production gap
- Team concentration features (HHI, star dependency, depth) added to ML model
- `live_adjustments.py` now uses data-driven impact instead of static STAR_PLAYERS dict
- EV analyzer shows full replacement analysis for injured star players

### #3: Venue-Specific Edge Modeling ✅
Some teams perform better at home/away vs model prediction (altitude, crowd, travel)
- `venue_edge.py`: fits per-team (team × venue) residuals from historical predictions
- James-Stein shrinkage toward zero (K=30) so small samples don't blow up
- Hard cap at ±4% per team-venue
- Applied at prediction time via `apply_venue_edge(home, away, base_prob)`
- Wired into `ev_analyzer.py` via `--extra-signals`
- Fit with: `python venue_edge.py --fit --predictions outputs/historical_predictions.csv`

### #4: Correlation-Aware Kelly Adjustments
Track correlation between consecutive bets; reduce sizing if bets correlated

### #5: Seasonal Momentum Cycles
Model hot/cold stretches beyond simple ELO; account for playoff positioning

### #6: Live Lineup Adjustments ✅
Integrate pre-game lineup confirmations (starting 5 changes, DNP situations)
- `late_lineups.py`: pulls ESPN team-injury JSON feed for both sides of a matchup
- Caches the morning injury snapshot to `outputs/cached/injuries_YYYYMMDD.csv`
- Diffs current vs. snapshot to classify: new scratches, escalations (Q→OUT), clearances
- Inside 4-hour tip window, late scratches get a 1.15x impact multiplier (markets lag these moves)
- Cleared players recover ~50% of previously-applied injury penalty
- Wired into `ev_analyzer.py` via `--late-check` flag; works in `--scan-all` mode too
- CLI for one-off checks: `python late_lineups.py --diff LAL,BOS`

### #7: Coach/Rotation Strategy Learning
Track coach tendency to rest players in B2B or specific game situations

### #8: Arbitrage Detection ✅ (new)
Price-divergence signal that detects Kalshi ↔ sportsbook-consensus mispricings and uses model-triangulation as a tier-up filter
- `consensus_divergence.py`: pure functions for proportional de-vig, median consensus across books, tier assignment, and threshold adjustment
- Three callers: `ev_analyzer.py --extra-signals` (bet-decision gating), `realtime_feature_store.py` (live training-data capture), and a `--scan` / `--settle` CLI
- Shadow-mode rollout (`config.CONSENSUS_TIER_MODE = "shadow" | "active" | "off"`): logs everything from day one without changing live bet decisions until evaluated
- Metrics log at `outputs/consensus_divergence_metrics.csv` captures per-game tier, divergences, and (post-settlement) PnL
- Thresholds and tier multipliers all live in `config.py` (`CONSENSUS_MIN_BOOKS`, `CONSENSUS_STALE_SECONDS`, `CONSENSUS_DIVERGENCE_THRESHOLD_PP = 3.5`, `TIER_THRESHOLD_MULTIPLIERS = {2: 0.7, 1: 1.0, -1: 1.5}`)
- Smoke: `python consensus_divergence.py scan --dry-run`
- Settle: `python consensus_divergence.py settle --game-id ... --home-win true --pnl 123.45 --settled-at 2026-04-17T03:30:00Z`
- NOTE: Historical divergence reconstruction is intentionally out of scope for v1 (no historical multi-book odds source). Shadow-mode evaluation of tier-2 PnL delta is flagged as follow-up after ≥50 tier-2 bets accumulate.

### #9: Portfolio Risk Management
Diversify bets across multiple markets (not just moneyline); hedge with spreads/props

### #10: Adaptive Vig Model
Learn true vig by market; adjust Kelly for markets with higher overhead

### #11: Trade Deadline Shock Modeling
Sudden probability shifts when trades announced; detect and reprrice

### #12: Predictive Line Movement
Estimate where line will close before tip-off; adjust entry strategy accordingly

### #15: Monte Carlo Variance Clustering
Detect and model variance clustering in P&L; adjust position sizing dynamically

### #17: Bayesian Model Updating ✅
Update model weights intra-season based on recent prediction accuracy
- `bayesian_update.py`: tracks per-base-model (XGB, LightGBM, LR) probabilities + outcomes
- Posterior weights via softmax over -tau × log-loss on rolling window (default 100 games)
- Auto-refits every 25 new settled games; prior = uniform (1/3 each) until evidence accumulates
- `blend(base_probs, weights)` to combine at prediction time
- CLI: `python bayesian_update.py --report` / `--refit`

### #18: Cross-Sport Correlation Analysis
Model how NFL/NHL outcomes affect NBA (news cycle, player attention, betting volume)

### #19: Referee Crew Bias ✅ (new)
Officiating crew tendencies affect foul rates, pace, and home-court effects
- `referee_bias.py`: fetches official crew from NBA CDN (assignments posted ~2h pre-tip)
- Maintains tendency table (outputs/ref_tendencies.csv) seeded with public estimates for 16 high-profile refs
- Combines FTA diff, home-foul tilt, pace tilt into a moneyline shift (capped at ±1.5%)
- Wired into `ev_analyzer.py` via `--extra-signals`
- Seed/update with: `python referee_bias.py --seed`

### #20: Model Calibration Monitoring ✅ (new)
Detect and correct for mid-season model drift
- `calibration_monitor.py`: rolling Brier on last 50 settled predictions
- Reliability diagram (pred bucket vs. empirical rate) via `--plot`
- Adaptive edge threshold: inflates from 3% → 4% (Brier 1.10×) or 5% (Brier 1.25×)
- Protects against post-trade-deadline drift, rule-change regime shifts
- Every prediction auto-logged; record outcomes with `record_outcome(game_id, home_win)`
- Report: `python calibration_monitor.py --report`

### #21: Real-Time Feature Store + Event Logger ✅ (new)
Capture live NBA game state and Kalshi market microstructure for in-game model training
- `realtime_feature_store.py`: polls ESPN scoreboard + Kalshi markets/orderbooks
- Writes append-only snapshots to:
  - `data/live/games/game_states_YYYY-MM-DD.csv`
  - `data/live/markets/market_snapshots_YYYY-MM-DD.csv`
  - `data/live/features/live_features_YYYY-MM-DD.csv`
  - `data/live/events/live_events_YYYY-MM-DD.jsonl`
- Feature rows include:
  - live score / period / clock / elapsed time
  - Kalshi bid/ask + orderbook depth
  - sportsbook consensus where available
  - saved pregame model priors for the matchup
- Event logger emits discrete `score_change`, `period_change`, `price_change`, `liquidity_change`, and `volume_change` records
- Latest state persisted at `data/live/state/latest_state.json`
- One-shot capture: `python realtime_feature_store.py`
- Continuous polling: `python realtime_feature_store.py --loop --interval 15`
- Rebuild joined features from raw snapshots: `python realtime_feature_store.py --rebuild-history`

### #22: Live Label Builder ✅ (new)
Turn captured live features into supervised training targets for the first in-game model
- `live_labeler.py`: reads `data/live/features`, `data/live/games`, and `data/live/markets`
- Primary labels:
  - `label_final_home_win`
  - `label_yes_mid_move_5m`
  - `label_market_home_implied_move_5m`
  - `label_beats_close_yes`
  - `label_beats_close_home`
- Uses the last observed tradable market snapshot as the close-price proxy
- Saves labeled sets to:
  - `data/live/labels/labeled_features_h5m.csv`
  - `data/live/labels/latest_labeled_features.csv`
- Build labels: `python live_labeler.py`
- Keep only resolved games: `python live_labeler.py --resolved-only`

### #23: Bootstrap In-Game Model Trainer ✅ (new)
Train the first contract-group-aware live model while the true in-game dataset is still thin
- `live_bootstrap_model.py`: trains the initial in-game classifier on `label_home_up_5m`
- Uses grouped cross-validation by `ticker` to avoid leakage across repeated snapshots from the same contract
- Uses inverse-frequency sample weighting so one market does not dominate the fit
- Saves:
  - `models/live_home_up_5m_bootstrap.pkl`
  - `outputs/live_home_up_5m_bootstrap_report.json`
  - `outputs/live_home_up_5m_bootstrap_oof_predictions.csv`
  - `outputs/live_home_up_5m_bootstrap_scored_rows.csv`
- Train bootstrap model: `python live_bootstrap_model.py`
- Train only rows with matched game-state context: `python live_bootstrap_model.py --require-game-state`
- Train only true live rows once captured: `python live_bootstrap_model.py --live-only`
- Override target: `python live_bootstrap_model.py --target label_beats_close_home`

---

## Testing Checklist

- [ ] Run v1 baseline: `python backtester.py --no-historical-odds --no-dynamic-kelly --no-clv-filter`
- [ ] Run v2 full: `python backtester.py`
- [ ] Compare: ROI, Sharpe, max drawdown, CLV rate between v1 and v2
- [ ] Monte Carlo: Check P(profit), P(ruin), P(2x), P(5x)
- [ ] Health check: All 6 points passing (ROI, win rate, drawdown, Sharpe, CLV rate, avg CLV)
- [ ] Live test: Run `ev_analyzer.py` on a current game and verify output

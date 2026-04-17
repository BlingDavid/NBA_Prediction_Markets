# Arbitrage Detection — Price Divergence Signal

**Date:** 2026-04-16
**Status:** Design approved; awaiting implementation plan
**Tracks:** IMPROVEMENTS.md item #8
**Owner:** David Blau

## Goal

Detect when Kalshi YES prices diverge meaningfully from sportsbook consensus, and use that divergence as a soft-arbitrage *edge signal* (not risk-free arbitrage) that sharpens pre-game bet decisions in `ev_analyzer.py`. Concurrently, log the divergence on every live poll tick so it accumulates as training data for a future real-time model that will eventually find cross-venue trade opportunities autonomously.

## Non-goals (v1)

- **Pure risk-free cross-market arbitrage** (no simultaneous hedged positions across venues).
- **Historical divergence reconstruction / backtester integration.** Requires paid historical sportsbook odds (e.g. The Odds API historical tier); deferred until a source is available.
- **Book-sharpness weighting.** All non-stale books are equal-weight for v1.
- **Pinnacle / sharp book special treatment.** Not available in current Odds API free tier.
- **Kelly sizing coupling.** Tier affects the edge *threshold* only; Kelly fraction is untouched.
- **Model retraining on divergence features.** Signal is logged now, trained on later when David says go.
- **Automated shadow-to-active promotion.** Manual decision based on shadow-mode metrics.

## Design decisions (settled during brainstorm)

| Decision | Chosen value | Rationale |
|---|---|---|
| Signal type | Soft arbitrage / divergence (not risk-free arb) | Higher opportunity count; leverages the existing model as directional prior. |
| Reference price | Sportsbook consensus (A), with model-triangulation as tier-up filter (D) | Consensus is the most honest non-model baseline (independent opinions, partial vig cancellation); model-consensus agreement promotes high-conviction cases. |
| Integration point | `ev_analyzer.py --extra-signals` + realtime feature-store columns + thin CLI | Matches existing `venue_edge` / `referee_bias` pattern; logs training data at zero marginal cost. |
| De-vig method | Proportional | Simplest method without meaningful NBA-range bias; swappable behind a `method=` argument. |
| Aggregation across books | Median | Robust to stale or suspended lines. |
| Divergence-to-action mapping | Tier adjusts edge threshold (Approach 2) | Mirrors `calibration_monitor` pattern; preserves model as sole probability source; tier is emitted as a column regardless, so Kelly-coupling (Approach 3) can be added later as a 10-line swap. |
| Kalshi side | YES mid = (yes_bid + yes_ask) / 2 | Fairest reference for signal comparison; raw bid/ask still used by downstream EV math. |
| Consensus unavailable | Tier defaults to 1 (neutral); columns = NaN | Common off-hours; shouldn't gate the whole pipeline. |
| Meaningful divergence cutoff | 3.5pp | Balanced signal count; ~5-10% of games expected to trigger. |
| Historical reconstruction | Skipped for v1 | No historical sportsbook data source available. Revisit if acquired. |

## Module layout

### New module: `consensus_divergence.py` (~300 LOC, pure functions)

Single source of truth for the divergence signal. No I/O inside the public functions; callers fetch data and pass it in.

```python
devig_proportional(home_ml: int | None, away_ml: int | None) -> tuple[float, float] | None
    # moneyline pair → (p_home_devigged, p_away_devigged)
    # Returns None if either moneyline is None, 0, empty-string, or otherwise invalid.
    # Upstream parsers (ESPN, Odds API) may return None/"" for suspended or missing lines,
    # so the function must accept those gracefully rather than assuming valid ints.

consensus_implied_prob(
    sportsbooks: list[dict],
    stale_seconds: int = 300,
    min_books: int = 3,
) -> dict | None
    # books list (shape from live_data.get_odds_comparison) →
    # {"home_prob": float, "away_prob": float, "n_books": int, "books_used": list[str]}
    # Drops quotes older than stale_seconds using book.last_update from The Odds API.
    # Returns None if fewer than min_books remain after filtering.
    # Final home/away probs are clamped to [0.01, 0.99] and re-normalized to sum to 1.

compute_divergence(
    kalshi_yes_mid: float | None,
    model_prob_home: float | None,
    consensus: dict | None,
) -> dict
    # {
    #   "kalshi_vs_consensus_pp": float | None,   # consensus - kalshi (percentage points)
    #   "model_vs_consensus_pp":  float | None,
    #   "kalshi_vs_model_pp":     float | None,
    #   "triangulation_tier":     int,            # -1, 1, or 2
    #   "tier_reason":            str,            # human-readable
    # }
    # Missing inputs → corresponding divergences are None; tier defaults to 1.

tier_to_threshold_multiplier(tier: int, base_threshold: float) -> float
    # Tier 2 → base * 0.7  (easier to bet)
    # Tier 1 → base * 1.0  (neutral)
    # Tier -1 → base * 1.5 (stricter)
    # Multipliers live in config.TIER_THRESHOLD_MULTIPLIERS; tunable without code change.
```

### Tier assignment rules (deterministic)

Let `CUTOFF = config.CONSENSUS_DIVERGENCE_THRESHOLD_PP` (default 3.5pp).

- **Tier 2 (high-conviction triangulation)**:
  `|kalshi_vs_consensus_pp| >= CUTOFF` **and** `|kalshi_vs_model_pp| >= CUTOFF` **and** both disagreements point in the same direction (model and consensus agree that Kalshi is mispriced the same way).
- **Tier -1 (model ↔ consensus contradiction)**:
  `|kalshi_vs_consensus_pp| >= CUTOFF` **and** `|kalshi_vs_model_pp| >= CUTOFF` **and** the two signals point in *opposite* directions.
- **Tier 1 (neutral)**: everything else, including cases where consensus or model is missing.

### Callers

1. **`ev_analyzer.py`** — new `--consensus` flag (also triggered by `--extra-signals`). Fetches comparison via existing `live_data.get_odds_comparison`, calls the four functions, emits the four new columns (`kalshi_vs_consensus_pp`, `model_vs_consensus_pp`, `triangulation_tier`, `consensus_n_books`), applies `tier_to_threshold_multiplier` to the edge threshold before bet decision — respecting `CONSENSUS_TIER_MODE`.
2. **`realtime_feature_store.py`** — inside the existing polling loop, calls the pure functions using already-fetched Kalshi orderbook + Odds API data, appends the four columns to `live_features_YYYY-MM-DD.csv`. Emits a `divergence_change` record to `live_events_*.jsonl` when `triangulation_tier` changes between polls (reuses existing event-emitter pattern).
3. **Thin CLI**: `python consensus_divergence.py --scan` — ~30 lines, prints ranked divergence table for today's games. Used for ad-hoc sanity checks.

### Config additions (`config.py`)

```python
CONSENSUS_MIN_BOOKS = 3
CONSENSUS_STALE_SECONDS = 300
CONSENSUS_DIVERGENCE_THRESHOLD_PP = 3.5
TIER_THRESHOLD_MULTIPLIERS = {2: 0.7, 1: 1.0, -1: 1.5}
CONSENSUS_TIER_MODE = "shadow"   # "shadow" | "active" | "off"
```

### Untouched

`model.py`, `features.py`, `backtester.py`, training pipelines. Signal is logged but not trained on; backtester receives NaN for the new columns.

## Data flow

### Flow A — `ev_analyzer.py` (pre-game scan)

```
For each matchup:
  1. live_data.get_odds_comparison(home, away)
         → {kalshi: {...}, sportsbooks: [{name, home_moneyline, away_moneyline, last_update}, ...]}
  2. model.predict(features)                 → model_prob_home
  3. kalshi_yes_mid = (kalshi.yes_bid + kalshi.yes_ask) / 2
  4. consensus = consensus_implied_prob(comparison["sportsbooks"])
         → None if < CONSENSUS_MIN_BOOKS remain after staleness filter
  5. div = compute_divergence(kalshi_yes_mid, model_prob_home, consensus)
  6. if CONSENSUS_TIER_MODE == "active":
         adjusted_threshold = tier_to_threshold_multiplier(div.tier, config.EDGE_THRESHOLD)
     elif CONSENSUS_TIER_MODE == "shadow":
         adjusted_threshold = config.EDGE_THRESHOLD          # unchanged; shadow value printed separately
     else:  # "off"
         adjusted_threshold = config.EDGE_THRESHOLD
  7. bet = (abs(model_edge) >= adjusted_threshold)
  8. emit row: matchup, model_prob, kalshi_yes, consensus_prob, divergences, tier,
               base_threshold, adjusted_threshold, would_bet_at_base, would_bet_at_adjusted, bet_decision
```

Downstream CLV / Kelly / order-execution logic is unchanged.

### Flow B — `realtime_feature_store.py` (every poll tick)

```
Inside the existing poll-and-snapshot loop, after Kalshi + ESPN + Odds API fetches:
  For each live game:
    kalshi_yes_mid = from already-fetched Kalshi orderbook
    book_list      = from already-fetched Odds API payload
    consensus      = consensus_implied_prob(book_list)
    model_prob     = pregame prior loaded from saved predictions (already done today)
    div            = compute_divergence(kalshi_yes_mid, model_prob, consensus)

  Append to the existing live_features_YYYY-MM-DD.csv row:
    kalshi_vs_consensus_pp, model_vs_consensus_pp, kalshi_vs_model_pp,
    triangulation_tier, consensus_n_books
  (NaN fields when consensus was None.)

  If triangulation_tier changed since previous poll for this game:
    emit 'divergence_change' event to live_events_YYYY-MM-DD.jsonl
```

No new file paths, no new schemas — additive columns in the existing per-day CSV. `live_labeler.py` will pick these up automatically when the user decides to train on them.

### Flow C — Thin CLI `consensus_divergence.py --scan`

Lightweight subset of Flow A (steps 1, 4, 5, skipping model). Prints a ranked table sorted by `|kalshi_vs_consensus_pp|` for today's games.

## Edge cases

| # | Case | Behavior |
|---|---|---|
| 1 | `home_ml == 0` or malformed American odds | `devig_proportional` returns None; caller skips that book. |
| 2 | Book's `last_update` is older than `CONSENSUS_STALE_SECONDS` (300s) | Book dropped from consensus input. Uses The Odds API's per-book timestamp; no external clocking. |
| 3 | Fewer than `CONSENSUS_MIN_BOOKS` (3) remain after staleness filter | `consensus_implied_prob` returns None. Downstream: tier = 1, divergence columns = NaN. |
| 4 | Kalshi YES bid/ask missing (market not yet live) | Divergences populated where possible; `kalshi_vs_*_pp` = None. Tier = 1. |
| 5 | Model prediction missing (game not in today's pregame batch) | `kalshi_vs_consensus_pp` still computed; `model_vs_consensus_pp` = None; tier = 1 (no triangulation possible). |
| 6 | Book reports suspended line (Odds API price=None on one side) | Book dropped for that game's de-vig. |
| 7 | De-vigged probs don't sum to 1 exactly (float drift) | Clamp to [0.01, 0.99], re-normalize to sum to 1, return. |
| 8 | Concurrent writers to `live_features_YYYY-MM-DD.csv` | Already handled by existing append-only pattern; we only add columns. |

### Policies

- **Silent degradation over hard failure.** Feature store must never crash because an odds API hiccuped. Every failure returns None / NaN, logs a warning through the existing logger, continues polling.
- **No retries at this layer.** Retries belong in the HTTP clients of `live_data.py`. `consensus_divergence.py` treats inputs as already-fetched data.

## Testing strategy

Framework: **pytest**. No existing test suite to extend.

### Layer 1 — Unit tests (`tests/test_consensus_divergence.py`)

| Function | Cases |
|---|---|
| `devig_proportional` | Standard (-150/+130 → ~60/40); pick'em (-110/-110 → 50/50); malformed (0, missing) → None |
| `consensus_implied_prob` | 5 fresh books → median correct; 3 fresh + 2 stale → stale dropped; 2 fresh + 3 stale → None; all malformed → None |
| `compute_divergence` | **Tier boundaries are critical**: at exactly CUTOFF (boundary inclusion), just below CUTOFF (neutral), same-direction vs opposite-direction, model missing → tier 1, Kalshi missing → tier 1 |
| `tier_to_threshold_multiplier` | One assertion per tier |

Tier-boundary tests are the single most important guard — silent regressions here change live bet decisions.

### Layer 2 — Integration smoke tests

1. **`ev_analyzer.py` path.** Synthetic game: model_prob=0.65, kalshi_yes=0.55, 5 books implying 0.63 → assert tier=2, `adjusted_threshold = base * 0.7`, bet flips to True when it was False at base. One assertion per claim.
2. **`realtime_feature_store.py` path.** Feed one poll cycle, assert new columns present with correct values. Rerun with empty Odds API mock, assert columns are NaN and no exception raised.

### Layer 3 — Golden-data snapshot

Hand-curated ~20-row scenario CSV checked into the repo. Run `compute_divergence` across all rows, compare to expected tier/divergence columns. Detects any quiet change to tier rules.

### Layer 4 — Live smoke test (not automated)

`python consensus_divergence.py --scan --dry-run` pulls real current odds, prints the table, writes nothing. Run once after deploy. Not in CI (requires API keys + live games).

### Not tested

- Odds API parser (already covered upstream in `live_data.py`).
- Kelly / EV math downstream of tier adjustment (unchanged behavior).
- Historical backfill (not in v1).

## Rollout plan

### Three-phase rollout gated by `CONSENSUS_TIER_MODE`

| Phase | Mode | Duration | Behavior |
|---|---|---|---|
| 1. Shadow | `"shadow"` | ~4 weeks of live data | Divergences + tiers logged everywhere. `ev_analyzer` *prints* tier and "what adjusted_threshold would have been" but uses unmodified base threshold for bets. Zero capital risk. |
| 2. Active | `"active"` | Ongoing, once shadow-mode PnL delta is positive | Tier multipliers applied. Bet decisions shift. |
| 3. Disabled | `"off"` | Rollback | Divergence still logged (cheap data capture); tier hardcoded to 1, multiplier 1.0. Preserves data collection. |

Shadow-first is non-negotiable: the tier multipliers (0.7 / 1.0 / 1.5) are priors. With no historical odds data, shadow mode is the only way to calibrate responsibly.

### Metrics logged per settled game

Append one row per settled game to `outputs/consensus_divergence_metrics.csv`:

```
game_id, settled_at, mode, tier,
kalshi_vs_consensus_pp, model_vs_consensus_pp,
bet_taken, stake, pnl,
would_have_bet_at_base_threshold,
would_have_bet_at_adjusted_threshold
```

The last two columns drive the shadow-mode evaluation described below.

### Alerting

Two print-line warnings in `market_scanner.py` / `ev_analyzer.py` output only:
- "⚠ Consensus unavailable for N games today" (probable API outage or off-hours).
- "⚠ N games show tier -1 (model ↔ consensus disagreement)" (human eyeball recommended).

No email / Slack / push alerts in v1.

### Rollback criteria

Each criterion applies only once the relevant mode is reached:

- **(Active mode only)** Tier-2 bets placed under active mode post negative cumulative ROI after 50 settled tier-2 bets → flip back to `"shadow"`.
- **(Shadow or active)** Consensus availability drops below 60% of poll ticks over a rolling 7-day window → flip back to `"shadow"` if in active mode; investigate upstream odds fetch regardless.
- **(Any mode)** Any `consensus_divergence.py` exception crashes `realtime_feature_store.py` → immediate flip to `"off"`, file a bug (silent-degradation policy violated).

## Shadow-mode evaluation (FOLLOW-UP — not in v1)

After ~4 weeks of shadow-mode data, run a one-off analysis script (not included in v1 scope) that computes:

> **PnL delta** = ROI of bets that the adjusted threshold would have taken but the base threshold would not have taken, minus ROI of bets that the adjusted threshold would have rejected but the base threshold would have accepted.

Decision rules:
- **PnL delta ≥ +2% ROI on ≥50 divergent cases** → promote to `"active"`.
- **PnL delta near zero** → multipliers too timid; widen (e.g., 0.6 / 1.0 / 1.7); shadow another 2 weeks.
- **PnL delta negative** → tier logic is wrong; do not promote; revisit tier-assignment rules.

This analysis is flagged here so future-David doesn't forget. Write the script once sufficient data has accumulated (~50 tier-2 cases minimum).

## Definition of done (v1)

- [ ] `consensus_divergence.py` exists with four public pure functions + ~30-line `--scan` CLI. No I/O inside function bodies.
- [ ] Unit tests (Layer 1) pass.
- [ ] Integration smoke tests (Layer 2) pass.
- [ ] Golden snapshot (Layer 3) committed and passing.
- [ ] `ev_analyzer.py --extra-signals` emits the four new columns and respects `CONSENSUS_TIER_MODE=shadow` (no bet-decision change in shadow mode).
- [ ] `realtime_feature_store.py` appends the four new columns to `live_features_YYYY-MM-DD.csv` every poll; writes NaN silently when odds incomplete; never crashes the loop. Emits `divergence_change` events to the JSONL log on tier transitions.
- [ ] `outputs/consensus_divergence_metrics.csv` populated with at least one row per settled game.
- [ ] Live smoke test run once, output visually sane.
- [ ] Config flags (`CONSENSUS_MIN_BOOKS`, `CONSENSUS_STALE_SECONDS`, `CONSENSUS_DIVERGENCE_THRESHOLD_PP`, `TIER_THRESHOLD_MULTIPLIERS`, `CONSENSUS_TIER_MODE`) in `config.py`, each with an inline comment explaining its purpose and expected range.
- [ ] `IMPROVEMENTS.md` updated: #8 marked ✅ with a brief description matching the pattern of other completed items.

## Open questions deferred to follow-up work

- When to run the shadow-mode PnL delta analysis script (flagged above; to be scheduled after sufficient data).
- Whether to acquire a historical multi-book odds source to enable backtester integration and multiplier recalibration.
- Whether tier-2 / tier -1 thresholds should be asymmetric (e.g., Tier -1 may want a different cutoff than Tier 2).

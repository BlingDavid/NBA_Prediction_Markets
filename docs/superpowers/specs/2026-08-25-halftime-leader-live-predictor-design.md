# Live Halftime-Leader Predictor — Design

**Date:** 2026-08-25
**Status:** Approved design; implementation plan pending.
**Author:** Advisor(Opus) + user, via brainstorming.

## Overview

Pivot the live modeling target away from the 5-minute microstructure price move
(now proven NO-GO — Kalshi's winner fee ≈ the expected move; see
`outputs/liveonly_holdout_report.md`) toward a **live in-game predictor of the
halftime leader**.

At every ~15s live tick during the first half (period 1–2), predict
`P(home team is ahead at the end of Q2)`. This is a **research/prediction-first**
effort: the deliverable is a well-calibrated predictor evaluated honestly.
Trading/monetization is explicitly deferred until edge is demonstrated.

The interesting research question is **not** raw accuracy (trivially high late in
Q2 when you can just read the score) but **lead time**: how early in the half can
the model call the halftime leader *better than dumb baselines* — "whoever is
ahead now" and "the pre-game ELO favorite."

## Target / label

- `label_halftime_home_lead` ∈ {1, 0}: 1 if `home_score > away_score` at the
  Q2→Q3 boundary, else 0.
- Built in `live_labeler.py` from `data/live/games/game_states_<date>.csv`: per
  game, take the score at end of Q2 (last period≤2 tick before period 3 begins).
- **Halftime ties** (`home_score == away_score` at the break): dropped from
  training, counted and reported separately (rare but must not be silently
  mislabeled).
- The single game-level label is **joined to every H1 tick** (period ∈ {1,2}) of
  that game, so each tick is a training row carrying that game's outcome.

## Data & sufficiency (read this first)

The label is **one per game**, and ticks within a game are highly correlated, so
the **effective sample size ≈ number of games** (~60–100 across the captured
Apr–May 2026 playoff days), NOT the ~millions of ticks. Consequences, baked into
the design:

- Keep the model **simple** (logistic + calibration), lean on a strong
  analytical baseline, and group all CV/hold-out splits **by game**.
- Set expectations up front: as in Round 4, the honest verdict may be "promising
  but thin — needs more game-days to confirm." The evaluation must report
  confidence intervals / significance, not point estimates alone.

## Features (H1 training matrix)

Reuse `live_training_matrix.py` patterns; filter to period ∈ {1,2}. Feature set:

- `score_margin` = home_score − away_score (dominant signal; grows informative
  toward halftime).
- `seconds_elapsed_in_half` — game-time, so a **single** model spans all of H1.
- `pace` / possessions-so-far (expected remaining scoring).
- `score_run_3min` — the momentum feature already added and parity-tested in
  `live_training_matrix.compute_momentum_features`.
- `pregame_home_win_prob` — the ELO/ensemble pre-game prior for the matchup.
- Additional live context (foul trouble / lineup) **only if** reliably present in
  captured data; otherwise omitted (no fabricated features).

Grouped by game/`game_key` for CV to prevent cross-tick leakage.

## Model

- A calibrated classifier: start with logistic regression + isotonic calibration,
  `GroupKFold` by game. Output `P(home leads at halftime)` per tick.
- Reuse `live_bootstrap_model.py` machinery (it already accepts `--target`), with
  a thin H1-specific wrapper for the period filter and feature set. Candidate
  artifacts written under `models/candidates/` and `outputs/candidates/`
  (non-destructive; never overwrite deployed artifacts).

## Baselines to beat (the point of the exercise)

1. **Current-leader**: predict home leads at half iff home currently ahead
   (or a sigmoid of current margin for a probabilistic version).
2. **ELO prior**: the pre-game home-win probability, held constant through H1.
3. **Diffusion baseline**: halftime margin ≈ current margin + Brownian-bridge of
   expected remaining H1 scoring (0-drift, variance scaled by pace and
   time-to-halftime) → `P(home leads) = Φ(margin_est / σ)`. This is the strong,
   interpretable benchmark the ML model must beat to justify itself.

## Evaluation (honest, Round-4 discipline)

- **Temporal hold-out by date**: train on earlier game-days, test on later ones;
  Phase-style seeds/baselines fit on train only. No leakage from test dates.
- Metrics on the hold-out: ROC-AUC, Brier, log-loss, reliability/calibration.
- **Headline: lead-time curve.** Bin H1 ticks by minutes-elapsed (e.g. every 2
  min); at each bin compare model accuracy/Brier against each baseline. The
  deliverable statement: "from minute *M*, the model beats current-leader by *X*
  (± CI)." Report significance, given the thin effective N.
- New `tools/halftime_eval.py`, following the patterns of `tools/live_model_eval.py`
  and `tools/compare_models.py`.

## Reuse / new-file map

| Action | File |
|---|---|
| Edit (+ label) | `live_labeler.py` |
| New | H1 training-matrix builder (thin, reuses `live_training_matrix` helpers) |
| Reuse | `live_bootstrap_model.py --target label_halftime_home_lead` (+ H1 wrapper) |
| New | `tools/halftime_eval.py` (baselines + lead-time curve) |
| Tests | TDD throughout; parity tests where train/live feature code is shared |

## Out of scope (YAGNI)

- No trading, order execution, or market integration.
- No time-sliced ensemble (that is iteration 2 if v1 shows promise).
- No new data capture; use existing `data/live/*` captures only.

## Success criteria

- A calibrated live halftime-leader predictor with an honest, leakage-free
  hold-out evaluation.
- A lead-time curve showing whether — and from what minute — the model beats the
  current-leader and ELO baselines, with significance given the thin N.
- An explicit, honest verdict (including "not enough data to conclude" if that is
  the truth), never a number massaged to look good.

## Risks / open questions

- Effective N (~60–100 games) may be too small for a stable ML win over the
  diffusion baseline; v1 may conclude "thin."
- Q2-end boundary detection must be robust to capture gaps (games with missing
  end-of-Q2 ticks); define a tolerance and drop games that can't be labeled.
- Whether captured data cleanly separates period 2 end from period 3 start across
  all game-days (to verify during implementation).

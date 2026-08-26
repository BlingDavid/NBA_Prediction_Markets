# Halftime-Leader Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a live, calibrated predictor of the halftime leader (who is ahead at the end of Q2), predicting at every first-half tick, evaluated honestly against dumb baselines with a lead-time curve.

**Architecture:** Reuse the existing live-model pipeline end to end. Add a game-level `label_halftime_home_lead` in `live_labeler.py`, flow it through the existing training matrix, train with the existing `live_bootstrap_model.py --target` machinery restricted to first-half ticks, and add a standalone `tools/halftime_eval.py` for baselines + lead-time evaluation on a temporal hold-out. Prediction/research only — no trading.

**Tech Stack:** Python, pandas, numpy, scikit-learn, pytest. Env for any model-touching command: `source .venv/bin/activate && export DYLD_FALLBACK_LIBRARY_PATH="$DYLD_FALLBACK_LIBRARY_PATH:/opt/homebrew/opt/libomp/lib"`.

**Design ref:** `docs/superpowers/specs/2026-08-25-halftime-leader-live-predictor-design.md`

**Global constraints (apply to every task):**
- Work on branch `feat/halftime-leader`. Do NOT commit the intentionally-untracked root modules (model.py, features.py, data_ingest.py, kalshi_auth.py, late_lineups.py, live_adjustments.py, player_usage.py, prediction_utils.py, referee_bias.py, strength_of_schedule.py, venue_edge.py, bayesian_update.py, calibration_monitor.py, backtester.py) — note `live_labeler.py`, `live_training_matrix.py`, `live_bootstrap_model.py` ARE tracked and editable.
- Non-destructive: never overwrite `models/live_home_up_5m_bootstrap.pkl`, `outputs/paper_trades/pooled_means.json`, or existing `outputs/live_home_up_5m_bootstrap_*`. Candidate artifacts go under `models/candidates/` and `outputs/candidates/`.
- Keep the full suite green (currently 417). Run `python -m pytest tests/ -q` before each commit that touches shared code.

---

### Task 1: Halftime-leader label resolver

**Files:**
- Modify: `live_labeler.py` (add `resolve_halftime_leaders` near `resolve_final_game_outcomes`, ~line 39)
- Test: `tests/test_halftime_label.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_halftime_label.py
import pandas as pd
import pytest
from live_labeler import resolve_halftime_leaders


def _game_history():
    # game A: home leads at last Q2 tick; game B: away leads; game C: tie at Q2 end
    return pd.DataFrame([
        {"game_key": "A", "captured_at": "2026-04-18T00:10:00Z", "period": 1, "home_score": 20, "away_score": 22},
        {"game_key": "A", "captured_at": "2026-04-18T00:40:00Z", "period": 2, "home_score": 55, "away_score": 50},
        {"game_key": "A", "captured_at": "2026-04-18T01:30:00Z", "period": 4, "home_score": 99, "away_score": 90},
        {"game_key": "B", "captured_at": "2026-04-18T00:42:00Z", "period": 2, "home_score": 48, "away_score": 60},
        {"game_key": "C", "captured_at": "2026-04-18T00:43:00Z", "period": 2, "home_score": 51, "away_score": 51},
    ])


def test_resolves_end_of_q2_leader_per_game():
    out = resolve_halftime_leaders(_game_history()).set_index("game_key")
    assert out.loc["A", "label_halftime_home_lead"] == 1
    assert out.loc["A", "label_halftime_margin_home"] == 5
    assert out.loc["B", "label_halftime_home_lead"] == 0


def test_tie_at_halftime_is_flagged_and_label_is_na():
    out = resolve_halftime_leaders(_game_history()).set_index("game_key")
    assert out.loc["C", "label_halftime_is_tie"] == 1
    assert pd.isna(out.loc["C", "label_halftime_home_lead"])


def test_game_without_q2_rows_is_dropped():
    hist = pd.DataFrame([
        {"game_key": "D", "captured_at": "2026-04-18T00:05:00Z", "period": 1, "home_score": 10, "away_score": 8},
    ])
    out = resolve_halftime_leaders(hist)
    assert "D" not in set(out["game_key"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_halftime_label.py -q`
Expected: FAIL — `ImportError: cannot import name 'resolve_halftime_leaders'`.

- [ ] **Step 3: Implement the resolver**

Add to `live_labeler.py` (reuses existing module-level `_as_timestamp` and `_nullable_binary`):

```python
def resolve_halftime_leaders(game_history: pd.DataFrame) -> pd.DataFrame:
    """Per game, who leads at the end of Q2 (last observed period==2 tick).

    Returns game_key + label_halftime_home_lead (1 home / 0 away / NA on a
    halftime tie), label_halftime_margin_home, label_halftime_is_tie. Games
    with no period==2 capture are dropped (cannot be labeled).
    """
    cols = [
        "game_key",
        "label_halftime_home_lead",
        "label_halftime_margin_home",
        "label_halftime_is_tie",
    ]
    if game_history.empty:
        return pd.DataFrame(columns=cols)

    hist = game_history.copy()
    hist["captured_at"] = _as_timestamp(hist["captured_at"])
    hist["period"] = pd.to_numeric(hist["period"], errors="coerce")
    q2 = hist[hist["period"] == 2]
    if q2.empty:
        return pd.DataFrame(columns=cols)

    last_q2 = (
        q2.sort_values(["game_key", "captured_at"])
        .groupby("game_key", as_index=False)
        .last()
    )
    margin = last_q2["home_score"] - last_q2["away_score"]
    last_q2["label_halftime_margin_home"] = margin
    last_q2["label_halftime_is_tie"] = (margin == 0).astype(int)
    last_q2["label_halftime_home_lead"] = _nullable_binary(margin > 0)
    last_q2.loc[margin == 0, "label_halftime_home_lead"] = pd.NA
    return last_q2[cols]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_halftime_label.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add live_labeler.py tests/test_halftime_label.py
git commit -m "feat(labeler): resolve halftime leader (end-of-Q2) per game"
```

---

### Task 2: Merge the halftime label into the labeled training set

**Files:**
- Modify: `live_labeler.py` — `build_labeled_training_set` (merge site at ~line 315)
- Test: `tests/test_halftime_label.py` (add a test)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_halftime_label.py
from live_labeler import build_labeled_training_set  # add to imports at top


def test_labeled_set_carries_halftime_label(monkeypatch):
    features = pd.DataFrame([
        {"game_key": "A", "captured_at": "2026-04-18T00:40:00Z", "bet_side": "home",
         "period": 2, "home_score": 55, "away_score": 50, "ticker": "T-A"},
    ])
    games = pd.DataFrame([
        {"game_key": "A", "captured_at": "2026-04-18T00:40:00Z", "period": 2,
         "home_score": 55, "away_score": 50, "status_state": "in"},
    ])
    # Inject the synthetic frames instead of reading data/live/*
    monkeypatch.setattr("live_labeler._load_feature_history", lambda *a, **k: features)
    monkeypatch.setattr("live_labeler._load_game_history", lambda *a, **k: games)
    monkeypatch.setattr("live_labeler._load_market_history", lambda *a, **k: pd.DataFrame())
    labeled = build_labeled_training_set()
    assert "label_halftime_home_lead" in labeled.columns
    assert labeled.iloc[0]["label_halftime_home_lead"] == 1
```

Note: confirm the exact loader function names inside `build_labeled_training_set` (read `live_labeler.py:291-330`) and adjust the three `monkeypatch.setattr` targets to match. If the function reads frames differently, patch whatever it calls to obtain `features`/`games`/`market` before the merges.

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_halftime_label.py::test_labeled_set_carries_halftime_label -q`
Expected: FAIL — column `label_halftime_home_lead` missing.

- [ ] **Step 3: Add the merge**

In `live_labeler.py`, immediately after the existing final-outcomes merge (currently `labeled = features.merge(final_outcomes, on="game_key", how="left")` at ~line 315), add:

```python
    halftime_outcomes = resolve_halftime_leaders(games)
    labeled = labeled.merge(halftime_outcomes, on="game_key", how="left")
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_halftime_label.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the full suite (shared-file change)**

Run: `python -m pytest tests/ -q`
Expected: all pass (was 417; now 421).

- [ ] **Step 6: Commit**

```bash
git add live_labeler.py tests/test_halftime_label.py
git commit -m "feat(labeler): join halftime-leader label into training set"
```

---

### Task 3: Flow the label through the training matrix

**Files:**
- Modify: `live_training_matrix.py` — `build_in_game_training_matrix` (numeric_cols list ~line 222-260, and the pass-through column list ~line 362 where `label_final_home_win` appears)
- Test: `tests/test_halftime_matrix.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_halftime_matrix.py
import pandas as pd
from live_training_matrix import build_in_game_training_matrix


def _labeled():
    return pd.DataFrame([
        {"game_key": "A", "ticker": "T-A", "bet_side": "home",
         "captured_at": "2026-04-18T00:40:00Z", "status_state": "in",
         "period": 2, "seconds_elapsed": 1400, "seconds_left_in_period": 100,
         "home_score": 55, "away_score": 50, "score_margin_home": 5,
         "total_points": 105, "label_halftime_home_lead": 1,
         "label_final_home_win": 1},
    ])


def test_matrix_retains_halftime_label_column():
    matrix, feature_cols = build_in_game_training_matrix(_labeled())
    assert "label_halftime_home_lead" in matrix.columns
    assert matrix.iloc[0]["label_halftime_home_lead"] == 1
    # label must NOT leak into the model feature set
    assert "label_halftime_home_lead" not in feature_cols
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_halftime_matrix.py -q`
Expected: FAIL — `label_halftime_home_lead` not in matrix columns.

- [ ] **Step 3: Add the label to numeric coercion and pass-through**

In `build_in_game_training_matrix`, add `"label_halftime_home_lead"` and `"label_halftime_margin_home"` to the `numeric_cols` list (next to the other `label_*` entries, ~line 258), AND to the returned pass-through column block (the list that already includes `"label_final_home_win"`, ~line 362). Do NOT add it to `feature_cols` (it is a target, not a feature).

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_halftime_matrix.py -q`
Expected: PASS.

- [ ] **Step 5: Full suite + commit**

```bash
python -m pytest tests/ -q   # expect all pass
git add live_training_matrix.py tests/test_halftime_matrix.py
git commit -m "feat(matrix): pass halftime-leader label through training matrix"
```

---

### Task 4: First-half filter in the dataset builder

**Files:**
- Modify: `live_bootstrap_model.py` — add pure `filter_first_half`, thread `first_half_only` through `build_bootstrap_dataset` and `train_bootstrap_model`, add `--first-half-only` and (if absent) `--model-name` CLI args.
- Test: `tests/test_first_half_filter.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_first_half_filter.py
import pandas as pd
from live_bootstrap_model import filter_first_half


def test_keeps_only_period_1_and_2():
    df = pd.DataFrame({"period": [0, 1, 2, 3, 4], "x": range(5)})
    out = filter_first_half(df)
    assert sorted(out["period"].tolist()) == [1, 2]


def test_noop_when_flag_false():
    df = pd.DataFrame({"period": [0, 1, 2, 3, 4]})
    assert len(filter_first_half(df, enabled=False)) == 5
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_first_half_filter.py -q`
Expected: FAIL — `cannot import name 'filter_first_half'`.

- [ ] **Step 3: Implement the filter + threading**

Add near `filter_matrix_by_cutoff` (~line 253) in `live_bootstrap_model.py`:

```python
def filter_first_half(matrix: pd.DataFrame, enabled: bool = True) -> pd.DataFrame:
    """Restrict to first-half ticks (period in {1, 2})."""
    if not enabled or "period" not in matrix.columns:
        return matrix
    period = pd.to_numeric(matrix["period"], errors="coerce")
    return matrix[period.isin([1, 2])].copy()
```

In `build_bootstrap_dataset`, add parameter `first_half_only: bool = False` and, right after the existing `live_only` filter block (before/near the `filter_matrix_by_cutoff` call), insert:

```python
    matrix = filter_first_half(matrix, enabled=first_half_only)
```

In `train_bootstrap_model`, add parameter `first_half_only: bool = False` and pass it into the `build_bootstrap_dataset(...)` call. In the argparse block (`main`), add:

```python
    parser.add_argument("--first-half-only", action="store_true",
                        help="Train only on first-half ticks (period 1-2).")
    # add only if not already present:
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                        help="Base name for the output model/artifacts.")
```

and thread `first_half_only=args.first_half_only` and `model_name=args.model_name` into the `train_bootstrap_model(...)` call. (If `--model-name` already exists, skip re-adding it.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_first_half_filter.py -q`
Expected: PASS.

- [ ] **Step 5: Full suite + commit**

```bash
python -m pytest tests/ -q   # expect all pass
git add live_bootstrap_model.py tests/test_first_half_filter.py
git commit -m "feat(train): --first-half-only filter for halftime target"
```

---

### Task 5: Train the halftime-leader candidate model (temporal hold-out)

**Files:**
- No code changes; a training run producing candidate artifacts.

- [ ] **Step 1: Train on the TRAIN window only**

Run (env exported):

```bash
python live_bootstrap_model.py \
  --target label_halftime_home_lead \
  --first-half-only \
  --train-cutoff-date 2026-04-29 \
  --model-name live_halftime_leader \
  --models-dir models/candidates \
  --outputs-dir outputs/candidates
```

Expected: writes `models/candidates/live_halftime_leader.pkl`,
`outputs/candidates/live_halftime_leader_report.json`,
`outputs/candidates/live_halftime_leader_oof_predictions.csv`. The report's
`metadata.groups` = number of TRAIN games; confirm the `target` field is
`label_halftime_home_lead` and base rate is near 0.5 (not 0.047).

- [ ] **Step 2: Sanity-check the candidate loads and predicts**

Run:

```bash
python -c "
import pickle, sklearn
b = pickle.load(open('models/candidates/live_halftime_leader.pkl','rb'))
print('target', b.get('target'), 'sklearn', b.get('sklearn_version'), 'features', len(b.get('feature_cols', [])))
"
```

Expected: `target label_halftime_home_lead`, sklearn version present, feature count > 0.

- [ ] **Step 3: Commit the report only (pkl is under gitignored/candidate path)**

```bash
git add -f outputs/candidates/live_halftime_leader_report.json 2>/dev/null || true
# If outputs/ is gitignored (it is), skip committing artifacts; just note the run in the next task's report.
echo "candidate trained"
```

(No commit needed if artifacts are gitignored — proceed.)

---

### Task 6: Baseline probability functions

**Files:**
- Create: `tools/halftime_eval.py` (baseline functions first)
- Test: `tests/test_halftime_eval.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_halftime_eval.py
import numpy as np
import pytest
from tools.halftime_eval import (
    current_leader_prob,
    elo_prior_prob,
    diffusion_prob,
)


def test_current_leader_prob_monotone_in_margin():
    assert current_leader_prob(10.0) > current_leader_prob(0.0) > current_leader_prob(-10.0)
    assert current_leader_prob(0.0) == pytest.approx(0.5)
    assert 0.0 < current_leader_prob(3.0) < 1.0


def test_elo_prior_passthrough():
    assert elo_prior_prob(0.62) == pytest.approx(0.62)
    # clips into [0,1]
    assert elo_prior_prob(1.5) == pytest.approx(1.0)


def test_diffusion_prob_tightens_as_time_runs_out():
    # same +4 margin: with almost no time left, prob(home leads) -> ~1
    near_end = diffusion_prob(margin=4.0, seconds_left_in_half=5.0)
    early = diffusion_prob(margin=4.0, seconds_left_in_half=1200.0)
    assert near_end > early
    assert diffusion_prob(margin=0.0, seconds_left_in_half=600.0) == pytest.approx(0.5)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_halftime_eval.py -q`
Expected: FAIL — module `tools.halftime_eval` does not exist.

- [ ] **Step 3: Implement the baselines**

```python
# tools/halftime_eval.py
"""Honest evaluation of the live halftime-leader predictor: dumb baselines +
a lead-time curve on a temporal hold-out. Research/prediction only."""
from __future__ import annotations

import math

import numpy as np


def current_leader_prob(margin: float, k: float = 0.15) -> float:
    """Probabilistic 'current leader stays' baseline: logistic in current margin."""
    return 1.0 / (1.0 + math.exp(-k * float(margin)))


def elo_prior_prob(pregame_home_win_prob: float) -> float:
    """Pre-game ELO prior held constant through the half (clipped to [0,1])."""
    return float(min(1.0, max(0.0, pregame_home_win_prob)))


def diffusion_prob(margin: float, seconds_left_in_half: float,
                   points_std_per_sec: float = 0.11) -> float:
    """Brownian-bridge baseline: P(home leads at half) = Phi(margin / sigma),
    sigma grows with remaining time. points_std_per_sec is tuned so a full half
    (~1440s) gives a realistic scoring spread."""
    remaining = max(0.0, float(seconds_left_in_half))
    sigma = points_std_per_sec * math.sqrt(remaining) if remaining > 0 else 1e-9
    if sigma <= 1e-9:
        return 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
    z = float(margin) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_halftime_eval.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/halftime_eval.py tests/test_halftime_eval.py
git commit -m "feat(eval): halftime baselines (current-leader, ELO, diffusion)"
```

---

### Task 7: Lead-time curve aggregation

**Files:**
- Modify: `tools/halftime_eval.py` (add `lead_time_curve`)
- Test: `tests/test_halftime_eval.py` (add a test)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_halftime_eval.py
import pandas as pd
from tools.halftime_eval import lead_time_curve


def test_lead_time_curve_reports_per_bin_brier_for_model_and_baseline():
    df = pd.DataFrame({
        "minutes_elapsed": [1, 1, 13, 13],          # two H1 bins
        "y_true":          [1, 0, 1, 1],
        "model_prob":      [0.6, 0.4, 0.9, 0.8],
        "baseline_prob":   [0.5, 0.5, 0.55, 0.55],
    })
    out = lead_time_curve(df, bin_minutes=6).set_index("minute_bin")
    # model Brier must be computed per bin and be <= baseline in the late bin
    assert set(["minute_bin", "n", "model_brier", "baseline_brier"]).issubset(out.reset_index().columns)
    assert out.loc[12, "model_brier"] < out.loc[12, "baseline_brier"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_halftime_eval.py::test_lead_time_curve_reports_per_bin_brier_for_model_and_baseline -q`
Expected: FAIL — `cannot import name 'lead_time_curve'`.

- [ ] **Step 3: Implement the aggregation**

```python
# add to tools/halftime_eval.py
import pandas as pd


def lead_time_curve(df: pd.DataFrame, bin_minutes: int = 2) -> pd.DataFrame:
    """Per game-time bin, compare model vs baseline Brier / accuracy.

    df needs columns: minutes_elapsed, y_true, model_prob, baseline_prob.
    Returns one row per bin with n, model_brier, baseline_brier,
    model_acc, baseline_acc.
    """
    d = df.dropna(subset=["y_true", "model_prob", "baseline_prob"]).copy()
    d["minute_bin"] = (d["minutes_elapsed"] // bin_minutes) * bin_minutes
    rows = []
    for b, g in d.groupby("minute_bin"):
        yt = g["y_true"].to_numpy(dtype=float)
        rows.append({
            "minute_bin": int(b),
            "n": int(len(g)),
            "model_brier": float(np.mean((g["model_prob"].to_numpy() - yt) ** 2)),
            "baseline_brier": float(np.mean((g["baseline_prob"].to_numpy() - yt) ** 2)),
            "model_acc": float(np.mean((g["model_prob"].to_numpy() >= 0.5) == (yt == 1))),
            "baseline_acc": float(np.mean((g["baseline_prob"].to_numpy() >= 0.5) == (yt == 1))),
        })
    return pd.DataFrame(rows).sort_values("minute_bin").reset_index(drop=True)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_halftime_eval.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/halftime_eval.py tests/test_halftime_eval.py
git commit -m "feat(eval): lead-time curve aggregation"
```

---

### Task 8: End-to-end hold-out evaluation + honest report

**Files:**
- Modify: `tools/halftime_eval.py` — add a `main()` CLI that scores the TEST-window H1 matrix with the candidate model, computes baselines per row, builds the lead-time curve, and writes a report.
- Test: covered by the pure-function tests above; this task is an integration run.

- [ ] **Step 1: Add the CLI driver**

Add a `main()` to `tools/halftime_eval.py` that:
1. Accepts `--model-path models/candidates/live_halftime_leader.pkl`, `--dates-from 2026-04-29`, `--dates-to 2026-05-08`, `--out outputs/halftime_eval_report.md`.
2. Builds the TEST-window first-half matrix by reusing the same dataset machinery the training run uses, restricted to the test dates (mirror how `tools/cost_replay.py` selects date-ranged `data/live/features/live_features_<date>.csv` files; reuse `live_bootstrap_model.filter_first_half` and the matrix builder). Score each row with the loaded model to get `model_prob`; compute `current_leader_prob`, `elo_prior_prob`, `diffusion_prob` per row from `score_margin_home`, `pregame_home_win_prob`, and remaining-seconds-in-half (`= max(0, 1440 - seconds_elapsed)`); derive `minutes_elapsed = seconds_elapsed / 60`.
3. Computes hold-out ROC-AUC / Brier / log-loss (model vs each baseline) and the `lead_time_curve` for model vs the strongest baseline (diffusion).
4. Writes `outputs/halftime_eval_report.md` with: TRAIN/TEST split + game counts, overall metrics table, the lead-time curve, and an explicit GO / CAUTION / NO-GO / "too thin to conclude" verdict with confidence caveats given effective N ≈ number of games.

- [ ] **Step 2: Run the evaluation**

Run (env exported):

```bash
python tools/halftime_eval.py \
  --model-path models/candidates/live_halftime_leader.pkl \
  --dates-from 2026-04-29 --dates-to 2026-05-08 \
  --out outputs/halftime_eval_report.md
```

Expected: prints an overall metrics table and lead-time curve; writes `outputs/halftime_eval_report.md`. Model Brier should beat `current_leader` in early bins if there is real early-game signal; report honestly if it does not.

- [ ] **Step 3: Full suite**

Run: `python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 4: Commit the tool (report is under gitignored outputs/)**

```bash
git add tools/halftime_eval.py
git commit -m "feat(eval): end-to-end halftime hold-out evaluation + report"
```

---

## Self-review notes (completed)

- **Spec coverage:** label (T1–2), H1 matrix/features incl. momentum + ELO prior (T3, already emitted by builder), single game-time-aware model (T4–5), baselines current-leader/ELO/diffusion (T6), temporal hold-out + lead-time curve + honest verdict (T7–8), thin-N caveat (T8 verdict). Trading explicitly out of scope — no tasks add it. ✔
- **Consistency:** `label_halftime_home_lead` name used identically across T1–5; `filter_first_half`, `current_leader_prob`, `elo_prior_prob`, `diffusion_prob`, `lead_time_curve` signatures match between their defining task and their callers in T8. ✔
- **Known verification points for the implementer:** exact loader function names inside `build_labeled_training_set` (T2 Step 1 note) and whether `--model-name` already exists (T4 Step 3 note). Both are flagged inline.

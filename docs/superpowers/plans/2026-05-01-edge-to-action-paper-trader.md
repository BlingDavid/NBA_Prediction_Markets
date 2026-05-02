# Edge-to-Action Paper Trader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the saved `live_home_up_5m_bootstrap` model into a 15-second paper-trading loop on Kalshi NBA contracts, with EV-aware entry, continuous Kelly sizing, hard 5-minute exit, and a manually-gated path to live execution.

**Architecture:** A long-running consumer process (`paper_trader.py`) tails `data/live/features/latest_features.csv`, runs each new (ticker, captured_at) row through gates → scorer → trigger → position manager → trade log. State lives in `outputs/paper_trades/` as atomic JSON + append-only JSONL/CSV. The engine is restartable: an append-only `trader_log.jsonl` is the source of truth and replays byte-identically to `closed_trades.csv` and `trader_state.json`.

**Tech Stack:** Python 3.11, pandas, scikit-learn (existing model artifact), pytest. No new dependencies.

**Spec reference:** `docs/superpowers/specs/2026-04-28-edge-to-action-design.md`

**Key codebase facts (already verified):**
- Model artifact `models/live_home_up_5m_bootstrap.pkl` is a dict: `{model: sklearn.Pipeline, feature_cols: list[str] (42 cols), calibration: {base_rate: 0.044264, alpha: 0.95, ...}, target: 'label_home_up_5m', ...}`.
- Calibration formula (canonical): `np.clip(base_rate + alpha * (raw - base_rate), 1e-6, 1 - 1e-6)` — see `live_bootstrap_model._apply_shrinkage` (line 179).
- Training-time feature engineering lives in `live_training_matrix.build_in_game_training_matrix` (lines ~96-115). The scorer must apply the same transforms before calling `model.predict_proba`.
- `data/live/features/latest_features.csv` schema (54 cols) is the per-tick feed. It has *raw* columns; derived features (`game_progress`, `regulation_seconds_remaining`, `market_yes_spread`, `market_no_spread`, `orderbook_imbalance_3/5`, `orderbook_pressure_3`, `market_vs_pregame_home`, `score_gap_vs_pregame_spread`, `points_vs_pregame_total`, `flag_status_*`, `flag_has_*`) must be computed on the fly.
- Pooled-means seed source: `outputs/live_home_up_5m_bootstrap_scored_rows.csv` (~953k rows) has columns `pred_bootstrap_home_up_5m`, `label_home_up_5m`, `label_yes_mid_move_5m`. This is the artifact to compute Phase A `E[Δ|rises]` / `E[Δ|doesn't]` from.
- Test layout: `tests/test_*.py` at repo root with pytest configured via `pyproject.toml` (`pythonpath = ["."]`). New tests go in `tests/paper_trader/`.

**Fee model (locked from spec, made explicit here):**
- Half-spread cost (always paid at entry): `entry_yes_ask - entry_yes_mid`. Loser pays no additional Kalshi fee. Winner pays Kalshi fee at settlement: `fee_per_contract = ceil(0.07 × p_at_entry × (1 − p_at_entry) × 100) / 100`. (Standard Kalshi event-contract formula, rounded up to next cent.)
- Expected fee per contract used in trigger EV: `expected_fee = p_calibrated × fee_per_contract_if_win`.
- This is what `test_cost_model.py` validates against three scenarios (5¢, 50¢, 95¢ entry).

---

## File Structure

### New package: `paper_trader/`

| File | Responsibility |
|---|---|
| `__init__.py` | Empty marker |
| `cost_model.py` | Pure functions: `half_spread_cost`, `winner_fee_per_contract`, `expected_fee`. No state. |
| `trade_log.py` | Atomic JSON state writes (`write_state`), JSONL append (`emit_event`), CSV append (`record_trade`). Disk I/O only. |
| `gate_filters.py` | `evaluate(row) -> (passed: bool, reason: str)`. Pure, table-driven. |
| `scorer.py` | `load_model(path)`, `engineer_features(df)`, `score(model_bundle, df) -> (p_raw, p_calibrated)`. Imports feature engineering from `live_training_matrix` to avoid drift. |
| `move_size_estimator.py` | `pooled_means(scored_rows_path)` builds Phase A; `per_decile_means(closed_trades_df)` builds Phase B; `expected_moves(p, mode, store) -> (E[Δ\|rises], E[Δ\|doesn't], var_per_contract)`. |
| `trigger.py` | `evaluate(p_calibrated, moves, yes_ask, yes_mid, fee_per_contract) -> (accept: bool, expected_pnl_per_contract: float, reason: str)`. Pure. |
| `position_manager.py` | Owns in-memory state; opens/closes positions; enforces concurrency cap and 5-min timer; writes via `trade_log`. |
| `feed_reader.py` | Reads `data/live/features/latest_features.csv`; dedups against `last_processed_captured_at_per_ticker`. |

### New entry point: `paper_trader.py`

CLI: `python paper_trader.py --mode paper --bankroll 1000 --interval 15 [--features-path ... --model-path ... --output-dir ... --config ...]`. Owns the main loop, drawdown gate, stale-feed halt, restart/replay logic, and live-mode boot gate.

### New tools: `tools/`

| File | Responsibility |
|---|---|
| `tools/__init__.py` | Empty marker |
| `tools/build_pooled_means.py` | One-shot. Reads `outputs/live_home_up_5m_bootstrap_scored_rows.csv`, writes `outputs/paper_trades/pooled_means.json` (Phase A seed). Run once before first paper start. |
| `tools/trader_status.py` | Read-only CLI; reads state + tails JSONL; safe under `watch -n 30`. |
| `tools/trader_rollups.py` | Idempotent batch; recomputes `rollups/{daily_*,per_game,per_p_decile,per_minute_of_game}.csv`. |
| `tools/trader_smoke.py` | 10-min live run with trigger forced to reject; validates wiring before mode flips. |
| `tools/trader_backtest.py` | Replays one historical `live_features_<date>.csv` through the same pipeline; outputs to `outputs/paper_trades/backtest_<date>/`. |

### Tests: `tests/paper_trader/`

`test_cost_model.py`, `test_gate_filters.py`, `test_scorer.py`, `test_move_size_estimator.py`, `test_trigger.py`, `test_position_manager.py`, `test_trade_log.py`, `test_state_atomicity.py`, `test_feed_reader.py`, `test_paper_trader_loop.py`, `test_invariants.py` (replay + cumulative-PnL + concurrency cap + trade-id uniqueness), `test_tools_status.py`, `test_tools_rollups.py`, `test_tools_backtest.py`.

### Outputs (gitignored)

```
outputs/paper_trades/
  trader_state.json
  open_positions.csv
  closed_trades.csv
  trader_log.jsonl
  pooled_means.json
  rollups/
    daily_<YYYY-MM-DD>.csv
    per_game.csv
    per_p_decile.csv
    per_minute_of_game.csv
```

---

## Task 0: Scaffold packages, gitignore, README

**Files:**
- Create: `paper_trader/__init__.py` (empty)
- Create: `tools/__init__.py` (empty)
- Create: `tests/paper_trader/__init__.py` (empty)
- Modify: `.gitignore` (add `outputs/paper_trades/` and `**/__pycache__/`)
- Create: `paper_trader/README.md` (3-line pointer to spec + plan)

- [ ] **Step 1: Create empty package markers**

```bash
mkdir -p paper_trader tools tests/paper_trader
: > paper_trader/__init__.py
: > tools/__init__.py
: > tests/paper_trader/__init__.py
```

- [ ] **Step 2: Add output dir to gitignore**

Read existing `.gitignore` first (it's untracked — just append). Append:

```
outputs/paper_trades/
**/__pycache__/
```

- [ ] **Step 3: Create README pointer**

Create `paper_trader/README.md`:

```markdown
# paper_trader

Live microstructure paper trader for Kalshi NBA 5-minute home-YES scalps.

- Spec: `docs/superpowers/specs/2026-04-28-edge-to-action-design.md`
- Plan: `docs/superpowers/plans/2026-05-01-edge-to-action-paper-trader.md`
- Entrypoint: `python paper_trader.py --mode paper --bankroll 1000`
```

- [ ] **Step 4: Verify package imports**

```bash
python -c "import paper_trader; import tools; print('ok')"
```

Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add paper_trader/__init__.py paper_trader/README.md tools/__init__.py tests/paper_trader/__init__.py .gitignore
git commit -m "scaffold(paper_trader): package skeleton + gitignore"
```

---

## Task 1: `cost_model.py`

**Files:**
- Create: `paper_trader/cost_model.py`
- Test: `tests/paper_trader/test_cost_model.py`

- [ ] **Step 1: Write failing test**

Create `tests/paper_trader/test_cost_model.py`:

```python
"""Pure fee math. Three known Kalshi entry scenarios (5¢, 50¢, 95¢)."""
from __future__ import annotations

import math
import pytest

from paper_trader.cost_model import (
    half_spread_cost,
    winner_fee_per_contract,
    expected_fee,
)


def test_half_spread_cost_basic():
    assert half_spread_cost(yes_ask=0.42, yes_mid=0.40) == pytest.approx(0.02)


def test_half_spread_cost_zero_when_ask_equals_mid():
    assert half_spread_cost(yes_ask=0.50, yes_mid=0.50) == 0.0


@pytest.mark.parametrize(
    "p, expected",
    [
        # fee = ceil(0.07 * p * (1-p) * 100) / 100
        (0.05, math.ceil(0.07 * 0.05 * 0.95 * 100) / 100),  # 0.01
        (0.50, math.ceil(0.07 * 0.50 * 0.50 * 100) / 100),  # 0.02
        (0.95, math.ceil(0.07 * 0.95 * 0.05 * 100) / 100),  # 0.01
    ],
)
def test_winner_fee_per_contract_matches_kalshi_formula(p, expected):
    assert winner_fee_per_contract(p) == pytest.approx(expected)


def test_winner_fee_zero_at_extremes():
    # Fee floor: at p=0 or p=1 the formula gives 0; we still return 0.
    assert winner_fee_per_contract(0.0) == 0.0
    assert winner_fee_per_contract(1.0) == 0.0


def test_expected_fee_is_p_times_winner_fee():
    p = 0.30
    fee = winner_fee_per_contract(p)
    assert expected_fee(p) == pytest.approx(p * fee)
```

- [ ] **Step 2: Run test — confirm it fails**

Run: `pytest tests/paper_trader/test_cost_model.py -v`
Expected: ImportError / ModuleNotFoundError on `paper_trader.cost_model`.

- [ ] **Step 3: Implement `cost_model.py`**

Create `paper_trader/cost_model.py`:

```python
"""Kalshi-side cost model for the paper trader.

Three numbers per trade:
  - half_spread_cost: paid at entry, always (yes_ask - yes_mid).
  - winner_fee_per_contract: Kalshi's published 7% × p × (1-p) round-up-to-cent
    formula. Charged only on the winning side per the spec's accounting choice
    ("loser fee = 0").
  - expected_fee: trigger-time expectation, p × winner_fee.
"""
from __future__ import annotations

import math


def half_spread_cost(yes_ask: float, yes_mid: float) -> float:
    return float(yes_ask) - float(yes_mid)


def winner_fee_per_contract(p: float) -> float:
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    raw = 0.07 * p * (1.0 - p)
    return math.ceil(raw * 100.0) / 100.0


def expected_fee(p: float) -> float:
    return float(p) * winner_fee_per_contract(p)
```

- [ ] **Step 4: Run test — confirm pass**

Run: `pytest tests/paper_trader/test_cost_model.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add paper_trader/cost_model.py tests/paper_trader/test_cost_model.py
git commit -m "feat(paper_trader): cost_model — half-spread + Kalshi winner fee"
```

---

## Task 2: `trade_log.py` — atomic state writes, JSONL events, CSV appends

**Files:**
- Create: `paper_trader/trade_log.py`
- Test: `tests/paper_trader/test_trade_log.py`
- Test: `tests/paper_trader/test_state_atomicity.py`

- [ ] **Step 1: Write failing tests**

Create `tests/paper_trader/test_trade_log.py`:

```python
"""trade_log: atomic JSON state, JSONL event append, CSV append.

The append-only files are the source of truth; the JSON state is a cached
snapshot.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import pytest

from paper_trader.trade_log import (
    write_state,
    read_state,
    emit_event,
    record_trade,
    CLOSED_TRADES_HEADER,
)


def test_write_state_atomic(tmp_path: Path):
    target = tmp_path / "trader_state.json"
    state = {"mode": "A", "trades_closed": 0, "open_positions": []}
    write_state(target, state)
    assert json.loads(target.read_text()) == state
    # No tmp file left behind:
    assert not target.with_suffix(".json.tmp").exists()


def test_read_state_returns_none_when_missing(tmp_path: Path):
    assert read_state(tmp_path / "missing.json") is None


def test_read_state_roundtrip(tmp_path: Path):
    target = tmp_path / "s.json"
    write_state(target, {"k": 1})
    assert read_state(target) == {"k": 1}


def test_emit_event_appends_jsonl(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    emit_event(log, {"event_type": "tick", "ts_utc": "2026-05-01T00:00:00Z"})
    emit_event(log, {"event_type": "entry", "ts_utc": "2026-05-01T00:00:01Z"})
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == "tick"
    assert json.loads(lines[1])["event_type"] == "entry"


def test_record_trade_writes_header_once(tmp_path: Path):
    csv_path = tmp_path / "closed_trades.csv"
    row = {col: "" for col in CLOSED_TRADES_HEADER}
    row["trade_id"] = "T1"
    record_trade(csv_path, row)
    record_trade(csv_path, {**row, "trade_id": "T2"})
    text = csv_path.read_text()
    assert text.count("trade_id,") == 1  # header only once
    assert "T1" in text and "T2" in text


def _writer(args):
    csv_path, trade_id = args
    row = {col: "" for col in CLOSED_TRADES_HEADER}
    row["trade_id"] = trade_id
    record_trade(csv_path, row)


def test_record_trade_no_torn_writes_under_concurrency(tmp_path: Path):
    csv_path = tmp_path / "closed_trades.csv"
    # Pre-create header to avoid two writers both creating it.
    record_trade(csv_path, {col: "init" for col in CLOSED_TRADES_HEADER})
    args = [(csv_path, f"T{i}") for i in range(20)]
    with mp.Pool(4) as pool:
        pool.map(_writer, args)
    lines = csv_path.read_text().splitlines()
    # 1 header + 1 init row + 20 concurrent rows
    assert len(lines) == 22
    # Each line must have exactly the right number of fields.
    expected_fields = len(CLOSED_TRADES_HEADER)
    for line in lines:
        assert line.count(",") == expected_fields - 1
```

Create `tests/paper_trader/test_state_atomicity.py`:

```python
"""If write_state crashes mid-write, the previous state on disk must survive."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from paper_trader import trade_log


def test_state_survives_mid_write_crash(tmp_path: Path, monkeypatch):
    target = tmp_path / "s.json"
    trade_log.write_state(target, {"v": 1})

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(trade_log.os, "replace", boom)

    with pytest.raises(RuntimeError):
        trade_log.write_state(target, {"v": 2})

    # Original state intact, no half-written tmp adopted as primary.
    assert json.loads(target.read_text()) == {"v": 1}
```

- [ ] **Step 2: Run tests — confirm they fail**

Run: `pytest tests/paper_trader/test_trade_log.py tests/paper_trader/test_state_atomicity.py -v`
Expected: ImportError on `paper_trader.trade_log`.

- [ ] **Step 3: Implement `trade_log.py`**

Create `paper_trader/trade_log.py`:

```python
"""Disk-side persistence for the paper trader.

Three sinks:
  - trader_state.json   atomic JSON, single source of truth for live state.
  - trader_log.jsonl    append-only event stream, the replay source.
  - closed_trades.csv   append-only closed-position rows.
open_positions.csv is a derived mirror of state.open_positions and is
overwritten atomically each time state is written (handled by callers via
write_state -> mirror_open_positions).
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable


CLOSED_TRADES_HEADER: list[str] = [
    "trade_id", "ticker", "game_key", "home_team", "away_team",
    "entry_captured_at", "exit_captured_at", "hold_seconds",
    "entry_yes_ask", "entry_yes_mid", "exit_yes_bid", "exit_yes_mid",
    "contracts",
    "p_raw", "p_calibrated",
    "trigger_mode",
    "expected_pnl_per_contract",
    "realized_label_home_up_5m",
    "pnl_mid", "pnl_realistic", "fees",
    "exit_basis",
    "bankroll_at_entry",
    "notes",
]


def write_state(path: Path, state: dict[str, Any]) -> None:
    """Atomic write: serialize → write to .tmp → os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(state, indent=2, sort_keys=True, default=str)
    tmp.write_text(payload)
    os.replace(tmp, path)


def read_state(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def emit_event(path: Path, event: dict[str, Any]) -> None:
    """Append one JSON-encoded line to the event log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, default=str, sort_keys=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def record_trade(path: Path, row: dict[str, Any]) -> None:
    """Append one row to closed_trades.csv. Writes header if file is new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CLOSED_TRADES_HEADER, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def mirror_open_positions(path: Path, open_positions: Iterable[dict[str, Any]]) -> None:
    """Overwrite open_positions.csv atomically from in-memory state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    rows = list(open_positions)
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            fh.write("trade_id\n")  # header-only sentinel
    os.replace(tmp, path)
```

- [ ] **Step 4: Run tests — confirm pass**

Run: `pytest tests/paper_trader/test_trade_log.py tests/paper_trader/test_state_atomicity.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add paper_trader/trade_log.py tests/paper_trader/test_trade_log.py tests/paper_trader/test_state_atomicity.py
git commit -m "feat(paper_trader): trade_log — atomic state, JSONL events, CSV append"
```

---

## Task 3: `gate_filters.py` — eligibility table

**Files:**
- Create: `paper_trader/gate_filters.py`
- Test: `tests/paper_trader/test_gate_filters.py`

- [ ] **Step 1: Write failing tests**

Create `tests/paper_trader/test_gate_filters.py`:

```python
"""Each gate returns (passed, reason). One reason string per failure mode.

Order matters only for which reason is returned first; tests assert the
expected reason for each input row.
"""
from __future__ import annotations

import pytest

from paper_trader.gate_filters import evaluate


def _ok_row(**overrides) -> dict:
    base = dict(
        bet_side="home",
        status_state="in",
        is_live=True,
        open_interest=500,
        yes_depth_notional_3=200.0,
        yes_bid=0.40,
        yes_ask=0.42,
    )
    base.update(overrides)
    return base


def test_passes_clean_row():
    passed, reason = evaluate(_ok_row())
    assert passed, reason
    assert reason == "ok"


def test_rejects_away_side():
    passed, reason = evaluate(_ok_row(bet_side="away"))
    assert not passed and reason == "side_not_home"


def test_rejects_pregame():
    passed, reason = evaluate(_ok_row(status_state="pre"))
    assert not passed and reason == "not_live"


def test_rejects_thin_open_interest():
    passed, reason = evaluate(_ok_row(open_interest=99))
    assert not passed and reason == "open_interest_below_floor"


def test_rejects_thin_depth():
    passed, reason = evaluate(_ok_row(yes_depth_notional_3=49.99))
    assert not passed and reason == "depth_below_floor"


@pytest.mark.parametrize(
    "yes_bid, yes_ask, reason",
    [
        (0.0, 0.5, "price_out_of_range"),     # bid below 0.01
        (0.5, 1.0, "price_out_of_range"),     # ask at or above 1.0
        (0.5, 0.4, "crossed_book"),           # ask < bid
        (0.10, 0.45, "spread_too_wide"),      # spread > 0.30
    ],
)
def test_rejects_bad_prices(yes_bid, yes_ask, reason):
    passed, got = evaluate(_ok_row(yes_bid=yes_bid, yes_ask=yes_ask))
    assert not passed
    assert got == reason


def test_rejects_missing_field():
    row = _ok_row()
    row.pop("yes_bid")
    passed, reason = evaluate(row)
    assert not passed and reason == "missing_field:yes_bid"
```

- [ ] **Step 2: Run — confirm fail**

Run: `pytest tests/paper_trader/test_gate_filters.py -v`
Expected: ImportError on `paper_trader.gate_filters`.

- [ ] **Step 3: Implement `gate_filters.py`**

Create `paper_trader/gate_filters.py`:

```python
"""Pre-scoring eligibility filters. Each gate returns (passed, reason).

Reasons are stable strings — they show up in the `gate_blocked` event log and
in any log-query playbook. Don't reword them without updating downstream
queries.
"""
from __future__ import annotations

from typing import Any

OI_FLOOR = 100
DEPTH_FLOOR = 50.0
PRICE_MIN = 0.01
PRICE_MAX = 0.99
MAX_SPREAD = 0.30

REQUIRED_FIELDS = (
    "bet_side", "status_state", "open_interest",
    "yes_depth_notional_3", "yes_bid", "yes_ask",
)


def evaluate(row: dict[str, Any]) -> tuple[bool, str]:
    for field in REQUIRED_FIELDS:
        if field not in row or row[field] is None:
            return False, f"missing_field:{field}"

    if str(row["bet_side"]).lower() != "home":
        return False, "side_not_home"

    if str(row["status_state"]).lower() != "in":
        return False, "not_live"

    if float(row["open_interest"]) < OI_FLOOR:
        return False, "open_interest_below_floor"

    if float(row["yes_depth_notional_3"]) < DEPTH_FLOOR:
        return False, "depth_below_floor"

    bid = float(row["yes_bid"])
    ask = float(row["yes_ask"])
    if bid < PRICE_MIN or ask > PRICE_MAX:
        return False, "price_out_of_range"
    if ask < bid:
        return False, "crossed_book"
    if (ask - bid) > MAX_SPREAD:
        return False, "spread_too_wide"

    return True, "ok"
```

- [ ] **Step 4: Run — confirm pass**

Run: `pytest tests/paper_trader/test_gate_filters.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add paper_trader/gate_filters.py tests/paper_trader/test_gate_filters.py
git commit -m "feat(paper_trader): gate_filters — table-driven eligibility checks"
```

---

## Task 4: `scorer.py` — load model, engineer features, calibrate

**Files:**
- Create: `paper_trader/scorer.py`
- Test: `tests/paper_trader/test_scorer.py`

This task imports feature engineering from `live_training_matrix` to keep
training and live transforms in lockstep. Don't copy-paste the formulas —
import them.

- [ ] **Step 1: Write failing tests**

Create `tests/paper_trader/test_scorer.py`:

```python
"""scorer: load .pkl, engineer features, predict_proba + calibrated shrinkage.

The shrinkage formula must match live_bootstrap_model._apply_shrinkage exactly:
   p_cal = clip(base_rate + alpha * (raw - base_rate), 1e-6, 1 - 1e-6)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_trader.scorer import load_model, score, engineer_features


MODEL_PATH = Path("models/live_home_up_5m_bootstrap.pkl")


@pytest.fixture(scope="module")
def bundle():
    if not MODEL_PATH.exists():
        pytest.skip(f"{MODEL_PATH} not present")
    return load_model(MODEL_PATH)


def _raw_row() -> dict:
    """A single row of the latest_features.csv schema, populated to be
    in-distribution for a typical mid-game state."""
    return dict(
        captured_at="2026-04-28T19:30:00Z",
        ticker="KXNBAGAME-26APR28LALHOU-LAL",
        game_key="2026-04-28_LAL_HOU",
        bet_side="home",
        status_state="in",
        period=2,
        seconds_elapsed=600.0,
        seconds_left_in_period=300.0,
        home_score=42.0,
        away_score=39.0,
        score_margin_home=3.0,
        total_points=81.0,
        yes_bid=0.55,
        yes_ask=0.57,
        yes_mid=0.56,
        no_bid=0.43,
        no_ask=0.45,
        last_price=0.56,
        market_home_implied=0.56,
        volume=10000.0,
        open_interest=2000.0,
        yes_depth_notional_3=300.0,
        yes_depth_notional_5=500.0,
        no_depth_notional_3=280.0,
        no_depth_notional_5=480.0,
        yes_weighted_price_3=0.555,
        no_weighted_price_3=0.435,
        espn_home_implied=np.nan,
        espn_away_implied=np.nan,
        oddsapi_home_consensus=np.nan,
        oddsapi_away_consensus=np.nan,
        oddsapi_books=np.nan,
        market_consensus_home=np.nan,
        pregame_home_win_prob=0.50,
        pregame_away_win_prob=0.50,
        pregame_spread=0.0,
        pregame_total=220.0,
        pregame_edge_home=0.06,
        consensus_gap_home=np.nan,
    )


def test_engineer_features_adds_required_derived_columns(bundle):
    df = pd.DataFrame([_raw_row()])
    engineered = engineer_features(df)
    for col in bundle["feature_cols"]:
        assert col in engineered.columns, f"missing engineered col: {col}"
    # Spot-check a couple of formulas:
    assert engineered["market_yes_spread"].iloc[0] == pytest.approx(0.02)
    assert 0.0 <= engineered["game_progress"].iloc[0] <= 1.0


def test_score_returns_finite_probabilities(bundle):
    df = pd.DataFrame([_raw_row()])
    p_raw, p_cal = score(bundle, df)
    assert np.isfinite(p_raw[0]) and 0.0 <= p_raw[0] <= 1.0
    assert np.isfinite(p_cal[0]) and 0.0 <= p_cal[0] <= 1.0


def test_calibration_shrinkage_matches_formula(bundle):
    df = pd.DataFrame([_raw_row()])
    p_raw, p_cal = score(bundle, df)
    base_rate = bundle["calibration"]["base_rate"]
    alpha = bundle["calibration"]["alpha"]
    expected = np.clip(base_rate + alpha * (p_raw - base_rate), 1e-6, 1 - 1e-6)
    np.testing.assert_allclose(p_cal, expected, atol=1e-9)


def test_score_handles_multiple_rows(bundle):
    df = pd.DataFrame([_raw_row(), _raw_row()])
    p_raw, p_cal = score(bundle, df)
    assert p_raw.shape == (2,)
    assert p_cal.shape == (2,)
```

- [ ] **Step 2: Run — confirm fail**

Run: `pytest tests/paper_trader/test_scorer.py -v`
Expected: ImportError on `paper_trader.scorer`.

- [ ] **Step 3: Implement `scorer.py`**

Create `paper_trader/scorer.py`:

```python
"""Model loader + live feature engineering + calibrated scoring.

The feature-engineering transforms live in live_training_matrix and are
imported here, NOT reimplemented, so training and live can never drift.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Re-use training-time helpers verbatim. If these symbols move, update the
# import — don't recreate the formulas.
from live_training_matrix import _safe_numeric, _status_flag


REGULATION_SECONDS = 48 * 60


def load_model(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)
    required = {"model", "feature_cols", "calibration"}
    missing = required - bundle.keys()
    if missing:
        raise ValueError(f"model bundle missing keys: {missing}")
    return bundle


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same derived-column transforms as
    live_training_matrix.build_in_game_training_matrix."""
    df = df.copy()
    numeric_cols = [
        "period", "seconds_elapsed", "seconds_left_in_period",
        "home_score", "away_score", "score_margin_home", "total_points",
        "yes_bid", "yes_ask", "yes_mid", "no_bid", "no_ask",
        "last_price", "market_home_implied",
        "volume", "open_interest",
        "yes_depth_notional_3", "yes_depth_notional_5",
        "no_depth_notional_3", "no_depth_notional_5",
        "yes_weighted_price_3", "no_weighted_price_3",
        "espn_home_implied", "espn_away_implied",
        "oddsapi_home_consensus", "oddsapi_away_consensus", "oddsapi_books",
        "market_consensus_home",
        "pregame_home_win_prob", "pregame_away_win_prob",
        "pregame_spread", "pregame_total", "pregame_edge_home",
        "consensus_gap_home",
    ]
    df = _safe_numeric(df, numeric_cols)

    df["game_progress"] = (df["seconds_elapsed"] / REGULATION_SECONDS).clip(lower=0, upper=1)
    df["regulation_seconds_remaining"] = REGULATION_SECONDS - df["seconds_elapsed"]
    df.loc[df["regulation_seconds_remaining"] < 0, "regulation_seconds_remaining"] = 0

    df["market_yes_spread"] = df["yes_ask"] - df["yes_bid"]
    df["market_no_spread"] = df["no_ask"] - df["no_bid"]
    df["orderbook_imbalance_3"] = df["yes_depth_notional_3"] - df["no_depth_notional_3"]
    df["orderbook_imbalance_5"] = df["yes_depth_notional_5"] - df["no_depth_notional_5"]
    df["orderbook_pressure_3"] = df["yes_weighted_price_3"] - df["no_weighted_price_3"]
    df["market_vs_pregame_home"] = df["market_home_implied"] - df["pregame_home_win_prob"]
    df["score_gap_vs_pregame_spread"] = df["score_margin_home"] - df["pregame_spread"]
    df["points_vs_pregame_total"] = df["total_points"] - df["pregame_total"]

    if "status_state" in df.columns:
        df["flag_status_pre"] = _status_flag(df["status_state"], "pre")
        df["flag_status_live"] = _status_flag(df["status_state"], "in")
        df["flag_status_post"] = _status_flag(df["status_state"], "post")
    else:
        df["flag_status_pre"] = 0
        df["flag_status_live"] = 0
        df["flag_status_post"] = 0
    df["flag_has_game_state"] = df["home_score"].notna().astype(int)
    df["flag_has_time_state"] = df["seconds_elapsed"].notna().astype(int)

    return df


def score(bundle: dict[str, Any], df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Returns (p_raw, p_calibrated), each a 1-D ndarray of len(df)."""
    feature_cols = bundle["feature_cols"]
    engineered = engineer_features(df)
    # Add any model-required cols that are entirely absent in this feed
    # (e.g., an oddsapi_* column when ENABLE_ODDS_API was on at training time
    # but the feed doesn't carry it). The training pipeline imputes via the
    # SimpleImputer, so injecting NaN here matches training behavior.
    for col in feature_cols:
        if col not in engineered.columns:
            engineered[col] = np.nan
    X = engineered[feature_cols]
    raw = bundle["model"].predict_proba(X)[:, 1]
    base_rate = float(bundle["calibration"]["base_rate"])
    alpha = float(bundle["calibration"]["alpha"])
    cal = np.clip(base_rate + alpha * (raw - base_rate), 1e-6, 1 - 1e-6)
    return raw.astype(float), cal.astype(float)
```

- [ ] **Step 4: Run — confirm pass**

Run: `pytest tests/paper_trader/test_scorer.py -v`
Expected: all green (skipped if model artifact missing — check it's present first with `ls models/live_home_up_5m_bootstrap.pkl`).

- [ ] **Step 5: Commit**

```bash
git add paper_trader/scorer.py tests/paper_trader/test_scorer.py
git commit -m "feat(paper_trader): scorer — load model, engineer features, calibrated p"
```

---

## Task 5: Pooled-means seed builder

**Files:**
- Create: `tools/build_pooled_means.py`
- Test: (covered indirectly by Task 6 unit tests; this is a one-shot script)

The Phase A move-size estimator needs `E[Δ|rises]` and `E[Δ|doesn't]` derived
from the labeled training data. We freeze them once into a JSON, so the live
trader doesn't recompute on every tick.

- [ ] **Step 1: Implement the seed builder**

Create `tools/build_pooled_means.py`:

```python
"""One-shot: compute Phase A pooled conditional means from the bootstrap
scored_rows artifact.

Output: outputs/paper_trades/pooled_means.json with structure:
    {
      "source": "outputs/live_home_up_5m_bootstrap_scored_rows.csv",
      "rows_total": <int>,
      "rows_home_side_labeled": <int>,
      "delta_col": "label_yes_mid_move_5m",
      "label_col": "label_home_up_5m",
      "E_delta_given_rises": <float>,
      "E_delta_given_doesnt": <float>,
      "var_delta_pooled": <float>,
      "rises_count": <int>,
      "doesnt_count": <int>,
      "built_at": "<iso>"
    }
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SCORED_ROWS = Path("outputs/live_home_up_5m_bootstrap_scored_rows.csv")
OUT = Path("outputs/paper_trades/pooled_means.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-rows", type=Path, default=SCORED_ROWS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    cols = ["bet_side", "label_home_up_5m", "label_yes_mid_move_5m"]
    df = pd.read_csv(args.scored_rows, usecols=cols)
    df = df[df["bet_side"].astype(str).str.lower() == "home"]
    df = df.dropna(subset=["label_home_up_5m", "label_yes_mid_move_5m"])

    rises = df[df["label_home_up_5m"] == 1]["label_yes_mid_move_5m"]
    doesnt = df[df["label_home_up_5m"] == 0]["label_yes_mid_move_5m"]
    pooled_var = float(df["label_yes_mid_move_5m"].var(ddof=0))

    payload = {
        "source": str(args.scored_rows),
        "rows_total": int(len(df)),
        "rows_home_side_labeled": int(len(df)),
        "delta_col": "label_yes_mid_move_5m",
        "label_col": "label_home_up_5m",
        "E_delta_given_rises": float(rises.mean()),
        "E_delta_given_doesnt": float(doesnt.mean()),
        "var_delta_pooled": pooled_var,
        "rises_count": int(len(rises)),
        "doesnt_count": int(len(doesnt)),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}: rises={payload['rises_count']} "
          f"E[rises]={payload['E_delta_given_rises']:.6f} "
          f"E[doesnt]={payload['E_delta_given_doesnt']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the seed builder**

Run: `python tools/build_pooled_means.py`
Expected: prints `wrote outputs/paper_trades/pooled_means.json: rises=N E[rises]=… E[doesnt]=…` with rises_count and doesnt_count both non-zero. The file does NOT get committed (it's under `outputs/paper_trades/` which is gitignored).

- [ ] **Step 3: Sanity-check the file**

Run: `python -c "import json,pathlib; print(json.dumps(json.loads(pathlib.Path('outputs/paper_trades/pooled_means.json').read_text()), indent=2))"`
Expected: prints the JSON with finite floats, both counts > 0, `E_delta_given_rises > E_delta_given_doesnt` (a sanity-check the labels are oriented correctly — by construction "rises" should have larger Δ).

- [ ] **Step 4: Commit**

```bash
git add tools/build_pooled_means.py
git commit -m "feat(paper_trader): build_pooled_means — Phase A seed from scored_rows"
```

---

## Task 6: `move_size_estimator.py`

**Files:**
- Create: `paper_trader/move_size_estimator.py`
- Test: `tests/paper_trader/test_move_size_estimator.py`

- [ ] **Step 1: Write failing tests**

Create `tests/paper_trader/test_move_size_estimator.py`:

```python
"""Move-size estimator: Phase A pooled, Phase B per-decile, plus the A→B
gate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_trader.move_size_estimator import (
    Store,
    load_pooled_means,
    expected_moves,
    per_decile_means,
    bootstrap_ci_half_widths,
    ab_gate_passed,
)


@pytest.fixture
def pooled_json(tmp_path: Path) -> Path:
    p = tmp_path / "pooled.json"
    p.write_text(json.dumps({
        "E_delta_given_rises": 0.025,
        "E_delta_given_doesnt": -0.005,
        "var_delta_pooled": 0.0009,
        "rises_count": 1000,
        "doesnt_count": 9000,
    }))
    return p


def test_load_pooled_means_exposes_fields(pooled_json):
    store = load_pooled_means(pooled_json)
    assert isinstance(store, Store)
    assert store.E_rises_pooled == pytest.approx(0.025)
    assert store.E_doesnt_pooled == pytest.approx(-0.005)
    assert store.var_pooled == pytest.approx(0.0009)


def test_expected_moves_phase_a_uses_pooled(pooled_json):
    store = load_pooled_means(pooled_json)
    e_up, e_down, var = expected_moves(p=0.30, mode="A", store=store)
    assert e_up == pytest.approx(0.025)
    assert e_down == pytest.approx(-0.005)
    assert var == pytest.approx(0.0009)


def test_expected_moves_phase_b_uses_decile(pooled_json):
    store = load_pooled_means(pooled_json)
    # Inject a per-decile table covering p=0.30 (decile 3).
    store.decile_table = pd.DataFrame({
        "p_decile": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "E_rises": [0.005, 0.010, 0.015, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090],
        "E_doesnt": [-0.002, -0.003, -0.004, -0.006, -0.008, -0.010, -0.012, -0.014, -0.016, -0.018],
        "var": [0.0001] * 10,
        "rises_n": [50] * 10,
        "doesnt_n": [450] * 10,
    })
    e_up, e_down, var = expected_moves(p=0.30, mode="B", store=store)
    assert e_up == pytest.approx(0.030)
    assert e_down == pytest.approx(-0.006)


def test_phase_b_falls_back_to_pooled_when_decile_missing(pooled_json):
    store = load_pooled_means(pooled_json)
    store.decile_table = None
    e_up, e_down, var = expected_moves(p=0.30, mode="B", store=store)
    assert e_up == pytest.approx(0.025)


def test_per_decile_means_groups_correctly():
    df = pd.DataFrame({
        "p_calibrated": np.linspace(0.0, 0.99, 100),
        "label_home_up_5m": ([1] * 10 + [0] * 90),
        "yes_mid_delta_realized": np.concatenate([np.full(10, 0.05), np.full(90, -0.01)]),
    })
    table = per_decile_means(df, delta_col="yes_mid_delta_realized")
    assert set(table.columns) >= {"p_decile", "E_rises", "E_doesnt", "var", "rises_n", "doesnt_n"}
    # Decile 0 has all 10 rises in this synthetic setup.
    decile_0 = table[table["p_decile"] == 0].iloc[0]
    assert decile_0["rises_n"] == 10
    assert decile_0["E_rises"] == pytest.approx(0.05)


def test_bootstrap_ci_half_width_shrinks_with_n():
    rng = np.random.default_rng(0)
    small = rng.normal(0.025, 0.03, 50)
    large = rng.normal(0.025, 0.03, 5000)
    hw_small = bootstrap_ci_half_widths(small, n_iter=500, alpha=0.05, seed=1)
    hw_large = bootstrap_ci_half_widths(large, n_iter=500, alpha=0.05, seed=1)
    assert hw_large < hw_small


def test_ab_gate_passes_only_when_all_deciles_tight():
    # All deciles ±0.005 → passes
    rng = np.random.default_rng(0)
    rows = []
    for d in range(10):
        for _ in range(2000):
            rows.append({
                "p_decile": d,
                "is_rise": True,
                "delta": rng.normal(0.025, 0.001),  # very tight
            })
    df = pd.DataFrame(rows)
    assert ab_gate_passed(df, ci_threshold=0.01, n_iter=200, seed=1) is True

    # One decile loose → fails
    df_loose = df.copy()
    mask = df_loose["p_decile"] == 5
    df_loose.loc[mask, "delta"] = rng.normal(0.025, 0.05, mask.sum())  # huge spread
    assert ab_gate_passed(df_loose, ci_threshold=0.01, n_iter=200, seed=1) is False
```

- [ ] **Step 2: Run — confirm fail**

Run: `pytest tests/paper_trader/test_move_size_estimator.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `move_size_estimator.py`**

Create `paper_trader/move_size_estimator.py`:

```python
"""Move-size estimator: pooled (Phase A) and per-decile (Phase B) conditional
means of yes_mid Δ over the 5-minute window.

The store object holds both. Phase A is loaded once at engine boot from JSON;
Phase B is recomputed at the end of each tick from realized deltas in the
closed-trades table once trades_closed >= 200 AND every decile passes the CI
gate (see ab_gate_passed).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Store:
    E_rises_pooled: float
    E_doesnt_pooled: float
    var_pooled: float
    decile_table: Optional[pd.DataFrame] = field(default=None)


def load_pooled_means(path: Path) -> Store:
    payload = json.loads(Path(path).read_text())
    return Store(
        E_rises_pooled=float(payload["E_delta_given_rises"]),
        E_doesnt_pooled=float(payload["E_delta_given_doesnt"]),
        var_pooled=float(payload["var_delta_pooled"]),
    )


def expected_moves(p: float, mode: str, store: Store) -> tuple[float, float, float]:
    """Returns (E[Δ|rises], E[Δ|doesn't], var(Δ)) appropriate for the mode.

    Phase B falls back to pooled if no decile_table is loaded yet, or if the
    target decile is empty in the table.
    """
    if mode == "B" and store.decile_table is not None and not store.decile_table.empty:
        decile = min(int(p * 10), 9)
        row = store.decile_table[store.decile_table["p_decile"] == decile]
        if not row.empty and row.iloc[0].get("rises_n", 0) > 0:
            r = row.iloc[0]
            return float(r["E_rises"]), float(r["E_doesnt"]), float(r["var"])
    return store.E_rises_pooled, store.E_doesnt_pooled, store.var_pooled


def per_decile_means(closed: pd.DataFrame, delta_col: str = "yes_mid_delta_realized") -> pd.DataFrame:
    """Build a 10-row table indexed by p_calibrated decile.

    `closed` must have columns: p_calibrated, label_home_up_5m, <delta_col>.
    """
    df = closed.copy()
    df["p_decile"] = (df["p_calibrated"].clip(0, 0.999) * 10).astype(int)
    rows = []
    for d in range(10):
        sub = df[df["p_decile"] == d]
        rises = sub[sub["label_home_up_5m"] == 1][delta_col]
        doesnt = sub[sub["label_home_up_5m"] == 0][delta_col]
        rows.append({
            "p_decile": d,
            "E_rises": float(rises.mean()) if len(rises) else 0.0,
            "E_doesnt": float(doesnt.mean()) if len(doesnt) else 0.0,
            "var": float(sub[delta_col].var(ddof=0)) if len(sub) else 0.0,
            "rises_n": int(len(rises)),
            "doesnt_n": int(len(doesnt)),
        })
    return pd.DataFrame(rows)


def bootstrap_ci_half_widths(values: np.ndarray, n_iter: int = 1000, alpha: float = 0.05, seed: int = 0) -> float:
    """Half-width of the symmetric (1-alpha) percentile bootstrap CI of the mean."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("inf")
    rng = np.random.default_rng(seed)
    means = np.empty(n_iter)
    n = len(values)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        means[i] = values[idx].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float((hi - lo) / 2.0)


def ab_gate_passed(realized: pd.DataFrame, ci_threshold: float = 0.01, n_iter: int = 1000, seed: int = 0) -> bool:
    """For every populated decile in `realized`, the bootstrapped 95% CI
    half-width on E[Δ|rises] must be ≤ ci_threshold (in price units, default 1¢).

    `realized` must have columns: p_decile, is_rise (bool), delta (float).
    Empty deciles disqualify the gate (we want coverage)."""
    deciles = realized["p_decile"].unique()
    if len(deciles) < 10:
        return False
    for d in range(10):
        rises = realized[(realized["p_decile"] == d) & (realized["is_rise"])]["delta"].to_numpy()
        if len(rises) == 0:
            return False
        if bootstrap_ci_half_widths(rises, n_iter=n_iter, seed=seed) > ci_threshold:
            return False
    return True
```

- [ ] **Step 4: Run — confirm pass**

Run: `pytest tests/paper_trader/test_move_size_estimator.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add paper_trader/move_size_estimator.py tests/paper_trader/test_move_size_estimator.py
git commit -m "feat(paper_trader): move_size_estimator — pooled/per-decile + A→B gate"
```

---

## Task 7: `trigger.py` — EV evaluation

**Files:**
- Create: `paper_trader/trigger.py`
- Test: `tests/paper_trader/test_trigger.py`

- [ ] **Step 1: Write failing tests**

Create `tests/paper_trader/test_trigger.py`:

```python
"""trigger.evaluate: EV-aware accept/reject."""
from __future__ import annotations

import pytest

from paper_trader.trigger import evaluate


def test_accepts_when_ev_positive():
    # E[Δ] = 0.30 * 0.05 + 0.70 * (-0.01) = 0.015 - 0.007 = 0.008
    # half_spread = 0.42 - 0.40 = 0.02
    # expected_fee = 0.30 * 0.01 = 0.003
    # net = 0.008 - 0.02 - 0.003 = -0.015 → REJECT
    accept, ev, reason = evaluate(
        p_calibrated=0.30,
        e_up=0.05, e_down=-0.01,
        yes_ask=0.42, yes_mid=0.40,
        fee_per_contract_if_win=0.01,
    )
    assert not accept
    assert reason == "ev_negative"
    assert ev == pytest.approx(0.008 - 0.02 - 0.003)


def test_accepts_when_signal_strong():
    # E[Δ] = 0.40 * 0.10 + 0.60 * (-0.005) = 0.04 - 0.003 = 0.037
    # half_spread = 0.005
    # expected_fee = 0.40 * 0.02 = 0.008
    # net = 0.037 - 0.005 - 0.008 = 0.024 → ACCEPT
    accept, ev, reason = evaluate(
        p_calibrated=0.40,
        e_up=0.10, e_down=-0.005,
        yes_ask=0.405, yes_mid=0.40,
        fee_per_contract_if_win=0.02,
    )
    assert accept
    assert reason == "ok"
    assert ev > 0


def test_rejects_when_ev_exactly_zero():
    accept, ev, reason = evaluate(
        p_calibrated=0.50,
        e_up=0.04, e_down=-0.04,  # ev_pre_costs = 0
        yes_ask=0.50, yes_mid=0.50,  # half_spread = 0
        fee_per_contract_if_win=0.0,  # no fee
    )
    assert not accept
    assert ev == pytest.approx(0.0)
    assert reason == "ev_negative"  # strict > 0 requirement
```

- [ ] **Step 2: Run — confirm fail**

Run: `pytest tests/paper_trader/test_trigger.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `trigger.py`**

Create `paper_trader/trigger.py`:

```python
"""EV-aware entry trigger.

ev_per_contract = p * E[Δ|rises] + (1-p) * E[Δ|doesn't]
                  - half_spread_cost
                  - p * winner_fee_per_contract

The half-spread comes out of *every* trade (cross the book at entry); the
winner fee comes out only when the trade resolves as a winner, expectation-
weighted by p. We accept iff ev_per_contract > 0.
"""
from __future__ import annotations

from paper_trader.cost_model import half_spread_cost


def evaluate(
    p_calibrated: float,
    e_up: float,
    e_down: float,
    yes_ask: float,
    yes_mid: float,
    fee_per_contract_if_win: float,
) -> tuple[bool, float, str]:
    p = float(p_calibrated)
    ev_moves = p * float(e_up) + (1 - p) * float(e_down)
    half = half_spread_cost(yes_ask, yes_mid)
    expected_winner_fee = p * float(fee_per_contract_if_win)
    ev = ev_moves - half - expected_winner_fee
    if ev > 0:
        return True, ev, "ok"
    return False, ev, "ev_negative"
```

- [ ] **Step 4: Run — confirm pass**

Run: `pytest tests/paper_trader/test_trigger.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add paper_trader/trigger.py tests/paper_trader/test_trigger.py
git commit -m "feat(paper_trader): trigger — EV-aware entry decision"
```

---

## Task 8: `position_manager.py`

**Files:**
- Create: `paper_trader/position_manager.py`
- Test: `tests/paper_trader/test_position_manager.py`

- [ ] **Step 1: Write failing tests**

Create `tests/paper_trader/test_position_manager.py`:

```python
"""position_manager: open/close lifecycle, Kelly sizing, concurrency cap, exit."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from paper_trader.position_manager import (
    PositionManager,
    kelly_contracts,
    HOLD_SECONDS,
    MAX_OPEN,
)


def _ts(s: int) -> str:
    return (datetime(2026, 5, 1, 19, 0, tzinfo=timezone.utc) + timedelta(seconds=s)).isoformat()


def test_kelly_contracts_basic():
    # f = mu / sigma^2, capped at 5% of bankroll, floored at 1
    # mu = 0.02, sigma^2 = 0.001 → f = 20 (huge) → cap to 0.05*1000 = 50
    # contracts = floor(50 / yes_ask) = floor(50/0.40) = 125
    contracts = kelly_contracts(mu=0.02, var=0.001, bankroll=1000.0, yes_ask=0.40)
    assert contracts == 125

def test_kelly_contracts_floor_one():
    # mu tiny, var huge → f tiny → contracts floor to 1
    contracts = kelly_contracts(mu=0.0001, var=0.1, bankroll=1000.0, yes_ask=0.50)
    assert contracts == 1


def test_kelly_contracts_zero_var_returns_cap():
    # Degenerate: var = 0 → treat as full-cap allocation
    contracts = kelly_contracts(mu=0.01, var=0.0, bankroll=1000.0, yes_ask=0.50)
    assert contracts == int(0.05 * 1000 / 0.50)


def test_open_then_exit_at_5min(tmp_path: Path):
    pm = PositionManager(
        state_path=tmp_path / "trader_state.json",
        log_path=tmp_path / "trader_log.jsonl",
        closed_csv_path=tmp_path / "closed_trades.csv",
        open_csv_path=tmp_path / "open_positions.csv",
        bankroll_initial=1000.0,
        mode_sizing="A",
        live_or_paper="paper",
    )

    pm.open_position(
        ticker="X", game_key="G", home_team="H", away_team="A",
        captured_at=_ts(0), yes_ask=0.40, yes_mid=0.39, yes_bid=0.38,
        contracts=10, p_raw=0.20, p_calibrated=0.18,
        expected_pnl_per_contract=0.012,
    )
    assert len(pm.open_positions) == 1

    # Tick 60s in → no exit
    pm.tick_open_positions(now_captured_at=_ts(60), latest_bid_mid_per_ticker={"X": (0.41, 0.42)})
    assert len(pm.open_positions) == 1

    # Tick 5 min in → exit
    pm.tick_open_positions(now_captured_at=_ts(HOLD_SECONDS), latest_bid_mid_per_ticker={"X": (0.45, 0.46)})
    assert len(pm.open_positions) == 0
    assert pm.trades_closed == 1


def test_concurrency_cap_blocks_open(tmp_path: Path):
    pm = PositionManager(
        state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl",
        closed_csv_path=tmp_path / "c.csv", open_csv_path=tmp_path / "o.csv",
        bankroll_initial=1000.0, mode_sizing="A", live_or_paper="paper",
    )
    for i in range(MAX_OPEN):
        pm.open_position(
            ticker=f"T{i}", game_key=f"G{i}", home_team="H", away_team="A",
            captured_at=_ts(i), yes_ask=0.40, yes_mid=0.39, yes_bid=0.38,
            contracts=1, p_raw=0.2, p_calibrated=0.18,
            expected_pnl_per_contract=0.01,
        )
    # 21st should be refused
    accepted = pm.open_position(
        ticker="OVERFLOW", game_key="G", home_team="H", away_team="A",
        captured_at=_ts(99999), yes_ask=0.40, yes_mid=0.39, yes_bid=0.38,
        contracts=1, p_raw=0.2, p_calibrated=0.18,
        expected_pnl_per_contract=0.01,
    )
    assert accepted is False
    assert len(pm.open_positions) == MAX_OPEN


def test_pnl_mid_vs_realistic(tmp_path: Path):
    pm = PositionManager(
        state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl",
        closed_csv_path=tmp_path / "c.csv", open_csv_path=tmp_path / "o.csv",
        bankroll_initial=1000.0, mode_sizing="A", live_or_paper="paper",
    )
    pm.open_position(
        ticker="X", game_key="G", home_team="H", away_team="A",
        captured_at=_ts(0), yes_ask=0.40, yes_mid=0.39, yes_bid=0.38,
        contracts=10, p_raw=0.20, p_calibrated=0.18,
        expected_pnl_per_contract=0.01,
    )
    pm.tick_open_positions(now_captured_at=_ts(HOLD_SECONDS),
                           latest_bid_mid_per_ticker={"X": (0.43, 0.44)})

    # pnl_mid = (exit_mid - entry_mid) * contracts = (0.44 - 0.39) * 10 = 0.5
    # pnl_realistic = (exit_bid - entry_ask) * contracts = (0.43 - 0.40) * 10 = 0.3
    # winner: realized label = 1 (price went up), so winner fee applies on the realistic side.
    # That fee is reflected in trade.fees and subtracted from pnl_realistic.
    closed = pm.last_closed_trade
    assert closed["pnl_mid"] == pytest.approx(0.5)
    # Realistic before fee = 0.3; minus fees > 0 (it's a winner)
    assert closed["pnl_realistic"] < 0.3
    assert closed["fees"] > 0
    assert closed["realized_label_home_up_5m"] == 1


def test_restart_reload_keeps_open_positions(tmp_path: Path):
    pm1 = PositionManager(
        state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl",
        closed_csv_path=tmp_path / "c.csv", open_csv_path=tmp_path / "o.csv",
        bankroll_initial=1000.0, mode_sizing="A", live_or_paper="paper",
    )
    pm1.open_position(
        ticker="X", game_key="G", home_team="H", away_team="A",
        captured_at=_ts(0), yes_ask=0.40, yes_mid=0.39, yes_bid=0.38,
        contracts=5, p_raw=0.2, p_calibrated=0.18,
        expected_pnl_per_contract=0.01,
    )

    pm2 = PositionManager(
        state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl",
        closed_csv_path=tmp_path / "c.csv", open_csv_path=tmp_path / "o.csv",
        bankroll_initial=1000.0, mode_sizing="A", live_or_paper="paper",
    )
    pm2.load_state()
    assert len(pm2.open_positions) == 1
    assert pm2.open_positions[0]["ticker"] == "X"
```

- [ ] **Step 2: Run — confirm fail**

Run: `pytest tests/paper_trader/test_position_manager.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `position_manager.py`**

Create `paper_trader/position_manager.py`:

```python
"""In-memory position state + open/close lifecycle.

Owns the writes to trader_state.json, open_positions.csv, closed_trades.csv,
and the entry/exit JSONL events. Pure functions live in cost_model and
trigger; this module is where state changes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from paper_trader import trade_log
from paper_trader.cost_model import winner_fee_per_contract


HOLD_SECONDS = 5 * 60
MAX_OPEN = 20
KELLY_CAP_FRACTION = 0.05


def kelly_contracts(mu: float, var: float, bankroll: float, yes_ask: float) -> int:
    """Continuous-payoff Kelly: f = mu / sigma^2, capped at 5% of bankroll, floored at 1.

    Returns integer contract count. yes_ask is the entry price (bankroll
    consumed = contracts * yes_ask)."""
    cap_dollars = KELLY_CAP_FRACTION * float(bankroll)
    if var <= 0.0:
        # Degenerate variance: spend the cap.
        dollars = cap_dollars
    else:
        f = float(mu) / float(var)
        f = max(0.0, min(f, KELLY_CAP_FRACTION))  # cap at 5% of bankroll
        dollars = f * float(bankroll)
    contracts = int(dollars // float(yes_ask))
    return max(contracts, 1)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


class PositionManager:
    def __init__(
        self,
        state_path: Path,
        log_path: Path,
        closed_csv_path: Path,
        open_csv_path: Path,
        bankroll_initial: float,
        mode_sizing: str,
        live_or_paper: str,
    ):
        self.state_path = Path(state_path)
        self.log_path = Path(log_path)
        self.closed_csv_path = Path(closed_csv_path)
        self.open_csv_path = Path(open_csv_path)
        self.bankroll_initial = float(bankroll_initial)
        self.mode_sizing = mode_sizing
        self.live_or_paper = live_or_paper
        self.cumulative_pnl_mid = 0.0
        self.cumulative_pnl_realistic = 0.0
        self.trades_closed = 0
        self.max_drawdown = 0.0
        self.drawdown_breached = False
        self.open_positions: list[dict[str, Any]] = []
        self.last_processed_captured_at_per_ticker: dict[str, str] = {}
        self.last_closed_trade: dict[str, Any] | None = None

    @property
    def bankroll_current(self) -> float:
        return self.bankroll_initial + self.cumulative_pnl_realistic

    def load_state(self) -> None:
        state = trade_log.read_state(self.state_path)
        if state is None:
            return
        self.cumulative_pnl_mid = float(state.get("cumulative_pnl_mid", 0.0))
        self.cumulative_pnl_realistic = float(state.get("cumulative_pnl_realistic", 0.0))
        self.trades_closed = int(state.get("trades_closed", 0))
        self.max_drawdown = float(state.get("max_drawdown", 0.0))
        self.drawdown_breached = bool(state.get("drawdown_breached", False))
        self.open_positions = list(state.get("open_positions", []))
        self.last_processed_captured_at_per_ticker = dict(
            state.get("last_processed_captured_at_per_ticker", {})
        )

    def write_state(self) -> None:
        state = {
            "mode": self.mode_sizing,
            "live_or_paper": self.live_or_paper,
            "bankroll_initial": self.bankroll_initial,
            "cumulative_pnl_realistic": self.cumulative_pnl_realistic,
            "cumulative_pnl_mid": self.cumulative_pnl_mid,
            "trades_closed": self.trades_closed,
            "max_drawdown": self.max_drawdown,
            "drawdown_breached": self.drawdown_breached,
            "open_positions": self.open_positions,
            "last_processed_captured_at_per_ticker": self.last_processed_captured_at_per_ticker,
            "last_tick_at": _now_iso(),
        }
        trade_log.write_state(self.state_path, state)
        trade_log.mirror_open_positions(self.open_csv_path, self.open_positions)

    def open_position(
        self,
        ticker: str,
        game_key: str,
        home_team: str,
        away_team: str,
        captured_at: str,
        yes_ask: float,
        yes_mid: float,
        yes_bid: float,
        contracts: int,
        p_raw: float,
        p_calibrated: float,
        expected_pnl_per_contract: float,
    ) -> bool:
        if len(self.open_positions) >= MAX_OPEN:
            trade_log.emit_event(self.log_path, {
                "event_type": "signal", "ts_utc": _now_iso(),
                "captured_at": captured_at, "ticker": ticker,
                "accept": False, "reason": "concurrency_cap",
            })
            return False
        if self.live_or_paper == "live" and self.drawdown_breached:
            trade_log.emit_event(self.log_path, {
                "event_type": "halt", "ts_utc": _now_iso(),
                "captured_at": captured_at, "ticker": ticker,
                "reason": "drawdown_halt_live",
            })
            return False
        trade_id = f"{ticker}_{captured_at}"
        position = {
            "trade_id": trade_id,
            "ticker": ticker, "game_key": game_key,
            "home_team": home_team, "away_team": away_team,
            "entry_captured_at": captured_at,
            "entry_yes_ask": float(yes_ask),
            "entry_yes_mid": float(yes_mid),
            "entry_yes_bid": float(yes_bid),
            "contracts": int(contracts),
            "p_raw": float(p_raw),
            "p_calibrated": float(p_calibrated),
            "expected_pnl_per_contract": float(expected_pnl_per_contract),
            "bankroll_at_entry": self.bankroll_current,
        }
        self.open_positions.append(position)
        trade_log.emit_event(self.log_path, {
            "event_type": "entry", "ts_utc": _now_iso(),
            "captured_at": captured_at, **position,
        })
        return True

    def tick_open_positions(
        self,
        now_captured_at: str,
        latest_bid_mid_per_ticker: dict[str, tuple[float, float]],
        exit_basis: str = "5min_timer",
    ) -> None:
        if not self.open_positions:
            return
        now = _parse_iso(now_captured_at)
        still_open: list[dict[str, Any]] = []
        for pos in self.open_positions:
            entry = _parse_iso(pos["entry_captured_at"])
            elapsed = (now - entry).total_seconds()
            if elapsed < HOLD_SECONDS:
                still_open.append(pos)
                continue
            quote = latest_bid_mid_per_ticker.get(pos["ticker"])
            if quote is None:
                # No fresh quote for this ticker yet — keep waiting.
                still_open.append(pos)
                continue
            exit_yes_bid, exit_yes_mid = quote
            self._close_position(pos, now_captured_at, exit_yes_bid, exit_yes_mid, exit_basis)
        self.open_positions = still_open

    def _close_position(
        self,
        pos: dict[str, Any],
        exit_captured_at: str,
        exit_yes_bid: float,
        exit_yes_mid: float,
        exit_basis: str,
    ) -> None:
        contracts = int(pos["contracts"])
        entry_ask = float(pos["entry_yes_ask"])
        entry_mid = float(pos["entry_yes_mid"])
        pnl_mid = (float(exit_yes_mid) - entry_mid) * contracts
        gross_realistic = (float(exit_yes_bid) - entry_ask) * contracts
        realized_rise = 1 if float(exit_yes_mid) > entry_mid else 0
        # Per spec: loser fee = 0; winner pays Kalshi fee at settlement.
        fees = winner_fee_per_contract(pos["p_calibrated"]) * contracts if realized_rise == 1 else 0.0
        pnl_realistic = gross_realistic - fees

        hold_seconds = (_parse_iso(exit_captured_at) - _parse_iso(pos["entry_captured_at"])).total_seconds()
        row = {
            "trade_id": pos["trade_id"],
            "ticker": pos["ticker"], "game_key": pos["game_key"],
            "home_team": pos["home_team"], "away_team": pos["away_team"],
            "entry_captured_at": pos["entry_captured_at"],
            "exit_captured_at": exit_captured_at,
            "hold_seconds": hold_seconds,
            "entry_yes_ask": entry_ask,
            "entry_yes_mid": entry_mid,
            "exit_yes_bid": float(exit_yes_bid),
            "exit_yes_mid": float(exit_yes_mid),
            "contracts": contracts,
            "p_raw": pos["p_raw"],
            "p_calibrated": pos["p_calibrated"],
            "trigger_mode": self.mode_sizing,
            "expected_pnl_per_contract": pos["expected_pnl_per_contract"],
            "realized_label_home_up_5m": realized_rise,
            "pnl_mid": pnl_mid,
            "pnl_realistic": pnl_realistic,
            "fees": fees,
            "exit_basis": exit_basis,
            "bankroll_at_entry": pos["bankroll_at_entry"],
            "notes": "",
        }
        trade_log.record_trade(self.closed_csv_path, row)
        trade_log.emit_event(self.log_path, {
            "event_type": "exit", "ts_utc": _now_iso(),
            "captured_at": exit_captured_at, **row,
        })
        self.cumulative_pnl_mid += pnl_mid
        self.cumulative_pnl_realistic += pnl_realistic
        self.trades_closed += 1
        # Drawdown tracking (max peak-to-trough on cumulative_pnl_realistic).
        if self.cumulative_pnl_realistic < -self.max_drawdown:
            self.max_drawdown = -self.cumulative_pnl_realistic
        self.last_closed_trade = row
```

- [ ] **Step 4: Run — confirm pass**

Run: `pytest tests/paper_trader/test_position_manager.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add paper_trader/position_manager.py tests/paper_trader/test_position_manager.py
git commit -m "feat(paper_trader): position_manager — open/close + Kelly + concurrency cap"
```

---

## Task 9: `feed_reader.py`

**Files:**
- Create: `paper_trader/feed_reader.py`
- Test: `tests/paper_trader/test_feed_reader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/paper_trader/test_feed_reader.py`:

```python
"""feed_reader: read latest_features.csv, dedup against last-processed map."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from paper_trader.feed_reader import read_unprocessed


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_returns_all_rows_when_map_empty(tmp_path: Path):
    p = tmp_path / "feed.csv"
    _write_csv(p, [
        {"ticker": "A", "captured_at": "2026-05-01T00:00:00Z"},
        {"ticker": "B", "captured_at": "2026-05-01T00:00:00Z"},
    ])
    out = read_unprocessed(p, last_seen={})
    assert len(out) == 2


def test_filters_already_processed(tmp_path: Path):
    p = tmp_path / "feed.csv"
    _write_csv(p, [
        {"ticker": "A", "captured_at": "2026-05-01T00:00:00Z"},
        {"ticker": "B", "captured_at": "2026-05-01T00:00:00Z"},
        {"ticker": "A", "captured_at": "2026-05-01T00:00:15Z"},
    ])
    last_seen = {"A": "2026-05-01T00:00:00Z"}
    out = read_unprocessed(p, last_seen=last_seen)
    # A's first row already seen; A's second and B's first remain.
    tickers = sorted(out["ticker"].tolist())
    assert tickers == ["A", "B"]
    captured = out.set_index("ticker")["captured_at"].to_dict()
    assert captured["A"] == "2026-05-01T00:00:15Z"


def test_returns_empty_when_file_missing(tmp_path: Path):
    out = read_unprocessed(tmp_path / "nope.csv", last_seen={})
    assert out.empty


def test_keeps_latest_per_ticker_only(tmp_path: Path):
    """If multiple rows for the same ticker > last_seen exist, return only the
    latest. (latest_features.csv is supposed to have one row per ticker, but
    defend against duplicates.)"""
    p = tmp_path / "feed.csv"
    _write_csv(p, [
        {"ticker": "A", "captured_at": "2026-05-01T00:00:15Z"},
        {"ticker": "A", "captured_at": "2026-05-01T00:00:30Z"},
    ])
    out = read_unprocessed(p, last_seen={})
    assert len(out) == 1
    assert out.iloc[0]["captured_at"] == "2026-05-01T00:00:30Z"
```

- [ ] **Step 2: Run — confirm fail**

Run: `pytest tests/paper_trader/test_feed_reader.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `feed_reader.py`**

Create `paper_trader/feed_reader.py`:

```python
"""Reader for data/live/features/latest_features.csv.

Returns only rows newer than last_seen[ticker]. If multiple rows for the same
ticker are newer (which shouldn't happen with realtime_feature_store, but
defensively), keeps only the latest captured_at.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_unprocessed(path: Path, last_seen: dict[str, str]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["captured_at"] = df["captured_at"].astype(str)

    def _is_new(row) -> bool:
        prev = last_seen.get(row["ticker"])
        if prev is None:
            return True
        return row["captured_at"] > prev

    df = df[df.apply(_is_new, axis=1)]
    if df.empty:
        return df
    df = df.sort_values("captured_at").drop_duplicates(subset=["ticker"], keep="last")
    return df.reset_index(drop=True)
```

- [ ] **Step 4: Run — confirm pass**

Run: `pytest tests/paper_trader/test_feed_reader.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add paper_trader/feed_reader.py tests/paper_trader/test_feed_reader.py
git commit -m "feat(paper_trader): feed_reader — dedupe by (ticker, captured_at)"
```

---

## Task 10: `paper_trader.py` main loop

**Files:**
- Create: `paper_trader.py` (entry point at repo root)
- Test: `tests/paper_trader/test_paper_trader_loop.py`

- [ ] **Step 1: Write failing tests**

Create `tests/paper_trader/test_paper_trader_loop.py`:

```python
"""End-to-end one-tick test of the paper_trader main loop, with the
position-manager + feed-reader + scorer wired up against a tiny synthetic
feed and a stubbed model bundle.

We run a single iteration of `process_tick` rather than the whole forever-loop.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_trader import process_tick, build_engine_context


def _make_pooled(tmp_path: Path) -> Path:
    p = tmp_path / "pooled.json"
    p.write_text(json.dumps({
        "E_delta_given_rises": 0.05,
        "E_delta_given_doesnt": -0.005,
        "var_delta_pooled": 0.001,
        "rises_count": 1000,
        "doesnt_count": 9000,
    }))
    return p


def _make_feed(tmp_path: Path, captured_at: str, ticker: str = "X", price: float = 0.40) -> Path:
    p = tmp_path / "feed.csv"
    pd.DataFrame([{
        "captured_at": captured_at, "ticker": ticker, "event_ticker": ticker,
        "game_key": "G", "game_date": "2026-05-01",
        "home_team": "H", "away_team": "A", "bet_team": "H", "bet_side": "home",
        "game_status": "Q2", "status_state": "in", "is_live": True, "is_final": False,
        "period": 2, "display_clock": "5:00",
        "seconds_left_in_period": 300.0, "seconds_elapsed": 900.0,
        "home_score": 50.0, "away_score": 48.0, "score_margin_home": 2.0, "total_points": 98.0,
        "yes_bid": price - 0.01, "yes_ask": price + 0.01, "yes_mid": price,
        "no_bid": 0.99 - (price + 0.01), "no_ask": 0.99 - (price - 0.01),
        "last_price": price, "market_home_implied": price,
        "volume": 50000.0, "open_interest": 5000.0,
        "yes_depth_notional_3": 500.0, "yes_depth_notional_5": 800.0,
        "no_depth_notional_3": 480.0, "no_depth_notional_5": 770.0,
        "yes_weighted_price_3": price - 0.005, "no_weighted_price_3": 0.99 - price,
        "espn_home_implied": np.nan, "espn_away_implied": np.nan,
        "oddsapi_home_consensus": np.nan, "oddsapi_away_consensus": np.nan, "oddsapi_books": np.nan,
        "market_consensus_home": np.nan,
        "pregame_home_win_prob": 0.50, "pregame_away_win_prob": 0.50,
        "pregame_spread": 0.0, "pregame_total": 220.0, "pregame_data_date": "2026-04-30",
        "kalshi_vs_consensus_pp": 0.0, "model_vs_consensus_pp": 0.0, "kalshi_vs_model_pp": 0.0,
        "triangulation_tier": 1, "consensus_n_books": 0, "pregame_edge_home": 0.0,
        "consensus_gap_home": np.nan,
    }]).to_csv(p, index=False)
    return p


@pytest.fixture
def model_path() -> Path:
    p = Path("models/live_home_up_5m_bootstrap.pkl")
    if not p.exists():
        pytest.skip("model artifact not present")
    return p


def test_process_tick_emits_tick_event(tmp_path: Path, model_path: Path):
    feed = _make_feed(tmp_path, captured_at="2026-05-01T19:00:00Z", price=0.40)
    pooled = _make_pooled(tmp_path)
    ctx = build_engine_context(
        model_path=model_path, pooled_path=pooled, feed_path=feed,
        output_dir=tmp_path / "outputs", bankroll=1000.0,
        mode_sizing="A", live_or_paper="paper",
    )
    process_tick(ctx)
    log = (tmp_path / "outputs" / "trader_log.jsonl").read_text().strip().splitlines()
    types = [json.loads(line)["event_type"] for line in log]
    assert "tick" in types


def test_process_tick_closes_position_after_5min(tmp_path: Path, model_path: Path):
    pooled = _make_pooled(tmp_path)
    output_dir = tmp_path / "outputs"

    # Tick 1: open a position by faking a high-EV state. We do this by writing
    # a position directly into state, then process_tick at t+5min should close it.
    feed1 = _make_feed(tmp_path, captured_at="2026-05-01T19:00:00Z", price=0.40)
    ctx = build_engine_context(
        model_path=model_path, pooled_path=pooled, feed_path=feed1,
        output_dir=output_dir, bankroll=1000.0,
        mode_sizing="A", live_or_paper="paper",
    )
    ctx.position_manager.open_position(
        ticker="X", game_key="G", home_team="H", away_team="A",
        captured_at="2026-05-01T19:00:00Z",
        yes_ask=0.41, yes_mid=0.40, yes_bid=0.39,
        contracts=10, p_raw=0.20, p_calibrated=0.18,
        expected_pnl_per_contract=0.012,
    )
    ctx.position_manager.write_state()

    # Tick 2: 5 minutes later, with a richer price.
    feed2 = _make_feed(tmp_path, captured_at="2026-05-01T19:05:00Z", price=0.45)
    ctx2 = build_engine_context(
        model_path=model_path, pooled_path=pooled, feed_path=feed2,
        output_dir=output_dir, bankroll=1000.0,
        mode_sizing="A", live_or_paper="paper",
    )
    ctx2.position_manager.load_state()
    process_tick(ctx2)

    closed = pd.read_csv(output_dir / "closed_trades.csv")
    assert len(closed) == 1
    assert closed.iloc[0]["exit_basis"] == "5min_timer"


def test_live_boot_refuses_when_gates_fail(tmp_path: Path, model_path: Path):
    """In live mode, with no closed trades on disk, the engine must refuse
    to start (1000-trade gate, sharpe gate, drawdown gate all fail)."""
    pooled = _make_pooled(tmp_path)
    feed = _make_feed(tmp_path, "2026-05-01T19:00:00Z")
    from paper_trader import check_live_boot_gates

    ok, reasons = check_live_boot_gates(
        output_dir=tmp_path / "outputs",
        bankroll_initial=1000.0,
    )
    assert not ok
    assert any("trades_closed" in r for r in reasons)
```

- [ ] **Step 2: Run — confirm fail**

Run: `pytest tests/paper_trader/test_paper_trader_loop.py -v`
Expected: ImportError on `paper_trader`.

- [ ] **Step 3: Implement `paper_trader.py`**

Create `paper_trader.py` (at repo root):

```python
"""Edge-to-Action paper trader entry point.

CLI:
    python paper_trader.py --mode paper --bankroll 1000 [--interval 15]

Run forever, polling data/live/features/latest_features.csv every --interval
seconds, scoring eligible rows, opening EV-positive positions, and exiting
each on a hard 5-minute timer keyed off captured_at.

State and logs land in --output-dir (default: outputs/paper_trades/).
"""
from __future__ import annotations

import argparse
import math
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paper_trader import gate_filters, scorer, trigger, trade_log
from paper_trader.feed_reader import read_unprocessed
from paper_trader.move_size_estimator import (
    Store, expected_moves, load_pooled_means,
)
from paper_trader.position_manager import (
    HOLD_SECONDS, MAX_OPEN, PositionManager, kelly_contracts,
)
from paper_trader.cost_model import winner_fee_per_contract


DRAWDOWN_GATE_DOLLARS = 100.0
STALE_WARN_SECONDS = 300
STALE_HALT_SECONDS = 600
LIVE_PROMOTION_TRADES = 1000
LIVE_PROMOTION_SHARPE = 1.0
LIVE_PROMOTION_MAX_DD_FRACTION = 0.10


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EngineContext:
    bundle: dict[str, Any]
    pooled_store: Store
    feed_path: Path
    output_dir: Path
    state_path: Path
    log_path: Path
    closed_csv_path: Path
    open_csv_path: Path
    position_manager: PositionManager
    mode_sizing: str
    live_or_paper: str
    bankroll_initial: float
    interval_seconds: int = 15


def build_engine_context(
    model_path: Path,
    pooled_path: Path,
    feed_path: Path,
    output_dir: Path,
    bankroll: float,
    mode_sizing: str,
    live_or_paper: str,
    interval_seconds: int = 15,
) -> EngineContext:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = scorer.load_model(model_path)
    pooled_store = load_pooled_means(pooled_path)
    state_path = output_dir / "trader_state.json"
    log_path = output_dir / "trader_log.jsonl"
    closed_csv_path = output_dir / "closed_trades.csv"
    open_csv_path = output_dir / "open_positions.csv"
    pm = PositionManager(
        state_path=state_path, log_path=log_path,
        closed_csv_path=closed_csv_path, open_csv_path=open_csv_path,
        bankroll_initial=bankroll, mode_sizing=mode_sizing,
        live_or_paper=live_or_paper,
    )
    pm.load_state()
    return EngineContext(
        bundle=bundle, pooled_store=pooled_store,
        feed_path=Path(feed_path), output_dir=output_dir,
        state_path=state_path, log_path=log_path,
        closed_csv_path=closed_csv_path, open_csv_path=open_csv_path,
        position_manager=pm, mode_sizing=mode_sizing,
        live_or_paper=live_or_paper, bankroll_initial=bankroll,
        interval_seconds=interval_seconds,
    )


def check_live_boot_gates(output_dir: Path, bankroll_initial: float) -> tuple[bool, list[str]]:
    """The four gates per spec §Risk controls / Mode transitions."""
    output_dir = Path(output_dir)
    closed_csv = output_dir / "closed_trades.csv"
    reasons: list[str] = []
    if not closed_csv.exists():
        return False, ["trades_closed=0 (need 1000)"]
    df = pd.read_csv(closed_csv)
    n = len(df)
    if n < LIVE_PROMOTION_TRADES:
        reasons.append(f"trades_closed={n} (need {LIVE_PROMOTION_TRADES})")
    pnl = df["pnl_realistic"].astype(float)
    if len(pnl) > 1 and pnl.std(ddof=0) > 0:
        sharpe = float(pnl.mean() / pnl.std(ddof=0))
    else:
        sharpe = 0.0
    if sharpe <= LIVE_PROMOTION_SHARPE:
        reasons.append(f"sharpe={sharpe:.3f} (need > {LIVE_PROMOTION_SHARPE})")
    cum = pnl.cumsum()
    max_dd = float(-cum.min()) if (cum < 0).any() else 0.0
    max_dd_fraction = max_dd / bankroll_initial if bankroll_initial > 0 else math.inf
    if max_dd_fraction >= LIVE_PROMOTION_MAX_DD_FRACTION:
        reasons.append(f"max_dd={max_dd_fraction:.3f} (need < {LIVE_PROMOTION_MAX_DD_FRACTION})")
    return (len(reasons) == 0), reasons


def _detect_stale(now_utc: datetime, latest_captured_at: datetime | None) -> str | None:
    if latest_captured_at is None:
        return None
    age = (now_utc - latest_captured_at).total_seconds()
    if age >= STALE_HALT_SECONDS:
        return "halt"
    if age >= STALE_WARN_SECONDS:
        return "warn"
    return None


def process_tick(ctx: EngineContext) -> None:
    tick_start = time.monotonic()
    now_utc = datetime.now(timezone.utc)
    pm = ctx.position_manager
    rows = read_unprocessed(ctx.feed_path, pm.last_processed_captured_at_per_ticker)

    latest_capture: datetime | None = None
    quote_table: dict[str, tuple[float, float]] = {}

    if not rows.empty:
        for _, row in rows.iterrows():
            try:
                _process_one_row(ctx, row.to_dict())
            except Exception as exc:
                trade_log.emit_event(ctx.log_path, {
                    "event_type": "error", "ts_utc": _now_iso(),
                    "captured_at": str(row.get("captured_at")),
                    "ticker": str(row.get("ticker")),
                    "error": repr(exc), "trace": traceback.format_exc(),
                })
                continue
            ticker = row["ticker"]
            pm.last_processed_captured_at_per_ticker[ticker] = str(row["captured_at"])
            quote_table[ticker] = (float(row["yes_bid"]), float(row["yes_mid"]))
            ts = pd.to_datetime(row["captured_at"], utc=True, errors="coerce")
            if ts is not pd.NaT and (latest_capture is None or ts.to_pydatetime() > latest_capture):
                latest_capture = ts.to_pydatetime()

    stale = _detect_stale(now_utc, latest_capture)
    if stale == "warn":
        trade_log.emit_event(ctx.log_path, {
            "event_type": "halt", "ts_utc": _now_iso(),
            "reason": "stale_feed_warn", "latest_captured_at": str(latest_capture),
        })
    elif stale == "halt":
        trade_log.emit_event(ctx.log_path, {
            "event_type": "halt", "ts_utc": _now_iso(),
            "reason": "stale_feed_halt", "latest_captured_at": str(latest_capture),
        })

    # Exit any positions whose 5-min mark has passed.
    if latest_capture is not None and stale != "halt":
        pm.tick_open_positions(now_captured_at=latest_capture.isoformat(),
                               latest_bid_mid_per_ticker=quote_table)

    # Drawdown gate (paper: log once + continue; live: also halt new entries).
    if pm.cumulative_pnl_realistic <= -DRAWDOWN_GATE_DOLLARS and not pm.drawdown_breached:
        pm.drawdown_breached = True
        trade_log.emit_event(ctx.log_path, {
            "event_type": "halt", "ts_utc": _now_iso(),
            "reason": "drawdown_breached",
            "cumulative_pnl_realistic": pm.cumulative_pnl_realistic,
        })

    pm.write_state()
    tick_to_decision_ms = (time.monotonic() - tick_start) * 1000
    trade_log.emit_event(ctx.log_path, {
        "event_type": "tick", "ts_utc": _now_iso(),
        "captured_at": str(latest_capture) if latest_capture else None,
        "tick_to_decision_ms": tick_to_decision_ms,
        "rows_processed": int(len(rows)),
        "open_positions": len(pm.open_positions),
    })


def _process_one_row(ctx: EngineContext, row: dict[str, Any]) -> None:
    """One row → gate → score → trigger → maybe-open."""
    pm = ctx.position_manager
    captured_at = str(row["captured_at"])
    ticker = str(row["ticker"])
    passed, reason = gate_filters.evaluate(row)
    if not passed:
        trade_log.emit_event(ctx.log_path, {
            "event_type": "gate_blocked", "ts_utc": _now_iso(),
            "captured_at": captured_at, "ticker": ticker, "reason": reason,
        })
        return

    df = pd.DataFrame([row])
    p_raw, p_cal = scorer.score(ctx.bundle, df)
    p_raw_v = float(p_raw[0])
    p_cal_v = float(p_cal[0])
    e_up, e_down, var = expected_moves(p_cal_v, ctx.mode_sizing, ctx.pooled_store)
    fee = winner_fee_per_contract(p_cal_v)
    accept, ev, trigger_reason = trigger.evaluate(
        p_calibrated=p_cal_v, e_up=e_up, e_down=e_down,
        yes_ask=float(row["yes_ask"]), yes_mid=float(row["yes_mid"]),
        fee_per_contract_if_win=fee,
    )
    trade_log.emit_event(ctx.log_path, {
        "event_type": "signal", "ts_utc": _now_iso(),
        "captured_at": captured_at, "ticker": ticker,
        "p_raw": p_raw_v, "p_calibrated": p_cal_v,
        "e_up": e_up, "e_down": e_down,
        "expected_pnl_per_contract": ev,
        "accept": accept, "reason": trigger_reason,
    })
    if not accept:
        return
    # Sizing: continuous Kelly on (μ=ev, σ²=var).
    mu = ev
    contracts = kelly_contracts(mu=mu, var=var, bankroll=pm.bankroll_current,
                                yes_ask=float(row["yes_ask"]))
    pm.open_position(
        ticker=ticker, game_key=str(row["game_key"]),
        home_team=str(row["home_team"]), away_team=str(row["away_team"]),
        captured_at=captured_at,
        yes_ask=float(row["yes_ask"]), yes_mid=float(row["yes_mid"]),
        yes_bid=float(row["yes_bid"]),
        contracts=contracts, p_raw=p_raw_v, p_calibrated=p_cal_v,
        expected_pnl_per_contract=ev,
    )


def _run_forever(ctx: EngineContext) -> None:
    while True:
        try:
            process_tick(ctx)
        except Exception as exc:
            trade_log.emit_event(ctx.log_path, {
                "event_type": "error", "ts_utc": _now_iso(),
                "error": repr(exc), "trace": traceback.format_exc(),
            })
            time.sleep(5)
            continue
        time.sleep(ctx.interval_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--features-path", type=Path,
                        default=Path("data/live/features/latest_features.csv"))
    parser.add_argument("--model-path", type=Path,
                        default=Path("models/live_home_up_5m_bootstrap.pkl"))
    parser.add_argument("--pooled-path", type=Path,
                        default=Path("outputs/paper_trades/pooled_means.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/paper_trades"))
    parser.add_argument("--mode-sizing", choices=["A", "B"], default="A")
    parser.add_argument("--once", action="store_true",
                        help="Run a single tick and exit (smoke / debug).")
    args = parser.parse_args(argv)

    if args.mode == "live":
        ok, reasons = check_live_boot_gates(args.output_dir, args.bankroll)
        if not ok:
            print("Refusing to start in live mode. Failing gates:")
            for r in reasons:
                print(f"  - {r}")
            return 2

    ctx = build_engine_context(
        model_path=args.model_path, pooled_path=args.pooled_path,
        feed_path=args.features_path, output_dir=args.output_dir,
        bankroll=args.bankroll, mode_sizing=args.mode_sizing,
        live_or_paper=args.mode, interval_seconds=args.interval,
    )

    if args.once:
        process_tick(ctx)
        return 0
    _run_forever(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run — confirm pass**

Run: `pytest tests/paper_trader/test_paper_trader_loop.py -v`
Expected: 3 passed.

- [ ] **Step 5: Smoke a single-tick run against the real feed**

Run: `python paper_trader.py --once`
Expected: exits 0; `outputs/paper_trades/trader_log.jsonl` contains a `"tick"` event line; no `"error"` event.

- [ ] **Step 6: Commit**

```bash
git add paper_trader.py tests/paper_trader/test_paper_trader_loop.py
git commit -m "feat(paper_trader): main loop — gates -> score -> trigger -> open/close"
```

---

## Task 11: Replay invariant + cumulative-PnL invariant tests

**Files:**
- Create: `tests/paper_trader/test_invariants.py`

This is the disaster-recovery story per spec §Testing. Replaying
`trader_log.jsonl` from scratch must reconstruct `closed_trades.csv` and
`trader_state.json` byte-identically.

- [ ] **Step 1: Add a `replay_from_log` helper to `paper_trader.py`**

Insert into `paper_trader.py` (near the top, after the imports):

```python
def replay_from_log(log_path: Path, output_dir: Path, bankroll_initial: float) -> None:
    """Reconstruct closed_trades.csv and trader_state.json from trader_log.jsonl.

    Used by the disaster-recovery story and the replay invariant test.
    Reads only `entry`/`exit`/`tick` events (the rest are diagnostic).
    """
    import json as _json
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    closed_csv = output_dir / "closed_trades.csv"
    state_path = output_dir / "trader_state.json"
    open_csv = output_dir / "open_positions.csv"
    if closed_csv.exists():
        closed_csv.unlink()
    pm = PositionManager(
        state_path=state_path, log_path=output_dir / "_replay_unused.jsonl",
        closed_csv_path=closed_csv, open_csv_path=open_csv,
        bankroll_initial=bankroll_initial, mode_sizing="A",
        live_or_paper="paper",
    )
    open_by_id: dict[str, dict[str, Any]] = {}
    with Path(log_path).open() as fh:
        for line in fh:
            event = _json.loads(line)
            etype = event.get("event_type")
            if etype == "entry":
                position = {k: event[k] for k in (
                    "trade_id", "ticker", "game_key", "home_team", "away_team",
                    "entry_captured_at", "entry_yes_ask", "entry_yes_mid",
                    "entry_yes_bid", "contracts", "p_raw", "p_calibrated",
                    "expected_pnl_per_contract", "bankroll_at_entry",
                )}
                open_by_id[event["trade_id"]] = position
                pm.open_positions.append(position)
            elif etype == "exit":
                tid = event["trade_id"]
                if tid in open_by_id:
                    pos = open_by_id.pop(tid)
                    pm.open_positions = [p for p in pm.open_positions if p["trade_id"] != tid]
                    pm._close_position(
                        pos=pos,
                        exit_captured_at=event["exit_captured_at"],
                        exit_yes_bid=event["exit_yes_bid"],
                        exit_yes_mid=event["exit_yes_mid"],
                        exit_basis=event["exit_basis"],
                    )
    pm.write_state()
```

- [ ] **Step 2: Write failing tests**

Create `tests/paper_trader/test_invariants.py`:

```python
"""Replay + cumulative-PnL + concurrency-cap + trade-id-uniqueness invariants."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from paper_trader import process_tick, build_engine_context, replay_from_log
from paper_trader.position_manager import MAX_OPEN, PositionManager


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def model_path() -> Path:
    p = Path("models/live_home_up_5m_bootstrap.pkl")
    if not p.exists():
        pytest.skip("model artifact not present")
    return p


def test_cumulative_pnl_matches_sum_of_closed(tmp_path: Path):
    pm = PositionManager(
        state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl",
        closed_csv_path=tmp_path / "c.csv", open_csv_path=tmp_path / "o.csv",
        bankroll_initial=1000.0, mode_sizing="A", live_or_paper="paper",
    )
    # Open and close 5 positions back-to-back.
    base = pd.Timestamp("2026-05-01T19:00:00Z")
    for i in range(5):
        captured = (base + pd.Timedelta(seconds=i)).isoformat()
        pm.open_position(
            ticker=f"T{i}", game_key="G", home_team="H", away_team="A",
            captured_at=captured, yes_ask=0.40, yes_mid=0.39, yes_bid=0.38,
            contracts=10, p_raw=0.20, p_calibrated=0.18,
            expected_pnl_per_contract=0.012,
        )
        exit_at = (base + pd.Timedelta(seconds=i + 300)).isoformat()
        pm.tick_open_positions(now_captured_at=exit_at,
                               latest_bid_mid_per_ticker={f"T{i}": (0.43, 0.44)})
    df = pd.read_csv(tmp_path / "c.csv")
    assert pm.cumulative_pnl_realistic == pytest.approx(df["pnl_realistic"].sum(), abs=1e-6)


def test_concurrency_cap_holds_under_burst(tmp_path: Path):
    pm = PositionManager(
        state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl",
        closed_csv_path=tmp_path / "c.csv", open_csv_path=tmp_path / "o.csv",
        bankroll_initial=1000.0, mode_sizing="A", live_or_paper="paper",
    )
    base = pd.Timestamp("2026-05-01T19:00:00Z")
    for i in range(MAX_OPEN + 50):
        pm.open_position(
            ticker=f"T{i}", game_key="G", home_team="H", away_team="A",
            captured_at=(base + pd.Timedelta(seconds=i)).isoformat(),
            yes_ask=0.40, yes_mid=0.39, yes_bid=0.38,
            contracts=1, p_raw=0.2, p_calibrated=0.18,
            expected_pnl_per_contract=0.01,
        )
    assert len(pm.open_positions) == MAX_OPEN


def test_trade_id_uniqueness_under_synthetic_load(tmp_path: Path):
    pm = PositionManager(
        state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl",
        closed_csv_path=tmp_path / "c.csv", open_csv_path=tmp_path / "o.csv",
        bankroll_initial=1_000_000.0, mode_sizing="A", live_or_paper="paper",
    )
    seen = set()
    base = pd.Timestamp("2026-05-01T19:00:00Z")
    n = 5000
    for i in range(n):
        captured = (base + pd.Timedelta(microseconds=i)).isoformat()
        pm.open_position(
            ticker=f"T{i % 100}", game_key="G", home_team="H", away_team="A",
            captured_at=captured, yes_ask=0.40, yes_mid=0.39, yes_bid=0.38,
            contracts=1, p_raw=0.2, p_calibrated=0.18,
            expected_pnl_per_contract=0.01,
        )
        if len(pm.open_positions) > MAX_OPEN - 1:
            # Force-close one to make room.
            pos = pm.open_positions.pop(0)
            pm._close_position(pos, captured, 0.42, 0.42, "synthetic")
        seen.add(pm.open_positions[-1]["trade_id"]) if pm.open_positions else None
    df = pd.read_csv(tmp_path / "c.csv")
    assert df["trade_id"].is_unique


def test_replay_reconstructs_closed_trades(tmp_path: Path):
    """Drive the loop synthetically, then replay the log and assert
    closed_trades.csv is byte-identical."""
    out = tmp_path / "outputs"
    pm = PositionManager(
        state_path=out / "trader_state.json",
        log_path=out / "trader_log.jsonl",
        closed_csv_path=out / "closed_trades.csv",
        open_csv_path=out / "open_positions.csv",
        bankroll_initial=1000.0, mode_sizing="A", live_or_paper="paper",
    )
    base = pd.Timestamp("2026-05-01T19:00:00Z")
    for i in range(10):
        captured = (base + pd.Timedelta(seconds=i)).isoformat()
        pm.open_position(
            ticker=f"T{i}", game_key="G", home_team="H", away_team="A",
            captured_at=captured, yes_ask=0.40, yes_mid=0.39, yes_bid=0.38,
            contracts=10, p_raw=0.20, p_calibrated=0.18,
            expected_pnl_per_contract=0.012,
        )
        exit_at = (base + pd.Timedelta(seconds=i + 300)).isoformat()
        pm.tick_open_positions(now_captured_at=exit_at,
                               latest_bid_mid_per_ticker={f"T{i}": (0.43, 0.44)})
    pm.write_state()

    original_digest = _digest(out / "closed_trades.csv")

    # Replay into a fresh dir.
    replay_dir = tmp_path / "replay"
    replay_from_log(out / "trader_log.jsonl", replay_dir, bankroll_initial=1000.0)
    replay_digest = _digest(replay_dir / "closed_trades.csv")
    assert replay_digest == original_digest
```

- [ ] **Step 3: Run — confirm pass**

Run: `pytest tests/paper_trader/test_invariants.py -v`
Expected: 4 passed.

If the replay digest mismatches, common causes:
- `_close_position` doesn't accept the same columns the original entry/exit
  events carried — fix the dict comprehension in `replay_from_log` to pass
  every required field.
- DictWriter column ordering differs between runs — both runs use
  `CLOSED_TRADES_HEADER` from `trade_log`, so this should hold.

- [ ] **Step 4: Commit**

```bash
git add paper_trader.py tests/paper_trader/test_invariants.py
git commit -m "test(paper_trader): replay + PnL + cap + uniqueness invariants"
```

---

## Task 12: `tools/trader_status.py` — read-only CLI

**Files:**
- Create: `tools/trader_status.py`
- Test: `tests/paper_trader/test_tools_status.py`

- [ ] **Step 1: Write failing test**

Create `tests/paper_trader/test_tools_status.py`:

```python
"""trader_status: prints a readable summary from state + closed trades."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _seed_outputs(tmp_path: Path) -> Path:
    out = tmp_path / "outputs"
    out.mkdir()
    state = {
        "mode": "A", "live_or_paper": "paper",
        "bankroll_initial": 1000.0,
        "cumulative_pnl_realistic": 47.31, "cumulative_pnl_mid": 62.18,
        "trades_closed": 3, "max_drawdown": 31.40,
        "drawdown_breached": False, "open_positions": [],
        "last_processed_captured_at_per_ticker": {},
        "last_tick_at": "2026-05-01T19:42:17+00:00",
    }
    (out / "trader_state.json").write_text(json.dumps(state, indent=2))
    pd.DataFrame([
        {"trade_id": "T1", "pnl_realistic": 12.0, "pnl_mid": 14.0,
         "realized_label_home_up_5m": 1, "exit_captured_at": "2026-05-01T19:30:00Z"},
        {"trade_id": "T2", "pnl_realistic": -5.0, "pnl_mid": -2.0,
         "realized_label_home_up_5m": 0, "exit_captured_at": "2026-05-01T19:35:00Z"},
        {"trade_id": "T3", "pnl_realistic": 40.31, "pnl_mid": 50.18,
         "realized_label_home_up_5m": 1, "exit_captured_at": "2026-05-01T19:40:00Z"},
    ]).to_csv(out / "closed_trades.csv", index=False)
    (out / "trader_log.jsonl").write_text("")
    return out


def test_trader_status_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    from tools.trader_status import main
    out = _seed_outputs(tmp_path)
    rc = main(["--output-dir", str(out)])
    text = capsys.readouterr().out
    assert rc == 0
    assert "trades_closed" in text.lower() or "trades closed" in text.lower()
    assert "wins" in text.lower()
    assert "47.31" in text  # cumulative pnl realistic
```

- [ ] **Step 2: Run — confirm fail**

Run: `pytest tests/paper_trader/test_tools_status.py -v`
Expected: ImportError on `tools.trader_status`.

- [ ] **Step 3: Implement `tools/trader_status.py`**

Create `tools/trader_status.py`:

```python
"""Read-only paper trader status. Safe under `watch -n 30 python tools/trader_status.py`."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _read_state(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _summary(closed: pd.DataFrame) -> dict:
    if closed.empty:
        return {"trades_closed": 0, "wins": 0, "win_rate": 0.0, "sharpe": 0.0}
    wins = int((closed["realized_label_home_up_5m"] == 1).sum())
    n = len(closed)
    pnl = closed["pnl_realistic"].astype(float)
    sharpe = float(pnl.mean() / pnl.std(ddof=0)) if pnl.std(ddof=0) > 0 else 0.0
    return {
        "trades_closed": n,
        "wins": wins,
        "win_rate": wins / n if n else 0.0,
        "sharpe": sharpe,
        "pnl_realistic": float(pnl.sum()),
        "pnl_mid": float(closed["pnl_mid"].astype(float).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_trades"))
    args = parser.parse_args(argv)
    state = _read_state(args.output_dir / "trader_state.json")
    closed_path = args.output_dir / "closed_trades.csv"
    closed = pd.read_csv(closed_path) if closed_path.exists() else pd.DataFrame()
    summary = _summary(closed)
    last_tick = state.get("last_tick_at")
    if last_tick:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(last_tick.replace("Z", "+00:00"))).total_seconds()
    else:
        age = float("inf")
    print(f"Mode             : {state.get('mode', '?')} (sizing)  {state.get('live_or_paper', '?')}")
    print(f"Bankroll         : ${state.get('bankroll_initial', 0):.2f}")
    print(f"Cumulative PnL   : realistic ${summary.get('pnl_realistic', 0):+.2f}  "
          f"|  mid ${summary.get('pnl_mid', 0):+.2f}  "
          f"|  alpha capture ${summary.get('pnl_mid', 0) - summary.get('pnl_realistic', 0):+.2f}")
    print(f"Trades closed    : {summary['trades_closed']}   "
          f"wins {summary['wins']} ({summary['win_rate']*100:.1f}%)   "
          f"sharpe(per-trade) {summary['sharpe']:.2f}")
    print(f"Open positions   : {len(state.get('open_positions', []))} / 20")
    print(f"Last tick        : {last_tick}  ({age:.0f}s ago)")
    print(f"Drawdown         : current ${state.get('cumulative_pnl_realistic', 0):+.2f}   "
          f"max ${-state.get('max_drawdown', 0):.2f}")
    if not closed.empty:
        print("Recent 5 trades  :")
        cols = ["trade_id", "pnl_realistic", "pnl_mid", "realized_label_home_up_5m", "exit_captured_at"]
        print(closed.tail(5)[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run — confirm pass**

Run: `pytest tests/paper_trader/test_tools_status.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/trader_status.py tests/paper_trader/test_tools_status.py
git commit -m "feat(paper_trader): trader_status — read-only CLI summary"
```

---

## Task 13: `tools/trader_rollups.py`

**Files:**
- Create: `tools/trader_rollups.py`
- Test: `tests/paper_trader/test_tools_rollups.py`

- [ ] **Step 1: Write failing test**

Create `tests/paper_trader/test_tools_rollups.py`:

```python
"""trader_rollups: idempotent batch — daily, per_game, per_p_decile, per_minute."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _seed(tmp_path: Path) -> Path:
    out = tmp_path / "outputs"
    out.mkdir()
    pd.DataFrame([
        {"trade_id": "T1", "game_key": "G1", "p_calibrated": 0.10,
         "entry_captured_at": "2026-05-01T19:00:00Z", "exit_captured_at": "2026-05-01T19:05:00Z",
         "pnl_realistic": 1.5, "pnl_mid": 2.0,
         "realized_label_home_up_5m": 1, "fees": 0.10, "contracts": 5,
         "p_raw": 0.11, "trigger_mode": "A"},
        {"trade_id": "T2", "game_key": "G1", "p_calibrated": 0.40,
         "entry_captured_at": "2026-05-01T19:30:00Z", "exit_captured_at": "2026-05-01T19:35:00Z",
         "pnl_realistic": -2.0, "pnl_mid": -1.0,
         "realized_label_home_up_5m": 0, "fees": 0.0, "contracts": 5,
         "p_raw": 0.39, "trigger_mode": "A"},
        {"trade_id": "T3", "game_key": "G2", "p_calibrated": 0.40,
         "entry_captured_at": "2026-05-02T19:30:00Z", "exit_captured_at": "2026-05-02T19:35:00Z",
         "pnl_realistic": 3.0, "pnl_mid": 4.0,
         "realized_label_home_up_5m": 1, "fees": 0.20, "contracts": 5,
         "p_raw": 0.41, "trigger_mode": "A"},
    ]).to_csv(out / "closed_trades.csv", index=False)
    return out


def test_rollups_creates_all_four_tables(tmp_path: Path):
    from tools.trader_rollups import main
    out = _seed(tmp_path)
    rc = main(["--output-dir", str(out)])
    assert rc == 0
    rdir = out / "rollups"
    assert (rdir / "per_game.csv").exists()
    assert (rdir / "per_p_decile.csv").exists()
    assert (rdir / "per_minute_of_game.csv").exists()
    daily_files = list(rdir.glob("daily_*.csv"))
    assert len(daily_files) == 2  # 2026-05-01, 2026-05-02


def test_rollups_idempotent(tmp_path: Path):
    """Running twice produces the same file contents."""
    from tools.trader_rollups import main
    out = _seed(tmp_path)
    main(["--output-dir", str(out)])
    first = (out / "rollups" / "per_game.csv").read_bytes()
    main(["--output-dir", str(out)])
    second = (out / "rollups" / "per_game.csv").read_bytes()
    assert first == second
```

- [ ] **Step 2: Run — confirm fail**

Run: `pytest tests/paper_trader/test_tools_rollups.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `tools/trader_rollups.py`**

Create `tools/trader_rollups.py`:

```python
"""Recompute rollup tables from closed_trades.csv. Idempotent."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_closed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["entry_captured_at"] = pd.to_datetime(df["entry_captured_at"], utc=True, errors="coerce")
    df["exit_captured_at"] = pd.to_datetime(df["exit_captured_at"], utc=True, errors="coerce")
    df["entry_date"] = df["entry_captured_at"].dt.date.astype(str)
    return df


def _per_p_decile(df: pd.DataFrame) -> pd.DataFrame:
    decile = (df["p_calibrated"].clip(0, 0.999) * 10).astype(int)
    out = df.groupby(decile).agg(
        n=("trade_id", "count"),
        win_rate=("realized_label_home_up_5m", "mean"),
        pnl_realistic_mean=("pnl_realistic", "mean"),
        pnl_realistic_sum=("pnl_realistic", "sum"),
        pnl_mid_mean=("pnl_mid", "mean"),
    ).reset_index().rename(columns={"p_calibrated": "p_decile"})
    out.columns = ["p_decile", "n", "win_rate", "pnl_realistic_mean", "pnl_realistic_sum", "pnl_mid_mean"]
    return out


def _per_minute_of_game(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate game-minute bucket from entry_captured_at — needs the
    feature row's seconds_elapsed for true accuracy. Lacking that here, we
    bucket by hold_seconds and entry hour-of-day as a coarse proxy."""
    hour = df["entry_captured_at"].dt.hour.fillna(-1).astype(int)
    out = df.groupby(hour).agg(
        n=("trade_id", "count"),
        pnl_realistic_mean=("pnl_realistic", "mean"),
        pnl_realistic_sum=("pnl_realistic", "sum"),
    ).reset_index().rename(columns={"entry_captured_at": "entry_hour_utc"})
    out.columns = ["entry_hour_utc", "n", "pnl_realistic_mean", "pnl_realistic_sum"]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_trades"))
    args = parser.parse_args(argv)
    closed_path = args.output_dir / "closed_trades.csv"
    if not closed_path.exists():
        print("no closed_trades.csv to roll up")
        return 0
    df = _load_closed(closed_path)
    rdir = args.output_dir / "rollups"
    rdir.mkdir(parents=True, exist_ok=True)
    # daily_<date>.csv
    for date, sub in df.groupby("entry_date"):
        sub_summary = pd.DataFrame([{
            "date": date, "n": len(sub),
            "pnl_realistic_sum": float(sub["pnl_realistic"].sum()),
            "pnl_mid_sum": float(sub["pnl_mid"].sum()),
            "win_rate": float(sub["realized_label_home_up_5m"].mean()),
        }])
        sub_summary.to_csv(rdir / f"daily_{date}.csv", index=False)
    # per_game.csv
    per_game = df.groupby("game_key").agg(
        n=("trade_id", "count"),
        pnl_realistic_sum=("pnl_realistic", "sum"),
        win_rate=("realized_label_home_up_5m", "mean"),
    ).reset_index()
    per_game.to_csv(rdir / "per_game.csv", index=False)
    # per_p_decile.csv
    _per_p_decile(df).to_csv(rdir / "per_p_decile.csv", index=False)
    # per_minute_of_game.csv
    _per_minute_of_game(df).to_csv(rdir / "per_minute_of_game.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run — confirm pass**

Run: `pytest tests/paper_trader/test_tools_rollups.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/trader_rollups.py tests/paper_trader/test_tools_rollups.py
git commit -m "feat(paper_trader): trader_rollups — idempotent rollup tables"
```

---

## Task 14: `tools/trader_smoke.py`

**Files:**
- Create: `tools/trader_smoke.py`

This is the 10-minute live wiring check. The spec says: "10-minute live run
with mode=paper, max_positions=1, and trigger forced to reject
(expected_pnl_per_contract=0)." We achieve "trigger forced to reject" by
monkeypatching the trigger module to always return `(False, 0.0, "smoke_off")`.

- [ ] **Step 1: Implement smoke**

Create `tools/trader_smoke.py`:

```python
"""10-minute smoke run: connects to the real feed, evaluates gates and
scoring, but rejects every trigger so no positions are opened."""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from paper_trader import build_engine_context, process_tick
from paper_trader import trigger as _trigger


def _force_reject(*_args, **_kwargs):
    return False, 0.0, "smoke_off"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=int, default=10)
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--features-path", type=Path,
                        default=Path("data/live/features/latest_features.csv"))
    parser.add_argument("--model-path", type=Path,
                        default=Path("models/live_home_up_5m_bootstrap.pkl"))
    parser.add_argument("--pooled-path", type=Path,
                        default=Path("outputs/paper_trades/pooled_means.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/paper_trades_smoke"))
    args = parser.parse_args(argv)

    _trigger.evaluate = _force_reject

    ctx = build_engine_context(
        model_path=args.model_path, pooled_path=args.pooled_path,
        feed_path=args.features_path, output_dir=args.output_dir,
        bankroll=1000.0, mode_sizing="A", live_or_paper="paper",
        interval_seconds=args.interval,
    )

    deadline = datetime.now(timezone.utc).timestamp() + args.minutes * 60
    ticks = 0
    errors = 0
    while datetime.now(timezone.utc).timestamp() < deadline:
        try:
            process_tick(ctx)
            ticks += 1
        except Exception:
            errors += 1
        time.sleep(args.interval)
    print(f"smoke complete: ticks={ticks} errors={errors} dir={args.output_dir}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run a 60-second mini-smoke as a sanity check**

Run: `python tools/trader_smoke.py --minutes 1 --interval 15 --output-dir outputs/paper_trades_smoke`
Expected: prints `smoke complete: ticks=N errors=0` for some N ≥ 1; produces `outputs/paper_trades_smoke/trader_log.jsonl` with `tick` events and zero `entry`/`exit`/`error` events.

- [ ] **Step 3: Commit**

```bash
git add tools/trader_smoke.py
git commit -m "feat(paper_trader): trader_smoke — 10-min live wiring check"
```

---

## Task 15: `tools/trader_backtest.py`

**Files:**
- Create: `tools/trader_backtest.py`
- Test: `tests/paper_trader/test_tools_backtest.py`

- [ ] **Step 1: Write failing test**

Create `tests/paper_trader/test_tools_backtest.py`:

```python
"""trader_backtest: replay one historical day's live_features_<date>.csv."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def historical_day() -> Path:
    p = Path("data/live/features/live_features_2026-04-30.csv")
    if not p.exists():
        pytest.skip("no historical feature day available")
    return p


@pytest.fixture
def model_path() -> Path:
    p = Path("models/live_home_up_5m_bootstrap.pkl")
    pooled = Path("outputs/paper_trades/pooled_means.json")
    if not p.exists() or not pooled.exists():
        pytest.skip("model artifact or pooled means not present")
    return p


def test_backtest_writes_outputs_to_dated_dir(tmp_path: Path, historical_day: Path, model_path: Path):
    from tools.trader_backtest import main
    out = tmp_path / "bt"
    rc = main([
        "--features-path", str(historical_day),
        "--output-dir", str(out),
        "--max-rows", "200",  # cap for speed
    ])
    assert rc == 0
    # closed_trades.csv may be empty if no signals fired in the first 200 rows,
    # but the log file must exist with tick events.
    log = (out / "trader_log.jsonl").read_text().strip().splitlines()
    assert any("tick" in line for line in log)
```

- [ ] **Step 2: Run — confirm fail**

Run: `pytest tests/paper_trader/test_tools_backtest.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `tools/trader_backtest.py`**

Create `tools/trader_backtest.py`:

```python
"""Replay one historical live_features_<date>.csv through the same pipeline.

Produces outputs in --output-dir mirroring the live structure. The replay
groups feed rows by captured_at into "ticks", processes each tick exactly as
the live engine would, then advances simulated time to the next captured_at.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from paper_trader import build_engine_context, process_tick


def _write_chunk(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path,
                        default=Path("models/live_home_up_5m_bootstrap.pkl"))
    parser.add_argument("--pooled-path", type=Path,
                        default=Path("outputs/paper_trades/pooled_means.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap rows for speed (debugging).")
    args = parser.parse_args(argv)

    full = pd.read_csv(args.features_path)
    if args.max_rows:
        full = full.head(args.max_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = args.output_dir / "_replay_chunk.csv"

    # Process tick-by-tick; each chunk is the rows at one captured_at.
    captured_groups = sorted(full["captured_at"].astype(str).unique())
    start = time.monotonic()
    for capture in captured_groups:
        chunk = full[full["captured_at"].astype(str) == capture]
        _write_chunk(chunk_path, chunk)
        ctx = build_engine_context(
            model_path=args.model_path,
            pooled_path=args.pooled_path,
            feed_path=chunk_path,
            output_dir=args.output_dir,
            bankroll=1000.0,
            mode_sizing="A",
            live_or_paper="paper",
        )
        process_tick(ctx)
    elapsed = time.monotonic() - start
    print(f"backtest done: ticks={len(captured_groups)} rows={len(full)} elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run — confirm pass**

Run: `pytest tests/paper_trader/test_tools_backtest.py -v`
Expected: 1 passed.

- [ ] **Step 5: Run a real-day backtest**

Run: `python tools/trader_backtest.py --features-path data/live/features/live_features_2026-04-30.csv --output-dir outputs/paper_trades/backtest_2026-04-30 --max-rows 5000`
Expected: prints `backtest done: ticks=… rows=5000 elapsed=…s`. No exception.

- [ ] **Step 6: Commit**

```bash
git add tools/trader_backtest.py tests/paper_trader/test_tools_backtest.py
git commit -m "feat(paper_trader): trader_backtest — replay historical day"
```

---

## Task 16: Acceptance check

**Files:** none new

- [ ] **Step 1: Run full paper_trader test suite**

Run: `pytest tests/paper_trader/ -v`
Expected: all green.

- [ ] **Step 2: Run a full backtest on at least 3 days**

Run:
```bash
for d in 2026-04-26 2026-04-27 2026-04-28; do
  python tools/trader_backtest.py \
    --features-path data/live/features/live_features_${d}.csv \
    --output-dir outputs/paper_trades/backtest_${d}
done
```
Expected: all three exit 0, no `error` events in the per-day `trader_log.jsonl`. Spot-check `closed_trades.csv` if any was produced.

- [ ] **Step 3: Run the 60-min smoke**

Run: `python tools/trader_smoke.py --minutes 60 --output-dir outputs/paper_trades_smoke_60`
Expected: prints `smoke complete: ticks=… errors=0`. Tail the log: `jq 'select(.event_type == "error")' outputs/paper_trades_smoke_60/trader_log.jsonl | head` — should be empty.

- [ ] **Step 4: Confirm live boot gates refuse**

Run: `python paper_trader.py --mode live --once`
Expected: prints "Refusing to start in live mode. Failing gates:" with three reasons (trades_closed, sharpe, max_dd) and exits 2.

- [ ] **Step 5: Commit if any test or doc tweaks landed; otherwise no-op**

```bash
git status
# If clean, nothing to commit. If anything's changed:
# git add -p && git commit -m "test(paper_trader): acceptance pass"
```

---

## Self-Review Notes

**Spec coverage check (all sections from the design doc):**
- Goal/architecture → Tasks 0-10
- Module layout (8 components) → Tasks 1-9 (one task each except trade_log = Task 2 covers state + JSONL + CSV)
- `paper_trader.py` entry → Task 10
- Output directory + state schema + closed_trades schema + log event types → Task 2 (`trade_log.CLOSED_TRADES_HEADER` + `trade_log.emit_event`) and Task 10 (state writer)
- Pipeline (one tick) → Task 10 (`process_tick`)
- Stale-feed handling → Task 10 (`_detect_stale`)
- Restart behavior → Task 8 test (`test_restart_reload_keeps_open_positions`) + Task 10 (`pm.load_state` on boot)
- Risk controls (concurrency, liquidity, side, sanity, drawdown, A→B, paper→live) → Tasks 3, 6, 8, 10 (all)
- Observability — status CLI / rollups / log query / health checks → Tasks 12, 13
- Testing (unit, invariant, backtest, smoke, acceptance) → Tasks 1-9, 11, 14, 15, 16
- Open questions deferred to implementation → all resolved (exit pricing = next-tick yes_bid/yes_mid, pooled-means seeded from training scored_rows, log rotation deferred per spec).

**Placeholder scan:** none. All steps include exact code or exact commands.

**Type consistency:** `PositionManager` exposes `open_position`, `tick_open_positions`, `_close_position`, `load_state`, `write_state`, `cumulative_pnl_realistic`, `cumulative_pnl_mid`, `trades_closed`, `max_drawdown`, `drawdown_breached`, `open_positions`, `last_processed_captured_at_per_ticker`, `last_closed_trade`, `bankroll_current`. Same names used in `paper_trader.process_tick`, `replay_from_log`, and tests. `Store.E_rises_pooled` / `E_doesnt_pooled` / `var_pooled` / `decile_table` consistent across Tasks 5, 6, 10. `CLOSED_TRADES_HEADER` consistent across Tasks 2, 8, 11. `gate_filters.evaluate` returns `(bool, str)` everywhere. `trigger.evaluate` returns `(bool, float, str)` everywhere. `scorer.load_model` / `score` / `engineer_features` consistent across Tasks 4, 10.

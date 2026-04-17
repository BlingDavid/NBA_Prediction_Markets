# Arbitrage Detection (Price Divergence Signal) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect when Kalshi YES prices diverge meaningfully from a de-vigged sportsbook consensus, classify the disagreement into three tiers (triangulated / neutral / contradicting), use the tier to adjust the edge threshold in `ev_analyzer.py`, and log the divergence on every live poll for future model training.

**Architecture:** One new pure-function module `consensus_divergence.py` is the single source of truth for all divergence math. Three callers — `ev_analyzer.py` (pre-game bet decisions), `realtime_feature_store.py` (live polling + feature columns), and a thin `--scan` CLI — invoke those functions. The existing `_build_consensus_map` in `realtime_feature_store.py` is **refactored** (not parallel-implemented) to call into the new function so there is one place to fix consensus logic. A rollout mode flag (`CONSENSUS_TIER_MODE`) gates whether tier adjustments affect live bet decisions (`"shadow"` logs only; `"active"` applies the multiplier; `"off"` hardcodes tier=1).

**Tech Stack:** Python 3.10+, pandas, numpy, pytest, existing codebase (no new runtime deps). Spec: `docs/superpowers/specs/2026-04-16-arbitrage-detection-design.md`.

---

## File Structure

**Create:**
- `consensus_divergence.py` — pure functions + `--scan` / `--settle` CLI (~350 LOC).
- `pyproject.toml` — minimal pytest config so `from consensus_divergence import ...` resolves.
- `tests/__init__.py` — empty.
- `tests/test_consensus_divergence.py` — unit tests (Layer 1) + golden snapshot runner (Layer 3).
- `tests/test_integrations.py` — integration smokes (Layer 2).
- `tests/fixtures/consensus_golden_cases.csv` — ~20-row hand-curated snapshot.

**Modify:**
- `config.py` — add five config constants (lines appended at end).
- `live_data.py` — `fetch_odds_api_lines` book dict to include `last_update` field.
- `realtime_feature_store.py` — refactor `_build_consensus_map`, extend `build_feature_rows` with 5 new columns, add `_detect_divergence_events` wired into `capture_once`.
- `ev_analyzer.py` — extend `--extra-signals` block (starting at line 374) with consensus-tier adjustment.
- `requirements.txt` — add pytest.
- `IMPROVEMENTS.md` — mark #8 ✅.

**Not touched:** `model.py`, `features.py`, `backtester.py`, training pipelines. Historical reconstruction is deferred per spec non-goals.

---

### Task 1: Scaffolding — pytest deps, test dir, pyproject.toml

**Files:**
- Modify: `requirements.txt` (append pytest)
- Create: `pyproject.toml`
- Create: `tests/__init__.py`

- [ ] **Step 1: Append pytest to `requirements.txt`**

Edit `requirements.txt`, append at end (after the final `joblib>=1.3.0` line):

```
# Testing
pytest>=7.4.0
```

- [ ] **Step 2: Create `pyproject.toml` at repo root**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 3: Create empty `tests/__init__.py`**

```python
```

(File exists but is empty — this marks the directory as a package for import resolution.)

- [ ] **Step 4: Install pytest and verify discovery**

```bash
pip install pytest
pytest --collect-only
```

Expected output: `no tests collected` (no tests yet; this verifies config is readable).

- [ ] **Step 5: Commit**

```bash
git add requirements.txt pyproject.toml tests/__init__.py
git commit -m "chore: add pytest scaffolding for consensus_divergence tests"
```

---

### Task 2: Add config constants

**Files:**
- Modify: `config.py` (append at end)

- [ ] **Step 1: Append config constants after line 95**

Open `config.py`. After the final line `TEAM_NAME_TO_ABBREV = {v: k for k, v in TEAM_ABBREV_MAP.items()}`, append:

```python

# ── Consensus Divergence Signal (IMPROVEMENTS #8) ─────────────────────
# Minimum number of non-stale sportsbooks required to compute a consensus.
# Below this count, consensus is None and tier defaults to 1 (neutral).
CONSENSUS_MIN_BOOKS = 3

# Drop book quotes older than this many seconds (uses Odds API last_update).
CONSENSUS_STALE_SECONDS = 300

# Tier-2 / Tier -1 trigger: minimum |Kalshi - consensus| AND |Kalshi - model|
# disagreement in percentage points. See spec for tier assignment rules.
CONSENSUS_DIVERGENCE_THRESHOLD_PP = 3.5

# Multiplier applied to MIN_EDGE_THRESHOLD per tier:
#   2  → multiplier < 1  (easier to bet; high-conviction triangulation)
#   1  → multiplier = 1  (neutral)
#  -1  → multiplier > 1  (stricter; model ↔ consensus contradiction)
TIER_THRESHOLD_MULTIPLIERS = {2: 0.7, 1: 1.0, -1: 1.5}

# Rollout gate: "shadow" logs tiers but does NOT modify bet decisions;
# "active" applies TIER_THRESHOLD_MULTIPLIERS; "off" pins tier to 1 (disabled).
CONSENSUS_TIER_MODE = "shadow"
```

- [ ] **Step 2: Verify config imports cleanly**

```bash
python -c "import config; print(config.CONSENSUS_MIN_BOOKS, config.TIER_THRESHOLD_MULTIPLIERS, config.CONSENSUS_TIER_MODE)"
```

Expected output: `3 {2: 0.7, 1: 1.0, -1: 1.5} shadow`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat(config): add consensus divergence thresholds and rollout mode flag"
```

---

### Task 3: `devig_proportional` — create module + first pure function

**Files:**
- Create: `consensus_divergence.py`
- Create: `tests/test_consensus_divergence.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_consensus_divergence.py`:

```python
"""
Unit tests for consensus_divergence.py — the price-divergence signal
used by ev_analyzer and realtime_feature_store.
"""

import pytest

from consensus_divergence import devig_proportional


class TestDevigProportional:
    def test_standard_favorite_underdog(self):
        # -150 / +130: raw implied 0.600 / 0.4348, sum 1.0348,
        # de-vigged ≈ 0.580 / 0.420
        home, away = devig_proportional(-150, 130)
        assert home == pytest.approx(0.580, abs=0.005)
        assert away == pytest.approx(0.420, abs=0.005)
        assert home + away == pytest.approx(1.0, abs=1e-9)

    def test_pick_em(self):
        home, away = devig_proportional(-110, -110)
        assert home == pytest.approx(0.5, abs=1e-9)
        assert away == pytest.approx(0.5, abs=1e-9)

    def test_none_input_returns_none(self):
        assert devig_proportional(None, -110) is None
        assert devig_proportional(-110, None) is None
        assert devig_proportional(None, None) is None

    def test_zero_moneyline_returns_none(self):
        assert devig_proportional(0, -110) is None
        assert devig_proportional(-110, 0) is None

    def test_empty_string_returns_none(self):
        assert devig_proportional("", -110) is None
        assert devig_proportional(-110, "") is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_consensus_divergence.py::TestDevigProportional -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'consensus_divergence'`.

- [ ] **Step 3: Create `consensus_divergence.py` with `devig_proportional`**

```python
"""
Consensus Divergence Signal (IMPROVEMENTS #8)
─────────────────────────────────────────────
Detects when Kalshi YES prices diverge meaningfully from a de-vigged
sportsbook consensus. Classifies disagreements into three tiers and
exposes pure functions consumed by:

  1. ev_analyzer.py               (pre-game bet decisions)
  2. realtime_feature_store.py    (live polling + feature columns)
  3. the --scan / --settle CLI    (ad-hoc checks and settlement logging)

No I/O happens inside the public math functions — callers fetch data
and pass it in. See docs/superpowers/specs/2026-04-16-arbitrage-detection-design.md.
"""

from __future__ import annotations


def _american_to_raw_prob(moneyline) -> float | None:
    """American moneyline → raw implied probability (with vig)."""
    if moneyline is None or moneyline == "" or moneyline == 0:
        return None
    try:
        ml = int(moneyline)
    except (TypeError, ValueError):
        return None
    if ml > 0:
        return 100 / (ml + 100)
    return abs(ml) / (abs(ml) + 100)


def devig_proportional(home_ml, away_ml) -> tuple[float, float] | None:
    """
    Strip vig from a two-sided American moneyline via proportional scaling.

    Returns (p_home_devigged, p_away_devigged) summing to 1.0, or None
    when either side is None / 0 / empty-string / unparseable.
    """
    p_home_raw = _american_to_raw_prob(home_ml)
    p_away_raw = _american_to_raw_prob(away_ml)
    if p_home_raw is None or p_away_raw is None:
        return None
    total = p_home_raw + p_away_raw
    if total <= 0:
        return None
    return p_home_raw / total, p_away_raw / total
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_consensus_divergence.py::TestDevigProportional -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add consensus_divergence.py tests/test_consensus_divergence.py
git commit -m "feat(consensus): add devig_proportional for two-sided American odds"
```

---

### Task 4: Expose `last_update` from Odds API parser

**Files:**
- Modify: `live_data.py` (lines 584-602, `fetch_odds_api_lines` inner loop)
- Modify: `tests/test_consensus_divergence.py` (new test class)

The existing parser drops the per-book `last_update` timestamp that the Odds API provides. The staleness filter (edge case #2 in the spec) needs it. We extract the per-game parsing into a helper so it can be unit-tested without HTTP mocking.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_consensus_divergence.py`:

```python
from live_data import _parse_odds_api_game


class TestParseOddsApiGame:
    def _name_map(self):
        return {
            "Boston Celtics": "BOS",
            "Celtics": "BOS",
            "Miami Heat": "MIA",
            "Heat": "MIA",
        }

    def test_book_dict_includes_last_update(self):
        game = {
            "home_team": "Boston Celtics",
            "away_team": "Miami Heat",
            "commence_time": "2026-04-16T23:30:00Z",
            "bookmakers": [
                {
                    "title": "DraftKings",
                    "last_update": "2026-04-16T22:55:12Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Boston Celtics", "price": -150},
                                {"name": "Miami Heat", "price": 130},
                            ],
                        }
                    ],
                }
            ],
        }
        result = _parse_odds_api_game(game, self._name_map())
        assert result is not None
        assert result["books"][0]["name"] == "DraftKings"
        assert result["books"][0]["last_update"] == "2026-04-16T22:55:12Z"
        assert result["books"][0]["markets"]["h2h"]["BOS"]["price"] == -150
        assert result["books"][0]["markets"]["h2h"]["MIA"]["price"] == 130

    def test_missing_last_update_becomes_none(self):
        game = {
            "home_team": "Boston Celtics",
            "away_team": "Miami Heat",
            "bookmakers": [
                {
                    "title": "FanDuel",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Boston Celtics", "price": -140},
                                {"name": "Miami Heat", "price": 120},
                            ],
                        }
                    ],
                }
            ],
        }
        result = _parse_odds_api_game(game, self._name_map())
        assert result["books"][0]["last_update"] is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_consensus_divergence.py::TestParseOddsApiGame -v
```

Expected: FAIL with `ImportError: cannot import name '_parse_odds_api_game'` (function does not exist yet).

- [ ] **Step 3: Refactor `fetch_odds_api_lines` to use a new `_parse_odds_api_game` helper**

Open `live_data.py`. Replace the body of `fetch_odds_api_lines` from line 553 onward and add the new helper just above it.

First, insert this new helper function **immediately above** `def fetch_odds_api_lines(...)` at line 533:

```python
def _parse_odds_api_game(game: dict, name_to_abbrev: dict) -> dict | None:
    """
    Parse a single game payload from The Odds API /odds response into our
    normalized shape. Pure function so it can be unit-tested without HTTP.
    """
    h_team = game.get("home_team", "")
    a_team = game.get("away_team", "")
    h_abbr = name_to_abbrev.get(h_team, h_team)
    a_abbr = name_to_abbrev.get(a_team, a_team)

    game_odds = {
        "home": h_abbr,
        "away": a_abbr,
        "commence": game.get("commence_time", ""),
        "books": [],
    }

    for book in game.get("bookmakers", []):
        book_data = {
            "name": book.get("title", ""),
            "last_update": book.get("last_update"),
            "markets": {},
        }
        for market in book.get("markets", []):
            mkey = market.get("key", "")
            outcomes = {}
            for outcome in market.get("outcomes", []):
                team_name = outcome.get("name", "")
                abbr = name_to_abbrev.get(team_name, team_name)
                outcomes[abbr] = {
                    "price": outcome.get("price", 0),
                    "point": outcome.get("point", None),
                }
            book_data["markets"][mkey] = outcomes
        game_odds["books"].append(book_data)

    return game_odds
```

Then replace the existing inner-loop body of `fetch_odds_api_lines` (lines 565-604 — the `for game in data: ...` block) with a call to the helper. The new body from the `for game in data:` onward becomes:

```python
        for game in data:
            h_team = game.get("home_team", "")
            a_team = game.get("away_team", "")
            h_abbr = name_to_abbrev.get(h_team, h_team)
            a_abbr = name_to_abbrev.get(a_team, a_team)

            # Filter if specific teams requested
            if home_team and away_team:
                if not ((h_abbr == home_team and a_abbr == away_team) or
                        (h_abbr == away_team and a_abbr == home_team)):
                    continue

            game_odds = _parse_odds_api_game(game, name_to_abbrev)
            if game_odds is not None:
                results.append(game_odds)

        return results
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_consensus_divergence.py::TestParseOddsApiGame -v
```

Expected: 2 passed.

- [ ] **Step 5: Verify existing behavior still works**

```bash
pytest tests/test_consensus_divergence.py -v
```

Expected: all previous tests still pass (7 total so far).

- [ ] **Step 6: Commit**

```bash
git add live_data.py tests/test_consensus_divergence.py
git commit -m "refactor(live_data): extract _parse_odds_api_game and capture last_update per book"
```

---

### Task 5: `consensus_implied_prob` — de-vigged median consensus with staleness + min_books

**Files:**
- Modify: `consensus_divergence.py` (add function)
- Modify: `tests/test_consensus_divergence.py` (add test class)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_consensus_divergence.py`:

```python
from consensus_divergence import consensus_implied_prob


def _book(name, home_ml, away_ml, last_update="2026-04-16T22:55:00Z"):
    """Shape-compatible with live_data._parse_odds_api_game output."""
    return {
        "name": name,
        "last_update": last_update,
        "home_moneyline": home_ml,
        "away_moneyline": away_ml,
    }


class TestConsensusImpliedProb:
    NOW = "2026-04-16T23:00:00Z"  # reference "current time" for staleness

    def test_five_fresh_books_uses_median(self):
        # Five books roughly agreeing on -140 / +120 (home_prob ~0.575 after devig)
        books = [
            _book("DK", -140, 120),
            _book("FD", -145, 125),
            _book("MGM", -135, 115),
            _book("Caesars", -150, 130),
            _book("PointsBet", -138, 118),
        ]
        result = consensus_implied_prob(books, now_iso=self.NOW)
        assert result is not None
        assert result["n_books"] == 5
        assert len(result["books_used"]) == 5
        # Home prob should be near median of de-vigged home probs ~0.575
        assert 0.55 < result["home_prob"] < 0.60
        assert result["home_prob"] + result["away_prob"] == pytest.approx(1.0, abs=1e-9)

    def test_drops_stale_quotes(self):
        # 3 fresh + 2 stale (>300s old relative to NOW)
        books = [
            _book("DK", -140, 120),
            _book("FD", -145, 125),
            _book("MGM", -135, 115),
            _book("Caesars", -150, 130, last_update="2026-04-16T22:30:00Z"),
            _book("PointsBet", -138, 118, last_update="2026-04-16T22:40:00Z"),
        ]
        result = consensus_implied_prob(books, now_iso=self.NOW)
        assert result is not None
        assert result["n_books"] == 3
        assert "Caesars" not in result["books_used"]
        assert "PointsBet" not in result["books_used"]

    def test_below_min_books_returns_none(self):
        # Only 2 fresh books (below default min_books=3)
        books = [
            _book("DK", -140, 120),
            _book("FD", -145, 125, last_update="2026-04-16T22:30:00Z"),  # stale
            _book("MGM", -135, 115, last_update="2026-04-16T22:30:00Z"),  # stale
        ]
        result = consensus_implied_prob(books, now_iso=self.NOW)
        assert result is None

    def test_all_malformed_returns_none(self):
        books = [
            _book("DK", None, 120),
            _book("FD", -145, 0),
            _book("MGM", "", 115),
        ]
        result = consensus_implied_prob(books, now_iso=self.NOW)
        assert result is None

    def test_clamps_extreme_probabilities(self):
        # Heavy favorite: -2000 / +1000 devigs to ~0.95 / 0.05 — should clamp fine
        books = [
            _book("DK", -2000, 1000),
            _book("FD", -2200, 1100),
            _book("MGM", -1800, 900),
        ]
        result = consensus_implied_prob(books, now_iso=self.NOW)
        assert result is not None
        assert result["home_prob"] <= 0.99
        assert result["away_prob"] >= 0.01
        assert result["home_prob"] + result["away_prob"] == pytest.approx(1.0, abs=1e-9)

    def test_empty_book_list(self):
        assert consensus_implied_prob([], now_iso=self.NOW) is None

    def test_missing_last_update_is_considered_fresh(self):
        # If a book lacks a timestamp, it's treated as fresh (don't penalize upstream bugs)
        books = [
            _book("DK", -140, 120, last_update=None),
            _book("FD", -145, 125, last_update=None),
            _book("MGM", -135, 115, last_update=None),
        ]
        result = consensus_implied_prob(books, now_iso=self.NOW)
        assert result is not None
        assert result["n_books"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_consensus_divergence.py::TestConsensusImpliedProb -v
```

Expected: FAIL with `ImportError: cannot import name 'consensus_implied_prob'`.

- [ ] **Step 3: Implement `consensus_implied_prob` in `consensus_divergence.py`**

Append to `consensus_divergence.py`:

```python
from statistics import median

from config import CONSENSUS_MIN_BOOKS, CONSENSUS_STALE_SECONDS


def _parse_iso_utc_seconds(iso_str):
    """ISO-8601 string → POSIX seconds (UTC). Returns None on failure."""
    if not iso_str:
        return None
    try:
        # Accept both 'Z' and '+00:00' suffixes
        normalized = iso_str.replace("Z", "+00:00") if isinstance(iso_str, str) else iso_str
        from datetime import datetime
        return datetime.fromisoformat(normalized).timestamp()
    except (TypeError, ValueError):
        return None


def _is_fresh(last_update_iso, now_iso, stale_seconds) -> bool:
    """True if the quote is fresh enough OR if we cannot determine its age
    (missing timestamps must not silently discard the whole consensus)."""
    if last_update_iso is None:
        return True
    quote_ts = _parse_iso_utc_seconds(last_update_iso)
    now_ts = _parse_iso_utc_seconds(now_iso)
    if quote_ts is None or now_ts is None:
        return True
    return (now_ts - quote_ts) <= stale_seconds


def consensus_implied_prob(
    sportsbooks,
    now_iso=None,
    stale_seconds: int = CONSENSUS_STALE_SECONDS,
    min_books: int = CONSENSUS_MIN_BOOKS,
) -> dict | None:
    """
    Build a de-vigged consensus probability across sportsbooks.

    sportsbooks: list of dicts with keys:
        name (str), last_update (ISO-8601 str or None),
        home_moneyline (int | None), away_moneyline (int | None).

    now_iso: reference "now" for staleness. None → uses datetime.utcnow() ISO.

    Returns {home_prob, away_prob, n_books, books_used} or None when fewer
    than `min_books` non-stale, well-formed books remain.
    """
    if not sportsbooks:
        return None

    if now_iso is None:
        from datetime import datetime, timezone
        now_iso = datetime.now(tz=timezone.utc).isoformat()

    home_probs: list[float] = []
    away_probs: list[float] = []
    books_used: list[str] = []

    for book in sportsbooks:
        if not _is_fresh(book.get("last_update"), now_iso, stale_seconds):
            continue
        devigged = devig_proportional(
            book.get("home_moneyline"),
            book.get("away_moneyline"),
        )
        if devigged is None:
            continue
        home_probs.append(devigged[0])
        away_probs.append(devigged[1])
        books_used.append(book.get("name", ""))

    if len(home_probs) < min_books:
        return None

    home_prob = median(home_probs)
    away_prob = median(away_probs)

    # Clamp and re-normalize so the pair sums exactly to 1.
    home_prob = min(max(home_prob, 0.01), 0.99)
    away_prob = min(max(away_prob, 0.01), 0.99)
    total = home_prob + away_prob
    home_prob /= total
    away_prob /= total

    return {
        "home_prob": round(home_prob, 6),
        "away_prob": round(away_prob, 6),
        "n_books": len(books_used),
        "books_used": books_used,
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_consensus_divergence.py::TestConsensusImpliedProb -v
```

Expected: 7 passed.

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -v
```

Expected: all tests still pass (14 total so far).

- [ ] **Step 6: Commit**

```bash
git add consensus_divergence.py tests/test_consensus_divergence.py
git commit -m "feat(consensus): de-vigged median consensus with staleness + min_books filters"
```

---

### Task 6: `compute_divergence` + tier-assignment rules

**Files:**
- Modify: `consensus_divergence.py` (add function)
- Modify: `tests/test_consensus_divergence.py` (add test class)

The most important tests in this whole plan are the **tier boundary tests** — silent regressions here change live bet decisions.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_consensus_divergence.py`:

```python
from consensus_divergence import compute_divergence


class TestComputeDivergence:
    """
    CUTOFF = config.CONSENSUS_DIVERGENCE_THRESHOLD_PP (3.5pp).
    Tier 2: Kalshi AND model both disagree with consensus by >= CUTOFF,
            and they disagree in the SAME direction.
    Tier -1: Kalshi AND consensus disagree with model by >= CUTOFF,
             pointing in OPPOSITE directions.
    Tier 1: everything else (including any missing inputs).
    """

    def _consensus(self, home_prob):
        return {"home_prob": home_prob, "away_prob": 1 - home_prob,
                "n_books": 4, "books_used": ["DK", "FD", "MGM", "Caesars"]}

    def test_tier_2_triangulated_same_direction(self):
        # Kalshi says home 50%, consensus says 60%, model says 62%.
        # kalshi_vs_consensus = 10pp, kalshi_vs_model = 12pp, same direction.
        div = compute_divergence(
            kalshi_yes_mid=0.50,
            model_prob_home=0.62,
            consensus=self._consensus(0.60),
        )
        assert div["triangulation_tier"] == 2
        assert div["kalshi_vs_consensus_pp"] == pytest.approx(10.0, abs=0.01)
        assert div["model_vs_consensus_pp"] == pytest.approx(2.0, abs=0.01)
        assert div["kalshi_vs_model_pp"] == pytest.approx(12.0, abs=0.01)

    def test_tier_minus_1_model_vs_consensus_opposite(self):
        # Kalshi 50%, consensus 58% (Kalshi under), model 44% (Kalshi over).
        # Consensus and model point Kalshi in opposite directions, both >= 3.5pp.
        div = compute_divergence(
            kalshi_yes_mid=0.50,
            model_prob_home=0.44,
            consensus=self._consensus(0.58),
        )
        assert div["triangulation_tier"] == -1

    def test_tier_1_below_cutoff(self):
        # All disagreements < 3.5pp
        div = compute_divergence(
            kalshi_yes_mid=0.50,
            model_prob_home=0.52,
            consensus=self._consensus(0.51),
        )
        assert div["triangulation_tier"] == 1

    def test_tier_boundary_exactly_at_cutoff_is_tier_2(self):
        # Kalshi 50%, consensus 53.5%, model 53.5% — exactly at CUTOFF, same direction.
        # Spec: tier 2 triggers at >= CUTOFF (inclusive).
        div = compute_divergence(
            kalshi_yes_mid=0.50,
            model_prob_home=0.535,
            consensus=self._consensus(0.535),
        )
        assert div["triangulation_tier"] == 2

    def test_tier_boundary_just_below_cutoff_is_tier_1(self):
        div = compute_divergence(
            kalshi_yes_mid=0.50,
            model_prob_home=0.534,
            consensus=self._consensus(0.534),
        )
        assert div["triangulation_tier"] == 1

    def test_missing_model_defaults_to_tier_1(self):
        div = compute_divergence(
            kalshi_yes_mid=0.50,
            model_prob_home=None,
            consensus=self._consensus(0.62),
        )
        assert div["triangulation_tier"] == 1
        assert div["model_vs_consensus_pp"] is None
        assert div["kalshi_vs_model_pp"] is None
        assert div["kalshi_vs_consensus_pp"] == pytest.approx(12.0, abs=0.01)

    def test_missing_kalshi_defaults_to_tier_1(self):
        div = compute_divergence(
            kalshi_yes_mid=None,
            model_prob_home=0.62,
            consensus=self._consensus(0.60),
        )
        assert div["triangulation_tier"] == 1
        assert div["kalshi_vs_consensus_pp"] is None
        assert div["kalshi_vs_model_pp"] is None

    def test_missing_consensus_defaults_to_tier_1(self):
        div = compute_divergence(
            kalshi_yes_mid=0.50,
            model_prob_home=0.62,
            consensus=None,
        )
        assert div["triangulation_tier"] == 1
        assert div["kalshi_vs_consensus_pp"] is None
        assert div["model_vs_consensus_pp"] is None

    def test_tier_reason_is_descriptive(self):
        div = compute_divergence(0.50, 0.62, self._consensus(0.60))
        assert isinstance(div["tier_reason"], str)
        assert len(div["tier_reason"]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_consensus_divergence.py::TestComputeDivergence -v
```

Expected: FAIL with `ImportError: cannot import name 'compute_divergence'`.

- [ ] **Step 3: Implement `compute_divergence`**

Append to `consensus_divergence.py`:

```python
from config import CONSENSUS_DIVERGENCE_THRESHOLD_PP


def compute_divergence(
    kalshi_yes_mid,
    model_prob_home,
    consensus,
) -> dict:
    """
    Compute Kalshi-vs-consensus, model-vs-consensus, and kalshi-vs-model
    divergences (in percentage points) and assign a triangulation tier.

    All inputs may be None; missing inputs yield None divergences and tier=1.
    """
    consensus_home = consensus["home_prob"] if consensus else None

    kalshi_vs_consensus_pp = None
    if kalshi_yes_mid is not None and consensus_home is not None:
        kalshi_vs_consensus_pp = (consensus_home - kalshi_yes_mid) * 100.0

    model_vs_consensus_pp = None
    if model_prob_home is not None and consensus_home is not None:
        model_vs_consensus_pp = (consensus_home - model_prob_home) * 100.0

    kalshi_vs_model_pp = None
    if kalshi_yes_mid is not None and model_prob_home is not None:
        kalshi_vs_model_pp = (model_prob_home - kalshi_yes_mid) * 100.0

    tier, tier_reason = _assign_tier(
        kalshi_vs_consensus_pp,
        kalshi_vs_model_pp,
    )

    return {
        "kalshi_vs_consensus_pp": (
            round(kalshi_vs_consensus_pp, 4) if kalshi_vs_consensus_pp is not None else None
        ),
        "model_vs_consensus_pp": (
            round(model_vs_consensus_pp, 4) if model_vs_consensus_pp is not None else None
        ),
        "kalshi_vs_model_pp": (
            round(kalshi_vs_model_pp, 4) if kalshi_vs_model_pp is not None else None
        ),
        "triangulation_tier": tier,
        "tier_reason": tier_reason,
    }


def _assign_tier(kalshi_vs_consensus_pp, kalshi_vs_model_pp) -> tuple[int, str]:
    cutoff = CONSENSUS_DIVERGENCE_THRESHOLD_PP

    if kalshi_vs_consensus_pp is None or kalshi_vs_model_pp is None:
        return 1, "neutral (missing input)"

    big_consensus = abs(kalshi_vs_consensus_pp) >= cutoff
    big_model = abs(kalshi_vs_model_pp) >= cutoff

    if not (big_consensus and big_model):
        return 1, f"neutral (|k-c|={abs(kalshi_vs_consensus_pp):.1f}pp, |k-m|={abs(kalshi_vs_model_pp):.1f}pp)"

    same_direction = (kalshi_vs_consensus_pp * kalshi_vs_model_pp) > 0
    if same_direction:
        return 2, (
            f"triangulated: both consensus ({kalshi_vs_consensus_pp:+.1f}pp) and "
            f"model ({kalshi_vs_model_pp:+.1f}pp) disagree with Kalshi in the same direction"
        )
    return -1, (
        f"contradicting: consensus says {kalshi_vs_consensus_pp:+.1f}pp and "
        f"model says {kalshi_vs_model_pp:+.1f}pp — they point opposite ways"
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_consensus_divergence.py::TestComputeDivergence -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add consensus_divergence.py tests/test_consensus_divergence.py
git commit -m "feat(consensus): add compute_divergence with tier assignment rules"
```

---

### Task 7: `tier_to_threshold_multiplier`

**Files:**
- Modify: `consensus_divergence.py` (add function)
- Modify: `tests/test_consensus_divergence.py` (add test class)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_consensus_divergence.py`:

```python
from consensus_divergence import tier_to_threshold_multiplier


class TestTierToThresholdMultiplier:
    def test_tier_2_lowers_threshold(self):
        assert tier_to_threshold_multiplier(2, 0.03) == pytest.approx(0.021)

    def test_tier_1_unchanged(self):
        assert tier_to_threshold_multiplier(1, 0.03) == pytest.approx(0.03)

    def test_tier_minus_1_raises_threshold(self):
        assert tier_to_threshold_multiplier(-1, 0.03) == pytest.approx(0.045)

    def test_unknown_tier_falls_back_to_neutral(self):
        # Defensive: an unexpected tier should not crash; behave as tier 1.
        assert tier_to_threshold_multiplier(99, 0.03) == pytest.approx(0.03)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_consensus_divergence.py::TestTierToThresholdMultiplier -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `tier_to_threshold_multiplier`**

Append to `consensus_divergence.py`:

```python
from config import TIER_THRESHOLD_MULTIPLIERS


def tier_to_threshold_multiplier(tier: int, base_threshold: float) -> float:
    """
    Map a triangulation tier to an adjusted edge threshold.

    Multipliers are read from config.TIER_THRESHOLD_MULTIPLIERS so they
    can be tuned without code changes. Unknown tiers fall back to 1.0.
    """
    multiplier = TIER_THRESHOLD_MULTIPLIERS.get(tier, 1.0)
    return base_threshold * multiplier
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_consensus_divergence.py::TestTierToThresholdMultiplier -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add consensus_divergence.py tests/test_consensus_divergence.py
git commit -m "feat(consensus): add tier_to_threshold_multiplier"
```

---

### Task 8: Golden-data snapshot test

**Files:**
- Create: `tests/fixtures/consensus_golden_cases.csv`
- Modify: `tests/test_consensus_divergence.py` (add snapshot test)

- [ ] **Step 1: Create `tests/fixtures/consensus_golden_cases.csv`**

```
case_id,kalshi_yes,model_prob,consensus_home,expected_kalshi_vs_consensus_pp,expected_model_vs_consensus_pp,expected_kalshi_vs_model_pp,expected_tier
1_all_aligned,0.50,0.50,0.50,0.00,0.00,0.00,1
2_tiny_drift_below_cutoff,0.50,0.52,0.51,1.00,-1.00,2.00,1
3_just_below_cutoff,0.50,0.534,0.534,3.40,0.00,3.40,1
4_exact_cutoff_same_direction,0.50,0.535,0.535,3.50,0.00,3.50,2
5_above_cutoff_same_direction,0.50,0.60,0.60,10.00,0.00,10.00,2
6_heavy_triangulation,0.40,0.55,0.52,12.00,-3.00,15.00,2
7_kalshi_high_consensus_model_agree,0.70,0.55,0.58,-12.00,3.00,-15.00,2
8_opposite_direction_at_cutoff,0.50,0.465,0.535,3.50,7.00,-3.50,-1
9_opposite_direction_large,0.55,0.40,0.63,8.00,23.00,-15.00,-1
10_consensus_neutral_only_model_diverges,0.50,0.55,0.51,1.00,-4.00,5.00,1
11_model_neutral_only_consensus_diverges,0.50,0.51,0.55,5.00,4.00,1.00,1
12_both_diverge_same_sign_large,0.35,0.50,0.48,13.00,-2.00,15.00,2
13_heavy_favorite_triangulated,0.85,0.75,0.78,-7.00,3.00,-10.00,2
14_heavy_favorite_contradicting,0.85,0.90,0.80,-5.00,-10.00,5.00,-1
15_kalshi_underpriced_triangulated,0.30,0.42,0.40,10.00,-2.00,12.00,2
16_kalshi_slightly_off,0.48,0.50,0.49,1.00,-1.00,2.00,1
17_tier_minus_one_mirror,0.50,0.56,0.44,-6.00,-12.00,6.00,-1
18_wide_triangulated_away_side,0.25,0.10,0.12,-13.00,2.00,-15.00,2
19_neutral_model_missing,0.50,,0.62,12.00,,,1
20_neutral_kalshi_missing,,0.60,0.62,,2.00,,1
```

Note: rows 19 and 20 use empty cells for missing inputs; the test reads these as `None`.

- [ ] **Step 2: Append the snapshot test**

Append to `tests/test_consensus_divergence.py`:

```python
import csv
from pathlib import Path


class TestGoldenSnapshot:
    FIXTURE = Path(__file__).parent / "fixtures" / "consensus_golden_cases.csv"

    def _maybe_float(self, s):
        return None if s == "" else float(s)

    def _maybe_int(self, s):
        return None if s == "" else int(s)

    def test_golden_cases_all_match(self):
        rows = list(csv.DictReader(self.FIXTURE.open()))
        assert len(rows) >= 20, "Golden fixture should have >= 20 cases"

        failures = []
        for row in rows:
            kalshi = self._maybe_float(row["kalshi_yes"])
            model = self._maybe_float(row["model_prob"])
            cons_home = self._maybe_float(row["consensus_home"])
            consensus = (
                {"home_prob": cons_home, "away_prob": 1 - cons_home,
                 "n_books": 5, "books_used": []}
                if cons_home is not None else None
            )

            result = compute_divergence(kalshi, model, consensus)

            expected_tier = int(row["expected_tier"])
            if result["triangulation_tier"] != expected_tier:
                failures.append(
                    f"{row['case_id']}: tier expected={expected_tier} "
                    f"got={result['triangulation_tier']} reason={result['tier_reason']}"
                )

            for csv_key, result_key in (
                ("expected_kalshi_vs_consensus_pp", "kalshi_vs_consensus_pp"),
                ("expected_model_vs_consensus_pp",  "model_vs_consensus_pp"),
                ("expected_kalshi_vs_model_pp",     "kalshi_vs_model_pp"),
            ):
                exp = self._maybe_float(row[csv_key])
                got = result[result_key]
                if exp is None and got is None:
                    continue
                if exp is None or got is None:
                    failures.append(f"{row['case_id']}: {result_key} expected={exp} got={got}")
                    continue
                if abs(exp - got) > 0.01:
                    failures.append(f"{row['case_id']}: {result_key} expected={exp} got={got}")

        assert not failures, "Golden snapshot mismatches:\n" + "\n".join(failures)
```

- [ ] **Step 3: Run the snapshot test**

```bash
pytest tests/test_consensus_divergence.py::TestGoldenSnapshot -v
```

Expected: 1 passed (all 20 rows match).

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/consensus_golden_cases.csv tests/test_consensus_divergence.py
git commit -m "test(consensus): golden snapshot covering tier boundaries and edge cases"
```

---

### Task 9: `apply_consensus_tier` — orchestrator for ev_analyzer

**Files:**
- Modify: `consensus_divergence.py` (add function)
- Modify: `tests/test_consensus_divergence.py` (add test class)

This is the single entry point that `ev_analyzer.py` will call. It takes the pieces ev_analyzer already has (comparison dict from `get_odds_comparison`, model prob, kalshi bid/ask, base threshold) and returns an "adjusted" threshold + full divergence context for logging.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_consensus_divergence.py`:

```python
from consensus_divergence import apply_consensus_tier


class TestApplyConsensusTier:
    def _comparison(self, books):
        return {"home_team": "BOS", "away_team": "MIA", "sportsbooks": books}

    def _good_books(self):
        now = "2026-04-16T23:00:00Z"
        return [
            {"name": "DK", "last_update": now, "home_moneyline": -140, "away_moneyline": 120},
            {"name": "FD", "last_update": now, "home_moneyline": -145, "away_moneyline": 125},
            {"name": "MGM", "last_update": now, "home_moneyline": -135, "away_moneyline": 115},
            {"name": "Caesars", "last_update": now, "home_moneyline": -150, "away_moneyline": 130},
        ]

    def test_shadow_mode_does_not_adjust_threshold(self):
        result = apply_consensus_tier(
            comparison=self._comparison(self._good_books()),
            model_prob_home=0.60,
            kalshi_yes_mid=0.48,
            bet_side="home",
            base_threshold=0.03,
            mode="shadow",
            now_iso="2026-04-16T23:00:00Z",
        )
        assert result["signal_threshold"] == pytest.approx(0.03)
        assert result["shadow_signal_threshold"] != result["signal_threshold"]
        assert result["shadow_signal_threshold"] == pytest.approx(0.03 * 0.7)
        assert result["triangulation_tier"] == 2
        assert result["mode"] == "shadow"

    def test_active_mode_applies_multiplier(self):
        result = apply_consensus_tier(
            comparison=self._comparison(self._good_books()),
            model_prob_home=0.60,
            kalshi_yes_mid=0.48,
            bet_side="home",
            base_threshold=0.03,
            mode="active",
        )
        assert result["signal_threshold"] == pytest.approx(0.03 * 0.7)
        assert result["triangulation_tier"] == 2

    def test_off_mode_pins_tier_one(self):
        result = apply_consensus_tier(
            comparison=self._comparison(self._good_books()),
            model_prob_home=0.60,
            kalshi_yes_mid=0.48,
            bet_side="home",
            base_threshold=0.03,
            mode="off",
        )
        assert result["triangulation_tier"] == 1
        assert result["signal_threshold"] == pytest.approx(0.03)

    def test_away_bet_side_inverts_kalshi_yes_mid(self):
        # On an "away YES" ticker, yes_mid=0.48 means the market thinks AWAY wins 48%,
        # so home wins 52%. Divergence against consensus should use 0.52.
        result = apply_consensus_tier(
            comparison=self._comparison(self._good_books()),
            model_prob_home=0.60,
            kalshi_yes_mid=0.48,
            bet_side="away",
            base_threshold=0.03,
            mode="active",
        )
        # Consensus home ~0.575; kalshi-inverted 0.52 → disagreement ~5.5pp (>cutoff)
        # model 0.60 vs consensus 0.575 → disagreement ~2.5pp (<cutoff) → tier 1
        assert result["triangulation_tier"] == 1

    def test_missing_sportsbooks_yields_tier_one(self):
        result = apply_consensus_tier(
            comparison={"home_team": "BOS", "away_team": "MIA", "sportsbooks": []},
            model_prob_home=0.60,
            kalshi_yes_mid=0.48,
            bet_side="home",
            base_threshold=0.03,
            mode="active",
        )
        assert result["triangulation_tier"] == 1
        assert result["consensus_n_books"] == 0
        assert result["signal_threshold"] == pytest.approx(0.03)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_consensus_divergence.py::TestApplyConsensusTier -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `apply_consensus_tier`**

Append to `consensus_divergence.py`:

```python
def apply_consensus_tier(
    comparison: dict,
    model_prob_home: float | None,
    kalshi_yes_mid: float | None,
    bet_side: str,
    base_threshold: float,
    mode: str = "shadow",
    now_iso: str | None = None,
) -> dict:
    """
    Single entry point for ev_analyzer.

    Converts kalshi_yes_mid to a home-side probability (if bet_side='away',
    the yes side represents away winning, so home_prob = 1 - yes_mid),
    computes consensus, divergences, and tier, then emits an adjusted
    threshold gated by `mode`:

        shadow  → signal_threshold = base_threshold (unchanged);
                  shadow_signal_threshold = what it WOULD be in active.
        active  → signal_threshold = base_threshold * TIER_THRESHOLD_MULTIPLIERS[tier].
        off     → tier forced to 1; signal_threshold = base_threshold.

    Returns a dict the caller can log directly.
    """
    kalshi_home_prob = None
    if kalshi_yes_mid is not None:
        kalshi_home_prob = kalshi_yes_mid if bet_side == "home" else (1.0 - kalshi_yes_mid)

    books = comparison.get("sportsbooks", []) if comparison else []
    books_input = [
        {
            "name": b.get("name", ""),
            "last_update": b.get("last_update"),
            "home_moneyline": b.get("home_moneyline"),
            "away_moneyline": b.get("away_moneyline"),
        }
        for b in books
    ]
    consensus = consensus_implied_prob(books_input, now_iso=now_iso)
    div = compute_divergence(kalshi_home_prob, model_prob_home, consensus)

    tier = div["triangulation_tier"]
    if mode == "off":
        tier = 1
    elif mode not in ("shadow", "active"):
        # Defensive default: any unknown mode behaves like "off" (safer than "active").
        tier = 1

    shadow_signal_threshold = tier_to_threshold_multiplier(
        div["triangulation_tier"], base_threshold
    )

    if mode == "active":
        signal_threshold = tier_to_threshold_multiplier(tier, base_threshold)
    else:
        signal_threshold = base_threshold

    return {
        "mode": mode,
        "triangulation_tier": tier,
        "tier_reason": div["tier_reason"],
        "kalshi_vs_consensus_pp": div["kalshi_vs_consensus_pp"],
        "model_vs_consensus_pp": div["model_vs_consensus_pp"],
        "kalshi_vs_model_pp": div["kalshi_vs_model_pp"],
        "consensus_home_prob": consensus["home_prob"] if consensus else None,
        "consensus_n_books": consensus["n_books"] if consensus else 0,
        "signal_threshold": signal_threshold,
        "shadow_signal_threshold": shadow_signal_threshold,
        "base_threshold": base_threshold,
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_consensus_divergence.py::TestApplyConsensusTier -v
```

Expected: 5 passed.

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add consensus_divergence.py tests/test_consensus_divergence.py
git commit -m "feat(consensus): add apply_consensus_tier orchestrator with mode gating"
```

---

### Task 10: Refactor `_build_consensus_map` to use the new function

**Files:**
- Modify: `realtime_feature_store.py` (lines 550-579)
- Create: `tests/test_integrations.py`

The existing implementation averages raw implied probabilities (no de-vig, no median, no staleness filter, no min_books). After this task, it uses `consensus_implied_prob`, which changes behavior: the stored `oddsapi_home_consensus` / `oddsapi_away_consensus` values will shift slightly (vig is now removed) and may be `None` when fewer than 3 fresh books are available.

- [ ] **Step 1: Write the failing test**

Create `tests/test_integrations.py`:

```python
"""
Integration smokes — Layer 2 tests from the spec.
Exercise the refactored _build_consensus_map and the extended
build_feature_rows columns in realtime_feature_store.py.
"""

from unittest.mock import patch

import pandas as pd
import pytest


def _odds_api_payload_shape(home, away, books_price_pairs, last_update="2026-04-16T22:58:00Z"):
    """Return a list[dict] shaped like fetch_odds_api_lines' output."""
    return [{
        "home": home,
        "away": away,
        "commence": "2026-04-16T23:30:00Z",
        "books": [
            {
                "name": name,
                "last_update": last_update,
                "markets": {
                    "h2h": {
                        home: {"price": h_ml, "point": None},
                        away: {"price": a_ml, "point": None},
                    }
                },
            }
            for (name, h_ml, a_ml) in books_price_pairs
        ],
    }]


class TestBuildConsensusMapRefactor:
    def test_uses_devigged_median_when_five_books(self):
        from realtime_feature_store import LiveFeatureStore

        payload = _odds_api_payload_shape(
            "BOS", "MIA",
            [
                ("DK", -140, 120),
                ("FD", -145, 125),
                ("MGM", -135, 115),
                ("Caesars", -150, 130),
                ("PointsBet", -138, 118),
            ],
        )

        with patch("realtime_feature_store.fetch_odds_api_lines", return_value=payload):
            store = LiveFeatureStore.__new__(LiveFeatureStore)
            store.state = {"games": {}, "markets": {}}
            result = store._build_consensus_map()

        assert ("BOS", "MIA") in result
        consensus = result[("BOS", "MIA")]
        # De-vigged median home prob should be ~0.575, below the un-devigged 0.583
        assert 0.56 < consensus["home_prob"] < 0.60
        assert consensus["n_books"] == 5
        # Sum to 1 after re-normalization (de-vigged)
        assert consensus["home_prob"] + consensus["away_prob"] == pytest.approx(1.0, abs=1e-6)

    def test_returns_empty_when_below_min_books(self):
        from realtime_feature_store import LiveFeatureStore

        payload = _odds_api_payload_shape("BOS", "MIA", [("DK", -140, 120)])
        with patch("realtime_feature_store.fetch_odds_api_lines", return_value=payload):
            store = LiveFeatureStore.__new__(LiveFeatureStore)
            store.state = {"games": {}, "markets": {}}
            result = store._build_consensus_map()
        # Game key absent or present with None probs — either signals "no consensus".
        consensus = result.get(("BOS", "MIA"))
        assert consensus is None or consensus.get("home_prob") is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_integrations.py::TestBuildConsensusMapRefactor -v
```

Expected: FAIL — the first test gets a `home_prob` close to 0.583 (un-devigged), failing the `< 0.60` bound OR the strict sum-to-1 check; the second test gets a non-None consensus with 1 book (no min_books check).

- [ ] **Step 3: Refactor `_build_consensus_map` to call `consensus_implied_prob`**

Open `realtime_feature_store.py`. Add import near the top (after line 39):

```python
from consensus_divergence import consensus_implied_prob
```

Replace the entire `_build_consensus_map` method (lines 550-579) with:

```python
    def _build_consensus_map(self) -> dict[tuple[str, str], dict]:
        """
        Build per-matchup de-vigged consensus from The Odds API.

        Uses consensus_divergence.consensus_implied_prob so behavior matches
        ev_analyzer and the CLI (proportional de-vig, median across books,
        staleness filter, min_books gate).
        """
        consensus_map: dict[tuple[str, str], dict] = {}
        odds_games = fetch_odds_api_lines()
        for game in odds_games:
            home = game.get("home")
            away = game.get("away")
            if not home or not away:
                continue

            books_input = []
            for book in game.get("books", []):
                h2h = book.get("markets", {}).get("h2h", {})
                books_input.append({
                    "name": book.get("name", ""),
                    "last_update": book.get("last_update"),
                    "home_moneyline": h2h.get(home, {}).get("price"),
                    "away_moneyline": h2h.get(away, {}).get("price"),
                })

            consensus = consensus_implied_prob(books_input)
            if consensus is None:
                continue
            consensus_map[(home, away)] = {
                "home_prob": round(consensus["home_prob"], 4),
                "away_prob": round(consensus["away_prob"], 4),
                "n_books": consensus["n_books"],
                "commence": game.get("commence"),
            }
        return consensus_map
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_integrations.py::TestBuildConsensusMapRefactor -v
```

Expected: 2 passed.

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add realtime_feature_store.py tests/test_integrations.py
git commit -m "refactor(feature_store): de-vigged median consensus via consensus_divergence"
```

---

### Task 11: Add five divergence columns to `build_feature_rows`

**Files:**
- Modify: `realtime_feature_store.py` (`build_feature_rows` method, around line 677-741)
- Modify: `tests/test_integrations.py` (add test class)

Existing columns `oddsapi_home_consensus`, `consensus_gap_home` stay; we *add* `kalshi_vs_consensus_pp`, `model_vs_consensus_pp`, `kalshi_vs_model_pp`, `triangulation_tier`, `consensus_n_books` alongside them. (`consensus_gap_home` is expressed in raw probability units; the new `kalshi_vs_consensus_pp` is in percentage points — both kept for backwards compatibility.)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_integrations.py`:

```python
class TestFeatureRowDivergenceColumns:
    REQUIRED_COLS = {
        "kalshi_vs_consensus_pp",
        "model_vs_consensus_pp",
        "kalshi_vs_model_pp",
        "triangulation_tier",
        "consensus_n_books",
    }

    def _minimal_store(self, consensus_map):
        """Create a LiveFeatureStore with stubbed consensus."""
        from realtime_feature_store import LiveFeatureStore
        store = LiveFeatureStore.__new__(LiveFeatureStore)
        store.state = {"games": {}, "markets": {}}
        store._build_consensus_map = lambda: consensus_map
        return store

    def _markets_df(self, bet_side="home", yes_mid=0.48):
        yes_bid = yes_mid - 0.01
        yes_ask = yes_mid + 0.01
        return pd.DataFrame([{
            "captured_at": pd.Timestamp("2026-04-16T23:00:00", tz="UTC"),
            "ticker": "KXNBAGAME-26APR16BOSMIA-BOS",
            "event_ticker": "KXNBAGAME-26APR16BOSMIA",
            "home_team": "BOS",
            "away_team": "MIA",
            "bet_team": "BOS",
            "bet_side": bet_side,
            "status": "active",
            "game_date": "2026-04-16",
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "yes_mid": yes_mid,
            "no_bid": 1 - yes_ask,
            "no_ask": 1 - yes_bid,
            "last_price": yes_mid,
            "volume": 100.0,
            "open_interest": 50.0,
            "yes_depth_notional_3": 200.0,
            "yes_depth_notional_5": 300.0,
            "no_depth_notional_3": 200.0,
            "no_depth_notional_5": 300.0,
            "yes_weighted_price_3": yes_mid,
            "no_weighted_price_3": 1 - yes_mid,
            "market_home_implied": yes_mid if bet_side == "home" else 1 - yes_mid,
            "espn_home_implied": None,
            "espn_away_implied": None,
            "game_key": "2026-04-16|BOS|MIA",
        }])

    def test_columns_present_when_consensus_available(self):
        consensus_map = {("BOS", "MIA"): {
            "home_prob": 0.60, "away_prob": 0.40, "n_books": 4, "commence": None,
        }}
        store = self._minimal_store(consensus_map)
        with patch("realtime_feature_store.get_model_prediction",
                   return_value={"home_win_prob": 0.62, "away_win_prob": 0.38,
                                 "predicted_spread": -3.5, "predicted_total": 225,
                                 "data_date": "2026-04-16"}):
            df = store.build_feature_rows(
                games_df=pd.DataFrame(),
                markets_df=self._markets_df(),
                include_consensus=True,
            )
        row = df.iloc[0]
        assert set(df.columns) >= self.REQUIRED_COLS
        # kalshi_home = 0.48, consensus_home = 0.60 → 12pp
        assert row["kalshi_vs_consensus_pp"] == pytest.approx(12.0, abs=0.01)
        # model 0.62 vs consensus 0.60 → -2pp
        assert row["model_vs_consensus_pp"] == pytest.approx(-2.0, abs=0.01)
        # kalshi 0.48 vs model 0.62 → 14pp
        assert row["kalshi_vs_model_pp"] == pytest.approx(14.0, abs=0.01)
        # |k-c|=12 >= 3.5, |k-m|=14 >= 3.5, same direction → tier 2
        assert row["triangulation_tier"] == 2
        assert row["consensus_n_books"] == 4

    def test_columns_nan_when_consensus_missing(self):
        store = self._minimal_store({})  # no consensus entries
        with patch("realtime_feature_store.get_model_prediction",
                   return_value={"home_win_prob": 0.62, "away_win_prob": 0.38,
                                 "predicted_spread": -3.5, "predicted_total": 225,
                                 "data_date": "2026-04-16"}):
            df = store.build_feature_rows(
                games_df=pd.DataFrame(),
                markets_df=self._markets_df(),
                include_consensus=True,
            )
        row = df.iloc[0]
        assert set(df.columns) >= self.REQUIRED_COLS
        # Without consensus, kalshi_vs_consensus and model_vs_consensus are NaN.
        assert pd.isna(row["kalshi_vs_consensus_pp"])
        assert pd.isna(row["model_vs_consensus_pp"])
        # Tier defaults to 1 when consensus is missing.
        assert row["triangulation_tier"] == 1
        assert row["consensus_n_books"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_integrations.py::TestFeatureRowDivergenceColumns -v
```

Expected: FAIL — the new columns are not in the dataframe.

- [ ] **Step 3: Extend `build_feature_rows` with the new columns**

Open `realtime_feature_store.py`. Add import at the top (after the `consensus_implied_prob` import added in Task 10):

```python
from consensus_divergence import compute_divergence
```

Inside `build_feature_rows`, locate the `feature_row = { ... }` assignment (starts at line 677). **Immediately before** the dict is built (i.e., just before line 677), add:

```python
            # Divergence signal (IMPROVEMENTS #8)
            consensus_for_div = (
                {"home_prob": consensus.get("home_prob"),
                 "away_prob": consensus.get("away_prob"),
                 "n_books": consensus.get("n_books", 0)}
                if consensus and consensus.get("home_prob") is not None
                else None
            )
            kalshi_home_prob = market_row.get("market_home_implied")
            model_home = pregame.get("home_win_prob")
            div = compute_divergence(
                kalshi_yes_mid=kalshi_home_prob,
                model_prob_home=model_home,
                consensus=consensus_for_div,
            )
```

Then, at the end of the `feature_row` dict (after `"pregame_data_date": pregame.get("data_date"),` at line 724), add these keys:

```python
                "kalshi_vs_consensus_pp": div["kalshi_vs_consensus_pp"],
                "model_vs_consensus_pp": div["model_vs_consensus_pp"],
                "kalshi_vs_model_pp": div["kalshi_vs_model_pp"],
                "triangulation_tier": div["triangulation_tier"],
                "consensus_n_books": consensus.get("n_books", 0) if consensus else 0,
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_integrations.py::TestFeatureRowDivergenceColumns -v
```

Expected: 2 passed.

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add realtime_feature_store.py tests/test_integrations.py
git commit -m "feat(feature_store): emit divergence + triangulation_tier columns per poll"
```

---

### Task 12: `_detect_divergence_events` — emit event on tier transitions

**Files:**
- Modify: `realtime_feature_store.py` (add method + wire into `capture_once`)
- Modify: `tests/test_integrations.py` (add test class)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_integrations.py`:

```python
class TestDetectDivergenceEvents:
    def test_tier_change_emits_divergence_change_event(self):
        from realtime_feature_store import LiveFeatureStore

        store = LiveFeatureStore.__new__(LiveFeatureStore)
        store.state = {
            "games": {},
            "markets": {},
            "divergence": {
                "2026-04-16|BOS|MIA": {"triangulation_tier": 1},
            },
        }

        features_df = pd.DataFrame([{
            "captured_at": pd.Timestamp("2026-04-16T23:00:00", tz="UTC"),
            "game_key": "2026-04-16|BOS|MIA",
            "ticker": "KXNBAGAME-26APR16BOSMIA-BOS",
            "home_team": "BOS",
            "away_team": "MIA",
            "triangulation_tier": 2,
            "kalshi_vs_consensus_pp": 12.0,
            "model_vs_consensus_pp": -2.0,
            "kalshi_vs_model_pp": 14.0,
        }])

        events = store._detect_divergence_events(features_df)
        assert len(events) == 1
        event = events[0]
        assert event["event_type"] == "divergence_change"
        assert event["entity_type"] == "divergence"
        assert event["entity_key"] == "2026-04-16|BOS|MIA"
        assert event["previous"]["triangulation_tier"] == 1
        assert event["current"]["triangulation_tier"] == 2

    def test_unchanged_tier_emits_no_event(self):
        from realtime_feature_store import LiveFeatureStore

        store = LiveFeatureStore.__new__(LiveFeatureStore)
        store.state = {
            "games": {},
            "markets": {},
            "divergence": {
                "2026-04-16|BOS|MIA": {"triangulation_tier": 1},
            },
        }

        features_df = pd.DataFrame([{
            "captured_at": pd.Timestamp("2026-04-16T23:00:00", tz="UTC"),
            "game_key": "2026-04-16|BOS|MIA",
            "ticker": "KXNBAGAME-26APR16BOSMIA-BOS",
            "home_team": "BOS",
            "away_team": "MIA",
            "triangulation_tier": 1,
            "kalshi_vs_consensus_pp": 1.0,
            "model_vs_consensus_pp": 0.5,
            "kalshi_vs_model_pp": 1.5,
        }])

        events = store._detect_divergence_events(features_df)
        assert events == []

    def test_first_observation_emits_divergence_observed(self):
        from realtime_feature_store import LiveFeatureStore

        store = LiveFeatureStore.__new__(LiveFeatureStore)
        store.state = {"games": {}, "markets": {}, "divergence": {}}

        features_df = pd.DataFrame([{
            "captured_at": pd.Timestamp("2026-04-16T23:00:00", tz="UTC"),
            "game_key": "2026-04-16|BOS|MIA",
            "ticker": "KXNBAGAME-26APR16BOSMIA-BOS",
            "home_team": "BOS",
            "away_team": "MIA",
            "triangulation_tier": 2,
            "kalshi_vs_consensus_pp": 12.0,
            "model_vs_consensus_pp": -2.0,
            "kalshi_vs_model_pp": 14.0,
        }])

        events = store._detect_divergence_events(features_df)
        assert len(events) == 1
        assert events[0]["event_type"] == "divergence_observed"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_integrations.py::TestDetectDivergenceEvents -v
```

Expected: FAIL with `AttributeError: 'LiveFeatureStore' object has no attribute '_detect_divergence_events'`.

- [ ] **Step 3: Add the `_detect_divergence_events` method**

Open `realtime_feature_store.py`. Locate the `_detect_market_events` method that ends around line 999. **Immediately after** that method, add:

```python
    def _detect_divergence_events(self, features_df: pd.DataFrame) -> list[dict]:
        """
        Emit divergence_change events when triangulation_tier flips between
        polls, and divergence_observed events on first sighting.
        """
        if features_df.empty or "triangulation_tier" not in features_df.columns:
            return []

        events = []
        previous = self.state.get("divergence", {})
        captured_at = _now_utc().isoformat()

        # Take the latest row per game_key to avoid duplicate events for
        # multi-ticker matchups in the same poll cycle.
        latest = (
            features_df.dropna(subset=["game_key"])
            .sort_values("captured_at")
            .groupby("game_key", as_index=False)
            .tail(1)
        )

        for _, row in latest.iterrows():
            game_key = row.get("game_key")
            tier = _safe_int(row.get("triangulation_tier"))
            current = {
                "triangulation_tier": tier,
                "kalshi_vs_consensus_pp": _safe_float(row.get("kalshi_vs_consensus_pp")),
                "model_vs_consensus_pp": _safe_float(row.get("model_vs_consensus_pp")),
                "kalshi_vs_model_pp": _safe_float(row.get("kalshi_vs_model_pp")),
            }
            prev = previous.get(game_key)

            if prev is None:
                events.append({
                    "captured_at": captured_at,
                    "entity_type": "divergence",
                    "event_type": "divergence_observed",
                    "entity_key": game_key,
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "current": current,
                })
            elif prev.get("triangulation_tier") != tier:
                events.append({
                    "captured_at": captured_at,
                    "entity_type": "divergence",
                    "event_type": "divergence_change",
                    "entity_key": game_key,
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "previous": prev,
                    "current": current,
                })

        return events
```

- [ ] **Step 4: Wire `_detect_divergence_events` into `capture_once`**

In the same file, locate `capture_once` around line 1001. Replace the event-gathering section that reads:

```python
        game_events = self._detect_game_events(games_df)
        market_events = self._detect_market_events(markets_df)
        events = game_events + market_events
```

with:

```python
        game_events = self._detect_game_events(games_df)
        market_events = self._detect_market_events(markets_df)
        divergence_events = self._detect_divergence_events(features_df)
        events = game_events + market_events + divergence_events
```

- [ ] **Step 5: Persist divergence state inside `_save_state`**

Locate `_save_state` around line 333. Inspect its body. Where it saves `games` and `markets` state, also save `divergence`. Replace the body so it includes divergence state. Specifically, add after the existing state-building code (wherever `state["markets"] = ...` is written), add:

```python
        # Persist latest divergence tier per game_key for next-poll diffing.
        divergence_state = {}
        if hasattr(self, "_last_features_df") and self._last_features_df is not None:
            latest = (
                self._last_features_df.dropna(subset=["game_key"])
                .sort_values("captured_at")
                .groupby("game_key", as_index=False)
                .tail(1)
            )
            for _, row in latest.iterrows():
                game_key = row.get("game_key")
                divergence_state[game_key] = {
                    "triangulation_tier": _safe_int(row.get("triangulation_tier")),
                    "kalshi_vs_consensus_pp": _safe_float(row.get("kalshi_vs_consensus_pp")),
                    "model_vs_consensus_pp": _safe_float(row.get("model_vs_consensus_pp")),
                    "kalshi_vs_model_pp": _safe_float(row.get("kalshi_vs_model_pp")),
                }
        state["divergence"] = divergence_state
```

And inside `capture_once`, **before** `self._save_state(games_df, markets_df)`, stash the features df for `_save_state`:

```python
        self._last_features_df = features_df
```

- [ ] **Step 6: Run test to verify it passes**

```bash
pytest tests/test_integrations.py::TestDetectDivergenceEvents -v
```

Expected: 3 passed.

- [ ] **Step 7: Run full suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add realtime_feature_store.py tests/test_integrations.py
git commit -m "feat(feature_store): emit divergence_change events on tier transitions"
```

---

### Task 13: Wire `ev_analyzer.py --extra-signals` into `apply_consensus_tier`

**Files:**
- Modify: `ev_analyzer.py` (extend the `--extra-signals` block at lines 374-412)
- Modify: `tests/test_integrations.py` (add test class)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_integrations.py`:

```python
class TestEvAnalyzerConsensusWiring:
    """
    Smoke tests that ev_analyzer imports apply_consensus_tier and
    uses its signal_threshold in active mode.
    """

    def test_ev_analyzer_imports_apply_consensus_tier(self):
        import ev_analyzer  # noqa: F401
        assert hasattr(ev_analyzer, "apply_consensus_tier")

    def test_active_mode_tier_2_reduces_threshold(self, monkeypatch):
        import ev_analyzer
        monkeypatch.setattr(ev_analyzer, "CONSENSUS_TIER_MODE", "active")

        now = "2026-04-16T23:00:00Z"
        comparison = {
            "home_team": "BOS", "away_team": "MIA",
            "sportsbooks": [
                {"name": "DK", "last_update": now, "home_moneyline": -140, "away_moneyline": 120},
                {"name": "FD", "last_update": now, "home_moneyline": -145, "away_moneyline": 125},
                {"name": "MGM", "last_update": now, "home_moneyline": -135, "away_moneyline": 115},
                {"name": "Caesars", "last_update": now, "home_moneyline": -150, "away_moneyline": 130},
            ],
        }
        result = ev_analyzer.apply_consensus_tier(
            comparison=comparison,
            model_prob_home=0.60,
            kalshi_yes_mid=0.48,
            bet_side="home",
            base_threshold=0.03,
            mode="active",
            now_iso=now,
        )
        assert result["triangulation_tier"] == 2
        assert result["signal_threshold"] == pytest.approx(0.03 * 0.7)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_integrations.py::TestEvAnalyzerConsensusWiring -v
```

Expected: FAIL — `ev_analyzer` module does not import `apply_consensus_tier` yet.

- [ ] **Step 3: Add import to `ev_analyzer.py`**

Open `ev_analyzer.py`. After the existing imports (around line 64), add:

```python
from consensus_divergence import apply_consensus_tier
from config import CONSENSUS_TIER_MODE
```

- [ ] **Step 4: Extend the `--extra-signals` block with consensus tier adjustment**

Locate the calibration-threshold section (lines 404-412) which currently reads:

```python
        # Adaptive calibration-driven threshold
        try:
            signal_threshold = current_threshold()
            if signal_threshold > MIN_EDGE_THRESHOLD:
                print(f"    Edge threshold:  {signal_threshold:.2%} (inflated from {MIN_EDGE_THRESHOLD:.2%} due to Brier drift)")
            else:
                print(f"    Edge threshold:  {signal_threshold:.2%} (baseline)")
        except Exception as e:
            print(f"    Edge threshold:  (unavailable: {e})")
```

Replace with the following — which preserves the calibration baseline, then layers the consensus-tier multiplier on top:

```python
        # Adaptive calibration-driven threshold
        try:
            signal_threshold = current_threshold()
            if signal_threshold > MIN_EDGE_THRESHOLD:
                print(f"    Edge threshold:  {signal_threshold:.2%} (inflated from {MIN_EDGE_THRESHOLD:.2%} due to Brier drift)")
            else:
                print(f"    Edge threshold:  {signal_threshold:.2%} (baseline)")
        except Exception as e:
            print(f"    Edge threshold:  (unavailable: {e})")

        # Consensus-divergence signal (#8): adjust threshold based on Kalshi↔consensus↔model tier.
        try:
            comparison = get_odds_comparison(home, away)
            yes_mid = (yes_bid + yes_ask) / 2.0 if (yes_bid > 0 and yes_ask > 0) else None
            tier_result = apply_consensus_tier(
                comparison=comparison,
                model_prob_home=model_home_prob,
                kalshi_yes_mid=yes_mid,
                bet_side=bet_side,
                base_threshold=signal_threshold,
                mode=CONSENSUS_TIER_MODE,
            )
            tier = tier_result["triangulation_tier"]
            print(
                f"    Consensus tier:  {tier:+d} ({tier_result['tier_reason']})"
            )
            if CONSENSUS_TIER_MODE == "active" and tier_result["signal_threshold"] != signal_threshold:
                print(
                    f"    Tier adjusts edge threshold: "
                    f"{signal_threshold:.2%} → {tier_result['signal_threshold']:.2%} (active mode)"
                )
                signal_threshold = tier_result["signal_threshold"]
            elif CONSENSUS_TIER_MODE == "shadow":
                print(
                    f"    (Shadow mode — adjusted threshold would be "
                    f"{tier_result['shadow_signal_threshold']:.2%}, not applied)"
                )
        except Exception as e:
            print(f"    Consensus tier:  (unavailable: {e})")
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_integrations.py::TestEvAnalyzerConsensusWiring -v
```

Expected: 2 passed.

- [ ] **Step 6: Run full suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add ev_analyzer.py tests/test_integrations.py
git commit -m "feat(ev_analyzer): integrate consensus-tier multiplier behind shadow/active mode"
```

---

### Task 14: Metrics CSV — decision-time logging `record_divergence_decision`

**Files:**
- Modify: `consensus_divergence.py` (add function)
- Modify: `tests/test_consensus_divergence.py` (add test class)
- Modify: `ev_analyzer.py` (call the function after `apply_consensus_tier`)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_consensus_divergence.py`:

```python
import tempfile
from pathlib import Path


class TestRecordDivergenceDecision:
    def test_creates_file_with_header_on_first_call(self, tmp_path):
        from consensus_divergence import record_divergence_decision

        metrics_path = tmp_path / "consensus_divergence_metrics.csv"
        record_divergence_decision(
            metrics_path=metrics_path,
            game_id="GAME_2026-04-16_BOS_MIA",
            decided_at="2026-04-16T22:00:00Z",
            mode="shadow",
            tier=2,
            kalshi_vs_consensus_pp=12.0,
            model_vs_consensus_pp=-2.0,
            kalshi_vs_model_pp=14.0,
            bet_taken=True,
            stake=100.0,
            base_threshold=0.03,
            adjusted_threshold=0.021,
            would_have_bet_at_base=True,
            would_have_bet_at_adjusted=True,
        )
        assert metrics_path.exists()
        content = metrics_path.read_text()
        assert "game_id,decided_at,settled_at,mode,tier" in content.splitlines()[0]
        assert "GAME_2026-04-16_BOS_MIA" in content

    def test_appends_to_existing_file(self, tmp_path):
        from consensus_divergence import record_divergence_decision

        metrics_path = tmp_path / "consensus_divergence_metrics.csv"
        for game_id in ["A", "B", "C"]:
            record_divergence_decision(
                metrics_path=metrics_path,
                game_id=game_id,
                decided_at="2026-04-16T22:00:00Z",
                mode="shadow",
                tier=1,
                kalshi_vs_consensus_pp=1.0,
                model_vs_consensus_pp=0.5,
                kalshi_vs_model_pp=1.5,
                bet_taken=False,
                stake=0.0,
                base_threshold=0.03,
                adjusted_threshold=0.03,
                would_have_bet_at_base=False,
                would_have_bet_at_adjusted=False,
            )
        lines = metrics_path.read_text().splitlines()
        # 1 header + 3 data rows
        assert len(lines) == 4
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_consensus_divergence.py::TestRecordDivergenceDecision -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `record_divergence_decision`**

Append to `consensus_divergence.py`:

```python
import csv as _csv
from pathlib import Path as _Path

METRICS_COLUMNS = [
    "game_id",
    "decided_at",
    "settled_at",
    "mode",
    "tier",
    "kalshi_vs_consensus_pp",
    "model_vs_consensus_pp",
    "kalshi_vs_model_pp",
    "base_threshold",
    "adjusted_threshold",
    "bet_taken",
    "stake",
    "would_have_bet_at_base",
    "would_have_bet_at_adjusted",
    "pnl",
    "home_win",
]


def record_divergence_decision(
    metrics_path,
    game_id: str,
    decided_at: str,
    mode: str,
    tier: int,
    kalshi_vs_consensus_pp,
    model_vs_consensus_pp,
    kalshi_vs_model_pp,
    bet_taken: bool,
    stake: float,
    base_threshold: float,
    adjusted_threshold: float,
    would_have_bet_at_base: bool,
    would_have_bet_at_adjusted: bool,
) -> None:
    """
    Append a decision-time row to the metrics CSV. pnl, settled_at, home_win
    columns are left blank and filled in later by update_divergence_outcome.
    """
    path = _Path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=METRICS_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "game_id": game_id,
            "decided_at": decided_at,
            "settled_at": "",
            "mode": mode,
            "tier": tier,
            "kalshi_vs_consensus_pp": kalshi_vs_consensus_pp if kalshi_vs_consensus_pp is not None else "",
            "model_vs_consensus_pp": model_vs_consensus_pp if model_vs_consensus_pp is not None else "",
            "kalshi_vs_model_pp": kalshi_vs_model_pp if kalshi_vs_model_pp is not None else "",
            "base_threshold": base_threshold,
            "adjusted_threshold": adjusted_threshold,
            "bet_taken": bet_taken,
            "stake": stake,
            "would_have_bet_at_base": would_have_bet_at_base,
            "would_have_bet_at_adjusted": would_have_bet_at_adjusted,
            "pnl": "",
            "home_win": "",
        })
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_consensus_divergence.py::TestRecordDivergenceDecision -v
```

Expected: 2 passed.

- [ ] **Step 5: Wire the call into `ev_analyzer.py`**

Open `ev_analyzer.py`. Add imports at the top (after the import added in Task 13):

```python
from datetime import datetime, timezone

from consensus_divergence import record_divergence_decision
from config import OUTPUTS_DIR as _OUTPUTS_DIR  # already imported but alias for clarity
```

(If `OUTPUTS_DIR` is already imported at line 38, drop the alias and just use the existing `OUTPUTS_DIR`.)

Inside the `--extra-signals` block in Task 13, **replace** the `except Exception as e: print(f"    Consensus tier: (unavailable: {e})")` line with the same line PLUS a `tier_result = None` fallback, AND after the try/except block but while still inside `if extra_signals:`, add the recording call:

Replace:

```python
        except Exception as e:
            print(f"    Consensus tier:  (unavailable: {e})")
```

with:

```python
        except Exception as e:
            print(f"    Consensus tier:  (unavailable: {e})")
            tier_result = None

        if tier_result is not None:
            try:
                # Whether the bet would fire at each threshold. Uses model_home_prob
                # (set earlier in ev_analyzer's prediction block) as the model's home-win prob.
                model_edge = abs((model_home_prob if bet_side == "home" else 1.0 - model_home_prob) - ((yes_bid + yes_ask) / 2.0))
                would_bet_base = model_edge >= tier_result["base_threshold"]
                would_bet_adjusted = model_edge >= tier_result["signal_threshold"]
                record_divergence_decision(
                    metrics_path=OUTPUTS_DIR / "consensus_divergence_metrics.csv",
                    game_id=ticker,
                    decided_at=datetime.now(tz=timezone.utc).isoformat(),
                    mode=tier_result["mode"],
                    tier=tier_result["triangulation_tier"],
                    kalshi_vs_consensus_pp=tier_result["kalshi_vs_consensus_pp"],
                    model_vs_consensus_pp=tier_result["model_vs_consensus_pp"],
                    kalshi_vs_model_pp=tier_result["kalshi_vs_model_pp"],
                    bet_taken=False,  # order-placement is a separate manual step; settlement updater fills the rest
                    stake=0.0,
                    base_threshold=tier_result["base_threshold"],
                    adjusted_threshold=tier_result["signal_threshold"],
                    would_have_bet_at_base=would_bet_base,
                    would_have_bet_at_adjusted=would_bet_adjusted,
                )
            except Exception as e:
                print(f"    Metrics log:     (failed: {e})")
```

**Also**, just before the `try:` block that calls `apply_consensus_tier` (inside the existing `if extra_signals:`), add `tier_result = None` to initialize the variable so the downstream check is safe:

```python
        tier_result = None
        try:
            comparison = get_odds_comparison(home, away)
            ...
```

- [ ] **Step 6: Run full suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add consensus_divergence.py ev_analyzer.py tests/test_consensus_divergence.py
git commit -m "feat(consensus): log decision-time metrics row from ev_analyzer"
```

---

### Task 15: Metrics CSV — settlement updater + `--settle` CLI

**Files:**
- Modify: `consensus_divergence.py` (add `update_divergence_outcome` + CLI arg parsing)
- Modify: `tests/test_consensus_divergence.py` (add test class)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_consensus_divergence.py`:

```python
class TestUpdateDivergenceOutcome:
    def _seed(self, tmp_path, game_id="GAME_A"):
        from consensus_divergence import record_divergence_decision
        metrics_path = tmp_path / "consensus_divergence_metrics.csv"
        record_divergence_decision(
            metrics_path=metrics_path,
            game_id=game_id,
            decided_at="2026-04-16T22:00:00Z",
            mode="shadow",
            tier=2,
            kalshi_vs_consensus_pp=12.0,
            model_vs_consensus_pp=-2.0,
            kalshi_vs_model_pp=14.0,
            bet_taken=True,
            stake=100.0,
            base_threshold=0.03,
            adjusted_threshold=0.021,
            would_have_bet_at_base=True,
            would_have_bet_at_adjusted=True,
        )
        return metrics_path

    def test_updates_pnl_and_settled_at(self, tmp_path):
        from consensus_divergence import update_divergence_outcome

        path = self._seed(tmp_path)
        n = update_divergence_outcome(
            metrics_path=path,
            game_id="GAME_A",
            home_win=True,
            pnl=57.0,
            settled_at="2026-04-17T03:30:00Z",
        )
        assert n == 1

        import csv
        rows = list(csv.DictReader(path.open()))
        assert rows[0]["pnl"] == "57.0"
        assert rows[0]["settled_at"] == "2026-04-17T03:30:00Z"
        assert rows[0]["home_win"] == "True"

    def test_returns_zero_when_game_not_found(self, tmp_path):
        from consensus_divergence import update_divergence_outcome

        path = self._seed(tmp_path)
        n = update_divergence_outcome(
            metrics_path=path,
            game_id="UNKNOWN",
            home_win=True,
            pnl=0.0,
            settled_at="2026-04-17T03:30:00Z",
        )
        assert n == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_consensus_divergence.py::TestUpdateDivergenceOutcome -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `update_divergence_outcome`**

Append to `consensus_divergence.py`:

```python
def update_divergence_outcome(
    metrics_path,
    game_id: str,
    home_win: bool,
    pnl: float,
    settled_at: str,
) -> int:
    """
    Find rows in the metrics CSV matching `game_id` and fill in their
    pnl, settled_at, home_win fields. Returns the number of rows updated.

    Rewrites the CSV in place (small file; no scale concerns for v1).
    """
    path = _Path(metrics_path)
    if not path.exists():
        return 0

    with path.open() as f:
        reader = _csv.DictReader(f)
        rows = list(reader)

    updated = 0
    for row in rows:
        if row.get("game_id") == game_id:
            row["pnl"] = str(pnl)
            row["settled_at"] = settled_at
            row["home_win"] = str(home_win)
            updated += 1

    if updated > 0:
        with path.open("w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=METRICS_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    return updated
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_consensus_divergence.py::TestUpdateDivergenceOutcome -v
```

Expected: 2 passed.

- [ ] **Step 5: Add `--settle` CLI entry**

Append to `consensus_divergence.py` (at the very bottom, outside any class):

```python
def _cli_settle(args):
    from config import OUTPUTS_DIR
    path = OUTPUTS_DIR / "consensus_divergence_metrics.csv"
    n = update_divergence_outcome(
        metrics_path=path,
        game_id=args.game_id,
        home_win=(args.home_win.lower() == "true"),
        pnl=float(args.pnl),
        settled_at=args.settled_at,
    )
    print(f"Updated {n} row(s) for game_id={args.game_id} in {path}")
```

- [ ] **Step 6: Commit**

```bash
git add consensus_divergence.py tests/test_consensus_divergence.py
git commit -m "feat(consensus): add update_divergence_outcome settlement helper"
```

---

### Task 16: `--scan` CLI with formatter

**Files:**
- Modify: `consensus_divergence.py` (add formatter + CLI + `__main__`)
- Modify: `tests/test_consensus_divergence.py` (add formatter test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_consensus_divergence.py`:

```python
class TestFormatScanTable:
    def test_formatter_renders_rows_sorted_by_divergence_magnitude(self):
        from consensus_divergence import format_scan_table

        rows = [
            {"matchup": "BOS@MIA", "kalshi_yes": 0.48, "consensus_home": 0.60,
             "kalshi_vs_consensus_pp": 12.0, "n_books": 4, "tier": 2},
            {"matchup": "LAL@DEN", "kalshi_yes": 0.51, "consensus_home": 0.52,
             "kalshi_vs_consensus_pp": 1.0, "n_books": 5, "tier": 1},
            {"matchup": "OKC@GSW", "kalshi_yes": 0.70, "consensus_home": 0.58,
             "kalshi_vs_consensus_pp": -12.0, "n_books": 3, "tier": 2},
        ]
        table = format_scan_table(rows)
        # Tier-2 rows (highest |divergence|) should appear before tier-1
        assert table.index("BOS@MIA") < table.index("LAL@DEN")
        assert table.index("OKC@GSW") < table.index("LAL@DEN")
        assert "BOS@MIA" in table
        assert "LAL@DEN" in table
        assert "OKC@GSW" in table

    def test_formatter_handles_empty_list(self):
        from consensus_divergence import format_scan_table
        assert format_scan_table([]) == "No divergences to report.\n"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_consensus_divergence.py::TestFormatScanTable -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `format_scan_table` and wire CLI**

Append to `consensus_divergence.py`:

```python
def format_scan_table(rows) -> str:
    if not rows:
        return "No divergences to report.\n"

    # Sort by absolute Kalshi-vs-consensus divergence, desc
    sorted_rows = sorted(
        rows,
        key=lambda r: abs(r.get("kalshi_vs_consensus_pp") or 0.0),
        reverse=True,
    )
    lines = [
        f"{'Matchup':<16}{'Kalshi':>10}{'Consensus':>12}{'Δ(pp)':>10}{'Tier':>6}{'Books':>8}",
        "-" * 62,
    ]
    for r in sorted_rows:
        lines.append(
            f"{r.get('matchup', ''):<16}"
            f"{(r.get('kalshi_yes') or 0.0):>10.3f}"
            f"{(r.get('consensus_home') or 0.0):>12.3f}"
            f"{(r.get('kalshi_vs_consensus_pp') or 0.0):>+10.2f}"
            f"{r.get('tier', 0):>+6d}"
            f"{r.get('n_books', 0):>8d}"
        )
    return "\n".join(lines) + "\n"


def _cli_scan(args):
    from live_data import fetch_odds_api_lines, fetch_espn_odds
    from config import OUTPUTS_DIR

    games = fetch_odds_api_lines()
    espn_games = {(g.get("home", ""), g.get("away", "")): g for g in fetch_espn_odds()}

    # NOTE: This CLI does not know Kalshi contract prices without per-matchup lookup.
    # For v1 we print consensus divergences vs the ESPN quote (single-book proxy for
    # "Kalshi-like" price) when available; --scan is a sanity-check tool, not a
    # replacement for ev_analyzer's full decision flow.
    rows = []
    for game in games:
        books = [
            {
                "name": b.get("name", ""),
                "last_update": b.get("last_update"),
                "home_moneyline": b.get("markets", {}).get("h2h", {}).get(game["home"], {}).get("price"),
                "away_moneyline": b.get("markets", {}).get("h2h", {}).get(game["away"], {}).get("price"),
            }
            for b in game.get("books", [])
        ]
        consensus = consensus_implied_prob(books)
        if consensus is None:
            continue

        espn = espn_games.get((game["home"], game["away"]))
        kalshi_proxy = espn.get("home_implied_prob") if espn else None

        div = compute_divergence(
            kalshi_yes_mid=kalshi_proxy,
            model_prob_home=None,  # No model in plain --scan
            consensus=consensus,
        )
        rows.append({
            "matchup": f"{game['away']}@{game['home']}",
            "kalshi_yes": kalshi_proxy,
            "consensus_home": consensus["home_prob"],
            "kalshi_vs_consensus_pp": div["kalshi_vs_consensus_pp"],
            "n_books": consensus["n_books"],
            "tier": div["triangulation_tier"],
        })

    table = format_scan_table(rows)
    print(table)

    if not args.dry_run:
        out_path = OUTPUTS_DIR / f"consensus_scan_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt"
        out_path.write_text(table)
        print(f"Scan written to {out_path}")


def main():
    import argparse as _argparse
    parser = _argparse.ArgumentParser(description="Consensus divergence CLI (IMPROVEMENTS #8).")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Print consensus divergences for today's games.")
    scan.add_argument("--dry-run", action="store_true", help="Do not write output file.")

    settle = sub.add_parser("settle", help="Fill in pnl/settled_at for a logged game decision.")
    settle.add_argument("--game-id", required=True)
    settle.add_argument("--home-win", required=True, help="true|false")
    settle.add_argument("--pnl", required=True)
    settle.add_argument("--settled-at", required=True, help="ISO-8601 timestamp")

    args = parser.parse_args()
    if args.command == "scan":
        _cli_scan(args)
    elif args.command == "settle":
        _cli_settle(args)


if __name__ == "__main__":
    from datetime import datetime, timezone  # local import for CLI
    main()
```

Note: the top-level `from datetime import datetime, timezone` that `_cli_scan` uses is re-imported inside `__main__` to keep module import cheap. If unit tests already import `datetime` via earlier tasks, the top-level import can be moved to the top of the file instead. **Move** `from datetime import datetime, timezone` to the top of `consensus_divergence.py` (immediately below `from __future__ import annotations`) to keep imports clean and delete the one inside `if __name__ == "__main__"`.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_consensus_divergence.py::TestFormatScanTable -v
```

Expected: 2 passed.

- [ ] **Step 5: Verify the CLI boots**

```bash
python consensus_divergence.py scan --dry-run
```

Expected: either a table of today's divergences (if Odds API key is set and there are live NBA games) OR `No divergences to report.` — both are successes for smoke purposes.

- [ ] **Step 6: Commit**

```bash
git add consensus_divergence.py tests/test_consensus_divergence.py
git commit -m "feat(consensus): add --scan and --settle CLI with ranked divergence table"
```

---

### Task 17: Update `IMPROVEMENTS.md` to mark #8 ✅

**Files:**
- Modify: `IMPROVEMENTS.md` (lines 99-100, item #8 entry)

- [ ] **Step 1: Replace the #8 entry**

Open `IMPROVEMENTS.md`. Find the block:

```markdown
### #8: Arbitrage Detection
Flag moments when Kalshi vs ESPN vs DraftKings prices diverge significantly
```

Replace with:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add IMPROVEMENTS.md
git commit -m "docs(improvements): mark #8 Arbitrage Detection complete"
```

---

### Task 18: Final full-suite check + live smoke (manual)

- [ ] **Step 1: Run the entire test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass; no warnings; no collection errors.

- [ ] **Step 2: Live smoke test (requires valid `ODDS_API_KEY` in `.env` and today's NBA slate)**

```bash
python consensus_divergence.py scan --dry-run
```

Expected: either a formatted table listing today's divergences, or `No divergences to report.` (both valid — the former tests the full fetch→de-vig→tier path, the latter only signals no live slate).

- [ ] **Step 3: Verify ev_analyzer flow prints the consensus-tier line**

```bash
python ev_analyzer.py <any-live-ticker> --extra-signals
```

Expected: the printed "Extra signals" block includes either `Consensus tier:  +N (...)` or `Consensus tier:  (unavailable: ...)`. In `shadow` mode it also prints `(Shadow mode — adjusted threshold would be ...)`.

- [ ] **Step 4: No commit for Task 18 (verification only)**

If any step fails, investigate and file a follow-up rather than committing a workaround. All failures at this stage are signals that an upstream task merged incorrectly.

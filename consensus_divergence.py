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

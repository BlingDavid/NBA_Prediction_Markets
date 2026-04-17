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

from statistics import median
from datetime import datetime, timezone

from config import CONSENSUS_MIN_BOOKS, CONSENSUS_STALE_SECONDS, CONSENSUS_DIVERGENCE_THRESHOLD_PP, TIER_THRESHOLD_MULTIPLIERS


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


def _parse_iso_utc_seconds(iso_str):
    """ISO-8601 string → POSIX seconds (UTC). Returns None on failure."""
    if not iso_str:
        return None
    try:
        # Accept both 'Z' and '+00:00' suffixes
        normalized = iso_str.replace("Z", "+00:00") if isinstance(iso_str, str) else iso_str
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

    # Tolerance absorbs IEEE-754 noise so values exactly at the cutoff classify deterministically.
    eps = 1e-9
    big_consensus = abs(kalshi_vs_consensus_pp) >= cutoff - eps
    big_model = abs(kalshi_vs_model_pp) >= cutoff - eps

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


def tier_to_threshold_multiplier(tier: int, base_threshold: float) -> float:
    """
    Map a triangulation tier to an adjusted edge threshold.

    Multipliers are read from config.TIER_THRESHOLD_MULTIPLIERS so they
    can be tuned without code changes. Unknown tiers fall back to 1.0.
    """
    multiplier = TIER_THRESHOLD_MULTIPLIERS.get(tier, 1.0)
    return base_threshold * multiplier

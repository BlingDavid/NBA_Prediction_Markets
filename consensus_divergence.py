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
    the YES side represents away winning, so home_prob = 1 - yes_mid),
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
            model_prob_home=None,
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
    main()

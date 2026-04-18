"""
Unified bet-lifecycle ledger.

One row per EV recommendation — whether executed or not — joined across
three phases via `decision_id`:

  1. record_decision    → written by ev_analyzer at recommendation time
  2. update_fill        → written by order_executor when user places an order
  3. update_settlement  → written by the nightly settlement script

Pricing semantics assume the user crosses the spread (takes yes_ask or
1 - yes_bid depending on contract_side). The ledger stores
entry_limit_price explicitly so CLV can be computed post-close.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

DECISION_COLUMNS = [
    # core decision metadata
    "decision_id",
    "decided_at",
    "ticker",
    "home_team",
    "away_team",
    "bet_side",
    "contract_side",
    # probabilities
    "model_prob_home",
    "model_prob_away",
    # Kalshi price at decision
    "kalshi_yes_bid",
    "kalshi_yes_ask",
    "kalshi_yes_mid",
    # pricing
    "entry_limit_price",
    "edge_pp",
    "ev",
    # thresholds + sizing
    "signal_threshold",
    "base_threshold",
    "recommended_stake",
    # consensus-divergence signal (#8)
    "triangulation_tier",
    "kalshi_vs_consensus_pp",
    "model_vs_consensus_pp",
    "kalshi_vs_model_pp",
    "would_bet_at_base",
    "would_bet_at_adjusted",
    # fill-time (nullable until placed)
    "order_id",
    "filled_at",
    "filled_price",
    "filled_size",
    # settlement (nullable until resolved)
    "settled_at",
    "home_win",
    "realized_pnl",
    "clv_pp",
]


def make_decision_id(ticker: str, decided_at: datetime) -> str:
    """Canonical decision_id: '{ticker}_{YYYYMMDDTHHMMSSffffffZ}'.

    Microsecond precision prevents collisions within a single analyzer run.
    No colons — safe for filesystem use; sorts chronologically by suffix.
    """
    ts = decided_at.strftime("%Y%m%dT%H%M%S%f")
    return f"{ticker}_{ts}Z"


def record_decision(
    decisions_path,
    *,
    ticker: str,
    decided_at: str | None = None,
    home_team: str,
    away_team: str,
    bet_side: str,
    contract_side: str,
    model_prob_home: float,
    model_prob_away: float,
    kalshi_yes_bid: float,
    kalshi_yes_ask: float,
    kalshi_yes_mid: float,
    entry_limit_price: float,
    edge_pp: float,
    ev: float,
    signal_threshold: float,
    base_threshold: float,
    recommended_stake: float,
    triangulation_tier: int,
    kalshi_vs_consensus_pp,
    model_vs_consensus_pp,
    kalshi_vs_model_pp,
    would_bet_at_base: bool,
    would_bet_at_adjusted: bool,
) -> str:
    """Append a decision row. Returns the generated decision_id."""
    path = Path(decisions_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # If caller didn't supply decided_at, stamp one now (microsecond precision
    # keeps decision_ids unique across rapid-fire calls).
    if decided_at is None:
        decided_dt = datetime.now(tz=timezone.utc)
        decided_at = decided_dt.isoformat()
    else:
        normalized = decided_at.replace("Z", "+00:00") if decided_at.endswith("Z") else decided_at
        decided_dt = datetime.fromisoformat(normalized)
    decision_id = make_decision_id(ticker, decided_dt)

    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DECISION_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "decision_id": decision_id,
            "decided_at": decided_at,
            "ticker": ticker,
            "home_team": home_team,
            "away_team": away_team,
            "bet_side": bet_side,
            "contract_side": contract_side,
            "model_prob_home": model_prob_home,
            "model_prob_away": model_prob_away,
            "kalshi_yes_bid": kalshi_yes_bid,
            "kalshi_yes_ask": kalshi_yes_ask,
            "kalshi_yes_mid": kalshi_yes_mid,
            "entry_limit_price": entry_limit_price,
            "edge_pp": edge_pp,
            "ev": ev,
            "signal_threshold": signal_threshold,
            "base_threshold": base_threshold,
            "recommended_stake": recommended_stake,
            "triangulation_tier": triangulation_tier,
            "kalshi_vs_consensus_pp": kalshi_vs_consensus_pp if kalshi_vs_consensus_pp is not None else "",
            "model_vs_consensus_pp": model_vs_consensus_pp if model_vs_consensus_pp is not None else "",
            "kalshi_vs_model_pp": kalshi_vs_model_pp if kalshi_vs_model_pp is not None else "",
            "would_bet_at_base": would_bet_at_base,
            "would_bet_at_adjusted": would_bet_at_adjusted,
            # nullable fields left blank until fill / settlement
            "order_id": "",
            "filled_at": "",
            "filled_price": "",
            "filled_size": "",
            "settled_at": "",
            "home_win": "",
            "realized_pnl": "",
            "clv_pp": "",
        })
    return decision_id


def _rewrite_with_updates(path: Path, decision_id: str, updates: dict) -> bool:
    """Load, match by decision_id, merge updates, rewrite. Returns True if matched."""
    if not path.exists():
        return False

    with path.open() as f:
        rows = list(csv.DictReader(f))

    matched = False
    for row in rows:
        if row.get("decision_id") == decision_id:
            for k, v in updates.items():
                row[k] = "" if v is None else str(v)
            matched = True

    if not matched:
        return False

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DECISION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return True


def update_fill(
    decisions_path,
    *,
    decision_id: str,
    order_id: str,
    filled_at: str,
    filled_price: float,
    filled_size: float,
) -> bool:
    """Fill-time update. Returns False if decision_id not found."""
    return _rewrite_with_updates(
        Path(decisions_path),
        decision_id,
        {
            "order_id": order_id,
            "filled_at": filled_at,
            "filled_price": filled_price,
            "filled_size": filled_size,
        },
    )


def update_settlement(
    decisions_path,
    *,
    decision_id: str,
    settled_at: str,
    home_win: bool,
    realized_pnl: float,
    clv_pp: float,
) -> bool:
    """Settlement update. Returns False if decision_id not found."""
    return _rewrite_with_updates(
        Path(decisions_path),
        decision_id,
        {
            "settled_at": settled_at,
            "home_win": home_win,
            "realized_pnl": realized_pnl,
            "clv_pp": clv_pp,
        },
    )

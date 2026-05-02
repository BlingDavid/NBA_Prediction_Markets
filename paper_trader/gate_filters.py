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

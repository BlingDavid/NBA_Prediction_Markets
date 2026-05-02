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

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

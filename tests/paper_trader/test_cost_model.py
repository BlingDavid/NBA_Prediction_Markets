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

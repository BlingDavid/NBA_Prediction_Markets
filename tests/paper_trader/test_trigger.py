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

"""Each gate returns (passed, reason). One reason string per failure mode.

Order matters only for which reason is returned first; tests assert the
expected reason for each input row.
"""
from __future__ import annotations

import pytest

from paper_trader.gate_filters import evaluate


def _ok_row(**overrides) -> dict:
    base = dict(
        bet_side="home",
        status_state="in",
        is_live=True,
        open_interest=500,
        yes_depth_notional_3=200.0,
        yes_bid=0.40,
        yes_ask=0.42,
    )
    base.update(overrides)
    return base


def test_passes_clean_row():
    passed, reason = evaluate(_ok_row())
    assert passed, reason
    assert reason == "ok"


def test_rejects_away_side():
    passed, reason = evaluate(_ok_row(bet_side="away"))
    assert not passed and reason == "side_not_home"


def test_rejects_pregame():
    passed, reason = evaluate(_ok_row(status_state="pre"))
    assert not passed and reason == "not_live"


def test_rejects_thin_open_interest():
    passed, reason = evaluate(_ok_row(open_interest=99))
    assert not passed and reason == "open_interest_below_floor"


def test_rejects_thin_depth():
    passed, reason = evaluate(_ok_row(yes_depth_notional_3=49.99))
    assert not passed and reason == "depth_below_floor"


@pytest.mark.parametrize(
    "yes_bid, yes_ask, reason",
    [
        (0.0, 0.5, "price_out_of_range"),     # bid below 0.01
        (0.5, 1.0, "price_out_of_range"),     # ask at or above 1.0
        (0.5, 0.4, "crossed_book"),           # ask < bid
        (0.10, 0.45, "spread_too_wide"),      # spread > 0.30
    ],
)
def test_rejects_bad_prices(yes_bid, yes_ask, reason):
    passed, got = evaluate(_ok_row(yes_bid=yes_bid, yes_ask=yes_ask))
    assert not passed
    assert got == reason


def test_rejects_missing_field():
    row = _ok_row()
    row.pop("yes_bid")
    passed, reason = evaluate(row)
    assert not passed and reason == "missing_field:yes_bid"


def test_rejects_nan_open_interest():
    row = _ok_row(open_interest=float("nan"))
    passed, reason = evaluate(row)
    assert not passed and reason == "missing_field:open_interest"


def test_rejects_nan_yes_bid():
    row = _ok_row(yes_bid=float("nan"))
    passed, reason = evaluate(row)
    assert not passed and reason == "missing_field:yes_bid"

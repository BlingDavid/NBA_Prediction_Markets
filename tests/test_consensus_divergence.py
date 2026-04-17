"""
Unit tests for consensus_divergence.py — the price-divergence signal
used by ev_analyzer and realtime_feature_store.
"""

import pytest

from consensus_divergence import devig_proportional


class TestDevigProportional:
    def test_standard_favorite_underdog(self):
        # -150 / +130: raw implied 0.600 / 0.4348, sum 1.0348,
        # de-vigged ≈ 0.580 / 0.420
        home, away = devig_proportional(-150, 130)
        assert home == pytest.approx(0.580, abs=0.005)
        assert away == pytest.approx(0.420, abs=0.005)
        assert home + away == pytest.approx(1.0, abs=1e-9)

    def test_pick_em(self):
        home, away = devig_proportional(-110, -110)
        assert home == pytest.approx(0.5, abs=1e-9)
        assert away == pytest.approx(0.5, abs=1e-9)

    def test_none_input_returns_none(self):
        assert devig_proportional(None, -110) is None
        assert devig_proportional(-110, None) is None
        assert devig_proportional(None, None) is None

    def test_zero_moneyline_returns_none(self):
        assert devig_proportional(0, -110) is None
        assert devig_proportional(-110, 0) is None

    def test_empty_string_returns_none(self):
        assert devig_proportional("", -110) is None
        assert devig_proportional(-110, "") is None

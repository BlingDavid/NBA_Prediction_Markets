"""Unit tests for tools/cost_replay.py pure cost/PnL accounting.

All tests use tiny synthetic fills with known expected net PnL.
No CSVs are read; no models are loaded.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tools.cost_replay import (
    compute_gross_realistic,
    compute_pnl_realistic,
    compute_winner_fee,
    ev_gate_passes,
    sharpe_like,
    get_decile_expected_moves,
    load_expected_moves,
)


# ═══════════════════════════════════════════════════════════════════════════
# compute_gross_realistic
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeGrossRealistic:
    def test_positive_move(self):
        # Bought at 0.42 ask, sold at 0.46 bid → +0.04
        assert compute_gross_realistic(0.42, 0.46) == pytest.approx(0.04)

    def test_negative_move(self):
        # Bought at 0.42, sold at 0.38 → -0.04
        assert compute_gross_realistic(0.42, 0.38) == pytest.approx(-0.04)

    def test_zero_move(self):
        assert compute_gross_realistic(0.50, 0.50) == pytest.approx(0.0)

    def test_float_precision(self):
        # Confirm no floating-point surprise for common Kalshi prices
        assert compute_gross_realistic(0.70, 0.71) == pytest.approx(0.01)


# ═══════════════════════════════════════════════════════════════════════════
# compute_winner_fee
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeWinnerFee:
    def test_fee_charged_when_rose(self):
        # At p=0.75: ceil(0.07*0.75*0.25*100)/100 = ceil(1.3125)/100 = 2/100 = 0.02
        fee = compute_winner_fee(0.75, realized_rose=True)
        expected = math.ceil(0.07 * 0.75 * 0.25 * 100) / 100
        assert fee == pytest.approx(expected)

    def test_no_fee_when_didnt_rise(self):
        # Even at high p, if price didn't rise, no fee.
        assert compute_winner_fee(0.75, realized_rose=False) == 0.0

    def test_fee_at_p05(self):
        # ceil(0.07 * 0.05 * 0.95 * 100) / 100 = ceil(0.3325)/100 = 0.01
        fee = compute_winner_fee(0.05, realized_rose=True)
        expected = math.ceil(0.07 * 0.05 * 0.95 * 100) / 100
        assert fee == pytest.approx(expected)

    def test_fee_at_p50(self):
        # ceil(0.07 * 0.50 * 0.50 * 100) / 100 = ceil(1.75)/100 = 0.02
        fee = compute_winner_fee(0.50, realized_rose=True)
        assert fee == pytest.approx(0.02)


# ═══════════════════════════════════════════════════════════════════════════
# compute_pnl_realistic — synthetic fill accounting
# ═══════════════════════════════════════════════════════════════════════════

class TestComputePnlRealistic:
    def test_winner_trade(self):
        """Synthetic winner: price rose, fee charged.

        Setup:
            entry_yes_ask = 0.60,  entry_yes_mid = 0.595
            exit_yes_bid  = 0.65,  exit_yes_mid  = 0.655
            p_cal = 0.75

        gross_realistic = 0.65 - 0.60 = 0.05
        realized_rose = True (0.655 > 0.595)
        winner_fee = ceil(0.07 * 0.75 * 0.25 * 100) / 100 = 0.02
        pnl_realistic = 0.05 - 0.02 = 0.03
        half_spread_entry = 0.60 - 0.595 = 0.005
        pnl_mid = 0.655 - 0.595 = 0.060
        """
        result = compute_pnl_realistic(
            entry_yes_ask=0.60,
            entry_yes_mid=0.595,
            exit_yes_bid=0.65,
            exit_yes_mid=0.655,
            p_cal=0.75,
        )
        assert result["gross_realistic"] == pytest.approx(0.05)
        assert result["winner_fee"] == pytest.approx(0.02)
        assert result["pnl_realistic"] == pytest.approx(0.03)
        assert result["half_spread_entry"] == pytest.approx(0.005)
        assert result["pnl_mid"] == pytest.approx(0.060)
        assert result["realized_rose"] == 1

    def test_loser_trade_no_fee(self):
        """Synthetic loser: price fell, no fee, full spread cost absorbed.

        Setup:
            entry_yes_ask = 0.60,  entry_yes_mid = 0.595
            exit_yes_bid  = 0.55,  exit_yes_mid  = 0.555
            p_cal = 0.75

        gross_realistic = 0.55 - 0.60 = -0.05
        realized_rose = False (0.555 < 0.595)
        winner_fee = 0.0
        pnl_realistic = -0.05 - 0.0 = -0.05
        """
        result = compute_pnl_realistic(
            entry_yes_ask=0.60,
            entry_yes_mid=0.595,
            exit_yes_bid=0.55,
            exit_yes_mid=0.555,
            p_cal=0.75,
        )
        assert result["gross_realistic"] == pytest.approx(-0.05)
        assert result["winner_fee"] == pytest.approx(0.0)
        assert result["pnl_realistic"] == pytest.approx(-0.05)
        assert result["realized_rose"] == 0

    def test_flat_trade_no_fee(self):
        """Mid stayed flat → not a winner → no fee."""
        result = compute_pnl_realistic(
            entry_yes_ask=0.51,
            entry_yes_mid=0.505,
            exit_yes_bid=0.49,
            exit_yes_mid=0.505,  # identical mid = not a winner
            p_cal=0.50,
        )
        assert result["realized_rose"] == 0
        assert result["winner_fee"] == 0.0
        # gross = 0.49 - 0.51 = -0.02 (lost the spread both ways)
        assert result["gross_realistic"] == pytest.approx(-0.02)

    def test_known_net_pnl_portfolio_of_three(self):
        """Portfolio of 3 synthetic trades → net PnL must equal sum of parts.

        Trade 1 (winner): entry_ask=0.40, entry_mid=0.395, exit_bid=0.44, exit_mid=0.445, p=0.50
            gross = 0.04, fee = 0.02 (ceil(0.07*0.5*0.5*100)/100), pnl = 0.02
        Trade 2 (loser):  entry_ask=0.40, entry_mid=0.395, exit_bid=0.36, exit_mid=0.365, p=0.50
            gross = -0.04, fee = 0.0, pnl = -0.04
        Trade 3 (winner): entry_ask=0.70, entry_mid=0.695, exit_bid=0.73, exit_mid=0.735, p=0.30
            gross = 0.03, fee = ceil(0.07*0.3*0.7*100)/100 = ceil(1.47)/100 = 0.02, pnl = 0.01
        Expected net = 0.02 - 0.04 + 0.01 = -0.01
        """
        trades = [
            compute_pnl_realistic(0.40, 0.395, 0.44, 0.445, 0.50),
            compute_pnl_realistic(0.40, 0.395, 0.36, 0.365, 0.50),
            compute_pnl_realistic(0.70, 0.695, 0.73, 0.735, 0.30),
        ]
        net = sum(t["pnl_realistic"] for t in trades)
        assert trades[0]["pnl_realistic"] == pytest.approx(0.02)
        assert trades[1]["pnl_realistic"] == pytest.approx(-0.04)
        assert trades[2]["pnl_realistic"] == pytest.approx(0.01)
        assert net == pytest.approx(-0.01)


# ═══════════════════════════════════════════════════════════════════════════
# ev_gate_passes
# ═══════════════════════════════════════════════════════════════════════════

class TestEvGatePasses:
    def test_high_ev_passes(self):
        # Decile 8 midpoint (p=0.85): E_rises=0.08791, E_doesnt=-0.04165
        # EV at 0.01 ≈ +0.04998 → should pass
        assert ev_gate_passes(0.85, e_up=0.08791, e_down=-0.04165, half_spread_assumption=0.01)

    def test_low_ev_fails(self):
        # Decile 0 midpoint (p=0.05): E_rises=0.01168, E_doesnt=-0.00037
        # EV is deeply negative at any spread → should fail
        assert not ev_gate_passes(0.05, e_up=0.01168, e_down=-0.00037, half_spread_assumption=0.005)

    def test_exactly_zero_fails(self):
        # EV exactly 0 → gate requires strictly > 0
        assert not ev_gate_passes(0.50, e_up=0.0, e_down=0.0, half_spread_assumption=0.0)

    def test_large_spread_kills_marginal_edge(self):
        # Decile 6 midpoint (p=0.65): E_rises=0.05425, E_doesnt=-0.04955
        # EV@0.5c ≈ -0.00008 (just below zero — fails at 0.5c too)
        # Use p=0.75 (decile 7) at 2c: EV ≈ +0.0039 (still passes)
        # So use pooled means for a scenario that demonstrably flips sign
        # pooled: E_up=0.02380, E_down=-0.00104
        # EV(p=0.25, pooled, hs=0.005) = 0.25*0.02380 + 0.75*(-0.00104) - 0.005 - fee
        # fee = ceil(0.07*0.25*0.75*100)/100 = ceil(1.3125)/100 = 0.02
        # EV = 0.00595 - 0.00078 - 0.005 - 0.25*0.02 = 0.00595-0.00078-0.005-0.005 = -0.00483
        # Use decile 8 (p=0.85): clearly positive at hs=0.005, negative at hs=0.10
        e_up, e_down = 0.08791, -0.04165
        assert ev_gate_passes(0.85, e_up=e_up, e_down=e_down, half_spread_assumption=0.005)
        assert not ev_gate_passes(0.85, e_up=e_up, e_down=e_down, half_spread_assumption=0.10)


# ═══════════════════════════════════════════════════════════════════════════
# sharpe_like
# ═══════════════════════════════════════════════════════════════════════════

class TestSharpeLike:
    def test_positive_mean_positive_sharpe(self):
        pnl = [0.01, 0.02, 0.01, 0.03, 0.01]
        s = sharpe_like(pnl)
        assert s > 0

    def test_negative_mean_negative_sharpe(self):
        pnl = [-0.01, -0.02, -0.01, -0.03]
        s = sharpe_like(pnl)
        assert s < 0

    def test_empty_is_nan(self):
        assert math.isnan(sharpe_like([]))

    def test_single_is_nan(self):
        assert math.isnan(sharpe_like([0.01]))

    def test_zero_std_is_nan(self):
        assert math.isnan(sharpe_like([0.01, 0.01, 0.01]))

    def test_known_value(self):
        # [0.0, 0.02]: mean=0.01, std(ddof=1)=0.01*sqrt(2)≈0.01414
        # sharpe = 0.01 / (0.01*sqrt(2)) = 1/sqrt(2) ≈ 0.7071
        arr = [0.0, 0.02]
        s = sharpe_like(arr)
        assert s == pytest.approx(1.0 / math.sqrt(2))


# ═══════════════════════════════════════════════════════════════════════════
# get_decile_expected_moves
# ═══════════════════════════════════════════════════════════════════════════

class TestGetDecileExpectedMoves:
    _em_data = {
        "e_up_pooled": 0.024,
        "e_down_pooled": -0.001,
        "decile_table": [
            {"p_decile": 7, "E_rises": 0.06462, "E_doesnt": -0.03845, "rises_n": 1335, "doesnt_n": 1773},
            {"p_decile": 8, "E_rises": 0.08791, "E_doesnt": -0.04165, "rises_n": 460, "doesnt_n": 601},
        ],
    }

    def test_decile_7_lookup(self):
        e_up, e_down = get_decile_expected_moves(0.75, self._em_data)
        assert e_up == pytest.approx(0.06462)
        assert e_down == pytest.approx(-0.03845)

    def test_decile_8_lookup(self):
        e_up, e_down = get_decile_expected_moves(0.85, self._em_data)
        assert e_up == pytest.approx(0.08791)
        assert e_down == pytest.approx(-0.04165)

    def test_missing_decile_falls_back_to_pooled(self):
        # Decile 5 (p=0.55) not in the stub table → pooled fallback
        e_up, e_down = get_decile_expected_moves(0.55, self._em_data)
        assert e_up == pytest.approx(0.024)
        assert e_down == pytest.approx(-0.001)

    def test_decile_0_falls_back(self):
        e_up, e_down = get_decile_expected_moves(0.05, self._em_data)
        assert e_up == pytest.approx(0.024)

    def test_zero_rises_falls_back_to_pooled(self):
        # Decile row exists but rises_n=0 → pooled
        em = {
            **self._em_data,
            "decile_table": [
                {"p_decile": 3, "E_rises": 0.05, "E_doesnt": -0.02, "rises_n": 0, "doesnt_n": 100},
            ],
        }
        e_up, e_down = get_decile_expected_moves(0.35, em)
        assert e_up == pytest.approx(0.024)

    def test_p_clipped_at_999(self):
        # p=0.999 → decile = min(int(0.999*10), 9) = 9 → fallback if not in table
        e_up, e_down = get_decile_expected_moves(0.999, self._em_data)
        assert e_up == pytest.approx(0.024)  # 9 not in stub table


# ═══════════════════════════════════════════════════════════════════════════
# End-to-end accounting: portfolio of synthetic trades with known net PnL
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEndAccounting:
    def test_portfolio_net_pnl_is_sum_of_individual(self):
        """Verify the aggregation formula: net_pnl == sum(pnl_realistic)."""
        fills = [
            # (ask, mid, fbid, fmid, p_cal)
            (0.60, 0.595, 0.64, 0.645, 0.75),  # winner: gross=+0.04, fee=0.02 → pnl=+0.02
            (0.60, 0.595, 0.55, 0.545, 0.75),  # loser:  gross=-0.05, fee=0.0  → pnl=-0.05
            (0.80, 0.795, 0.83, 0.835, 0.80),  # winner: gross=+0.03, fee=0.02 → pnl=+0.01
        ]
        pnls = [
            compute_pnl_realistic(a, m, fb, fm, p)["pnl_realistic"]
            for (a, m, fb, fm, p) in fills
        ]
        net = sum(pnls)
        expected_net = 0.02 + (-0.05) + 0.01
        assert net == pytest.approx(expected_net, abs=1e-9)

    def test_all_losers_no_fees_no_upside(self):
        """All-loser portfolio: net == sum of gross (no winner fees to reduce)."""
        fills = [
            (0.50, 0.495, 0.44, 0.435, 0.50),
            (0.50, 0.495, 0.42, 0.415, 0.50),
        ]
        pnls = [compute_pnl_realistic(a, m, fb, fm, p) for (a, m, fb, fm, p) in fills]
        for p in pnls:
            assert p["winner_fee"] == 0.0
        net = sum(p["pnl_realistic"] for p in pnls)
        gross = sum(p["gross_realistic"] for p in pnls)
        assert net == pytest.approx(gross)

    def test_fees_always_reduce_winner_pnl(self):
        """Winner fee must always be >= 0 and <= gross for a winning trade."""
        result = compute_pnl_realistic(0.70, 0.695, 0.75, 0.755, 0.70)
        assert result["winner_fee"] >= 0.0
        assert result["pnl_realistic"] <= result["gross_realistic"]

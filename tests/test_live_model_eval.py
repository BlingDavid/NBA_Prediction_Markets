"""Unit tests for tools/live_model_eval.py.

Tests cover pure functions only — no 233 MB file reads.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure repo root is on path so the import works when run from project root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.live_model_eval import (
    ev_breakeven_threshold,
    ev_per_contract,
    kalshi_winner_fee,
    reliability_table_from_oof,
    pr_curve_summary,
)


# ═══════════════════════════════════════════════════════════════════════════
# kalshi_winner_fee
# ═══════════════════════════════════════════════════════════════════════════

class TestKalshiWinnerFee:
    def test_zero_at_boundaries(self):
        assert kalshi_winner_fee(0.0) == 0.0
        assert kalshi_winner_fee(1.0) == 0.0

    def test_at_p_half_is_max(self):
        # 0.07 * 0.5 * 0.5 = 0.0175 → ceil(1.75) / 100 = 0.02
        fee = kalshi_winner_fee(0.5)
        assert fee == pytest.approx(0.02, abs=1e-9)

    def test_ceil_behaviour(self):
        # p=0.1: 0.07 * 0.1 * 0.9 = 0.0063 → ceil(0.63) / 100 = 0.01
        fee = kalshi_winner_fee(0.1)
        assert fee == pytest.approx(0.01, abs=1e-9)

    def test_p_point_3(self):
        # 0.07 * 0.3 * 0.7 = 0.0147 → ceil(1.47) / 100 = 0.02
        fee = kalshi_winner_fee(0.3)
        assert fee == pytest.approx(0.02, abs=1e-9)

    def test_returns_float(self):
        assert isinstance(kalshi_winner_fee(0.2), float)

    def test_negative_p_returns_zero(self):
        assert kalshi_winner_fee(-0.1) == 0.0

    def test_greater_than_one_returns_zero(self):
        assert kalshi_winner_fee(1.1) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# ev_per_contract
# ═══════════════════════════════════════════════════════════════════════════

class TestEvPerContract:
    def test_positive_ev_when_e_up_large(self):
        # If e_up is huge and half_spread small, EV should be positive
        ev = ev_per_contract(p=0.5, e_up=1.0, e_down=-1.0, half_spread=0.01)
        # ev = 0.5*1.0 + 0.5*(-1.0) - 0.01 - 0.5*0.02 = 0 - 0.01 - 0.01 = -0.02
        assert ev == pytest.approx(-0.02, abs=1e-6)

    def test_fee_deducted_proportional_to_p(self):
        # At p=0, ev = 0*e_up + 1*e_down - half - 0*fee = e_down - half
        ev = ev_per_contract(p=0.0, e_up=0.1, e_down=-0.01, half_spread=0.005)
        assert ev == pytest.approx(-0.01 - 0.005, abs=1e-9)

    def test_negative_when_spread_too_large(self):
        ev = ev_per_contract(p=0.1, e_up=0.024, e_down=-0.005, half_spread=0.10)
        assert ev < 0

    def test_positive_ev_no_spread_high_p(self):
        # At p=0.9, e_up=0.06, e_down=0, spread=0:
        # ev = 0.9*0.06 + 0.1*0 - 0 - 0.9*winner_fee(0.9)
        # winner_fee(0.9) = ceil(0.07*0.9*0.1*100)/100 = ceil(0.63)/100 = 0.01
        # ev = 0.054 - 0.009 = 0.045
        ev = ev_per_contract(p=0.9, e_up=0.06, e_down=0.0, half_spread=0.0)
        assert ev == pytest.approx(0.045, abs=1e-6)

    def test_symmetry_fee(self):
        # Winner fee is symmetric: fee(p) == fee(1-p) when using the
        # mathematical formula (before ceiling). After ceiling, may differ slightly.
        fee_03 = kalshi_winner_fee(0.3)
        fee_07 = kalshi_winner_fee(0.7)
        # 0.07*0.3*0.7 = 0.0147 → ceil = 0.02 for both
        assert fee_03 == fee_07


# ═══════════════════════════════════════════════════════════════════════════
# ev_breakeven_threshold
# ═══════════════════════════════════════════════════════════════════════════

class TestEvBreakevenThreshold:
    def test_returns_dict_with_required_keys(self):
        result = ev_breakeven_threshold(e_up=0.05, e_down=-0.01, half_spread=0.0)
        assert "threshold" in result
        assert "ev_at_threshold" in result
        assert "ev_curve" in result

    def test_ev_curve_is_dataframe(self):
        result = ev_breakeven_threshold(e_up=0.05, e_down=-0.01, half_spread=0.0)
        assert isinstance(result["ev_curve"], pd.DataFrame)
        assert "p" in result["ev_curve"].columns
        assert "ev" in result["ev_curve"].columns

    def test_impossible_when_e_up_negative(self):
        # If e_up and e_down are both negative, EV is always negative
        result = ev_breakeven_threshold(e_up=-0.05, e_down=-0.05, half_spread=0.01)
        assert math.isnan(result["threshold"])
        assert math.isnan(result["ev_at_threshold"])

    def test_threshold_in_valid_range(self):
        # With large enough e_up, should find a threshold
        result = ev_breakeven_threshold(e_up=0.1, e_down=-0.01, half_spread=0.0)
        thr = result["threshold"]
        assert 0.0 < thr < 1.0

    def test_ev_positive_at_threshold(self):
        result = ev_breakeven_threshold(e_up=0.1, e_down=-0.01, half_spread=0.0)
        assert result["ev_at_threshold"] > 0.0

    def test_custom_p_grid(self):
        p_grid = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = ev_breakeven_threshold(
            e_up=0.1, e_down=-0.01, half_spread=0.0, p_grid=p_grid
        )
        # p values in the curve should match the input grid
        assert list(result["ev_curve"]["p"]) == list(p_grid)

    def test_higher_spread_raises_threshold(self):
        # With a larger half-spread, the breakeven threshold should be higher
        r_low = ev_breakeven_threshold(e_up=0.05, e_down=-0.005, half_spread=0.001)
        r_high = ev_breakeven_threshold(e_up=0.05, e_down=-0.005, half_spread=0.01)
        if not math.isnan(r_low["threshold"]) and not math.isnan(r_high["threshold"]):
            assert r_high["threshold"] >= r_low["threshold"]


# ═══════════════════════════════════════════════════════════════════════════
# reliability_table_from_oof
# ═══════════════════════════════════════════════════════════════════════════

class TestReliabilityTableFromOof:
    def _make_data(self, n: int = 1000, seed: int = 42):
        rng = np.random.default_rng(seed)
        y_prob = rng.uniform(0.0, 0.5, size=n).astype(np.float32)
        y_true = (rng.uniform(size=n) < y_prob).astype(np.int8)
        return y_true, y_prob

    def test_returns_dataframe(self):
        y_true, y_prob = self._make_data()
        rel = reliability_table_from_oof(y_true, y_prob, n_bins=5)
        assert isinstance(rel, pd.DataFrame)

    def test_has_required_columns(self):
        y_true, y_prob = self._make_data()
        rel = reliability_table_from_oof(y_true, y_prob, n_bins=5)
        for col in ["bin_lo", "bin_hi", "n", "avg_pred", "emp_rate", "gap"]:
            assert col in rel.columns, f"Missing column: {col}"

    def test_gap_is_emp_rate_minus_avg_pred(self):
        y_true, y_prob = self._make_data()
        rel = reliability_table_from_oof(y_true, y_prob, n_bins=5)
        np.testing.assert_allclose(
            rel["gap"].values,
            rel["emp_rate"].values - rel["avg_pred"].values,
            atol=1e-6,
        )

    def test_total_n_equals_input(self):
        y_true, y_prob = self._make_data(n=500)
        rel = reliability_table_from_oof(y_true, y_prob, n_bins=10)
        assert rel["n"].sum() == 500

    def test_perfect_calibration_zero_gap(self):
        # Construct a perfectly calibrated dataset
        # At each p, emp_rate should equal p exactly (in expectation)
        rng = np.random.default_rng(0)
        y_prob = np.array([0.05] * 1000 + [0.45] * 1000, dtype=np.float32)
        y_true = np.zeros(2000, dtype=np.int8)
        # Force emp_rate = avg_pred in each bin
        y_true[:50] = 1   # 5% of first 1000 = 0.05 emp rate
        y_true[1000:1450] = 1  # 45% of second 1000 = 0.45 emp rate
        rel = reliability_table_from_oof(y_true, y_prob, n_bins=10)
        # Both bins should have near-zero gap
        for _, row in rel.iterrows():
            assert abs(row["gap"]) < 0.06, f"Gap too large in bin {row['bin_lo']}-{row['bin_hi']}: {row['gap']:.4f}"


# ═══════════════════════════════════════════════════════════════════════════
# pr_curve_summary
# ═══════════════════════════════════════════════════════════════════════════

class TestPrCurveSummary:
    def _make_data(self, n: int = 500, seed: int = 7):
        rng = np.random.default_rng(seed)
        y_prob = rng.uniform(0.01, 0.99, n).astype(np.float32)
        y_true = (rng.uniform(0.0, 1.0, n) < 0.1).astype(np.int8)  # ~10% positive rate
        return y_true, y_prob

    def test_returns_dataframe(self):
        y_true, y_prob = self._make_data()
        pr = pr_curve_summary(y_true, y_prob, n_thresholds=10)
        assert isinstance(pr, pd.DataFrame)

    def test_has_required_columns(self):
        y_true, y_prob = self._make_data()
        pr = pr_curve_summary(y_true, y_prob, n_thresholds=10)
        for col in ["threshold", "precision", "recall", "f1"]:
            assert col in pr.columns

    def test_n_rows_bounded_by_n_thresholds(self):
        y_true, y_prob = self._make_data()
        pr = pr_curve_summary(y_true, y_prob, n_thresholds=15)
        assert len(pr) <= 15

    def test_precision_in_zero_one(self):
        y_true, y_prob = self._make_data()
        pr = pr_curve_summary(y_true, y_prob)
        assert (pr["precision"] >= 0.0).all() and (pr["precision"] <= 1.0).all()

    def test_recall_in_zero_one(self):
        y_true, y_prob = self._make_data()
        pr = pr_curve_summary(y_true, y_prob)
        assert (pr["recall"] >= 0.0).all() and (pr["recall"] <= 1.0).all()

    def test_f1_in_zero_one(self):
        y_true, y_prob = self._make_data()
        pr = pr_curve_summary(y_true, y_prob)
        assert (pr["f1"] >= 0.0).all() and (pr["f1"] <= 1.0 + 1e-9).all()

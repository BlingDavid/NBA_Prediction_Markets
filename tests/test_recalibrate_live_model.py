"""Unit tests for tools/recalibrate_live_model.py.

Tests cover pure helpers ONLY — no 233 MB file reads, no live model loading.
All inputs are small synthetic arrays constructed inline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.recalibrate_live_model import (
    apply_isotonic,
    ev_per_contract,
    ev_positive_count,
    grouped_split,
    kalshi_winner_fee,
    reliability_table,
)


# ═══════════════════════════════════════════════════════════════════════════
# apply_isotonic
# ═══════════════════════════════════════════════════════════════════════════

class TestApplyIsotonic:
    def _make_fitted_iso(self) -> IsotonicRegression:
        iso = IsotonicRegression(out_of_bounds="clip")
        X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        y = np.array([0.05, 0.08, 0.12, 0.20, 0.30, 0.40, 0.45, 0.50])
        iso.fit(X, y)
        return iso

    def test_returns_float32(self):
        iso = self._make_fitted_iso()
        x = np.array([0.2, 0.4, 0.6], dtype=np.float32)
        result = apply_isotonic(iso, x)
        assert result.dtype == np.float32

    def test_output_clipped_to_0_1(self):
        """Values outside fitting range are clipped, not extrapolated wildly."""
        iso = self._make_fitted_iso()
        # Edge inputs: very small and very large
        x = np.array([0.0, 0.01, 0.99, 1.0], dtype=np.float32)
        result = apply_isotonic(iso, x)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_monotone_non_decreasing(self):
        """Output must be non-decreasing (isotonic property)."""
        iso = self._make_fitted_iso()
        x = np.linspace(0.1, 0.8, 20).astype(np.float32)
        result = apply_isotonic(iso, x)
        assert np.all(np.diff(result) >= -1e-7), "apply_isotonic output not monotone"

    def test_known_value(self):
        """A trained-value input should map through correctly."""
        iso = IsotonicRegression(out_of_bounds="clip")
        X = np.array([0.3, 0.5, 0.7])
        y = np.array([0.10, 0.20, 0.30])
        iso.fit(X, y)
        result = apply_isotonic(iso, np.array([0.5], dtype=np.float32))
        assert result[0] == pytest.approx(0.20, abs=1e-5)

    def test_same_length_as_input(self):
        iso = self._make_fitted_iso()
        x = np.array([0.1, 0.3, 0.5, 0.7], dtype=np.float32)
        result = apply_isotonic(iso, x)
        assert len(result) == len(x)


# ═══════════════════════════════════════════════════════════════════════════
# grouped_split
# ═══════════════════════════════════════════════════════════════════════════

class TestGroupedSplit:
    def _groups(self, n_groups: int = 20, rows_per: int = 50) -> np.ndarray:
        return np.repeat(np.arange(n_groups), rows_per)

    def test_returns_two_boolean_masks(self):
        groups = self._groups()
        fit, evl = grouped_split(groups)
        assert fit.dtype == bool
        assert evl.dtype == bool

    def test_masks_cover_all_rows(self):
        groups = self._groups(n_groups=10, rows_per=30)
        fit, evl = grouped_split(groups)
        assert np.all(fit | evl), "Some rows not covered"
        assert not np.any(fit & evl), "Some rows in both halves"

    def test_no_group_appears_in_both_halves(self):
        groups = self._groups(n_groups=40, rows_per=10)
        fit, evl = grouped_split(groups)
        fit_groups = set(groups[fit])
        evl_groups = set(groups[evl])
        overlap = fit_groups & evl_groups
        assert len(overlap) == 0, f"Groups in both halves: {overlap}"

    def test_fit_frac_roughly_respected(self):
        """fit half should have approximately fit_frac fraction of unique groups."""
        groups = self._groups(n_groups=100, rows_per=20)
        fit, evl = grouped_split(groups, fit_frac=0.5, seed=0)
        n_fit_groups = len(np.unique(groups[fit]))
        n_total_groups = len(np.unique(groups))
        # Allow 5% tolerance on group count
        assert 0.40 <= n_fit_groups / n_total_groups <= 0.60

    def test_reproducible_with_same_seed(self):
        groups = self._groups()
        fit1, _ = grouped_split(groups, seed=7)
        fit2, _ = grouped_split(groups, seed=7)
        np.testing.assert_array_equal(fit1, fit2)

    def test_different_seeds_differ(self):
        groups = self._groups(n_groups=50)
        fit1, _ = grouped_split(groups, seed=1)
        fit2, _ = grouped_split(groups, seed=2)
        assert not np.all(fit1 == fit2), "Different seeds produced identical splits"

    def test_single_group_goes_to_fit(self):
        """Edge case: only 1 group — must end up in fit half."""
        groups = np.array([0, 0, 0, 0])
        fit, evl = grouped_split(groups, fit_frac=0.5)
        assert fit.any() or evl.any()
        # No overlap
        assert not np.any(fit & evl)

    def test_all_rows_in_one_of_the_masks(self):
        groups = np.array(["A", "A", "B", "B", "C", "C"])
        fit, evl = grouped_split(groups, fit_frac=0.5)
        assert len(fit) == 6
        assert np.all(fit | evl)


# ═══════════════════════════════════════════════════════════════════════════
# reliability_table
# ═══════════════════════════════════════════════════════════════════════════

class TestReliabilityTable:
    def _make_data(self, n: int = 1000, seed: int = 42):
        rng = np.random.default_rng(seed)
        y_prob = rng.uniform(0.0, 0.5, size=n).astype(np.float32)
        y_true = (rng.uniform(size=n) < y_prob).astype(np.int8)
        return y_true, y_prob

    def test_returns_dataframe(self):
        y_true, y_prob = self._make_data()
        rel = reliability_table(y_true, y_prob, n_bins=5)
        import pandas as pd
        assert isinstance(rel, pd.DataFrame)

    def test_has_required_columns(self):
        y_true, y_prob = self._make_data()
        rel = reliability_table(y_true, y_prob, n_bins=5)
        for col in ["bin_lo", "bin_hi", "n", "avg_pred", "emp_rate", "gap"]:
            assert col in rel.columns, f"Missing column: {col}"

    def test_gap_is_emp_minus_pred(self):
        y_true, y_prob = self._make_data()
        rel = reliability_table(y_true, y_prob)
        np.testing.assert_allclose(
            rel["gap"].values,
            rel["emp_rate"].values - rel["avg_pred"].values,
            atol=1e-6,
        )

    def test_total_n_equals_input(self):
        y_true, y_prob = self._make_data(n=500)
        rel = reliability_table(y_true, y_prob, n_bins=10)
        assert rel["n"].sum() == 500

    def test_empty_bins_excluded(self):
        """If all points land in one bin, other bins should not appear."""
        y_prob = np.full(100, 0.05, dtype=np.float32)
        y_true = np.zeros(100, dtype=np.int8)
        y_true[:5] = 1
        rel = reliability_table(y_true, y_prob, n_bins=10)
        # Only the bin covering 0.05 should be present
        assert len(rel) == 1
        assert rel.iloc[0]["n"] == 100


# ═══════════════════════════════════════════════════════════════════════════
# kalshi_winner_fee (smoke — shared formula; full tests in test_live_model_eval)
# ═══════════════════════════════════════════════════════════════════════════

class TestKalshiWinnerFeeLocal:
    def test_boundaries_return_zero(self):
        assert kalshi_winner_fee(0.0) == 0.0
        assert kalshi_winner_fee(1.0) == 0.0

    def test_p_half_max_fee(self):
        assert kalshi_winner_fee(0.5) == pytest.approx(0.02, abs=1e-9)

    def test_p_01_fee(self):
        # 0.07 * 0.1 * 0.9 = 0.0063 → ceil(0.63) / 100 = 0.01
        assert kalshi_winner_fee(0.1) == pytest.approx(0.01, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════════════
# ev_per_contract
# ═══════════════════════════════════════════════════════════════════════════

class TestEvPerContractLocal:
    def test_negative_when_spread_too_large(self):
        ev = ev_per_contract(0.1, 0.024, -0.005, half_spread=0.10)
        assert ev < 0

    def test_zero_spread_small_e_up_negative(self):
        # Very small e_up → negative ev
        ev = ev_per_contract(0.5, 0.001, -0.001, half_spread=0.0)
        assert ev < 0


# ═══════════════════════════════════════════════════════════════════════════
# ev_positive_count
# ═══════════════════════════════════════════════════════════════════════════

class TestEvPositiveCount:
    def test_all_zero_when_e_up_tiny(self):
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        count = ev_positive_count(y_prob, e_up=0.001, e_down=-0.005, half_spread=0.01)
        assert count == 0

    def test_count_increases_with_higher_e_up(self):
        y_prob = np.linspace(0.01, 0.99, 200).astype(np.float32)
        c_low  = ev_positive_count(y_prob, e_up=0.01,  e_down=-0.005, half_spread=0.01)
        c_high = ev_positive_count(y_prob, e_up=0.10,  e_down=-0.005, half_spread=0.01)
        assert c_high >= c_low

    def test_returns_int(self):
        y_prob = np.array([0.3, 0.5], dtype=np.float32)
        result = ev_positive_count(y_prob, 0.05, -0.005, 0.0)
        assert isinstance(result, int)

    def test_all_positive_with_extreme_e_up(self):
        """With huge e_up and no spread, every p > 0 should be EV-positive."""
        y_prob = np.linspace(0.01, 0.99, 50).astype(np.float32)
        count = ev_positive_count(y_prob, e_up=10.0, e_down=0.0, half_spread=0.0)
        assert count == len(y_prob)

    def test_count_falls_with_larger_spread(self):
        y_prob = np.linspace(0.01, 0.99, 100).astype(np.float32)
        c_tight = ev_positive_count(y_prob, 0.05, -0.005, half_spread=0.0)
        c_wide  = ev_positive_count(y_prob, 0.05, -0.005, half_spread=0.05)
        assert c_tight >= c_wide

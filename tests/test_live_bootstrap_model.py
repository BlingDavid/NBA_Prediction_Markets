"""
Tests for live_bootstrap_model._apply_final_predictions.

Ensures the scored_rows artifact is honest: labeled rows get their
out-of-fold predictions (not in-sample final-model scores), unlabeled
rows get the final-model prediction, and prediction_basis flags the
split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from live_bootstrap_model import (
    _apply_final_predictions,
    _apply_shrinkage,
    _build_group_weights,
    _lift_table,
    _select_feature_columns,
    _tune_shrinkage,
)


def _fit_tiny_model() -> Pipeline:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f0": rng.normal(size=40), "f1": rng.normal(size=40)})
    y = (X["f0"] + X["f1"] > 0).astype(int)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=500, random_state=0)),
    ])
    pipe.fit(X, y)
    return pipe


def _make_matrix(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "f0": rng.normal(size=n),
        "f1": rng.normal(size=n),
        "ticker": [f"T{i}" for i in range(n)],
    })


class TestApplyFinalPredictions:
    def test_unlabeled_rows_get_final_model_predictions(self):
        model = _fit_tiny_model()
        scored = _make_matrix(5)
        empty_labeled = pd.DataFrame(columns=["oof_prediction_raw", "oof_prediction"])

        out = _apply_final_predictions(
            scored_matrix=scored,
            labeled_rows=empty_labeled,
            final_model=model,
            feature_cols=["f0", "f1"],
            base_rate=0.5,
            calibration_alpha=1.0,
        )

        expected_raw = model.predict_proba(scored[["f0", "f1"]])[:, 1]
        np.testing.assert_allclose(out["pred_bootstrap_home_up_5m_raw"], expected_raw)
        assert (out["prediction_basis"] == "final_model").all()

    def test_labeled_rows_get_oof_predictions(self):
        model = _fit_tiny_model()
        scored = _make_matrix(5)
        # Labeled = rows 1 and 3 of scored; OOF values are clearly distinguishable.
        labeled = scored.loc[[1, 3]].copy()
        labeled["oof_prediction_raw"] = [0.11, 0.22]
        labeled["oof_prediction"] = [0.15, 0.25]

        out = _apply_final_predictions(
            scored_matrix=scored,
            labeled_rows=labeled,
            final_model=model,
            feature_cols=["f0", "f1"],
            base_rate=0.5,
            calibration_alpha=1.0,
        )

        assert out.loc[1, "pred_bootstrap_home_up_5m_raw"] == pytest.approx(0.11)
        assert out.loc[3, "pred_bootstrap_home_up_5m_raw"] == pytest.approx(0.22)
        assert out.loc[1, "pred_bootstrap_home_up_5m"] == pytest.approx(0.15)
        assert out.loc[3, "pred_bootstrap_home_up_5m"] == pytest.approx(0.25)
        assert out.loc[1, "prediction_basis"] == "oof"
        assert out.loc[3, "prediction_basis"] == "oof"

    def test_unlabeled_rows_unchanged_when_mixed(self):
        """In a mix, unlabeled rows must keep shrunk final-model scores."""
        model = _fit_tiny_model()
        scored = _make_matrix(5)
        labeled = scored.loc[[1, 3]].copy()
        labeled["oof_prediction_raw"] = [0.11, 0.22]
        labeled["oof_prediction"] = [0.15, 0.25]

        out = _apply_final_predictions(
            scored_matrix=scored,
            labeled_rows=labeled,
            final_model=model,
            feature_cols=["f0", "f1"],
            base_rate=0.4,
            calibration_alpha=0.6,
        )

        raw_all = model.predict_proba(scored[["f0", "f1"]])[:, 1]
        shrunk_all = _apply_shrinkage(raw_all, 0.4, 0.6)

        for i in [0, 2, 4]:
            assert out.loc[i, "pred_bootstrap_home_up_5m_raw"] == pytest.approx(raw_all[i])
            assert out.loc[i, "pred_bootstrap_home_up_5m"] == pytest.approx(shrunk_all[i])
            assert out.loc[i, "prediction_basis"] == "final_model"

    def test_prediction_basis_column_covers_every_row(self):
        model = _fit_tiny_model()
        scored = _make_matrix(5)
        labeled = scored.loc[[1, 3]].copy()
        labeled["oof_prediction_raw"] = [0.11, 0.22]
        labeled["oof_prediction"] = [0.15, 0.25]

        out = _apply_final_predictions(
            scored_matrix=scored,
            labeled_rows=labeled,
            final_model=model,
            feature_cols=["f0", "f1"],
            base_rate=0.5,
            calibration_alpha=1.0,
        )

        assert out["prediction_basis"].isin({"oof", "final_model"}).all()
        assert out["prediction_basis"].notna().all()
        assert (out["prediction_basis"] == "oof").sum() == 2
        assert (out["prediction_basis"] == "final_model").sum() == 3

    def test_does_not_mutate_input_scored_matrix(self):
        model = _fit_tiny_model()
        scored = _make_matrix(5)
        scored_before = scored.copy(deep=True)
        labeled = pd.DataFrame(columns=["oof_prediction_raw", "oof_prediction"])

        _apply_final_predictions(
            scored_matrix=scored,
            labeled_rows=labeled,
            final_model=model,
            feature_cols=["f0", "f1"],
            base_rate=0.5,
            calibration_alpha=1.0,
        )

        pd.testing.assert_frame_equal(scored, scored_before)


class TestApplyShrinkage:
    def test_alpha_zero_collapses_to_base_rate(self):
        y = np.array([0.1, 0.5, 0.9])
        out = _apply_shrinkage(y, base_rate=0.4, alpha=0.0)
        np.testing.assert_allclose(out, [0.4, 0.4, 0.4])

    def test_alpha_one_leaves_probs_unchanged(self):
        y = np.array([0.1, 0.5, 0.9])
        out = _apply_shrinkage(y, base_rate=0.3, alpha=1.0)
        np.testing.assert_allclose(out, [0.1, 0.5, 0.9])

    def test_partial_shrinkage_is_linear_interpolation(self):
        # 0.5 * (0.8 - 0.5) + 0.5 = 0.65
        out = _apply_shrinkage(np.array([0.8]), base_rate=0.5, alpha=0.5)
        assert out[0] == pytest.approx(0.65)

    def test_clipping_prevents_zero_and_one(self):
        # With base=0 and alpha=1, shrunk = y_prob; clip lower bound at 1e-6
        out = _apply_shrinkage(np.array([0.0, 1.0]), base_rate=0.0, alpha=1.0)
        assert out[0] >= 1e-6
        assert out[1] <= 1 - 1e-6


class TestTuneShrinkage:
    def test_returns_21_grid_candidates(self):
        rng = np.random.default_rng(0)
        y = pd.Series(rng.integers(0, 2, size=100))
        y_prob = rng.uniform(size=100)
        result = _tune_shrinkage(y, y_prob)
        assert len(result["grid"]) == 21
        assert result["selection_metric"] == "logloss"

    def test_best_alpha_minimizes_logloss(self):
        rng = np.random.default_rng(1)
        y = pd.Series(rng.integers(0, 2, size=200))
        y_prob = rng.uniform(size=200)
        result = _tune_shrinkage(y, y_prob)
        best = min(result["grid"], key=lambda r: r["metrics"]["logloss"])
        assert result["alpha"] == best["alpha"]

    def test_base_rate_matches_empirical_positive_rate(self):
        y = pd.Series([0, 0, 0, 1, 1])
        y_prob = np.array([0.2, 0.3, 0.4, 0.6, 0.7])
        result = _tune_shrinkage(y, y_prob)
        assert result["base_rate"] == pytest.approx(0.4, abs=1e-6)


class TestLiftTable:
    def test_returns_row_per_quantile(self):
        rng = np.random.default_rng(0)
        y = pd.Series(rng.integers(0, 2, size=100))
        y_prob = rng.uniform(size=100)
        rows = _lift_table(y, y_prob, quantiles=(0.70, 0.90))
        assert [r["quantile"] for r in rows] == [0.70, 0.90]

    def test_top_bucket_rate_higher_for_correlated_signal(self):
        # y correlates with y_prob → top bucket should beat the base rate
        y_prob = np.linspace(0.0, 1.0, 200)
        y = pd.Series((y_prob > 0.5).astype(int))
        rows = _lift_table(y, y_prob, quantiles=(0.90,))
        assert rows[0]["lift_vs_base"] > 1.0

    def test_lift_vs_base_none_when_no_positives(self):
        y = pd.Series([0] * 10)
        y_prob = np.linspace(0.1, 0.9, 10)
        rows = _lift_table(y, y_prob, quantiles=(0.70,))
        assert rows[0]["lift_vs_base"] is None


class TestSelectFeatureColumns:
    def test_keeps_valid_varying_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
        selected, dropped = _select_feature_columns(df, ["a", "b"])
        assert selected == ["a", "b"]
        assert dropped == {}

    def test_drops_missing_column_with_reason(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        selected, dropped = _select_feature_columns(df, ["a", "b"])
        assert selected == ["a"]
        assert dropped == {"b": "missing_from_matrix"}

    def test_drops_all_null_column(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
        selected, dropped = _select_feature_columns(df, ["a", "b"])
        assert selected == ["a"]
        assert dropped == {"b": "all_missing"}

    def test_drops_constant_column(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [5, 5, 5]})
        selected, dropped = _select_feature_columns(df, ["a", "b"])
        assert selected == ["a"]
        assert dropped == {"b": "constant_or_single_value"}


class TestBuildGroupWeights:
    def test_sum_equals_row_count(self):
        groups = pd.Series(["a", "a", "a", "b", "b", "c"])
        w = _build_group_weights(groups)
        assert w.sum() == pytest.approx(len(groups))

    def test_larger_group_gets_smaller_per_row_weight(self):
        groups = pd.Series(["big", "big", "big", "big", "small"])
        w = _build_group_weights(groups)
        big_weight = w[groups == "big"].iloc[0]
        small_weight = w[groups == "small"].iloc[0]
        assert small_weight > big_weight

    def test_equal_group_sizes_give_equal_weights(self):
        groups = pd.Series(["a", "a", "b", "b"])
        w = _build_group_weights(groups)
        assert w.nunique() == 1
        assert w.iloc[0] == pytest.approx(1.0)

    def test_preserves_input_index(self):
        groups = pd.Series(["a", "b", "a"], index=[10, 20, 30])
        w = _build_group_weights(groups)
        assert list(w.index) == [10, 20, 30]

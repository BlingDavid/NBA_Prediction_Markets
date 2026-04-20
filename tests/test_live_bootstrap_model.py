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

from live_bootstrap_model import _apply_final_predictions, _apply_shrinkage


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

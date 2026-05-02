"""scorer: load .pkl, engineer features, predict_proba + calibrated shrinkage.

The shrinkage formula must match live_bootstrap_model._apply_shrinkage exactly:
   p_cal = clip(base_rate + alpha * (raw - base_rate), 1e-6, 1 - 1e-6)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_trader.scorer import load_model, score, engineer_features


MODEL_PATH = Path("models/live_home_up_5m_bootstrap.pkl")


@pytest.fixture(scope="module")
def bundle():
    if not MODEL_PATH.exists():
        pytest.skip(f"{MODEL_PATH} not present")
    return load_model(MODEL_PATH)


def _raw_row() -> dict:
    """A single row of the latest_features.csv schema, populated to be
    in-distribution for a typical mid-game state."""
    return dict(
        captured_at="2026-04-28T19:30:00Z",
        ticker="KXNBAGAME-26APR28LALHOU-LAL",
        game_key="2026-04-28_LAL_HOU",
        bet_side="home",
        status_state="in",
        period=2,
        seconds_elapsed=600.0,
        seconds_left_in_period=300.0,
        home_score=42.0,
        away_score=39.0,
        score_margin_home=3.0,
        total_points=81.0,
        yes_bid=0.55,
        yes_ask=0.57,
        yes_mid=0.56,
        no_bid=0.43,
        no_ask=0.45,
        last_price=0.56,
        market_home_implied=0.56,
        volume=10000.0,
        open_interest=2000.0,
        yes_depth_notional_3=300.0,
        yes_depth_notional_5=500.0,
        no_depth_notional_3=280.0,
        no_depth_notional_5=480.0,
        yes_weighted_price_3=0.555,
        no_weighted_price_3=0.435,
        espn_home_implied=np.nan,
        espn_away_implied=np.nan,
        oddsapi_home_consensus=np.nan,
        oddsapi_away_consensus=np.nan,
        oddsapi_books=np.nan,
        market_consensus_home=np.nan,
        pregame_home_win_prob=0.50,
        pregame_away_win_prob=0.50,
        pregame_spread=0.0,
        pregame_total=220.0,
        pregame_edge_home=0.06,
        consensus_gap_home=np.nan,
    )


def test_engineer_features_adds_required_derived_columns(bundle):
    df = pd.DataFrame([_raw_row()])
    engineered = engineer_features(df)
    for col in bundle["feature_cols"]:
        assert col in engineered.columns, f"missing engineered col: {col}"
    # Spot-check a couple of formulas:
    assert engineered["market_yes_spread"].iloc[0] == pytest.approx(0.02)
    assert 0.0 <= engineered["game_progress"].iloc[0] <= 1.0


def test_score_returns_finite_probabilities(bundle):
    df = pd.DataFrame([_raw_row()])
    p_raw, p_cal = score(bundle, df)
    assert np.isfinite(p_raw[0]) and 0.0 <= p_raw[0] <= 1.0
    assert np.isfinite(p_cal[0]) and 0.0 <= p_cal[0] <= 1.0


def test_calibration_shrinkage_matches_formula(bundle):
    df = pd.DataFrame([_raw_row()])
    p_raw, p_cal = score(bundle, df)
    base_rate = bundle["calibration"]["base_rate"]
    alpha = bundle["calibration"]["alpha"]
    expected = np.clip(base_rate + alpha * (p_raw - base_rate), 1e-6, 1 - 1e-6)
    np.testing.assert_allclose(p_cal, expected, atol=1e-9)


def test_score_handles_multiple_rows(bundle):
    df = pd.DataFrame([_raw_row(), _raw_row()])
    p_raw, p_cal = score(bundle, df)
    assert p_raw.shape == (2,)
    assert p_cal.shape == (2,)


def test_score_injects_nan_for_missing_feature_columns(bundle, monkeypatch):
    """The score() function must inject NaN for any feature_col that's
    absent from the engineered df, so the model's SimpleImputer can fill
    it. Simulates a missing column by patching engineer_features to drop
    a column that's in feature_cols."""
    from paper_trader.scorer import engineer_features as orig_engineer_features

    # Pick the first feature column to drop during engineering
    col_to_drop = bundle["feature_cols"][0]

    def mock_engineer_features(df):
        engineered = orig_engineer_features(df)
        # Drop a column that score() expects, forcing injection to fire
        if col_to_drop in engineered.columns:
            engineered = engineered.drop(columns=[col_to_drop])
        return engineered

    monkeypatch.setattr("paper_trader.scorer.engineer_features", mock_engineer_features)

    df = pd.DataFrame([_raw_row()])
    # If injection is broken, the model will raise KeyError when it tries to
    # access the missing column. With injection working, score() adds NaN for
    # the missing column, and the pipeline's SimpleImputer fills it with an
    # imputed value (median from training, or 0.0 if all-NaN at train time).
    # The test asserts the call succeeds and returns finite scalars.
    p_raw, p_cal = score(bundle, df)
    assert p_raw.shape == (1,)
    assert p_cal.shape == (1,)
    assert np.isfinite(p_raw[0])
    assert np.isfinite(p_cal[0])

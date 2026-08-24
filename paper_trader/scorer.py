"""Model loader + live feature engineering + calibrated scoring.

The feature-engineering transforms live in live_training_matrix and are
imported here, NOT reimplemented, so training and live can never drift.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

# Re-use training-time helpers verbatim. If these symbols move, update the
# import — don't recreate the formulas.
from live_training_matrix import _safe_numeric, _status_flag


REGULATION_SECONDS = 48 * 60

logger = logging.getLogger(__name__)


def load_model(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)
    required = {"model", "feature_cols", "calibration"}
    missing = required - bundle.keys()
    if missing:
        raise ValueError(f"model bundle missing keys: {missing}")

    # ------------------------------------------------------------------
    # sklearn version guard — fail loud instead of silently broken trader
    # ------------------------------------------------------------------
    # When the venv's sklearn drifts past the version that pickled this
    # bundle, sklearn emits InconsistentVersionWarning at load time and
    # every predict() raises AttributeError('SimpleImputer' object has no
    # attribute '_fill_dtype'). The paper trader catches those as
    # event_type=error events and keeps ticking — the loop looks healthy
    # while no trade is ever placed. Raising here surfaces the skew clearly
    # so the operator knows to retrain: python live_bootstrap_model.py
    # ------------------------------------------------------------------
    bundle_version = bundle.get("sklearn_version")
    runtime_version = sklearn.__version__

    if bundle_version is None:
        logger.warning(
            "sklearn_version key absent from model bundle at %s — bundle was "
            "pickled before version-tagging was added. Runtime sklearn is %s. "
            "If predict() errors appear, retrain via: python live_bootstrap_model.py",
            path,
            runtime_version,
        )
    elif bundle_version != runtime_version:
        raise RuntimeError(
            f"sklearn version mismatch: model bundle at '{path}' was pickled "
            f"with sklearn {bundle_version}, but the runtime has sklearn "
            f"{runtime_version}. This causes silent predict() failures "
            f"(AttributeError on SimpleImputer._fill_dtype). "
            f"Retrain the model: python live_bootstrap_model.py"
        )

    return bundle


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same derived-column transforms as
    live_training_matrix.build_in_game_training_matrix."""
    df = df.copy()
    numeric_cols = [
        "period", "seconds_elapsed", "seconds_left_in_period",
        "home_score", "away_score", "score_margin_home", "total_points",
        "yes_bid", "yes_ask", "yes_mid", "no_bid", "no_ask",
        "last_price", "market_home_implied",
        "volume", "open_interest",
        "yes_depth_notional_3", "yes_depth_notional_5",
        "no_depth_notional_3", "no_depth_notional_5",
        "yes_weighted_price_3", "no_weighted_price_3",
        "espn_home_implied", "espn_away_implied",
        "oddsapi_home_consensus", "oddsapi_away_consensus", "oddsapi_books",
        "market_consensus_home",
        "pregame_home_win_prob", "pregame_away_win_prob",
        "pregame_spread", "pregame_total", "pregame_edge_home",
        "consensus_gap_home",
    ]
    df = _safe_numeric(df, numeric_cols)

    df["game_progress"] = (df["seconds_elapsed"] / REGULATION_SECONDS).clip(lower=0, upper=1)
    df["regulation_seconds_remaining"] = REGULATION_SECONDS - df["seconds_elapsed"]
    df.loc[df["regulation_seconds_remaining"] < 0, "regulation_seconds_remaining"] = 0

    df["market_yes_spread"] = df["yes_ask"] - df["yes_bid"]
    df["market_no_spread"] = df["no_ask"] - df["no_bid"]
    df["orderbook_imbalance_3"] = df["yes_depth_notional_3"] - df["no_depth_notional_3"]
    df["orderbook_imbalance_5"] = df["yes_depth_notional_5"] - df["no_depth_notional_5"]
    df["orderbook_pressure_3"] = df["yes_weighted_price_3"] - df["no_weighted_price_3"]
    df["market_vs_pregame_home"] = df["market_home_implied"] - df["pregame_home_win_prob"]
    df["score_gap_vs_pregame_spread"] = df["score_margin_home"] - df["pregame_spread"]
    df["points_vs_pregame_total"] = df["total_points"] - df["pregame_total"]

    if "status_state" in df.columns:
        df["flag_status_pre"] = _status_flag(df["status_state"], "pre")
        df["flag_status_live"] = _status_flag(df["status_state"], "in")
        df["flag_status_post"] = _status_flag(df["status_state"], "post")
    else:
        df["flag_status_pre"] = 0
        df["flag_status_live"] = 0
        df["flag_status_post"] = 0
    df["flag_has_game_state"] = df["home_score"].notna().astype(int)
    df["flag_has_time_state"] = df["seconds_elapsed"].notna().astype(int)

    return df


def score(bundle: dict[str, Any], df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Returns (p_raw, p_calibrated), each a 1-D ndarray of len(df)."""
    feature_cols = bundle["feature_cols"]
    engineered = engineer_features(df)
    # Add any model-required cols that are entirely absent in this feed
    # (e.g., an oddsapi_* column when ENABLE_ODDS_API was on at training time
    # but the feed doesn't carry it). The training pipeline imputes via the
    # SimpleImputer, so injecting NaN here matches training behavior.
    for col in feature_cols:
        if col not in engineered.columns:
            engineered[col] = np.nan
    X = engineered[feature_cols]
    raw = bundle["model"].predict_proba(X)[:, 1]
    base_rate = float(bundle["calibration"]["base_rate"])
    alpha = float(bundle["calibration"]["alpha"])
    cal = np.clip(base_rate + alpha * (raw - base_rate), 1e-6, 1 - 1e-6)
    return raw.astype(float), cal.astype(float)

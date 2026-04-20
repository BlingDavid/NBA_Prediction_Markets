"""
Build the first in-game training matrix and target coverage report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import ENABLE_ODDS_API, LIVE_LABELS_DIR
from live_labeler import build_labeled_training_set


ODDSAPI_FEATURES = ["oddsapi_home_consensus", "oddsapi_away_consensus", "oddsapi_books"]


def _as_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _status_flag(series: pd.Series, value: str) -> pd.Series:
    return series.fillna("").astype(str).str.lower().eq(value).astype(int)


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_in_game_training_matrix(
    labeled: pd.DataFrame,
    only_home_side: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Convert labeled contract snapshots into a model-ready matrix.

    For `final_home_win`, home-side rows are the cleanest first cut because the
    target is game-level while the raw data is contract-level.
    """
    if labeled.empty:
        return pd.DataFrame(), []

    df = labeled.copy()
    df["captured_at"] = _as_timestamp(df["captured_at"])

    if only_home_side:
        df = df[df["bet_side"] == "home"].copy()

    numeric_cols = [
        "seconds_elapsed",
        "seconds_left_in_period",
        "home_score",
        "away_score",
        "score_margin_home",
        "total_points",
        "yes_bid",
        "yes_ask",
        "yes_mid",
        "no_bid",
        "no_ask",
        "last_price",
        "market_home_implied",
        "volume",
        "open_interest",
        "yes_depth_notional_3",
        "yes_depth_notional_5",
        "no_depth_notional_3",
        "no_depth_notional_5",
        "yes_weighted_price_3",
        "no_weighted_price_3",
        "espn_home_implied",
        "espn_away_implied",
        *(ODDSAPI_FEATURES if ENABLE_ODDS_API else []),
        "market_consensus_home",
        "pregame_home_win_prob",
        "pregame_away_win_prob",
        "pregame_spread",
        "pregame_total",
        "pregame_edge_home",
        "consensus_gap_home",
        "label_yes_mid_move_5m",
        "label_market_home_implied_move_5m",
        "label_yes_up_5m",
        "label_home_up_5m",
        "label_beats_close_yes",
        "label_beats_close_home",
        "label_final_home_win",
    ]
    df = _safe_numeric(df, numeric_cols)

    regulation_seconds = 48 * 60
    df["game_progress"] = (df["seconds_elapsed"] / regulation_seconds).clip(lower=0, upper=1)
    df["regulation_seconds_remaining"] = regulation_seconds - df["seconds_elapsed"]
    df.loc[df["regulation_seconds_remaining"] < 0, "regulation_seconds_remaining"] = 0

    df["market_yes_spread"] = df["yes_ask"] - df["yes_bid"]
    df["market_no_spread"] = df["no_ask"] - df["no_bid"]
    df["orderbook_imbalance_3"] = df["yes_depth_notional_3"] - df["no_depth_notional_3"]
    df["orderbook_imbalance_5"] = df["yes_depth_notional_5"] - df["no_depth_notional_5"]
    df["orderbook_pressure_3"] = df["yes_weighted_price_3"] - df["no_weighted_price_3"]
    df["market_vs_pregame_home"] = df["market_home_implied"] - df["pregame_home_win_prob"]
    df["market_vs_consensus_home"] = df["market_home_implied"] - df["market_consensus_home"]
    df["espn_vs_market_home"] = df["espn_home_implied"] - df["market_home_implied"]
    df["score_gap_vs_pregame_spread"] = df["score_margin_home"] - df["pregame_spread"]
    df["points_vs_pregame_total"] = df["total_points"] - df["pregame_total"]

    df["flag_status_pre"] = _status_flag(df["status_state"], "pre")
    df["flag_status_live"] = _status_flag(df["status_state"], "in")
    df["flag_status_post"] = _status_flag(df["status_state"], "post")
    df["flag_has_game_state"] = df["home_score"].notna().astype(int)
    df["flag_has_time_state"] = df["seconds_elapsed"].notna().astype(int)

    feature_cols = [
        "period",
        "seconds_elapsed",
        "seconds_left_in_period",
        "game_progress",
        "regulation_seconds_remaining",
        "home_score",
        "away_score",
        "score_margin_home",
        "total_points",
        "yes_bid",
        "yes_ask",
        "yes_mid",
        "no_bid",
        "no_ask",
        "last_price",
        "market_home_implied",
        "market_yes_spread",
        "market_no_spread",
        "volume",
        "open_interest",
        "yes_depth_notional_3",
        "yes_depth_notional_5",
        "no_depth_notional_3",
        "no_depth_notional_5",
        "orderbook_imbalance_3",
        "orderbook_imbalance_5",
        "yes_weighted_price_3",
        "no_weighted_price_3",
        "orderbook_pressure_3",
        "espn_home_implied",
        "espn_away_implied",
        *(ODDSAPI_FEATURES if ENABLE_ODDS_API else []),
        "market_consensus_home",
        "pregame_home_win_prob",
        "pregame_away_win_prob",
        "pregame_spread",
        "pregame_total",
        "pregame_edge_home",
        "consensus_gap_home",
        "market_vs_pregame_home",
        "market_vs_consensus_home",
        "espn_vs_market_home",
        "score_gap_vs_pregame_spread",
        "points_vs_pregame_total",
        "flag_status_pre",
        "flag_status_live",
        "flag_status_post",
        "flag_has_game_state",
        "flag_has_time_state",
    ]

    matrix_cols = [
        "ticker",
        "event_ticker",
        "game_key",
        "game_date",
        "captured_at",
        "home_team",
        "away_team",
        "bet_side",
        "game_status",
        "status_state",
    ] + feature_cols + [
        "label_final_home_win",
        "label_yes_mid_move_5m",
        "label_market_home_implied_move_5m",
        "label_yes_up_5m",
        "label_home_up_5m",
        "label_beats_close_yes",
        "label_beats_close_home",
    ]

    matrix = df[matrix_cols].copy()
    return matrix, feature_cols


def summarize_targets(matrix: pd.DataFrame) -> dict:
    """
    Produce target coverage / class-balance diagnostics and a recommendation.
    """
    if matrix.empty:
        return {
            "rows": 0,
            "recommended_primary_target_current": None,
            "recommended_primary_target_architecture": "label_final_home_win",
        }

    targets = [
        "label_final_home_win",
        "label_home_up_5m",
        "label_yes_up_5m",
        "label_beats_close_home",
        "label_beats_close_yes",
        "label_market_home_implied_move_5m",
        "label_yes_mid_move_5m",
    ]

    summary = {
        "rows": int(len(matrix)),
        "rows_with_game_state": int(matrix["flag_has_game_state"].sum()),
        "rows_with_time_state": int(matrix["flag_has_time_state"].sum()),
        "rows_live": int(matrix["flag_status_live"].sum()),
        "rows_pre": int(matrix["flag_status_pre"].sum()),
        "targets": {},
    }

    for target in targets:
        if target not in matrix.columns:
            continue
        series = matrix[target].dropna()
        if series.empty:
            summary["targets"][target] = {"count": 0}
            continue

        target_summary = {"count": int(len(series))}
        if set(series.dropna().unique()).issubset({0, 1}):
            target_summary["positive_rate"] = round(float(series.astype(float).mean()), 4)
        else:
            target_summary["mean"] = round(float(series.mean()), 6)
            target_summary["std"] = round(float(series.std(ddof=0)), 6)
            target_summary["median"] = round(float(series.median()), 6)
        summary["targets"][target] = target_summary

    final_count = summary["targets"].get("label_final_home_win", {}).get("count", 0)
    move_count = summary["targets"].get("label_home_up_5m", {}).get("count", 0)
    close_count = summary["targets"].get("label_beats_close_home", {}).get("count", 0)
    live_rows = summary["rows_live"]

    if final_count >= 500 and live_rows >= 500:
        current = "label_final_home_win"
        reason = "Enough resolved live rows exist to train directly on game outcome."
    elif move_count >= 500:
        current = "label_home_up_5m"
        reason = "Short-horizon market-move labels are the only target with usable coverage today."
    elif close_count >= 500:
        current = "label_beats_close_home"
        reason = "Close-proxy labels have coverage, but they are secondary to forward market moves."
    else:
        current = None
        reason = "Not enough labeled live rows yet."

    summary["recommended_primary_target_current"] = current
    summary["recommended_primary_target_architecture"] = "label_final_home_win"
    summary["recommendation_reason"] = reason
    return summary


def save_training_matrix(
    matrix: pd.DataFrame,
    feature_cols: list[str],
    summary: dict,
    output_prefix: str = "in_game_training_matrix",
) -> dict:
    """
    Persist matrix and report artifacts.
    """
    matrix_path = LIVE_LABELS_DIR / f"{output_prefix}.csv"
    features_path = LIVE_LABELS_DIR / f"{output_prefix}_features.json"
    summary_path = LIVE_LABELS_DIR / f"{output_prefix}_summary.json"

    matrix.to_csv(matrix_path, index=False)
    features_path.write_text(json.dumps(feature_cols, indent=2))
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    latest_matrix = LIVE_LABELS_DIR / "latest_in_game_training_matrix.csv"
    latest_summary = LIVE_LABELS_DIR / "latest_in_game_training_summary.json"
    matrix.to_csv(latest_matrix, index=False)
    latest_summary.write_text(json.dumps(summary, indent=2, default=str))

    return {
        "matrix_path": str(matrix_path),
        "features_path": str(features_path),
        "summary_path": str(summary_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build the first in-game training matrix from labeled live feature rows.",
    )
    parser.add_argument("--horizon", type=int, default=5, help="Forward label horizon in minutes.")
    parser.add_argument("--max-lag", type=int, default=None, help="Maximum allowed delay past the horizon in minutes.")
    parser.add_argument("--all-sides", action="store_true", help="Keep both home and away contract rows.")
    parser.add_argument("--resolved-only", action="store_true", help="Keep only rows with resolved final outcomes.")
    args = parser.parse_args()

    labeled = build_labeled_training_set(
        horizon_minutes=args.horizon,
        max_lag_minutes=args.max_lag,
        only_resolved=args.resolved_only,
    )
    if labeled.empty:
        print("  No labeled rows available. Build live captures first.")
        return

    matrix, feature_cols = build_in_game_training_matrix(
        labeled,
        only_home_side=not args.all_sides,
    )
    summary = summarize_targets(matrix)
    paths = save_training_matrix(matrix, feature_cols, summary)

    print(f"  Matrix rows:                    {summary['rows']}")
    print(f"  Rows with game state:           {summary['rows_with_game_state']}")
    print(f"  Rows with time state:           {summary['rows_with_time_state']}")
    print(f"  Live rows:                      {summary['rows_live']}")
    print(f"  Current primary target:         {summary['recommended_primary_target_current']}")
    print(f"  Architectural primary target:   {summary['recommended_primary_target_architecture']}")
    print(f"  Reason:                         {summary['recommendation_reason']}")
    print(f"  Matrix saved:                   {paths['matrix_path']}")
    print(f"  Summary saved:                  {paths['summary_path']}")


if __name__ == "__main__":
    main()

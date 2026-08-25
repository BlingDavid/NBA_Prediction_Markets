"""
Build supervised labels for the real-time NBA feature store.

Primary targets:
  - final home win
  - 5-minute-ahead Kalshi move
  - beat-closing-price proxy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import LIVE_LABELS_DIR
from realtime_feature_store import (
    load_feature_history,
    load_game_history,
    load_market_history,
)


OPEN_STATUSES = {"open", "active"}


def _as_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _nullable_binary(mask: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=mask.index, dtype="Int64")
    valid = mask.notna()
    result.loc[valid] = mask.loc[valid].astype(int)
    return result


def resolve_final_game_outcomes(game_history: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve final scoreboard outcomes per game from live game snapshots.
    """
    if game_history.empty:
        return pd.DataFrame(columns=[
            "game_key",
            "label_final_home_win",
            "label_final_margin_home",
            "label_final_total_points",
            "label_final_captured_at",
        ])

    games = game_history.copy()
    games["captured_at"] = _as_timestamp(games["captured_at"])
    games["is_final"] = games["is_final"].fillna(False).astype(bool)

    final_games = games[games["is_final"]].copy()
    if final_games.empty:
        return pd.DataFrame(columns=[
            "game_key",
            "label_final_home_win",
            "label_final_margin_home",
            "label_final_total_points",
            "label_final_captured_at",
        ])

    final_games = (
        final_games.sort_values(["game_key", "captured_at"])
        .groupby("game_key", as_index=False)
        .tail(1)
        .copy()
    )

    final_games["label_final_home_win"] = (
        final_games["home_score"] > final_games["away_score"]
    ).astype("Int64")
    final_games["label_final_margin_home"] = final_games["home_score"] - final_games["away_score"]
    final_games["label_final_total_points"] = final_games["home_score"] + final_games["away_score"]
    final_games["label_final_captured_at"] = final_games["captured_at"]

    return final_games[
        [
            "game_key",
            "label_final_home_win",
            "label_final_margin_home",
            "label_final_total_points",
            "label_final_captured_at",
        ]
    ]


def resolve_halftime_leaders(game_history: pd.DataFrame) -> pd.DataFrame:
    """Per game, who leads at the end of Q2 (last observed period==2 tick).

    Returns game_key + label_halftime_home_lead (1 home / 0 away / NA on a
    halftime tie), label_halftime_margin_home, label_halftime_is_tie. Games
    with no period==2 capture are dropped (cannot be labeled).
    """
    cols = [
        "game_key",
        "label_halftime_home_lead",
        "label_halftime_margin_home",
        "label_halftime_is_tie",
    ]
    if game_history.empty:
        return pd.DataFrame(columns=cols)

    hist = game_history.copy()
    hist["captured_at"] = _as_timestamp(hist["captured_at"])
    hist["period"] = pd.to_numeric(hist["period"], errors="coerce")
    q2 = hist[hist["period"] == 2]
    if q2.empty:
        return pd.DataFrame(columns=cols)

    last_q2 = (
        q2.sort_values(["game_key", "captured_at"])
        .groupby("game_key", as_index=False)
        .last()
    )
    margin = last_q2["home_score"] - last_q2["away_score"]
    last_q2["label_halftime_margin_home"] = margin
    last_q2["label_halftime_is_tie"] = (margin == 0).astype(int)
    last_q2["label_halftime_home_lead"] = _nullable_binary(margin > 0)
    last_q2.loc[margin == 0, "label_halftime_home_lead"] = pd.NA
    return last_q2[cols]


def resolve_close_proxies(market_history: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve a close proxy per ticker from the last observed tradable market snapshot.
    """
    if market_history.empty:
        return pd.DataFrame(columns=[
            "ticker",
            "label_close_proxy_status",
            "label_close_proxy_captured_at",
            "label_close_yes_mid",
            "label_close_yes_ask",
            "label_close_market_home_implied",
            "label_close_last_price",
        ])

    markets = market_history.copy()
    markets["captured_at"] = _as_timestamp(markets["captured_at"])
    markets = markets.sort_values(["ticker", "captured_at"])

    open_rows = markets[markets["status"].astype(str).str.lower().isin(OPEN_STATUSES)].copy()
    close_proxy = (
        open_rows.groupby("ticker", as_index=False).tail(1).copy()
        if not open_rows.empty
        else pd.DataFrame()
    )

    unresolved = set(markets["ticker"]) - set(close_proxy["ticker"]) if not close_proxy.empty else set(markets["ticker"])
    if unresolved:
        fallback = markets[markets["ticker"].isin(unresolved)].groupby("ticker", as_index=False).tail(1)
        close_proxy = pd.concat([close_proxy, fallback], ignore_index=True) if not close_proxy.empty else fallback

    close_proxy = close_proxy.drop_duplicates("ticker", keep="last")
    close_proxy["label_close_proxy_status"] = close_proxy["status"]
    close_proxy["label_close_proxy_captured_at"] = close_proxy["captured_at"]
    close_proxy["label_close_yes_mid"] = close_proxy["yes_mid"]
    close_proxy["label_close_yes_ask"] = close_proxy["yes_ask"]
    close_proxy["label_close_market_home_implied"] = close_proxy["market_home_implied"]
    close_proxy["label_close_last_price"] = close_proxy["last_price"]

    return close_proxy[
        [
            "ticker",
            "label_close_proxy_status",
            "label_close_proxy_captured_at",
            "label_close_yes_mid",
            "label_close_yes_ask",
            "label_close_market_home_implied",
            "label_close_last_price",
        ]
    ]


def attach_forward_market_labels(
    feature_history: pd.DataFrame,
    horizon_minutes: int = 5,
    max_lag_minutes: int | None = None,
) -> pd.DataFrame:
    """
    Attach forward Kalshi move labels using the first snapshot observed at or after
    the requested horizon within each ticker.
    """
    if feature_history.empty:
        return feature_history.copy()

    max_lag_minutes = max_lag_minutes if max_lag_minutes is not None else max(horizon_minutes * 2, horizon_minutes + 2)
    suffix = f"{horizon_minutes}m"

    df = feature_history.copy()
    df["captured_at"] = _as_timestamp(df["captured_at"])
    df = df.sort_values("captured_at").reset_index(drop=True)
    df["target_time"] = df["captured_at"] + pd.Timedelta(minutes=horizon_minutes)

    label_future_col = f"label_future_captured_at_{suffix}"
    label_yes_mid = f"label_yes_mid_{suffix}"
    label_yes_ask = f"label_yes_ask_{suffix}"
    label_market = f"label_market_home_implied_{suffix}"
    label_last = f"label_last_price_{suffix}"
    label_volume = f"label_volume_{suffix}"
    label_delay = f"label_forward_delay_min_{suffix}"

    right = df[[
        "ticker", "captured_at", "yes_mid", "yes_ask",
        "market_home_implied", "last_price", "volume",
    ]].rename(columns={
        "captured_at": label_future_col,
        "yes_mid": label_yes_mid,
        "yes_ask": label_yes_ask,
        "market_home_implied": label_market,
        "last_price": label_last,
        "volume": label_volume,
    })

    df = pd.merge_asof(
        df,
        right,
        left_on="target_time",
        right_on=label_future_col,
        by="ticker",
        direction="forward",
        tolerance=pd.Timedelta(minutes=max_lag_minutes),
    )

    df[label_delay] = (df[label_future_col] - df["target_time"]).dt.total_seconds() / 60.0

    for col in [label_yes_mid, label_yes_ask, label_market, label_last, label_volume, label_delay]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = (
        df[f"label_future_captured_at_{suffix}"].notna()
        & df["captured_at"].notna()
        & (df[f"label_future_captured_at_{suffix}"] > df["captured_at"])
        & (df[f"label_forward_delay_min_{suffix}"] <= max_lag_minutes)
    )

    df[f"label_yes_mid_move_{suffix}"] = df[f"label_yes_mid_{suffix}"] - df["yes_mid"]
    df[f"label_yes_ask_move_{suffix}"] = df[f"label_yes_ask_{suffix}"] - df["yes_ask"]
    df[f"label_market_home_implied_move_{suffix}"] = (
        df[f"label_market_home_implied_{suffix}"] - df["market_home_implied"]
    )
    df[f"label_last_price_move_{suffix}"] = df[f"label_last_price_{suffix}"] - df["last_price"]
    df[f"label_volume_change_{suffix}"] = df[f"label_volume_{suffix}"] - df["volume"]

    eps = 0.001
    home_up = df[f"label_market_home_implied_move_{suffix}"] > eps
    yes_up = df[f"label_yes_mid_move_{suffix}"] > eps
    df[f"label_home_up_{suffix}"] = _nullable_binary(home_up.where(valid))
    df[f"label_yes_up_{suffix}"] = _nullable_binary(yes_up.where(valid))

    return df.drop(columns=["target_time"])


def attach_close_labels(
    feature_history: pd.DataFrame,
    close_proxy: pd.DataFrame,
    price_epsilon: float = 0.001,
) -> pd.DataFrame:
    """
    Attach closing-price-proxy labels for each ticker.
    """
    if feature_history.empty:
        return feature_history.copy()

    df = feature_history.copy()
    df["captured_at"] = _as_timestamp(df["captured_at"])

    if close_proxy.empty:
        for col in [
            "label_close_proxy_status",
            "label_close_proxy_captured_at",
            "label_close_yes_mid",
            "label_close_yes_ask",
            "label_close_market_home_implied",
            "label_close_last_price",
            "label_minutes_to_close_proxy",
            "label_yes_mid_to_close",
            "label_market_home_implied_to_close",
            "label_beats_close_yes",
            "label_beats_close_home",
        ]:
            df[col] = pd.NA
        return df

    close_df = close_proxy.copy()
    close_df["label_close_proxy_captured_at"] = _as_timestamp(close_df["label_close_proxy_captured_at"])
    df = df.merge(close_df, on="ticker", how="left")

    time_to_close = (
        df["label_close_proxy_captured_at"] - df["captured_at"]
    ).dt.total_seconds() / 60.0
    df["label_minutes_to_close_proxy"] = time_to_close

    valid = (
        df["label_close_proxy_captured_at"].notna()
        & df["captured_at"].notna()
        & (df["label_close_proxy_captured_at"] > df["captured_at"])
    )

    df["label_yes_mid_to_close"] = df["label_close_yes_mid"] - df["yes_mid"]
    df["label_market_home_implied_to_close"] = (
        df["label_close_market_home_implied"] - df["market_home_implied"]
    )

    beats_close_yes = (df["label_yes_mid_to_close"] > price_epsilon).where(valid)
    beats_close_home = (df["label_market_home_implied_to_close"] > price_epsilon).where(valid)
    df["label_beats_close_yes"] = _nullable_binary(beats_close_yes)
    df["label_beats_close_home"] = _nullable_binary(beats_close_home)

    invalid_cols = [
        "label_minutes_to_close_proxy",
        "label_yes_mid_to_close",
        "label_market_home_implied_to_close",
        "label_beats_close_yes",
        "label_beats_close_home",
    ]
    for col in invalid_cols:
        df.loc[~valid, col] = pd.NA

    return df


def build_labeled_training_set(
    feature_history: pd.DataFrame | None = None,
    game_history: pd.DataFrame | None = None,
    market_history: pd.DataFrame | None = None,
    horizon_minutes: int = 5,
    max_lag_minutes: int | None = None,
    only_resolved: bool = False,
) -> pd.DataFrame:
    """
    Build a training-ready labeled dataset from captured live history.
    """
    features = load_feature_history() if feature_history is None else feature_history.copy()
    games = load_game_history() if game_history is None else game_history.copy()
    markets = load_market_history() if market_history is None else market_history.copy()

    if features.empty:
        return pd.DataFrame()

    features["captured_at"] = _as_timestamp(features["captured_at"])
    features = features.sort_values(["ticker", "captured_at"]).reset_index(drop=True)

    final_outcomes = resolve_final_game_outcomes(games)
    close_proxy = resolve_close_proxies(markets)

    labeled = features.merge(final_outcomes, on="game_key", how="left")
    labeled = attach_forward_market_labels(
        labeled,
        horizon_minutes=horizon_minutes,
        max_lag_minutes=max_lag_minutes,
    )
    labeled = attach_close_labels(labeled, close_proxy)

    if only_resolved:
        labeled = labeled[labeled["label_final_home_win"].notna()].reset_index(drop=True)

    return labeled


def save_labeled_training_set(
    labeled: pd.DataFrame,
    horizon_minutes: int = 5,
    output_path: Path | None = None,
) -> Path:
    """
    Persist the labeled training set to disk.
    """
    if output_path is None:
        output_path = LIVE_LABELS_DIR / f"labeled_features_h{horizon_minutes}m.csv"

    labeled.to_csv(output_path, index=False)
    latest_path = LIVE_LABELS_DIR / "latest_labeled_features.csv"
    labeled.to_csv(latest_path, index=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Build final-outcome and forward-market labels for live NBA feature rows.",
    )
    parser.add_argument("--horizon", type=int, default=5, help="Forward market horizon in minutes.")
    parser.add_argument("--max-lag", type=int, default=None, help="Maximum allowed delay past the horizon in minutes.")
    parser.add_argument("--resolved-only", action="store_true", help="Keep only rows with resolved final-game outcomes.")
    parser.add_argument("--output", type=str, default=None, help="Optional output CSV path.")
    args = parser.parse_args()

    labeled = build_labeled_training_set(
        horizon_minutes=args.horizon,
        max_lag_minutes=args.max_lag,
        only_resolved=args.resolved_only,
    )

    if labeled.empty:
        print("  No live feature history available to label.")
        return

    output_path = save_labeled_training_set(
        labeled,
        horizon_minutes=args.horizon,
        output_path=Path(args.output) if args.output else None,
    )

    resolved_final = int(labeled["label_final_home_win"].notna().sum()) if "label_final_home_win" in labeled else 0
    resolved_forward = int(labeled[f"label_yes_mid_move_{args.horizon}m"].notna().sum())
    resolved_close = int(labeled["label_beats_close_yes"].notna().sum())

    print(f"  Rows labeled:                {len(labeled)}")
    print(f"  Final outcome labels:       {resolved_final}")
    print(f"  {args.horizon}-minute labels:          {resolved_forward}")
    print(f"  Beat-close labels:          {resolved_close}")
    print(f"  Saved labeled set:          {output_path}")


if __name__ == "__main__":
    main()

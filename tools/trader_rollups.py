"""Recompute rollup tables from closed_trades.csv. Idempotent."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_closed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["entry_captured_at"] = pd.to_datetime(df["entry_captured_at"], utc=True, errors="coerce")
    df["exit_captured_at"] = pd.to_datetime(df["exit_captured_at"], utc=True, errors="coerce")
    df["entry_date"] = df["entry_captured_at"].dt.date.astype(str)
    return df


def _per_p_decile(df: pd.DataFrame) -> pd.DataFrame:
    decile = (df["p_calibrated"].clip(0, 0.999) * 10).astype(int)
    out = df.groupby(decile).agg(
        n=("trade_id", "count"),
        win_rate=("realized_label_home_up_5m", "mean"),
        pnl_realistic_mean=("pnl_realistic", "mean"),
        pnl_realistic_sum=("pnl_realistic", "sum"),
        pnl_mid_mean=("pnl_mid", "mean"),
    ).reset_index().rename(columns={"p_calibrated": "p_decile"})
    out.columns = ["p_decile", "n", "win_rate", "pnl_realistic_mean", "pnl_realistic_sum", "pnl_mid_mean"]
    return out


def _per_minute_of_game(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate game-minute bucket from entry_captured_at — needs the
    feature row's seconds_elapsed for true accuracy. Lacking that here, we
    bucket by hold_seconds and entry hour-of-day as a coarse proxy."""
    hour = df["entry_captured_at"].dt.hour.fillna(-1).astype(int)
    out = df.groupby(hour).agg(
        n=("trade_id", "count"),
        pnl_realistic_mean=("pnl_realistic", "mean"),
        pnl_realistic_sum=("pnl_realistic", "sum"),
    ).reset_index().rename(columns={"entry_captured_at": "entry_hour_utc"})
    out.columns = ["entry_hour_utc", "n", "pnl_realistic_mean", "pnl_realistic_sum"]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_trades"))
    args = parser.parse_args(argv)
    closed_path = args.output_dir / "closed_trades.csv"
    if not closed_path.exists():
        print("no closed_trades.csv to roll up")
        return 0
    df = _load_closed(closed_path)
    rdir = args.output_dir / "rollups"
    rdir.mkdir(parents=True, exist_ok=True)
    # daily_<date>.csv
    for date, sub in df.groupby("entry_date"):
        sub_summary = pd.DataFrame([{
            "date": date, "n": len(sub),
            "pnl_realistic_sum": float(sub["pnl_realistic"].sum()),
            "pnl_mid_sum": float(sub["pnl_mid"].sum()),
            "win_rate": float(sub["realized_label_home_up_5m"].mean()),
        }])
        sub_summary.to_csv(rdir / f"daily_{date}.csv", index=False)
    # per_game.csv
    per_game = df.groupby("game_key").agg(
        n=("trade_id", "count"),
        pnl_realistic_sum=("pnl_realistic", "sum"),
        win_rate=("realized_label_home_up_5m", "mean"),
    ).reset_index()
    per_game.to_csv(rdir / "per_game.csv", index=False)
    # per_p_decile.csv
    _per_p_decile(df).to_csv(rdir / "per_p_decile.csv", index=False)
    # per_minute_of_game.csv
    _per_minute_of_game(df).to_csv(rdir / "per_minute_of_game.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

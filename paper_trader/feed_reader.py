"""Reader for data/live/features/latest_features.csv.

Returns only rows newer than last_seen[ticker]. If multiple rows for the same
ticker are newer (which shouldn't happen with realtime_feature_store, but
defensively), keeps only the latest captured_at.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_unprocessed(path: Path, last_seen: dict[str, str]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["captured_at"] = df["captured_at"].astype(str)

    def _is_new(row) -> bool:
        prev = last_seen.get(row["ticker"])
        if prev is None:
            return True
        return row["captured_at"] > prev

    df = df[df.apply(_is_new, axis=1)]
    if df.empty:
        return df
    df = df.sort_values("captured_at").drop_duplicates(subset=["ticker"], keep="last")
    return df.reset_index(drop=True)

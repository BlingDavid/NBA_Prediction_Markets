"""feed_reader: read latest_features.csv, dedup against last-processed map."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from paper_trader.feed_reader import read_unprocessed


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_returns_all_rows_when_map_empty(tmp_path: Path):
    p = tmp_path / "feed.csv"
    _write_csv(p, [
        {"ticker": "A", "captured_at": "2026-05-01T00:00:00Z"},
        {"ticker": "B", "captured_at": "2026-05-01T00:00:00Z"},
    ])
    out = read_unprocessed(p, last_seen={})
    assert len(out) == 2


def test_filters_already_processed(tmp_path: Path):
    p = tmp_path / "feed.csv"
    _write_csv(p, [
        {"ticker": "A", "captured_at": "2026-05-01T00:00:00Z"},
        {"ticker": "B", "captured_at": "2026-05-01T00:00:00Z"},
        {"ticker": "A", "captured_at": "2026-05-01T00:00:15Z"},
    ])
    last_seen = {"A": "2026-05-01T00:00:00Z"}
    out = read_unprocessed(p, last_seen=last_seen)
    # A's first row already seen; A's second and B's first remain.
    tickers = sorted(out["ticker"].tolist())
    assert tickers == ["A", "B"]
    captured = out.set_index("ticker")["captured_at"].to_dict()
    assert captured["A"] == "2026-05-01T00:00:15Z"


def test_returns_empty_when_file_missing(tmp_path: Path):
    out = read_unprocessed(tmp_path / "nope.csv", last_seen={})
    assert out.empty


def test_keeps_latest_per_ticker_only(tmp_path: Path):
    """If multiple rows for the same ticker > last_seen exist, return only the
    latest. (latest_features.csv is supposed to have one row per ticker, but
    defend against duplicates.)"""
    p = tmp_path / "feed.csv"
    _write_csv(p, [
        {"ticker": "A", "captured_at": "2026-05-01T00:00:15Z"},
        {"ticker": "A", "captured_at": "2026-05-01T00:00:30Z"},
    ])
    out = read_unprocessed(p, last_seen={})
    assert len(out) == 1
    assert out.iloc[0]["captured_at"] == "2026-05-01T00:00:30Z"

"""trader_backtest: replay one historical day's live_features_<date>.csv."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def historical_day() -> Path:
    p = Path("data/live/features/live_features_2026-04-30.csv")
    if not p.exists():
        pytest.skip("no historical feature day available")
    return p


@pytest.fixture
def model_path() -> Path:
    p = Path("models/live_home_up_5m_bootstrap.pkl")
    pooled = Path("outputs/paper_trades/pooled_means.json")
    if not p.exists() or not pooled.exists():
        pytest.skip("model artifact or pooled means not present")
    return p


def test_backtest_writes_outputs_to_dated_dir(tmp_path: Path, historical_day: Path, model_path: Path):
    from tools.trader_backtest import main
    out = tmp_path / "bt"
    rc = main([
        "--features-path", str(historical_day),
        "--output-dir", str(out),
        "--max-rows", "200",  # cap for speed
    ])
    assert rc == 0
    # closed_trades.csv may be empty if no signals fired in the first 200 rows,
    # but the log file must exist with tick events.
    log = (out / "trader_log.jsonl").read_text().strip().splitlines()
    assert any("tick" in line for line in log)

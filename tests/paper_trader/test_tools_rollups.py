"""trader_rollups: idempotent batch — daily, per_game, per_p_decile, per_minute."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _seed(tmp_path: Path) -> Path:
    out = tmp_path / "outputs"
    out.mkdir()
    pd.DataFrame([
        {"trade_id": "T1", "game_key": "G1", "p_calibrated": 0.10,
         "entry_captured_at": "2026-05-01T19:00:00Z", "exit_captured_at": "2026-05-01T19:05:00Z",
         "pnl_realistic": 1.5, "pnl_mid": 2.0,
         "realized_label_home_up_5m": 1, "fees": 0.10, "contracts": 5,
         "p_raw": 0.11, "trigger_mode": "A"},
        {"trade_id": "T2", "game_key": "G1", "p_calibrated": 0.40,
         "entry_captured_at": "2026-05-01T19:30:00Z", "exit_captured_at": "2026-05-01T19:35:00Z",
         "pnl_realistic": -2.0, "pnl_mid": -1.0,
         "realized_label_home_up_5m": 0, "fees": 0.0, "contracts": 5,
         "p_raw": 0.39, "trigger_mode": "A"},
        {"trade_id": "T3", "game_key": "G2", "p_calibrated": 0.40,
         "entry_captured_at": "2026-05-02T19:30:00Z", "exit_captured_at": "2026-05-02T19:35:00Z",
         "pnl_realistic": 3.0, "pnl_mid": 4.0,
         "realized_label_home_up_5m": 1, "fees": 0.20, "contracts": 5,
         "p_raw": 0.41, "trigger_mode": "A"},
    ]).to_csv(out / "closed_trades.csv", index=False)
    return out


def test_rollups_creates_all_four_tables(tmp_path: Path):
    from tools.trader_rollups import main
    out = _seed(tmp_path)
    rc = main(["--output-dir", str(out)])
    assert rc == 0
    rdir = out / "rollups"
    assert (rdir / "per_game.csv").exists()
    assert (rdir / "per_p_decile.csv").exists()
    assert (rdir / "per_minute_of_game.csv").exists()
    daily_files = list(rdir.glob("daily_*.csv"))
    assert len(daily_files) == 2  # 2026-05-01, 2026-05-02


def test_rollups_idempotent(tmp_path: Path):
    """Running twice produces the same file contents."""
    from tools.trader_rollups import main
    out = _seed(tmp_path)
    main(["--output-dir", str(out)])
    first = (out / "rollups" / "per_game.csv").read_bytes()
    main(["--output-dir", str(out)])
    second = (out / "rollups" / "per_game.csv").read_bytes()
    assert first == second

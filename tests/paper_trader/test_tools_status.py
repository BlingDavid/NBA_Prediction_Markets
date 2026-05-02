"""trader_status: prints a readable summary from state + closed trades."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _seed_outputs(tmp_path: Path) -> Path:
    out = tmp_path / "outputs"
    out.mkdir()
    state = {
        "mode": "A", "live_or_paper": "paper",
        "bankroll_initial": 1000.0,
        "cumulative_pnl_realistic": 47.31, "cumulative_pnl_mid": 62.18,
        "trades_closed": 3, "max_drawdown": 31.40,
        "drawdown_breached": False, "open_positions": [],
        "last_processed_captured_at_per_ticker": {},
        "last_tick_at": "2026-05-01T19:42:17+00:00",
    }
    (out / "trader_state.json").write_text(json.dumps(state, indent=2))
    pd.DataFrame([
        {"trade_id": "T1", "pnl_realistic": 12.0, "pnl_mid": 14.0,
         "realized_label_home_up_5m": 1, "exit_captured_at": "2026-05-01T19:30:00Z"},
        {"trade_id": "T2", "pnl_realistic": -5.0, "pnl_mid": -2.0,
         "realized_label_home_up_5m": 0, "exit_captured_at": "2026-05-01T19:35:00Z"},
        {"trade_id": "T3", "pnl_realistic": 40.31, "pnl_mid": 50.18,
         "realized_label_home_up_5m": 1, "exit_captured_at": "2026-05-01T19:40:00Z"},
    ]).to_csv(out / "closed_trades.csv", index=False)
    (out / "trader_log.jsonl").write_text("")
    return out


def test_trader_status_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    from tools.trader_status import main
    out = _seed_outputs(tmp_path)
    rc = main(["--output-dir", str(out)])
    text = capsys.readouterr().out
    assert rc == 0
    assert "trades_closed" in text.lower() or "trades closed" in text.lower()
    assert "wins" in text.lower()
    assert "47.31" in text  # cumulative pnl realistic

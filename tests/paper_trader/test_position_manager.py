"""position_manager: open/close lifecycle, Kelly sizing, concurrency cap, exit."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from paper_trader.position_manager import (
    PositionManager,
    kelly_contracts,
    HOLD_SECONDS,
    MAX_OPEN,
)


def _ts(s: int) -> str:
    return (datetime(2026, 5, 1, 19, 0, tzinfo=timezone.utc) + timedelta(seconds=s)).isoformat()


def test_kelly_contracts_basic():
    # f = mu / sigma^2, capped at 5% of bankroll, floored at 1
    # mu = 0.02, sigma^2 = 0.001 → f = 20 (huge) → cap to 0.05*1000 = 50
    # contracts = floor(50 / yes_ask) = floor(50/0.40) = 125
    contracts = kelly_contracts(mu=0.02, var=0.001, bankroll=1000.0, yes_ask=0.40)
    assert contracts == 125

def test_kelly_contracts_floor_one():
    # mu tiny, var huge → f tiny → contracts floor to 1
    contracts = kelly_contracts(mu=0.0001, var=0.1, bankroll=1000.0, yes_ask=0.50)
    assert contracts == 1


def test_kelly_contracts_zero_var_returns_cap():
    # Degenerate: var = 0 → treat as full-cap allocation
    contracts = kelly_contracts(mu=0.01, var=0.0, bankroll=1000.0, yes_ask=0.50)
    assert contracts == int(0.05 * 1000 / 0.50)


def test_open_then_exit_at_5min(tmp_path: Path):
    pm = PositionManager(
        state_path=tmp_path / "trader_state.json",
        log_path=tmp_path / "trader_log.jsonl",
        closed_csv_path=tmp_path / "closed_trades.csv",
        open_csv_path=tmp_path / "open_positions.csv",
        bankroll_initial=1000.0,
        mode_sizing="A",
        live_or_paper="paper",
    )

    pm.open_position(
        ticker="X", game_key="G", home_team="H", away_team="A",
        captured_at=_ts(0), yes_ask=0.40, yes_mid=0.39, yes_bid=0.38,
        contracts=10, p_raw=0.20, p_calibrated=0.18,
        expected_pnl_per_contract=0.012,
    )
    assert len(pm.open_positions) == 1

    # Tick 60s in → no exit
    pm.tick_open_positions(now_captured_at=_ts(60), latest_bid_mid_per_ticker={"X": (0.41, 0.42)})
    assert len(pm.open_positions) == 1

    # Tick 5 min in → exit
    pm.tick_open_positions(now_captured_at=_ts(HOLD_SECONDS), latest_bid_mid_per_ticker={"X": (0.45, 0.46)})
    assert len(pm.open_positions) == 0
    assert pm.trades_closed == 1


def test_concurrency_cap_blocks_open(tmp_path: Path):
    pm = PositionManager(
        state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl",
        closed_csv_path=tmp_path / "c.csv", open_csv_path=tmp_path / "o.csv",
        bankroll_initial=1000.0, mode_sizing="A", live_or_paper="paper",
    )
    for i in range(MAX_OPEN):
        pm.open_position(
            ticker=f"T{i}", game_key=f"G{i}", home_team="H", away_team="A",
            captured_at=_ts(i), yes_ask=0.40, yes_mid=0.39, yes_bid=0.38,
            contracts=1, p_raw=0.2, p_calibrated=0.18,
            expected_pnl_per_contract=0.01,
        )
    # 21st should be refused
    accepted = pm.open_position(
        ticker="OVERFLOW", game_key="G", home_team="H", away_team="A",
        captured_at=_ts(99999), yes_ask=0.40, yes_mid=0.39, yes_bid=0.38,
        contracts=1, p_raw=0.2, p_calibrated=0.18,
        expected_pnl_per_contract=0.01,
    )
    assert accepted is False
    assert len(pm.open_positions) == MAX_OPEN


def test_pnl_mid_vs_realistic(tmp_path: Path):
    pm = PositionManager(
        state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl",
        closed_csv_path=tmp_path / "c.csv", open_csv_path=tmp_path / "o.csv",
        bankroll_initial=1000.0, mode_sizing="A", live_or_paper="paper",
    )
    pm.open_position(
        ticker="X", game_key="G", home_team="H", away_team="A",
        captured_at=_ts(0), yes_ask=0.40, yes_mid=0.39, yes_bid=0.38,
        contracts=10, p_raw=0.20, p_calibrated=0.18,
        expected_pnl_per_contract=0.01,
    )
    pm.tick_open_positions(now_captured_at=_ts(HOLD_SECONDS),
                           latest_bid_mid_per_ticker={"X": (0.43, 0.44)})

    # pnl_mid = (exit_mid - entry_mid) * contracts = (0.44 - 0.39) * 10 = 0.5
    # pnl_realistic = (exit_bid - entry_ask) * contracts = (0.43 - 0.40) * 10 = 0.3
    # winner: realized label = 1 (price went up), so winner fee applies on the realistic side.
    # That fee is reflected in trade.fees and subtracted from pnl_realistic.
    closed = pm.last_closed_trade
    assert closed["pnl_mid"] == pytest.approx(0.5)
    # Realistic before fee = 0.3; minus fees > 0 (it's a winner)
    assert closed["pnl_realistic"] < 0.3
    assert closed["fees"] > 0
    assert closed["realized_label_home_up_5m"] == 1


def test_restart_reload_keeps_open_positions(tmp_path: Path):
    pm1 = PositionManager(
        state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl",
        closed_csv_path=tmp_path / "c.csv", open_csv_path=tmp_path / "o.csv",
        bankroll_initial=1000.0, mode_sizing="A", live_or_paper="paper",
    )
    pm1.open_position(
        ticker="X", game_key="G", home_team="H", away_team="A",
        captured_at=_ts(0), yes_ask=0.40, yes_mid=0.39, yes_bid=0.38,
        contracts=5, p_raw=0.2, p_calibrated=0.18,
        expected_pnl_per_contract=0.01,
    )
    pm1.write_state()

    pm2 = PositionManager(
        state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl",
        closed_csv_path=tmp_path / "c.csv", open_csv_path=tmp_path / "o.csv",
        bankroll_initial=1000.0, mode_sizing="A", live_or_paper="paper",
    )
    pm2.load_state()
    assert len(pm2.open_positions) == 1
    assert pm2.open_positions[0]["ticker"] == "X"

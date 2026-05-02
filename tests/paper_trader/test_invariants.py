"""Replay + cumulative-PnL + concurrency-cap + trade-id-uniqueness invariants."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from run_paper_trader import process_tick, build_engine_context, replay_from_log
from paper_trader.position_manager import MAX_OPEN, PositionManager


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def model_path() -> Path:
    p = Path("models/live_home_up_5m_bootstrap.pkl")
    if not p.exists():
        pytest.skip("model artifact not present")
    return p


def test_cumulative_pnl_matches_sum_of_closed(tmp_path: Path):
    pm = PositionManager(
        state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl",
        closed_csv_path=tmp_path / "c.csv", open_csv_path=tmp_path / "o.csv",
        bankroll_initial=1000.0, mode_sizing="A", live_or_paper="paper",
    )
    # Open and close 5 positions back-to-back.
    base = pd.Timestamp("2026-05-01T19:00:00Z")
    for i in range(5):
        captured = (base + pd.Timedelta(seconds=i)).isoformat()
        pm.open_position(
            ticker=f"T{i}", game_key="G", home_team="H", away_team="A",
            captured_at=captured, yes_ask=0.40, yes_mid=0.39, yes_bid=0.38,
            contracts=10, p_raw=0.20, p_calibrated=0.18,
            expected_pnl_per_contract=0.012,
        )
        exit_at = (base + pd.Timedelta(seconds=i + 300)).isoformat()
        pm.tick_open_positions(now_captured_at=exit_at,
                               latest_bid_mid_per_ticker={f"T{i}": (0.43, 0.44)})
    df = pd.read_csv(tmp_path / "c.csv")
    assert pm.cumulative_pnl_realistic == pytest.approx(df["pnl_realistic"].sum(), abs=1e-6)


def test_concurrency_cap_holds_under_burst(tmp_path: Path):
    pm = PositionManager(
        state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl",
        closed_csv_path=tmp_path / "c.csv", open_csv_path=tmp_path / "o.csv",
        bankroll_initial=1000.0, mode_sizing="A", live_or_paper="paper",
    )
    base = pd.Timestamp("2026-05-01T19:00:00Z")
    for i in range(MAX_OPEN + 50):
        pm.open_position(
            ticker=f"T{i}", game_key="G", home_team="H", away_team="A",
            captured_at=(base + pd.Timedelta(seconds=i)).isoformat(),
            yes_ask=0.40, yes_mid=0.39, yes_bid=0.38,
            contracts=1, p_raw=0.2, p_calibrated=0.18,
            expected_pnl_per_contract=0.01,
        )
    assert len(pm.open_positions) == MAX_OPEN


def test_trade_id_uniqueness_under_synthetic_load(tmp_path: Path):
    pm = PositionManager(
        state_path=tmp_path / "s.json", log_path=tmp_path / "l.jsonl",
        closed_csv_path=tmp_path / "c.csv", open_csv_path=tmp_path / "o.csv",
        bankroll_initial=1_000_000.0, mode_sizing="A", live_or_paper="paper",
    )
    seen = set()
    base = pd.Timestamp("2026-05-01T19:00:00Z")
    n = 5000
    for i in range(n):
        captured = (base + pd.Timedelta(microseconds=i)).isoformat()
        pm.open_position(
            ticker=f"T{i % 100}", game_key="G", home_team="H", away_team="A",
            captured_at=captured, yes_ask=0.40, yes_mid=0.39, yes_bid=0.38,
            contracts=1, p_raw=0.2, p_calibrated=0.18,
            expected_pnl_per_contract=0.01,
        )
        if len(pm.open_positions) > MAX_OPEN - 1:
            # Force-close one to make room.
            pos = pm.open_positions.pop(0)
            pm._close_position(pos, captured, 0.42, 0.42, "synthetic")
        seen.add(pm.open_positions[-1]["trade_id"]) if pm.open_positions else None
    df = pd.read_csv(tmp_path / "c.csv")
    assert df["trade_id"].is_unique


def test_replay_reconstructs_closed_trades(tmp_path: Path):
    """Drive the loop synthetically, then replay the log and assert
    closed_trades.csv is byte-identical."""
    out = tmp_path / "outputs"
    pm = PositionManager(
        state_path=out / "trader_state.json",
        log_path=out / "trader_log.jsonl",
        closed_csv_path=out / "closed_trades.csv",
        open_csv_path=out / "open_positions.csv",
        bankroll_initial=1000.0, mode_sizing="A", live_or_paper="paper",
    )
    base = pd.Timestamp("2026-05-01T19:00:00Z")
    for i in range(10):
        captured = (base + pd.Timedelta(seconds=i)).isoformat()
        pm.open_position(
            ticker=f"T{i}", game_key="G", home_team="H", away_team="A",
            captured_at=captured, yes_ask=0.40, yes_mid=0.39, yes_bid=0.38,
            contracts=10, p_raw=0.20, p_calibrated=0.18,
            expected_pnl_per_contract=0.012,
        )
        exit_at = (base + pd.Timedelta(seconds=i + 300)).isoformat()
        pm.tick_open_positions(now_captured_at=exit_at,
                               latest_bid_mid_per_ticker={f"T{i}": (0.43, 0.44)})
    pm.write_state()

    original_digest = _digest(out / "closed_trades.csv")

    # Replay into a fresh dir.
    replay_dir = tmp_path / "replay"
    replay_from_log(out / "trader_log.jsonl", replay_dir, bankroll_initial=1000.0)
    replay_digest = _digest(replay_dir / "closed_trades.csv")
    assert replay_digest == original_digest

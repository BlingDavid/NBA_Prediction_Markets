"""End-to-end one-tick test of the paper_trader main loop, with the
position-manager + feed-reader + scorer wired up against a tiny synthetic
feed and a stubbed model bundle.

We run a single iteration of `process_tick` rather than the whole forever-loop.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_paper_trader import process_tick, build_engine_context, check_live_boot_gates


def _make_pooled(tmp_path: Path) -> Path:
    p = tmp_path / "pooled.json"
    p.write_text(json.dumps({
        "E_delta_given_rises": 0.05,
        "E_delta_given_doesnt": -0.005,
        "var_delta_pooled": 0.001,
        "rises_count": 1000,
        "doesnt_count": 9000,
    }))
    return p


def _make_feed(tmp_path: Path, captured_at: str, ticker: str = "X", price: float = 0.40) -> Path:
    p = tmp_path / "feed.csv"
    pd.DataFrame([{
        "captured_at": captured_at, "ticker": ticker, "event_ticker": ticker,
        "game_key": "G", "game_date": "2026-05-01",
        "home_team": "H", "away_team": "A", "bet_team": "H", "bet_side": "home",
        "game_status": "Q2", "status_state": "in", "is_live": True, "is_final": False,
        "period": 2, "display_clock": "5:00",
        "seconds_left_in_period": 300.0, "seconds_elapsed": 900.0,
        "home_score": 50.0, "away_score": 48.0, "score_margin_home": 2.0, "total_points": 98.0,
        "yes_bid": price - 0.01, "yes_ask": price + 0.01, "yes_mid": price,
        "no_bid": 0.99 - (price + 0.01), "no_ask": 0.99 - (price - 0.01),
        "last_price": price, "market_home_implied": price,
        "volume": 50000.0, "open_interest": 5000.0,
        "yes_depth_notional_3": 500.0, "yes_depth_notional_5": 800.0,
        "no_depth_notional_3": 480.0, "no_depth_notional_5": 770.0,
        "yes_weighted_price_3": price - 0.005, "no_weighted_price_3": 0.99 - price,
        "espn_home_implied": np.nan, "espn_away_implied": np.nan,
        "oddsapi_home_consensus": np.nan, "oddsapi_away_consensus": np.nan, "oddsapi_books": np.nan,
        "market_consensus_home": np.nan,
        "pregame_home_win_prob": 0.50, "pregame_away_win_prob": 0.50,
        "pregame_spread": 0.0, "pregame_total": 220.0, "pregame_data_date": "2026-04-30",
        "kalshi_vs_consensus_pp": 0.0, "model_vs_consensus_pp": 0.0, "kalshi_vs_model_pp": 0.0,
        "triangulation_tier": 1, "consensus_n_books": 0, "pregame_edge_home": 0.0,
        "consensus_gap_home": np.nan,
    }]).to_csv(p, index=False)
    return p


@pytest.fixture
def model_path() -> Path:
    p = Path("models/live_home_up_5m_bootstrap.pkl")
    if not p.exists():
        pytest.skip("model artifact not present")
    return p


def test_process_tick_emits_tick_event(tmp_path: Path, model_path: Path):
    feed = _make_feed(tmp_path, captured_at="2026-05-01T19:00:00Z", price=0.40)
    pooled = _make_pooled(tmp_path)
    ctx = build_engine_context(
        model_path=model_path, pooled_path=pooled, feed_path=feed,
        output_dir=tmp_path / "outputs", bankroll=1000.0,
        mode_sizing="A", live_or_paper="paper",
    )
    process_tick(ctx)
    log = (tmp_path / "outputs" / "trader_log.jsonl").read_text().strip().splitlines()
    types = [json.loads(line)["event_type"] for line in log]
    assert "tick" in types


def test_process_tick_closes_position_after_5min(tmp_path: Path, model_path: Path):
    pooled = _make_pooled(tmp_path)
    output_dir = tmp_path / "outputs"
    feed_file = tmp_path / "feed.csv"

    # Tick 1: open a position by faking a high-EV state. We do this by writing
    # a position directly into state, then process_tick at t+5min should close it.
    _make_feed(tmp_path, captured_at="2026-05-01T19:00:00Z", price=0.40, ticker="X")
    ctx = build_engine_context(
        model_path=model_path, pooled_path=pooled, feed_path=feed_file,
        output_dir=output_dir, bankroll=1000.0,
        mode_sizing="A", live_or_paper="paper",
    )
    ctx.position_manager.open_position(
        ticker="X", game_key="G", home_team="H", away_team="A",
        captured_at="2026-05-01T19:00:00Z",
        yes_ask=0.41, yes_mid=0.40, yes_bid=0.39,
        contracts=10, p_raw=0.20, p_calibrated=0.18,
        expected_pnl_per_contract=0.012,
    )
    # Mark the first feed row as processed so we don't re-open it
    ctx.position_manager.last_processed_captured_at_per_ticker["X"] = "2026-05-01T19:00:00Z"
    ctx.position_manager.write_state()

    # Tick 2: 5 minutes later, with a richer price. Update feed to have both rows.
    df = pd.DataFrame([{
        "captured_at": "2026-05-01T19:00:00Z", "ticker": "X", "event_ticker": "X",
        "game_key": "G", "game_date": "2026-05-01",
        "home_team": "H", "away_team": "A", "bet_team": "H", "bet_side": "home",
        "game_status": "Q2", "status_state": "in", "is_live": True, "is_final": False,
        "period": 2, "display_clock": "5:00",
        "seconds_left_in_period": 300.0, "seconds_elapsed": 900.0,
        "home_score": 50.0, "away_score": 48.0, "score_margin_home": 2.0, "total_points": 98.0,
        "yes_bid": 0.39, "yes_ask": 0.41, "yes_mid": 0.40,
        "no_bid": 0.58, "no_ask": 0.60, "last_price": 0.40, "market_home_implied": 0.40,
        "volume": 50000.0, "open_interest": 5000.0,
        "yes_depth_notional_3": 500.0, "yes_depth_notional_5": 800.0,
        "no_depth_notional_3": 480.0, "no_depth_notional_5": 770.0,
        "yes_weighted_price_3": 0.395, "no_weighted_price_3": 0.595,
        "espn_home_implied": np.nan, "espn_away_implied": np.nan,
        "oddsapi_home_consensus": np.nan, "oddsapi_away_consensus": np.nan, "oddsapi_books": np.nan,
        "market_consensus_home": np.nan,
        "pregame_home_win_prob": 0.50, "pregame_away_win_prob": 0.50,
        "pregame_spread": 0.0, "pregame_total": 220.0, "pregame_data_date": "2026-04-30",
        "kalshi_vs_consensus_pp": 0.0, "model_vs_consensus_pp": 0.0, "kalshi_vs_model_pp": 0.0,
        "triangulation_tier": 1, "consensus_n_books": 0, "pregame_edge_home": 0.0,
        "consensus_gap_home": np.nan,
    }, {
        "captured_at": "2026-05-01T19:05:00Z", "ticker": "X", "event_ticker": "X",
        "game_key": "G", "game_date": "2026-05-01",
        "home_team": "H", "away_team": "A", "bet_team": "H", "bet_side": "home",
        "game_status": "Q2", "status_state": "in", "is_live": True, "is_final": False,
        "period": 2, "display_clock": "0:00",
        "seconds_left_in_period": 0.0, "seconds_elapsed": 1200.0,
        "home_score": 55.0, "away_score": 50.0, "score_margin_home": 5.0, "total_points": 105.0,
        "yes_bid": 0.44, "yes_ask": 0.46, "yes_mid": 0.45,
        "no_bid": 0.53, "no_ask": 0.55, "last_price": 0.45, "market_home_implied": 0.45,
        "volume": 60000.0, "open_interest": 6000.0,
        "yes_depth_notional_3": 600.0, "yes_depth_notional_5": 900.0,
        "no_depth_notional_3": 580.0, "no_depth_notional_5": 870.0,
        "yes_weighted_price_3": 0.445, "no_weighted_price_3": 0.545,
        "espn_home_implied": np.nan, "espn_away_implied": np.nan,
        "oddsapi_home_consensus": np.nan, "oddsapi_away_consensus": np.nan, "oddsapi_books": np.nan,
        "market_consensus_home": np.nan,
        "pregame_home_win_prob": 0.50, "pregame_away_win_prob": 0.50,
        "pregame_spread": 0.0, "pregame_total": 220.0, "pregame_data_date": "2026-04-30",
        "kalshi_vs_consensus_pp": 0.0, "model_vs_consensus_pp": 0.0, "kalshi_vs_model_pp": 0.0,
        "triangulation_tier": 1, "consensus_n_books": 0, "pregame_edge_home": 0.0,
        "consensus_gap_home": np.nan,
    }]).to_csv(feed_file, index=False)

    ctx2 = build_engine_context(
        model_path=model_path, pooled_path=pooled, feed_path=feed_file,
        output_dir=output_dir, bankroll=1000.0,
        mode_sizing="A", live_or_paper="paper",
    )
    ctx2.position_manager.load_state()
    process_tick(ctx2)

    closed = pd.read_csv(output_dir / "closed_trades.csv")
    assert len(closed) == 1
    assert closed.iloc[0]["exit_basis"] == "5min_timer"


def test_live_boot_refuses_when_gates_fail(tmp_path: Path, model_path: Path):
    """In live mode, with no closed trades on disk, the engine must refuse
    to start (1000-trade gate, sharpe gate, drawdown gate all fail)."""
    pooled = _make_pooled(tmp_path)
    feed = _make_feed(tmp_path, "2026-05-01T19:00:00Z")

    ok, reasons = check_live_boot_gates(
        output_dir=tmp_path / "outputs",
        bankroll_initial=1000.0,
    )
    assert not ok
    assert any("trades_closed" in r for r in reasons)

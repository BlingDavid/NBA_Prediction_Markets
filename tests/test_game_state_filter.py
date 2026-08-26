"""
Tests for live_bootstrap_model.filter_game_state_features.

Verifies that market/microstructure features are removed and true
game-state features (score, time, pregame priors) are kept.
"""

from __future__ import annotations

import pytest

from live_bootstrap_model import filter_game_state_features


# ---------------------------------------------------------------------------
# Features that must be DROPPED (contain market/microstructure substrings)
# ---------------------------------------------------------------------------

DROPPED_CASES = [
    "volume",
    "market_volume_1m",
    "yes_volume_5m",
    "open_interest",
    "open_interest_change",
    "depth_bid_1",
    "orderbook_depth",
    "pressure_buy",
    "weighted_price_yes",
    "bid_size",
    "best_bid",
    "ask_size",
    "yes_ask",
    "no_ask",
    "mid_price",
    "yes_mid",
    "implied_prob",
    "market_implied_move",
    "last_price",
    "last_price_change",
    "market_maker_spread",
    "espn_odds",
    "espn_pregame_line",
    "consensus_prob",
    "consensus_spread",
    "oddsapi_home_prob",
    "oddsapi_market_prob",
    "yes_bid",
    "no_bid",
    "no_price",
]


@pytest.mark.parametrize("col", DROPPED_CASES)
def test_market_features_are_removed(col):
    result = filter_game_state_features([col])
    assert result == [], f"Expected '{col}' to be dropped but it was kept"


# ---------------------------------------------------------------------------
# Features that must be KEPT (true game-state / pregame-prior signals)
# ---------------------------------------------------------------------------

KEPT_CASES = [
    "score_margin_home",
    "home_score",
    "away_score",
    "total_points",
    "period",
    "seconds_remaining",
    "seconds_elapsed",
    "score_gap_vs_pregame_spread",
    "pregame_home_win_prob",
    "pregame_spread",
    "pregame_total",
    "score_run_home_3min",
    "score_run_away_3min",
    "pace",
    "velocity_home",
    "flag_has_game_state",
    "flag_status_live",
    "momentum_flag",
]


@pytest.mark.parametrize("col", KEPT_CASES)
def test_game_state_features_are_kept(col):
    result = filter_game_state_features([col])
    assert result == [col], f"Expected '{col}' to be kept but it was dropped"


# ---------------------------------------------------------------------------
# Compound / multi-feature list tests
# ---------------------------------------------------------------------------

def test_mixed_list_removes_only_market_features():
    cols = [
        "score_margin_home",  # keep
        "volume",             # drop
        "period",             # keep
        "open_interest",      # drop
        "pregame_home_win_prob",  # keep
        "yes_ask",            # drop
    ]
    result = filter_game_state_features(cols)
    assert result == ["score_margin_home", "period", "pregame_home_win_prob"]


def test_empty_list_returns_empty():
    assert filter_game_state_features([]) == []


def test_all_game_state_list_unchanged():
    cols = ["score_margin_home", "period", "seconds_remaining", "pregame_home_win_prob"]
    assert filter_game_state_features(cols) == cols


def test_all_market_list_returns_empty():
    cols = ["volume", "open_interest", "yes_ask", "bid_size", "mid_price"]
    assert filter_game_state_features(cols) == []


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_returns_list():
    assert isinstance(filter_game_state_features(["score_margin_home"]), list)

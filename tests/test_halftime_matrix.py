import pandas as pd
from live_training_matrix import build_in_game_training_matrix


def _labeled():
    return pd.DataFrame([
        {"game_key": "A", "ticker": "T-A", "event_ticker": "E",
         "game_date": "2026-04-18", "home_team": "HOM", "away_team": "AWY",
         "game_status": "in", "bet_side": "home",
         "captured_at": "2026-04-18T00:40:00Z", "status_state": "in",
         "period": 2, "seconds_elapsed": 1400, "seconds_left_in_period": 100,
         "home_score": 55, "away_score": 50, "score_margin_home": 5,
         "total_points": 105,
         "yes_bid": 0.55, "yes_ask": 0.57, "yes_mid": 0.56,
         "no_bid": 0.43, "no_ask": 0.45, "last_price": 0.56,
         "market_home_implied": 0.56,
         "volume": 100.0, "open_interest": 50.0,
         "yes_depth_notional_3": 1.0, "yes_depth_notional_5": 1.0,
         "no_depth_notional_3": 1.0, "no_depth_notional_5": 1.0,
         "yes_weighted_price_3": 0.56, "no_weighted_price_3": 0.44,
         "espn_home_implied": 0.56, "espn_away_implied": 0.44,
         "oddsapi_home_consensus": 0.56, "oddsapi_away_consensus": 0.44,
         "oddsapi_books": 5.0,
         "market_consensus_home": 0.56,
         "pregame_home_win_prob": 0.50, "pregame_away_win_prob": 0.50,
         "pregame_spread": 0.0, "pregame_total": 220.0,
         "pregame_edge_home": 0.0, "consensus_gap_home": 0.0,
         "label_yes_mid_move_5m": 0.0,
         "label_market_home_implied_move_5m": 0.0,
         "label_yes_up_5m": 0, "label_home_up_5m": 1,
         "label_beats_close_yes": 0, "label_beats_close_home": 1,
         "label_halftime_home_lead": 1,
         "label_final_home_win": 1},
    ])


def test_matrix_retains_halftime_label_column():
    matrix, feature_cols = build_in_game_training_matrix(_labeled())
    assert "label_halftime_home_lead" in matrix.columns
    assert matrix.iloc[0]["label_halftime_home_lead"] == 1
    # label must NOT leak into the model feature set
    assert "label_halftime_home_lead" not in feature_cols

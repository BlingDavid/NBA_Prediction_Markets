"""
Tests for live_training_matrix helpers.

Covers the pure utility fns (_status_flag) and the summarize_targets
recommendation logic (the 500-row thresholds that steer which target
the bootstrap trainer picks up).
"""

from __future__ import annotations

import pandas as pd

from live_training_matrix import (
    _status_flag,
    build_in_game_training_matrix,
    summarize_targets,
)


class TestStatusFlag:
    def test_matches_case_insensitively(self):
        s = pd.Series(["in", "IN", "In"])
        out = _status_flag(s, "in")
        assert out.tolist() == [1, 1, 1]

    def test_non_match_returns_zero(self):
        s = pd.Series(["pre", "post", "final"])
        out = _status_flag(s, "in")
        assert out.tolist() == [0, 0, 0]

    def test_na_and_empty_treated_as_non_match(self):
        s = pd.Series(["in", None, "", float("nan")])
        out = _status_flag(s, "in")
        assert out.tolist() == [1, 0, 0, 0]


class TestSummarizeTargets:
    def test_empty_matrix_returns_no_rows(self):
        out = summarize_targets(pd.DataFrame())
        assert out["rows"] == 0
        assert out["recommended_primary_target_current"] is None
        assert out["recommended_primary_target_architecture"] == "label_final_home_win"

    @staticmethod
    def _make_matrix(
        final_count: int = 0,
        move_count: int = 0,
        close_count: int = 0,
        live_rows: int = 0,
    ) -> pd.DataFrame:
        """Build a synthetic matrix with the requested per-target coverage."""
        rows = max(final_count, move_count, close_count, live_rows, 1)
        matrix = pd.DataFrame({
            "flag_has_game_state": [1] * rows,
            "flag_has_time_state": [1] * rows,
            "flag_status_live": [1 if i < live_rows else 0 for i in range(rows)],
            "flag_status_pre": [0] * rows,
            "label_final_home_win": [1 if i < final_count else None for i in range(rows)],
            "label_home_up_5m": [1 if i < move_count else None for i in range(rows)],
            "label_beats_close_home": [1 if i < close_count else None for i in range(rows)],
        })
        return matrix

    def test_final_label_recommended_when_both_thresholds_met(self):
        matrix = self._make_matrix(final_count=600, live_rows=600)
        out = summarize_targets(matrix)
        assert out["recommended_primary_target_current"] == "label_final_home_win"

    def test_move_label_recommended_when_final_unavailable(self):
        # Final has enough rows but live_rows < 500 → fallback to move label
        matrix = self._make_matrix(final_count=600, move_count=600, live_rows=100)
        out = summarize_targets(matrix)
        assert out["recommended_primary_target_current"] == "label_home_up_5m"

    def test_close_label_recommended_when_move_also_unavailable(self):
        matrix = self._make_matrix(final_count=100, move_count=100, close_count=600, live_rows=50)
        out = summarize_targets(matrix)
        assert out["recommended_primary_target_current"] == "label_beats_close_home"

    def test_none_recommended_when_all_below_threshold(self):
        matrix = self._make_matrix(final_count=10, move_count=10, close_count=10, live_rows=10)
        out = summarize_targets(matrix)
        assert out["recommended_primary_target_current"] is None


class TestBuildInGameTrainingMatrix:
    def test_empty_input_returns_empty_matrix_and_features(self):
        matrix, feature_cols = build_in_game_training_matrix(pd.DataFrame())
        assert matrix.empty
        assert feature_cols == []

    def test_only_home_side_filters_away_rows(self):
        base = {
            "ticker": "T", "event_ticker": "E", "game_key": "G",
            "game_date": "2026-04-20",
            "captured_at": "2026-04-20T19:00:00Z",
            "home_team": "H", "away_team": "A",
            "game_status": "in", "status_state": "in",
            "period": 1, "seconds_elapsed": 60, "seconds_left_in_period": 600,
            "home_score": 10, "away_score": 8,
            "score_margin_home": 2, "total_points": 18,
            "yes_bid": 0.5, "yes_ask": 0.52, "yes_mid": 0.51,
            "no_bid": 0.48, "no_ask": 0.50, "last_price": 0.51,
            "market_home_implied": 0.51,
            "volume": 100, "open_interest": 50,
            "yes_depth_notional_3": 1.0, "yes_depth_notional_5": 1.0,
            "no_depth_notional_3": 1.0, "no_depth_notional_5": 1.0,
            "yes_weighted_price_3": 0.51, "no_weighted_price_3": 0.49,
            "espn_home_implied": 0.52, "espn_away_implied": 0.48,
            "oddsapi_home_consensus": 0.51, "oddsapi_away_consensus": 0.49, "oddsapi_books": 5,
            "market_consensus_home": 0.51,
            "pregame_home_win_prob": 0.55, "pregame_away_win_prob": 0.45,
            "pregame_spread": -1.0, "pregame_total": 220,
            "pregame_edge_home": 0.0, "consensus_gap_home": 0.0,
            "label_yes_mid_move_5m": 0.0,
            "label_market_home_implied_move_5m": 0.0,
            "label_yes_up_5m": 0, "label_home_up_5m": 0,
            "label_beats_close_yes": 0, "label_beats_close_home": 0,
            "label_final_home_win": 1,
        }
        labeled = pd.DataFrame([
            {**base, "bet_side": "home"},
            {**base, "bet_side": "away"},
        ])

        matrix, _ = build_in_game_training_matrix(labeled, only_home_side=True)
        assert len(matrix) == 1
        assert (matrix["bet_side"] == "home").all()

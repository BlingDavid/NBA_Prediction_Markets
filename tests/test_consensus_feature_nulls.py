"""
Regression tests for the 6 features that were 100% null in the live model:

    espn_home_implied, espn_away_implied  (ESPN/DraftKings moneyline absent)
    market_consensus_home, consensus_gap_home  (OddsAPI gate closed)
    market_vs_consensus_home, espn_vs_market_home  (derived from above — cascade)

Root causes diagnosed in 2026-08-24 investigation:

1. ESPN API (DraftKings): never returns homeTeamOdds.moneyLine for NBA games
   — it only provides spread+total. The capture code reads the right field but
   the field is always absent.  FIX: feed espn_home_implied from OddsAPI h2h
   (another agent's job) or accept it as permanently null (see recommendation).

2. OddsAPI gate (ENABLE_ODDS_API): ODDS_API_KEY is set in .env but
   ENABLE_ODDS_API defaults to False.  The key is wired; the gate is not open.
   FIX tested here: .env should contain ENABLE_ODDS_API=true.

3. market_consensus_home: computed in realtime_feature_store.build_feature_rows
   as mean(espn_home_implied, oddsapi_home_consensus) — if BOTH inputs are null
   the result is null.  With OddsAPI enabled, oddsapi_home_consensus becomes
   non-null and market_consensus_home becomes non-null.

4. consensus_gap_home: market_home_implied - market_consensus_home — null when
   market_consensus_home is null.  Resolved by fix (3).

5. market_vs_consensus_home: computed in live_training_matrix as
   market_home_implied - market_consensus_home — same dependency.

6. espn_vs_market_home: computed in live_training_matrix as
   espn_home_implied - market_home_implied — null whenever espn_home_implied is
   null (ESPN moneyline gap); requires a separate ESPN-moneyline wiring fix.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_game_row(**kwargs) -> dict:
    base = {
        "captured_at": "2026-04-20T19:00:00+00:00",
        "game_key": "2026-04-20_ATL_BOS",
        "game_date": "2026-04-20",
        "espn_event_id": "123",
        "home_team": "BOS",
        "away_team": "ATL",
        "home_score": 50,
        "away_score": 48,
        "score_margin_home": 2,
        "total_points": 98,
        "game_status": "In Progress",
        "status_detail": None,
        "status_state": "in",
        "is_live": True,
        "is_final": False,
        "period": 2,
        "display_clock": "6:00",
        "seconds_left_in_period": 360,
        "seconds_elapsed": 864,
        "scheduled_start": "2026-04-20T19:00:00Z",
        "espn_provider": None,
        # ESPN moneyline is the critical field — absent in practice
        "espn_home_moneyline": None,
        "espn_away_moneyline": None,
        "espn_home_implied": None,
        "espn_away_implied": None,
        "espn_spread": None,
        "espn_total": None,
    }
    base.update(kwargs)
    return base


def _make_market_row(**kwargs) -> dict:
    base = {
        "captured_at": "2026-04-20T19:00:00+00:00",
        "ticker": "KXNBAGAME-26APR20ATLBOS-BOS",
        "event_ticker": "KXNBAGAME-26APR20ATLBOS",
        "series_ticker": "KXNBAGAME",
        "title": "BOS vs ATL",
        "subtitle": None,
        "status": "open",
        "close_time": None,
        "game_date": "2026-04-20",
        "game_key": "2026-04-20_ATL_BOS",
        "home_team": "BOS",
        "away_team": "ATL",
        "bet_team": "BOS",
        "bet_side": "home",
        "yes_bid": 0.55,
        "yes_ask": 0.57,
        "yes_mid": 0.56,
        "no_bid": 0.43,
        "no_ask": 0.45,
        "last_price": 0.55,
        "volume": 200.0,
        "open_interest": 100.0,
        "market_home_implied": 0.57,
        "best_yes_bid_qty": 10.0,
        "best_no_bid_qty": 10.0,
        "yes_depth_qty_3": 30.0,
        "yes_depth_notional_3": 16.8,
        "yes_weighted_price_3": 0.56,
        "yes_depth_qty_5": 50.0,
        "yes_depth_notional_5": 28.0,
        "yes_weighted_price_5": 0.56,
        "no_depth_qty_3": 30.0,
        "no_depth_notional_3": 13.2,
        "no_weighted_price_3": 0.44,
        "no_depth_qty_5": 50.0,
        "no_depth_notional_5": 22.0,
        "no_weighted_price_5": 0.44,
        "yes_levels_top5": "[]",
        "no_levels_top5": "[]",
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Test 1: ESPN moneyline absent → espn_home_implied stays None in feature rows
# ---------------------------------------------------------------------------

class TestEspnMoneylineAbsent:
    """ESPN/DraftKings does NOT supply homeTeamOdds.moneyLine in the live scoreboard.
    Therefore espn_home_implied is always None in the captured game_states,
    and must propagate as None into feature rows — this is a data-source gap,
    not a code bug.  The test documents the limitation so no one introduces
    a silent masking workaround."""

    def test_espn_home_implied_is_none_when_moneyline_absent(self):
        """When ESPN odds blob has spread/total but no moneyLine,
        espn_home_implied in the feature row must be None."""
        from realtime_feature_store import LiveFeatureStore

        games_df = pd.DataFrame([_make_game_row()])
        markets_df = pd.DataFrame([_make_market_row()])

        store = LiveFeatureStore.__new__(LiveFeatureStore)
        store.state = {"games": {}, "markets": {}, "divergence": {}}

        with patch.object(store, "_build_consensus_map", return_value={}):
            with patch("realtime_feature_store.get_model_prediction", return_value=None):
                features = store.build_feature_rows(games_df, markets_df, include_consensus=False)

        assert not features.empty
        assert features["espn_home_implied"].isna().all(), (
            "espn_home_implied must be NaN when ESPN moneyline is absent in the raw data"
        )
        assert features["espn_away_implied"].isna().all(), (
            "espn_away_implied must be NaN when ESPN moneyline is absent in the raw data"
        )

    def test_espn_home_implied_is_populated_when_moneyline_present(self):
        """Positive control: when ESPN DOES provide a moneyLine, espn_home_implied
        is computed correctly in the game_state and flows into the feature row."""
        from realtime_feature_store import LiveFeatureStore

        # Simulate a game row where ESPN provided moneyline
        games_df = pd.DataFrame([_make_game_row(
            espn_home_moneyline=-150,
            espn_away_moneyline=130,
            espn_home_implied=round(150 / 250, 4),   # 0.6
            espn_away_implied=round(100 / 230, 4),   # ~0.4348
        )])
        markets_df = pd.DataFrame([_make_market_row()])

        store = LiveFeatureStore.__new__(LiveFeatureStore)
        store.state = {"games": {}, "markets": {}, "divergence": {}}

        with patch.object(store, "_build_consensus_map", return_value={}):
            with patch("realtime_feature_store.get_model_prediction", return_value=None):
                features = store.build_feature_rows(games_df, markets_df, include_consensus=False)

        assert not features.empty
        assert features["espn_home_implied"].notna().any(), (
            "espn_home_implied must be non-null when ESPN moneyline IS supplied"
        )
        val = features["espn_home_implied"].iloc[0]
        assert abs(val - 0.6) < 0.001


# ---------------------------------------------------------------------------
# Test 2: market_consensus_home null cascade when OddsAPI disabled
# ---------------------------------------------------------------------------

class TestConsensusNullCascade:
    """When ENABLE_ODDS_API is False (the historical state), _build_consensus_map
    returns {} for every matchup, so oddsapi_home_consensus is always None.
    Combined with espn_home_implied always being None (ESPN gap), the
    consensus_candidates list is empty, and market_consensus_home is None.
    This causes consensus_gap_home to also be None."""

    def _build_features_with_consensus_map(self, consensus_map: dict) -> pd.DataFrame:
        from realtime_feature_store import LiveFeatureStore

        games_df = pd.DataFrame([_make_game_row()])
        markets_df = pd.DataFrame([_make_market_row()])

        store = LiveFeatureStore.__new__(LiveFeatureStore)
        store.state = {"games": {}, "markets": {}, "divergence": {}}

        with patch.object(store, "_build_consensus_map", return_value=consensus_map):
            with patch("realtime_feature_store.get_model_prediction", return_value=None):
                return store.build_feature_rows(games_df, markets_df, include_consensus=True)

    def test_market_consensus_home_is_none_when_consensus_map_empty(self):
        """Empty consensus_map (OddsAPI disabled) → market_consensus_home is None."""
        features = self._build_features_with_consensus_map({})
        assert not features.empty
        assert features["market_consensus_home"].isna().all(), (
            "market_consensus_home must be null when both espn_home_implied and "
            "oddsapi_home_consensus are null"
        )

    def test_consensus_gap_home_is_none_when_consensus_map_empty(self):
        """consensus_gap_home = market_home_implied - market_consensus_home.
        When market_consensus_home is None, consensus_gap_home must be None too."""
        features = self._build_features_with_consensus_map({})
        assert not features.empty
        assert features["consensus_gap_home"].isna().all()

    def test_market_consensus_home_populated_when_oddsapi_returns_data(self):
        """Positive control: when _build_consensus_map returns a real consensus,
        market_consensus_home is non-null and matches the expected average."""
        consensus_map = {
            ("BOS", "ATL"): {
                "home_prob": 0.60,
                "away_prob": 0.40,
                "n_books": 5,
                "commence": None,
            }
        }
        features = self._build_features_with_consensus_map(consensus_map)
        assert not features.empty
        # consensus_home = mean([espn_home_implied(None is skipped), 0.60]) = 0.60
        assert features["market_consensus_home"].notna().any(), (
            "market_consensus_home must be non-null when oddsapi_home_consensus is available"
        )
        val = features["market_consensus_home"].iloc[0]
        assert abs(val - 0.60) < 0.001

    def test_consensus_gap_home_populated_when_oddsapi_returns_data(self):
        """consensus_gap_home = market_home_implied - market_consensus_home.
        With consensus = 0.60 and market_home_implied = 0.57, gap = -0.03."""
        consensus_map = {
            ("BOS", "ATL"): {
                "home_prob": 0.60,
                "away_prob": 0.40,
                "n_books": 5,
                "commence": None,
            }
        }
        features = self._build_features_with_consensus_map(consensus_map)
        assert features["consensus_gap_home"].notna().any()
        gap = features["consensus_gap_home"].iloc[0]
        # market_home_implied = 0.57, consensus_home = 0.60 → gap = 0.57 - 0.60 = -0.03
        assert abs(gap - (0.57 - 0.60)) < 0.001


# ---------------------------------------------------------------------------
# Test 3: derived features in live_training_matrix propagate nulls correctly
# ---------------------------------------------------------------------------

class TestDerivedFeatureNullPropagation:
    """market_vs_consensus_home and espn_vs_market_home are computed in
    live_training_matrix.build_in_game_training_matrix.
    When their inputs are null, they must be null too (not silently zero)."""

    @staticmethod
    def _labeled_row(**overrides) -> dict:
        base = {
            "ticker": "KXNBAGAME-26APR20ATLBOS-BOS",
            "event_ticker": "KXNBAGAME-26APR20ATLBOS",
            "game_key": "2026-04-20_ATL_BOS",
            "game_date": "2026-04-20",
            "captured_at": "2026-04-20T19:00:00+00:00",
            "home_team": "BOS",
            "away_team": "ATL",
            "bet_side": "home",
            "game_status": "In Progress",
            "status_state": "in",
            "period": 2,
            "seconds_elapsed": 864,
            "seconds_left_in_period": 360,
            "home_score": 50,
            "away_score": 48,
            "score_margin_home": 2,
            "total_points": 98,
            "yes_bid": 0.55,
            "yes_ask": 0.57,
            "yes_mid": 0.56,
            "no_bid": 0.43,
            "no_ask": 0.45,
            "last_price": 0.55,
            "market_home_implied": 0.57,
            "volume": 200.0,
            "open_interest": 100.0,
            "yes_depth_notional_3": 16.8,
            "yes_depth_notional_5": 28.0,
            "no_depth_notional_3": 13.2,
            "no_depth_notional_5": 22.0,
            "yes_weighted_price_3": 0.56,
            "no_weighted_price_3": 0.44,
            # The 6 null-in-prod features:
            "espn_home_implied": None,
            "espn_away_implied": None,
            # OddsAPI columns (required when ENABLE_ODDS_API=True)
            "oddsapi_home_consensus": None,
            "oddsapi_away_consensus": None,
            "oddsapi_books": None,
            "market_consensus_home": None,
            "consensus_gap_home": None,
            # Labels
            "pregame_home_win_prob": 0.55,
            "pregame_away_win_prob": 0.45,
            "pregame_spread": -2.0,
            "pregame_total": 218.0,
            "pregame_edge_home": -0.02,
            "label_yes_mid_move_5m": 0.01,
            "label_market_home_implied_move_5m": 0.02,
            "label_yes_up_5m": 1,
            "label_home_up_5m": 1,
            "label_beats_close_yes": 0,
            "label_beats_close_home": 0,
            "label_final_home_win": 1,
        }
        base.update(overrides)
        return base

    def test_market_vs_consensus_home_null_when_market_consensus_null(self):
        """market_vs_consensus_home = market_home_implied - market_consensus_home.
        When market_consensus_home is None, result must be NaN."""
        from live_training_matrix import build_in_game_training_matrix

        labeled = pd.DataFrame([self._labeled_row(market_consensus_home=None)])
        matrix, _ = build_in_game_training_matrix(labeled, only_home_side=True)

        assert not matrix.empty
        assert matrix["market_vs_consensus_home"].isna().all(), (
            "market_vs_consensus_home must be NaN when market_consensus_home is null"
        )

    def test_espn_vs_market_home_null_when_espn_implied_null(self):
        """espn_vs_market_home = espn_home_implied - market_home_implied.
        When espn_home_implied is None (ESPN moneyline gap), result must be NaN."""
        from live_training_matrix import build_in_game_training_matrix

        labeled = pd.DataFrame([self._labeled_row(espn_home_implied=None)])
        matrix, _ = build_in_game_training_matrix(labeled, only_home_side=True)

        assert not matrix.empty
        assert matrix["espn_vs_market_home"].isna().all(), (
            "espn_vs_market_home must be NaN when espn_home_implied is null"
        )

    def test_market_vs_consensus_home_populated_when_consensus_present(self):
        """Positive control: when market_consensus_home is supplied, the
        derived feature is computed correctly."""
        from live_training_matrix import build_in_game_training_matrix

        labeled = pd.DataFrame([self._labeled_row(market_consensus_home=0.60)])
        matrix, _ = build_in_game_training_matrix(labeled, only_home_side=True)

        assert not matrix.empty
        val = matrix["market_vs_consensus_home"].iloc[0]
        # market_home_implied = 0.57, market_consensus_home = 0.60 → diff = -0.03
        assert abs(val - (0.57 - 0.60)) < 0.001

    def test_espn_vs_market_home_populated_when_espn_implied_present(self):
        """Positive control: when espn_home_implied is supplied, the derived
        feature is computed correctly."""
        from live_training_matrix import build_in_game_training_matrix

        labeled = pd.DataFrame([self._labeled_row(
            espn_home_implied=0.62,
            espn_away_implied=0.38,
        )])
        matrix, _ = build_in_game_training_matrix(labeled, only_home_side=True)

        assert not matrix.empty
        val = matrix["espn_vs_market_home"].iloc[0]
        # espn_home_implied=0.62, market_home_implied=0.57 → diff = +0.05
        assert abs(val - (0.62 - 0.57)) < 0.001


# ---------------------------------------------------------------------------
# Test 4: ENABLE_ODDS_API gate — the fixable bug
# ---------------------------------------------------------------------------

class TestEnableOddsApiGateForConsensus:
    """The primary fixable bug: ODDS_API_KEY is in .env but ENABLE_ODDS_API is
    not set, so fetch_odds_api_lines() always returns []. This test documents
    the gate behavior and proves the fix (setting ENABLE_ODDS_API=true)
    allows oddsapi_home_consensus to flow through."""

    def test_build_consensus_map_returns_empty_when_odds_api_disabled(self):
        """With ENABLE_ODDS_API=False, _build_consensus_map returns {} → no consensus."""
        import live_data
        from realtime_feature_store import LiveFeatureStore

        store = LiveFeatureStore.__new__(LiveFeatureStore)
        store.state = {}

        # Simulate the disabled state (matching the production .env state)
        with patch.object(live_data, "ENABLE_ODDS_API", False):
            result = store._build_consensus_map()

        assert result == {}, (
            "_build_consensus_map must return empty dict when ENABLE_ODDS_API=False"
        )

    def test_build_consensus_map_returns_data_when_odds_api_enabled(self):
        """With ENABLE_ODDS_API=True and a mock API response, _build_consensus_map
        returns a populated dict — i.e., enabling the gate is sufficient to fix
        oddsapi_home_consensus."""
        import live_data
        from realtime_feature_store import LiveFeatureStore
        from datetime import datetime, timezone

        # Use a fresh timestamp so consensus_implied_prob's staleness check passes
        # (CONSENSUS_STALE_SECONDS=300; stale quotes from April 2026 would be filtered)
        fresh_ts = datetime.now(tz=timezone.utc).isoformat()

        # Fake The Odds API response shape (post-parse, as fetch_odds_api_lines returns)
        fake_game = {
            "home": "BOS",
            "away": "ATL",
            "commence": fresh_ts,
            "books": [
                {
                    "name": "DraftKings",
                    "last_update": fresh_ts,
                    "markets": {
                        "h2h": {
                            "BOS": {"price": -150, "point": None},
                            "ATL": {"price": 130, "point": None},
                        }
                    },
                },
                {
                    "name": "FanDuel",
                    "last_update": fresh_ts,
                    "markets": {
                        "h2h": {
                            "BOS": {"price": -145, "point": None},
                            "ATL": {"price": 125, "point": None},
                        }
                    },
                },
                {
                    "name": "BetMGM",
                    "last_update": fresh_ts,
                    "markets": {
                        "h2h": {
                            "BOS": {"price": -155, "point": None},
                            "ATL": {"price": 135, "point": None},
                        }
                    },
                },
            ],
        }

        store = LiveFeatureStore.__new__(LiveFeatureStore)
        store.state = {}

        with patch.object(live_data, "ENABLE_ODDS_API", True):
            with patch("realtime_feature_store.fetch_odds_api_lines", return_value=[fake_game]):
                result = store._build_consensus_map()

        assert ("BOS", "ATL") in result, (
            "With ENABLE_ODDS_API=True and valid API data, consensus map must be populated"
        )
        consensus = result[("BOS", "ATL")]
        assert "home_prob" in consensus
        assert 0.55 < consensus["home_prob"] < 0.65, (
            f"BOS at ~-150 should have home_prob ~0.60, got {consensus['home_prob']}"
        )

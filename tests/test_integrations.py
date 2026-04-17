"""
Integration smokes — Layer 2 tests from the spec.
Exercise the refactored _build_consensus_map and the extended
build_feature_rows columns in realtime_feature_store.py.
"""

from unittest.mock import patch

import pandas as pd
import pytest


def _odds_api_payload_shape(home, away, books_price_pairs, last_update=None):
    """Return a list[dict] shaped like fetch_odds_api_lines' output.

    Default last_update=None so books bypass the staleness filter — staleness
    semantics are covered by TestConsensusImpliedProb in test_consensus_divergence.
    """
    return [{
        "home": home,
        "away": away,
        "commence": "2026-04-16T23:30:00Z",
        "books": [
            {
                "name": name,
                "last_update": last_update,
                "markets": {
                    "h2h": {
                        home: {"price": h_ml, "point": None},
                        away: {"price": a_ml, "point": None},
                    }
                },
            }
            for (name, h_ml, a_ml) in books_price_pairs
        ],
    }]


class TestBuildConsensusMapRefactor:
    def test_uses_devigged_median_when_five_books(self):
        from realtime_feature_store import LiveFeatureStore

        payload = _odds_api_payload_shape(
            "BOS", "MIA",
            [
                ("DK", -140, 120),
                ("FD", -145, 125),
                ("MGM", -135, 115),
                ("Caesars", -150, 130),
                ("PointsBet", -138, 118),
            ],
        )

        with patch("realtime_feature_store.fetch_odds_api_lines", return_value=payload):
            store = LiveFeatureStore.__new__(LiveFeatureStore)
            store.state = {"games": {}, "markets": {}}
            result = store._build_consensus_map()

        assert ("BOS", "MIA") in result
        consensus = result[("BOS", "MIA")]
        # De-vigged median home prob should be ~0.575, below the un-devigged 0.583
        assert 0.56 < consensus["home_prob"] < 0.60
        assert consensus["n_books"] == 5
        # Sum to 1 after re-normalization (de-vigged)
        assert consensus["home_prob"] + consensus["away_prob"] == pytest.approx(1.0, abs=1e-6)

    def test_returns_empty_when_below_min_books(self):
        from realtime_feature_store import LiveFeatureStore

        payload = _odds_api_payload_shape("BOS", "MIA", [("DK", -140, 120)])
        with patch("realtime_feature_store.fetch_odds_api_lines", return_value=payload):
            store = LiveFeatureStore.__new__(LiveFeatureStore)
            store.state = {"games": {}, "markets": {}}
            result = store._build_consensus_map()
        # Game key absent or present with None probs — either signals "no consensus".
        consensus = result.get(("BOS", "MIA"))
        assert consensus is None or consensus.get("home_prob") is None


class TestFeatureRowDivergenceColumns:
    REQUIRED_COLS = {
        "kalshi_vs_consensus_pp",
        "model_vs_consensus_pp",
        "kalshi_vs_model_pp",
        "triangulation_tier",
        "consensus_n_books",
    }

    def _minimal_store(self, consensus_map):
        """Create a LiveFeatureStore with stubbed consensus."""
        from realtime_feature_store import LiveFeatureStore
        store = LiveFeatureStore.__new__(LiveFeatureStore)
        store.state = {"games": {}, "markets": {}}
        store._build_consensus_map = lambda: consensus_map
        return store

    def _markets_df(self, bet_side="home", yes_mid=0.48):
        yes_bid = yes_mid - 0.01
        yes_ask = yes_mid + 0.01
        return pd.DataFrame([{
            "captured_at": pd.Timestamp("2026-04-16T23:00:00", tz="UTC"),
            "ticker": "KXNBAGAME-26APR16BOSMIA-BOS",
            "event_ticker": "KXNBAGAME-26APR16BOSMIA",
            "home_team": "BOS",
            "away_team": "MIA",
            "bet_team": "BOS",
            "bet_side": bet_side,
            "status": "active",
            "game_date": "2026-04-16",
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "yes_mid": yes_mid,
            "no_bid": 1 - yes_ask,
            "no_ask": 1 - yes_bid,
            "last_price": yes_mid,
            "volume": 100.0,
            "open_interest": 50.0,
            "yes_depth_notional_3": 200.0,
            "yes_depth_notional_5": 300.0,
            "no_depth_notional_3": 200.0,
            "no_depth_notional_5": 300.0,
            "yes_weighted_price_3": yes_mid,
            "no_weighted_price_3": 1 - yes_mid,
            "market_home_implied": yes_mid if bet_side == "home" else 1 - yes_mid,
            "espn_home_implied": None,
            "espn_away_implied": None,
            "game_key": "2026-04-16|BOS|MIA",
        }])

    def test_columns_present_when_consensus_available(self):
        consensus_map = {("BOS", "MIA"): {
            "home_prob": 0.60, "away_prob": 0.40, "n_books": 4, "commence": None,
        }}
        store = self._minimal_store(consensus_map)
        with patch("realtime_feature_store.get_model_prediction",
                   return_value={"home_win_prob": 0.62, "away_win_prob": 0.38,
                                 "predicted_spread": -3.5, "predicted_total": 225,
                                 "data_date": "2026-04-16"}):
            df = store.build_feature_rows(
                games_df=pd.DataFrame(),
                markets_df=self._markets_df(),
                include_consensus=True,
            )
        row = df.iloc[0]
        assert set(df.columns) >= self.REQUIRED_COLS
        # kalshi_home = 0.48, consensus_home = 0.60 → 12pp
        assert row["kalshi_vs_consensus_pp"] == pytest.approx(12.0, abs=0.01)
        # model 0.62 vs consensus 0.60 → -2pp
        assert row["model_vs_consensus_pp"] == pytest.approx(-2.0, abs=0.01)
        # kalshi 0.48 vs model 0.62 → 14pp
        assert row["kalshi_vs_model_pp"] == pytest.approx(14.0, abs=0.01)
        # |k-c|=12 >= 3.5, |k-m|=14 >= 3.5, same direction → tier 2
        assert row["triangulation_tier"] == 2
        assert row["consensus_n_books"] == 4

    def test_columns_nan_when_consensus_missing(self):
        store = self._minimal_store({})  # no consensus entries
        with patch("realtime_feature_store.get_model_prediction",
                   return_value={"home_win_prob": 0.62, "away_win_prob": 0.38,
                                 "predicted_spread": -3.5, "predicted_total": 225,
                                 "data_date": "2026-04-16"}):
            df = store.build_feature_rows(
                games_df=pd.DataFrame(),
                markets_df=self._markets_df(),
                include_consensus=True,
            )
        row = df.iloc[0]
        assert set(df.columns) >= self.REQUIRED_COLS
        # Without consensus, kalshi_vs_consensus and model_vs_consensus are NaN.
        assert pd.isna(row["kalshi_vs_consensus_pp"])
        assert pd.isna(row["model_vs_consensus_pp"])
        # Tier defaults to 1 when consensus is missing.
        assert row["triangulation_tier"] == 1
        assert row["consensus_n_books"] == 0


class TestDetectDivergenceEvents:
    def test_tier_change_emits_divergence_change_event(self):
        from realtime_feature_store import LiveFeatureStore

        store = LiveFeatureStore.__new__(LiveFeatureStore)
        store.state = {
            "games": {},
            "markets": {},
            "divergence": {
                "2026-04-16|BOS|MIA": {"triangulation_tier": 1},
            },
        }

        features_df = pd.DataFrame([{
            "captured_at": pd.Timestamp("2026-04-16T23:00:00", tz="UTC"),
            "game_key": "2026-04-16|BOS|MIA",
            "ticker": "KXNBAGAME-26APR16BOSMIA-BOS",
            "home_team": "BOS",
            "away_team": "MIA",
            "triangulation_tier": 2,
            "kalshi_vs_consensus_pp": 12.0,
            "model_vs_consensus_pp": -2.0,
            "kalshi_vs_model_pp": 14.0,
        }])

        events = store._detect_divergence_events(features_df)
        assert len(events) == 1
        event = events[0]
        assert event["event_type"] == "divergence_change"
        assert event["entity_type"] == "divergence"
        assert event["entity_key"] == "2026-04-16|BOS|MIA"
        assert event["previous"]["triangulation_tier"] == 1
        assert event["current"]["triangulation_tier"] == 2

    def test_unchanged_tier_emits_no_event(self):
        from realtime_feature_store import LiveFeatureStore

        store = LiveFeatureStore.__new__(LiveFeatureStore)
        store.state = {
            "games": {},
            "markets": {},
            "divergence": {
                "2026-04-16|BOS|MIA": {"triangulation_tier": 1},
            },
        }

        features_df = pd.DataFrame([{
            "captured_at": pd.Timestamp("2026-04-16T23:00:00", tz="UTC"),
            "game_key": "2026-04-16|BOS|MIA",
            "ticker": "KXNBAGAME-26APR16BOSMIA-BOS",
            "home_team": "BOS",
            "away_team": "MIA",
            "triangulation_tier": 1,
            "kalshi_vs_consensus_pp": 1.0,
            "model_vs_consensus_pp": 0.5,
            "kalshi_vs_model_pp": 1.5,
        }])

        events = store._detect_divergence_events(features_df)
        assert events == []

    def test_first_observation_emits_divergence_observed(self):
        from realtime_feature_store import LiveFeatureStore

        store = LiveFeatureStore.__new__(LiveFeatureStore)
        store.state = {"games": {}, "markets": {}, "divergence": {}}

        features_df = pd.DataFrame([{
            "captured_at": pd.Timestamp("2026-04-16T23:00:00", tz="UTC"),
            "game_key": "2026-04-16|BOS|MIA",
            "ticker": "KXNBAGAME-26APR16BOSMIA-BOS",
            "home_team": "BOS",
            "away_team": "MIA",
            "triangulation_tier": 2,
            "kalshi_vs_consensus_pp": 12.0,
            "model_vs_consensus_pp": -2.0,
            "kalshi_vs_model_pp": 14.0,
        }])

        events = store._detect_divergence_events(features_df)
        assert len(events) == 1
        assert events[0]["event_type"] == "divergence_observed"

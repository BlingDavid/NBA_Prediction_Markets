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

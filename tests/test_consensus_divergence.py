"""
Unit tests for consensus_divergence.py — the price-divergence signal
used by ev_analyzer and realtime_feature_store.
"""

import pytest

from consensus_divergence import devig_proportional


class TestDevigProportional:
    def test_standard_favorite_underdog(self):
        # -150 / +130: raw implied 0.600 / 0.4348, sum 1.0348,
        # de-vigged ≈ 0.580 / 0.420
        home, away = devig_proportional(-150, 130)
        assert home == pytest.approx(0.580, abs=0.005)
        assert away == pytest.approx(0.420, abs=0.005)
        assert home + away == pytest.approx(1.0, abs=1e-9)

    def test_pick_em(self):
        home, away = devig_proportional(-110, -110)
        assert home == pytest.approx(0.5, abs=1e-9)
        assert away == pytest.approx(0.5, abs=1e-9)

    def test_none_input_returns_none(self):
        assert devig_proportional(None, -110) is None
        assert devig_proportional(-110, None) is None
        assert devig_proportional(None, None) is None

    def test_zero_moneyline_returns_none(self):
        assert devig_proportional(0, -110) is None
        assert devig_proportional(-110, 0) is None

    def test_empty_string_returns_none(self):
        assert devig_proportional("", -110) is None
        assert devig_proportional(-110, "") is None


from consensus_divergence import consensus_implied_prob


def _book(name, home_ml, away_ml, last_update="2026-04-16T22:55:00Z"):
    """Shape-compatible with live_data._parse_odds_api_game output."""
    return {
        "name": name,
        "last_update": last_update,
        "home_moneyline": home_ml,
        "away_moneyline": away_ml,
    }


class TestConsensusImpliedProb:
    NOW = "2026-04-16T23:00:00Z"  # reference "current time" for staleness

    def test_five_fresh_books_uses_median(self):
        books = [
            _book("DK", -140, 120),
            _book("FD", -145, 125),
            _book("MGM", -135, 115),
            _book("Caesars", -150, 130),
            _book("PointsBet", -138, 118),
        ]
        result = consensus_implied_prob(books, now_iso=self.NOW)
        assert result is not None
        assert result["n_books"] == 5
        assert len(result["books_used"]) == 5
        assert 0.55 < result["home_prob"] < 0.60
        assert result["home_prob"] + result["away_prob"] == pytest.approx(1.0, abs=1e-9)

    def test_drops_stale_quotes(self):
        books = [
            _book("DK", -140, 120),
            _book("FD", -145, 125),
            _book("MGM", -135, 115),
            _book("Caesars", -150, 130, last_update="2026-04-16T22:30:00Z"),
            _book("PointsBet", -138, 118, last_update="2026-04-16T22:40:00Z"),
        ]
        result = consensus_implied_prob(books, now_iso=self.NOW)
        assert result is not None
        assert result["n_books"] == 3
        assert "Caesars" not in result["books_used"]
        assert "PointsBet" not in result["books_used"]

    def test_below_min_books_returns_none(self):
        books = [
            _book("DK", -140, 120),
            _book("FD", -145, 125, last_update="2026-04-16T22:30:00Z"),
            _book("MGM", -135, 115, last_update="2026-04-16T22:30:00Z"),
        ]
        result = consensus_implied_prob(books, now_iso=self.NOW)
        assert result is None

    def test_all_malformed_returns_none(self):
        books = [
            _book("DK", None, 120),
            _book("FD", -145, 0),
            _book("MGM", "", 115),
        ]
        result = consensus_implied_prob(books, now_iso=self.NOW)
        assert result is None

    def test_clamps_extreme_probabilities(self):
        books = [
            _book("DK", -2000, 1000),
            _book("FD", -2200, 1100),
            _book("MGM", -1800, 900),
        ]
        result = consensus_implied_prob(books, now_iso=self.NOW)
        assert result is not None
        assert result["home_prob"] <= 0.99
        assert result["away_prob"] >= 0.01
        assert result["home_prob"] + result["away_prob"] == pytest.approx(1.0, abs=1e-9)

    def test_empty_book_list(self):
        assert consensus_implied_prob([], now_iso=self.NOW) is None

    def test_missing_last_update_is_considered_fresh(self):
        books = [
            _book("DK", -140, 120, last_update=None),
            _book("FD", -145, 125, last_update=None),
            _book("MGM", -135, 115, last_update=None),
        ]
        result = consensus_implied_prob(books, now_iso=self.NOW)
        assert result is not None
        assert result["n_books"] == 3


from live_data import _parse_odds_api_game


class TestParseOddsApiGame:
    def _name_map(self):
        return {
            "Boston Celtics": "BOS",
            "Celtics": "BOS",
            "Miami Heat": "MIA",
            "Heat": "MIA",
        }

    def test_book_dict_includes_last_update(self):
        game = {
            "home_team": "Boston Celtics",
            "away_team": "Miami Heat",
            "commence_time": "2026-04-16T23:30:00Z",
            "bookmakers": [
                {
                    "title": "DraftKings",
                    "last_update": "2026-04-16T22:55:12Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Boston Celtics", "price": -150},
                                {"name": "Miami Heat", "price": 130},
                            ],
                        }
                    ],
                }
            ],
        }
        result = _parse_odds_api_game(game, self._name_map())
        assert result is not None
        assert result["books"][0]["name"] == "DraftKings"
        assert result["books"][0]["last_update"] == "2026-04-16T22:55:12Z"
        assert result["books"][0]["markets"]["h2h"]["BOS"]["price"] == -150
        assert result["books"][0]["markets"]["h2h"]["MIA"]["price"] == 130

    def test_missing_last_update_becomes_none(self):
        game = {
            "home_team": "Boston Celtics",
            "away_team": "Miami Heat",
            "bookmakers": [
                {
                    "title": "FanDuel",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Boston Celtics", "price": -140},
                                {"name": "Miami Heat", "price": 120},
                            ],
                        }
                    ],
                }
            ],
        }
        result = _parse_odds_api_game(game, self._name_map())
        assert result["books"][0]["last_update"] is None

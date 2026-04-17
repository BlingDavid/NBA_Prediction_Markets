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


from consensus_divergence import compute_divergence


class TestComputeDivergence:
    """
    CUTOFF = config.CONSENSUS_DIVERGENCE_THRESHOLD_PP (3.5pp).
    Tier 2: Kalshi AND model both disagree with consensus by >= CUTOFF,
            and they disagree in the SAME direction.
    Tier -1: Kalshi AND consensus disagree with model by >= CUTOFF,
             pointing in OPPOSITE directions.
    Tier 1: everything else (including any missing inputs).
    """

    def _consensus(self, home_prob):
        return {"home_prob": home_prob, "away_prob": 1 - home_prob,
                "n_books": 4, "books_used": ["DK", "FD", "MGM", "Caesars"]}

    def test_tier_2_triangulated_same_direction(self):
        div = compute_divergence(
            kalshi_yes_mid=0.50,
            model_prob_home=0.62,
            consensus=self._consensus(0.60),
        )
        assert div["triangulation_tier"] == 2
        assert div["kalshi_vs_consensus_pp"] == pytest.approx(10.0, abs=0.01)
        assert div["model_vs_consensus_pp"] == pytest.approx(-2.0, abs=0.01)
        assert div["kalshi_vs_model_pp"] == pytest.approx(12.0, abs=0.01)

    def test_tier_minus_1_model_vs_consensus_opposite(self):
        div = compute_divergence(
            kalshi_yes_mid=0.50,
            model_prob_home=0.44,
            consensus=self._consensus(0.58),
        )
        assert div["triangulation_tier"] == -1

    def test_tier_1_below_cutoff(self):
        div = compute_divergence(
            kalshi_yes_mid=0.50,
            model_prob_home=0.52,
            consensus=self._consensus(0.51),
        )
        assert div["triangulation_tier"] == 1

    def test_tier_boundary_exactly_at_cutoff_is_tier_2(self):
        div = compute_divergence(
            kalshi_yes_mid=0.50,
            model_prob_home=0.535,
            consensus=self._consensus(0.535),
        )
        assert div["triangulation_tier"] == 2

    def test_tier_boundary_just_below_cutoff_is_tier_1(self):
        div = compute_divergence(
            kalshi_yes_mid=0.50,
            model_prob_home=0.534,
            consensus=self._consensus(0.534),
        )
        assert div["triangulation_tier"] == 1

    def test_missing_model_defaults_to_tier_1(self):
        div = compute_divergence(
            kalshi_yes_mid=0.50,
            model_prob_home=None,
            consensus=self._consensus(0.62),
        )
        assert div["triangulation_tier"] == 1
        assert div["model_vs_consensus_pp"] is None
        assert div["kalshi_vs_model_pp"] is None
        assert div["kalshi_vs_consensus_pp"] == pytest.approx(12.0, abs=0.01)

    def test_missing_kalshi_defaults_to_tier_1(self):
        div = compute_divergence(
            kalshi_yes_mid=None,
            model_prob_home=0.62,
            consensus=self._consensus(0.60),
        )
        assert div["triangulation_tier"] == 1
        assert div["kalshi_vs_consensus_pp"] is None
        assert div["kalshi_vs_model_pp"] is None

    def test_missing_consensus_defaults_to_tier_1(self):
        div = compute_divergence(
            kalshi_yes_mid=0.50,
            model_prob_home=0.62,
            consensus=None,
        )
        assert div["triangulation_tier"] == 1
        assert div["kalshi_vs_consensus_pp"] is None
        assert div["model_vs_consensus_pp"] is None

    def test_tier_reason_is_descriptive(self):
        div = compute_divergence(0.50, 0.62, self._consensus(0.60))
        assert isinstance(div["tier_reason"], str)
        assert len(div["tier_reason"]) > 0


from consensus_divergence import tier_to_threshold_multiplier


class TestTierToThresholdMultiplier:
    def test_tier_2_lowers_threshold(self):
        assert tier_to_threshold_multiplier(2, 0.03) == pytest.approx(0.021)

    def test_tier_1_unchanged(self):
        assert tier_to_threshold_multiplier(1, 0.03) == pytest.approx(0.03)

    def test_tier_minus_1_raises_threshold(self):
        assert tier_to_threshold_multiplier(-1, 0.03) == pytest.approx(0.045)

    def test_unknown_tier_falls_back_to_neutral(self):
        # Defensive: an unexpected tier should not crash; behave as tier 1.
        assert tier_to_threshold_multiplier(99, 0.03) == pytest.approx(0.03)


import csv
from pathlib import Path


class TestGoldenSnapshot:
    FIXTURE = Path(__file__).parent / "fixtures" / "consensus_golden_cases.csv"

    def _maybe_float(self, s):
        return None if s == "" else float(s)

    def _maybe_int(self, s):
        return None if s == "" else int(s)

    def test_golden_cases_all_match(self):
        rows = list(csv.DictReader(self.FIXTURE.open()))
        assert len(rows) >= 20, "Golden fixture should have >= 20 cases"

        failures = []
        for row in rows:
            kalshi = self._maybe_float(row["kalshi_yes"])
            model = self._maybe_float(row["model_prob"])
            cons_home = self._maybe_float(row["consensus_home"])
            consensus = (
                {"home_prob": cons_home, "away_prob": 1 - cons_home,
                 "n_books": 5, "books_used": []}
                if cons_home is not None else None
            )

            result = compute_divergence(kalshi, model, consensus)

            expected_tier = int(row["expected_tier"])
            if result["triangulation_tier"] != expected_tier:
                failures.append(
                    f"{row['case_id']}: tier expected={expected_tier} "
                    f"got={result['triangulation_tier']} reason={result['tier_reason']}"
                )

            for csv_key, result_key in (
                ("expected_kalshi_vs_consensus_pp", "kalshi_vs_consensus_pp"),
                ("expected_model_vs_consensus_pp",  "model_vs_consensus_pp"),
                ("expected_kalshi_vs_model_pp",     "kalshi_vs_model_pp"),
            ):
                exp = self._maybe_float(row[csv_key])
                got = result[result_key]
                if exp is None and got is None:
                    continue
                if exp is None or got is None:
                    failures.append(f"{row['case_id']}: {result_key} expected={exp} got={got}")
                    continue
                if abs(exp - got) > 0.01:
                    failures.append(f"{row['case_id']}: {result_key} expected={exp} got={got}")

        assert not failures, "Golden snapshot mismatches:\n" + "\n".join(failures)


from consensus_divergence import apply_consensus_tier


class TestApplyConsensusTier:
    NOW = "2026-04-16T23:00:00Z"

    def _comparison(self, books):
        return {"home_team": "BOS", "away_team": "MIA", "sportsbooks": books}

    def _good_books(self):
        return [
            {"name": "DK", "last_update": self.NOW, "home_moneyline": -140, "away_moneyline": 120},
            {"name": "FD", "last_update": self.NOW, "home_moneyline": -145, "away_moneyline": 125},
            {"name": "MGM", "last_update": self.NOW, "home_moneyline": -135, "away_moneyline": 115},
            {"name": "Caesars", "last_update": self.NOW, "home_moneyline": -150, "away_moneyline": 130},
        ]

    def test_shadow_mode_does_not_adjust_threshold(self):
        result = apply_consensus_tier(
            comparison=self._comparison(self._good_books()),
            model_prob_home=0.60,
            kalshi_yes_mid=0.48,
            bet_side="home",
            base_threshold=0.03,
            mode="shadow",
            now_iso=self.NOW,
        )
        assert result["signal_threshold"] == pytest.approx(0.03)
        assert result["shadow_signal_threshold"] != result["signal_threshold"]
        assert result["shadow_signal_threshold"] == pytest.approx(0.03 * 0.7)
        assert result["triangulation_tier"] == 2
        assert result["mode"] == "shadow"

    def test_active_mode_applies_multiplier(self):
        result = apply_consensus_tier(
            comparison=self._comparison(self._good_books()),
            model_prob_home=0.60,
            kalshi_yes_mid=0.48,
            bet_side="home",
            base_threshold=0.03,
            mode="active",
            now_iso=self.NOW,
        )
        assert result["signal_threshold"] == pytest.approx(0.03 * 0.7)
        assert result["triangulation_tier"] == 2

    def test_off_mode_pins_tier_one(self):
        result = apply_consensus_tier(
            comparison=self._comparison(self._good_books()),
            model_prob_home=0.60,
            kalshi_yes_mid=0.48,
            bet_side="home",
            base_threshold=0.03,
            mode="off",
            now_iso=self.NOW,
        )
        assert result["triangulation_tier"] == 1
        assert result["signal_threshold"] == pytest.approx(0.03)

    def test_away_bet_side_inverts_kalshi_yes_mid(self):
        # On an "away YES" ticker, yes_mid=0.48 means the market thinks AWAY wins 48%,
        # so home wins 52%. Divergence against consensus should use 0.52.
        # model 0.54 vs kalshi_home 0.52 → |k-m|=2pp (<cutoff) → tier 1.
        # Same books with home bet_side (kalshi_home=0.48) would land in tier 2,
        # so this case confirms the inversion flips the tier.
        result = apply_consensus_tier(
            comparison=self._comparison(self._good_books()),
            model_prob_home=0.54,
            kalshi_yes_mid=0.48,
            bet_side="away",
            base_threshold=0.03,
            mode="active",
            now_iso=self.NOW,
        )
        assert result["triangulation_tier"] == 1

    def test_missing_sportsbooks_yields_tier_one(self):
        result = apply_consensus_tier(
            comparison={"home_team": "BOS", "away_team": "MIA", "sportsbooks": []},
            model_prob_home=0.60,
            kalshi_yes_mid=0.48,
            bet_side="home",
            base_threshold=0.03,
            mode="active",
            now_iso=self.NOW,
        )
        assert result["triangulation_tier"] == 1
        assert result["consensus_n_books"] == 0
        assert result["signal_threshold"] == pytest.approx(0.03)

"""Edge detector math: spread cover probability + total over probability.

These tests lock in the fix for two bugs that surfaced when the bet card
showed +38% spread edges across every game with implied_prob hardcoded at
0.524:

  Bug A — _spread_cover_prob had a sign error: it computed P(margin > line)
          rather than P(margin > -line). The docstring already documented
          the correct convention; only the formula was wrong.

  Bug B — spread_impl / total_impl were hardcoded to 0.524 (the vigged
          price of a -110 line) instead of being derived from the actual
          spread_home_price / total_over_price columns.

The cover-probability formula assumes home margin ~ Normal(model_spread, std).
'spread_home' is signed in the standard sportsbook convention: negative
when the home team is favored (e.g. spread_home = -4.6 for "BOS -4.6").
"""
from __future__ import annotations

import pytest

ed = pytest.importorskip("edge_detector")


# ───────────────────────── _spread_cover_prob ─────────────────────────


def test_spread_cover_prob_pickem_returns_50_pct():
    # Even matchup, no spread → cover prob = 0.5.
    assert ed._spread_cover_prob(model_spread=0.0, line=0.0) == pytest.approx(0.5, abs=1e-6)


def test_spread_cover_prob_home_predicted_to_outperform_line_covers():
    # Home favored by 4.6; model predicts home wins by 10. Home should
    # comfortably cover. With std=10.5: z = (10 + (-4.6))/10.5 = 0.514,
    # cover prob ≈ 0.696.
    cover = ed._spread_cover_prob(model_spread=10.0, line=-4.6)
    assert 0.65 < cover < 0.75


def test_spread_cover_prob_model_below_line_does_not_cover():
    # Home favored by 13.9; model predicts only 12.5. Home unlikely to
    # cover. z = (12.5 + (-13.9))/10.5 = -0.133, cover ≈ 0.447.
    cover = ed._spread_cover_prob(model_spread=12.5, line=-13.9)
    assert 0.40 < cover < 0.50


def test_spread_cover_prob_huge_favorite_misvalued():
    # Home favored by 15.7; model thinks home wins by only 7.9. Home
    # should be a clear NO-cover. z ≈ -0.743, cover ≈ 0.229.
    cover = ed._spread_cover_prob(model_spread=7.9, line=-15.7)
    assert 0.20 < cover < 0.27


def test_spread_cover_prob_home_underdog_can_cover_by_losing_close():
    # Home is +4.6 underdog; model predicts home loses by 2 (margin = -2).
    # Home covers if it loses by less than 4.6. z = (-2 + 4.6)/10.5 = 0.248,
    # cover ≈ 0.598.
    cover = ed._spread_cover_prob(model_spread=-2.0, line=4.6)
    assert 0.55 < cover < 0.65


def test_spread_cover_prob_symmetric_around_pickem():
    # Equal-and-opposite predictions around the line should give
    # complementary probabilities.
    a = ed._spread_cover_prob(model_spread=5.0, line=0.0)
    b = ed._spread_cover_prob(model_spread=-5.0, line=0.0)
    assert a + b == pytest.approx(1.0, abs=1e-6)


# ───────────────────────── _total_over_prob ─────────────────────────


def test_total_over_prob_at_line_is_50_pct():
    assert ed._total_over_prob(model_total=215.0, line=215.0) == pytest.approx(0.5, abs=1e-6)


def test_total_over_prob_above_line():
    # Model predicts 10 above the line with std=18 → z=0.556 → ~0.71.
    p = ed._total_over_prob(model_total=225.0, line=215.0)
    assert 0.65 < p < 0.75


def test_total_over_prob_below_line():
    p = ed._total_over_prob(model_total=205.0, line=215.0)
    assert 0.25 < p < 0.35


# ───────────────────────── find_edges integration ─────────────────────


def _two_game_predictions() -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame([
        {
            "game_id": "g1", "date": "2026-05-02",
            "home_team": "BOS", "away_team": "PHI",
            "home_win_prob": 0.745, "away_win_prob": 0.255,
            "predicted_spread": 9.6, "predicted_total": 229.6,
        },
        {
            "game_id": "g2", "date": "2026-05-02",
            "home_team": "OKC", "away_team": "LAL",
            "home_win_prob": 0.645, "away_win_prob": 0.355,
            "predicted_spread": 7.9, "predicted_total": 229.6,
        },
    ])


def _two_game_market() -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame([
        {
            "home_team": "BOS", "away_team": "PHI",
            "ml_home": -174, "ml_away": 144,
            "spread_home": -4.6, "spread_home_price": -111,
            "total_line": 218.5, "total_over_price": -110,
        },
        {
            "home_team": "OKC", "away_team": "LAL",
            "ml_home": -180, "ml_away": 150,
            "spread_home": -15.7, "spread_home_price": -109,
            "total_line": 213.6, "total_over_price": -109,
        },
    ])


def test_find_edges_no_longer_inflates_spread_edges():
    """Before the fix the BOS -4.6 spread bet emerged with model_prob ≈ 0.91
    (edge ≈ +38%); after the fix it's ~0.68 (edge ~+18%). The OKC/LAL test
    case still produces a 27% LAL+15.7 edge, which is a real consequence of
    the model's predicted margin (7.9) being far from the line (-15.7).
    Both should land below the sign-bug regime."""
    bets = ed.find_edges(_two_game_predictions(), _two_game_market(), min_edge=0.05)
    spread_bets = bets[bets["market"] == "spread"]
    if len(spread_bets) == 0:
        pytest.skip("no spread edges over 5% — acceptable; we just need them not inflated")
    # The sign bug produced edges of 0.37–0.47. After fix, on these inputs,
    # the worst is ~0.27 (LAL+15.7 vs predicted +7.9). Anything ≥ 0.35 means
    # the sign bug regressed.
    assert spread_bets["edge"].max() < 0.35, (
        f"spread edge {spread_bets['edge'].max():.3f} is in the sign-bug "
        f"regime (>0.35).\n{spread_bets}"
    )
    # Sign bug produced model_prob in the 0.90+ range. Post-fix max here is
    # ~0.77. Anything ≥ 0.88 means the sign bug regressed.
    assert spread_bets["model_prob"].max() < 0.88


def test_find_edges_uses_actual_spread_price_for_implied_prob():
    """When spread_home_price = -111, implied_prob should be ~0.526, not the
    old hardcoded 0.524."""
    market = _two_game_market()
    market.loc[0, "spread_home_price"] = -120  # implied 0.5455
    bets = ed.find_edges(_two_game_predictions(), market, min_edge=0.0)
    spread_bets = bets[(bets["market"] == "spread") & (bets["home_team"] == "BOS")]
    if len(spread_bets) == 0:
        pytest.skip("no qualifying spread bet")
    impl = spread_bets.iloc[0]["implied_prob"]
    # Must reflect the -120 price, not be stuck at 0.524.
    assert impl == pytest.approx(0.5455, abs=0.01)


def test_find_edges_total_implied_prob_reflects_actual_price():
    market = _two_game_market()
    market.loc[0, "total_over_price"] = -120
    bets = ed.find_edges(_two_game_predictions(), market, min_edge=0.0)
    total_bets = bets[(bets["market"] == "total") & (bets["home_team"] == "BOS")]
    if len(total_bets) == 0:
        pytest.skip("no qualifying total bet")
    impl = total_bets.iloc[0]["implied_prob"]
    assert impl == pytest.approx(0.5455, abs=0.01)

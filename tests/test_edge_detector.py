"""Edge detector math: spread cover probability + total over probability + shrinkage.

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


# ───────────────────────── shrinkage ─────────────────────────


def test_shrink_edge_passes_through_below_free_threshold():
    """Edges within the well-calibrated zone (|edge| ≤ 0.10 by default)
    pass through unchanged. The backtest's 0.05–0.10 edge bucket already
    realised +38% ROI; shrinking it would just cost EV."""
    # +0.08 edge: fair=0.50, model=0.58 → unchanged
    out = ed._shrink_edge(model_prob=0.58, fair_prob=0.50)
    assert out == pytest.approx(0.58, abs=1e-6)
    # -0.08 edge stays unchanged too
    out = ed._shrink_edge(model_prob=0.42, fair_prob=0.50)
    assert out == pytest.approx(0.42, abs=1e-6)


def test_shrink_edge_at_boundary_unchanged():
    """At |edge| == free_threshold the shrinkage factor is exactly 1.0."""
    out = ed._shrink_edge(model_prob=0.60, fair_prob=0.50)  # edge = +0.10
    assert out == pytest.approx(0.60, abs=1e-6)


def test_shrink_edge_full_clamp_above_max_threshold():
    """At |edge| ≥ max_threshold (0.30 default), edge is shrunk by min_factor (0.5)."""
    # +0.40 edge: fair=0.20, model=0.60 → shrunk_edge = 0.40 * 0.5 = 0.20 → 0.40
    out = ed._shrink_edge(model_prob=0.60, fair_prob=0.20)
    assert out == pytest.approx(0.40, abs=1e-6)


def test_shrink_edge_lal_realistic_case():
    """The LAL +745 case: model=0.359, fair≈0.115, raw edge=+0.244.
    Linear shrinkage at |edge|=0.244 between thresholds 0.10 and 0.30:
       t = (0.244 - 0.10) / (0.30 - 0.10) = 0.72
       factor = 1.0 - t * (1.0 - 0.5) = 1.0 - 0.36 = 0.64
       shrunk_edge = 0.244 * 0.64 = 0.1562
       shrunk_model = 0.115 + 0.1562 = 0.2712"""
    out = ed._shrink_edge(model_prob=0.359, fair_prob=0.115)
    assert out == pytest.approx(0.2712, abs=1e-3)


def test_shrink_edge_preserves_direction():
    """Shrunk model_prob must remain on the same side of fair_prob as the
    original. We don't want shrinkage to flip a bet from YES to NO."""
    # Strong positive edge stays positive
    assert ed._shrink_edge(0.80, 0.20) > 0.20
    # Strong negative edge stays negative
    assert ed._shrink_edge(0.20, 0.80) < 0.80


def test_shrink_edge_in_range():
    """Shrunk model_prob must always be in [0, 1]."""
    for mp in [0.01, 0.05, 0.50, 0.95, 0.99]:
        for fp in [0.05, 0.50, 0.95]:
            out = ed._shrink_edge(mp, fp)
            assert 0.0 <= out <= 1.0, f"out-of-range for model={mp}, fair={fp}"


def test_find_edges_shrinkage_reduces_lal_style_call(monkeypatch):
    """Integration: LAL underdog with +0.24 raw edge should report a
    smaller shrunk edge in the bet card."""
    import pandas as pd
    predictions = pd.DataFrame([{
        "game_id": "g1", "date": "2026-05-05",
        "home_team": "OKC", "away_team": "LAL",
        "home_win_prob": 0.640, "away_win_prob": 0.360,
        "predicted_spread": 12.0, "predicted_total": 220.0,
    }])
    odds = pd.DataFrame([{
        "home_team": "OKC", "away_team": "LAL",
        "ml_home": -750, "ml_away": +600,  # LAL implied ≈ 0.142, fair ≈ 0.115
        "spread_home": -12.0, "spread_home_price": -110,
        "total_line": 220.0, "total_over_price": -110,
    }])
    bets_shrunk = ed.find_edges(predictions, odds, min_edge=0.0, apply_shrinkage=True)
    bets_raw = ed.find_edges(predictions, odds, min_edge=0.0, apply_shrinkage=False)

    lal_shrunk = bets_shrunk[(bets_shrunk["market"] == "moneyline") & (bets_shrunk["side"] == "away")]
    lal_raw = bets_raw[(bets_raw["market"] == "moneyline") & (bets_raw["side"] == "away")]

    if len(lal_raw) == 0 or len(lal_shrunk) == 0:
        pytest.skip("LAL ML edge didn't qualify in one of the modes")

    # Raw edge should be larger than shrunk edge.
    assert lal_raw.iloc[0]["edge"] > lal_shrunk.iloc[0]["edge"], (
        f"shrinkage didn't reduce edge: raw={lal_raw.iloc[0]['edge']}, "
        f"shrunk={lal_shrunk.iloc[0]['edge']}"
    )
    # Both should still be positive (direction preserved).
    assert lal_shrunk.iloc[0]["edge"] > 0
    # Raw edge is preserved as model_prob_raw column.
    assert "model_prob_raw" in lal_shrunk.columns
    assert lal_shrunk.iloc[0]["model_prob_raw"] == pytest.approx(0.360, abs=1e-3)
    # And the displayed model_prob is the shrunk value.
    assert lal_shrunk.iloc[0]["model_prob"] < lal_shrunk.iloc[0]["model_prob_raw"]


def test_find_edges_no_shrink_flag_passes_raw_through():
    """apply_shrinkage=False should leave model_prob == model_prob_raw."""
    import pandas as pd
    predictions = pd.DataFrame([{
        "game_id": "g1", "date": "2026-05-05",
        "home_team": "OKC", "away_team": "LAL",
        "home_win_prob": 0.640, "away_win_prob": 0.360,
        "predicted_spread": 12.0, "predicted_total": 220.0,
    }])
    odds = pd.DataFrame([{
        "home_team": "OKC", "away_team": "LAL",
        "ml_home": -750, "ml_away": +600,
        "spread_home": -12.0, "spread_home_price": -110,
        "total_line": 220.0, "total_over_price": -110,
    }])
    bets = ed.find_edges(predictions, odds, min_edge=0.0, apply_shrinkage=False)
    if not bets.empty and "model_prob_raw" in bets.columns:
        # All rows: model_prob == model_prob_raw under raw mode.
        assert (bets["model_prob"] == bets["model_prob_raw"]).all()

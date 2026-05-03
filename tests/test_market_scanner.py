"""Tests for market_scanner.

Locks in the fix for the bug where run_pipeline --scan only returned
finals-winner futures (KXNBA-26-{TEAM}) instead of per-game H2H markets
(KXNBAGAME-*). Without per-game markets the pipeline can't surface
Kalshi-vs-sportsbook divergence on individual games.
"""
from __future__ import annotations

import pytest

ms = pytest.importorskip("market_scanner")


# ───────────────────────── series list ─────────────────────────


def test_nba_series_list_includes_per_game_series():
    """KXNBAGAME is the per-game H2H series. If it falls out of this list,
    run_pipeline --scan goes back to surfacing only Finals-winner futures."""
    assert "KXNBAGAME" in ms.NBA_SERIES_TICKERS


def test_nba_series_list_includes_futures_series():
    """KXNBA is the season-long Finals-winner series. Keep both lanes alive."""
    assert "KXNBA" in ms.NBA_SERIES_TICKERS


def test_search_nba_markets_iterates_full_series_list(monkeypatch):
    """search_nba_markets must hit every entry in NBA_SERIES_TICKERS,
    not stop after the first hit. Otherwise dropping KXNBAGAME to second-rank
    would silently re-introduce the bug."""
    client = ms.KalshiClient()
    seen_tickers: list[str] = []

    def fake_get_markets(self, *, series_ticker, status, **_kwargs):
        seen_tickers.append(series_ticker)
        return {"markets": [
            {"ticker": f"FAKE-{series_ticker}-1", "title": "stub"},
        ]}

    monkeypatch.setattr(ms.KalshiClient, "get_markets", fake_get_markets)
    out = client.search_nba_markets()
    assert seen_tickers == ms.NBA_SERIES_TICKERS
    # Each series contributed one market → total = len(series).
    assert len(out) == len(ms.NBA_SERIES_TICKERS)


def test_search_nba_markets_dedupes_overlapping_tickers(monkeypatch):
    """If two series return the same ticker (rare but possible — Kalshi
    sometimes lists a market under multiple parents), keep only one copy."""
    def fake_get_markets(self, *, series_ticker, status, **_kwargs):
        return {"markets": [{"ticker": "DUP-1", "title": "dup"}]}

    monkeypatch.setattr(ms.KalshiClient, "get_markets", fake_get_markets)
    out = ms.KalshiClient().search_nba_markets()
    assert len(out) == 1


def test_search_nba_markets_skips_failing_series(monkeypatch):
    """If one series 404s or raises, the others must still be queried."""
    def fake_get_markets(self, *, series_ticker, status, **_kwargs):
        if series_ticker == "KXNBA":
            raise RuntimeError("simulated 404")
        return {"markets": [{"ticker": f"OK-{series_ticker}", "title": ""}]}

    monkeypatch.setattr(ms.KalshiClient, "get_markets", fake_get_markets)
    out = ms.KalshiClient().search_nba_markets()
    # All series except KXNBA returned a market.
    expected_count = len(ms.NBA_SERIES_TICKERS) - 1
    assert len(out) == expected_count


# ───────────────────────── parser ─────────────────────────


def test_parse_kalshi_markets_handles_per_game_payload():
    """Per-game KXNBAGAME-* markets carry the same field shape as KXNBA-*."""
    raw = [
        {
            "ticker": "KXNBAGAME-26MAY03DETORL-DET",
            "event_ticker": "KXNBAGAME-26MAY03DETORL",
            "series_ticker": "KXNBAGAME",
            "title": "Will the Detroit Pistons beat the Orlando Magic?",
            "subtitle": "May 3, 2026 — Pistons vs Magic",
            "yes_bid": 35,   # cents
            "yes_ask": 37,
            "no_bid": 63,
            "no_ask": 65,
            "volume": 12345,
            "open_interest": 6789,
            "status": "active",
            "close_time": "2026-05-03T23:30:00Z",
        }
    ]
    df = ms.parse_kalshi_markets(raw)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["platform"] == "kalshi"
    assert row["ticker"] == "KXNBAGAME-26MAY03DETORL-DET"
    assert row["series_ticker"] == "KXNBAGAME"
    # Cents converted to dollars (0.00–1.00 range).
    assert row["yes_bid"] == pytest.approx(0.35, abs=1e-6)
    assert row["yes_ask"] == pytest.approx(0.37, abs=1e-6)
    assert row["yes_mid"] == pytest.approx(0.36, abs=1e-6)


# ─────────────────────── matching to predictions ───────────────────────


def test_match_markets_to_predictions_handles_per_game_market():
    """Given a model prediction for DET vs ORL and a per-game Kalshi market,
    the matcher emits a row with model_prob keyed off whichever team appears
    first in the title."""
    import pandas as pd

    predictions = pd.DataFrame([{
        "home_team": "DET", "away_team": "ORL",
        "home_win_prob": 0.78, "away_win_prob": 0.22,
        "predicted_spread": 8.0, "predicted_total": 220.0,
    }])

    # First team in title = DET → market_prob is the YES price for DET winning.
    markets = pd.DataFrame([{
        "platform": "kalshi",
        "ticker": "KXNBAGAME-26MAY03DETORL-DET",
        "title": "Will the Detroit Pistons beat the Orlando Magic?",
        "subtitle": "",
        "yes_bid": 0.65, "yes_ask": 0.67, "yes_mid": 0.66,
        "no_bid": 0.33, "no_ask": 0.35,
        "volume": 1000, "open_interest": 500,
        "status": "active", "close_time": "", "url": "",
    }])

    result = ms.match_markets_to_predictions(markets, predictions)
    if result.empty:
        pytest.fail("expected at least one match for DET-vs-ORL per-game market")
    row = result.iloc[0]
    # First team is DET, which is the home team in predictions → use home_win_prob.
    assert row["yes_team"] == "DET"
    assert row["model_prob"] == pytest.approx(0.78, abs=1e-6)
    assert row["market_prob"] == pytest.approx(0.66, abs=1e-6)
    # Edge ≈ 0.78 - 0.66 = 0.12, well above the noise threshold.
    assert row["edge"] == pytest.approx(0.12, abs=1e-3)


def test_match_kxnbagame_uses_ticker_suffix_not_title_order():
    """KXNBAGAME contracts share one title ("LAL at OKC Winner?") across BOTH
    sides. The actual YES-team is the suffix on the ticker (-OKC vs -LAL).

    Before this fix the matcher used "first team in the title" as the yes
    team, so it would tag the OKC contract (yes_mid 0.885) as LAL — making
    every per-game match emit a fictitious ~50% edge."""
    import pandas as pd

    predictions = pd.DataFrame([{
        "home_team": "OKC", "away_team": "LAL",
        "home_win_prob": 0.64, "away_win_prob": 0.36,
        "predicted_spread": 6.0, "predicted_total": 220.0,
    }])

    markets = pd.DataFrame([
        {
            "platform": "kalshi",
            "ticker": "KXNBAGAME-26MAY05LALOKC-OKC",  # OKC YES contract
            "title": "Game 1: Los Angeles L at Oklahoma City Winner?",
            "subtitle": "",
            "yes_bid": 0.88, "yes_ask": 0.89, "yes_mid": 0.885,
            "no_bid": 0.11, "no_ask": 0.12,
            "volume": 78913, "open_interest": 50000,
            "status": "active", "close_time": "", "url": "",
        },
        {
            "platform": "kalshi",
            "ticker": "KXNBAGAME-26MAY05LALOKC-LAL",  # LAL YES contract
            "title": "Game 1: Los Angeles L at Oklahoma City Winner?",
            "subtitle": "",
            "yes_bid": 0.11, "yes_ask": 0.12, "yes_mid": 0.115,
            "no_bid": 0.88, "no_ask": 0.89,
            "volume": 299270, "open_interest": 50000,
            "status": "active", "close_time": "", "url": "",
        },
    ])

    result = ms.match_markets_to_predictions(markets, predictions)
    assert len(result) >= 2, f"expected both per-game contracts to match, got {len(result)} rows"

    by_ticker = {row["ticker"]: row for _, row in result.iterrows()}

    okc_row = by_ticker["KXNBAGAME-26MAY05LALOKC-OKC"]
    assert okc_row["yes_team"] == "OKC"
    assert okc_row["model_prob"] == pytest.approx(0.64, abs=1e-6)  # OKC is home → home_win_prob
    assert okc_row["market_prob"] == pytest.approx(0.885, abs=1e-6)
    # OKC: model 0.64 vs market 0.885 → negative edge (model thinks LAL more likely
    # to win than market does, so the OKC YES contract is overpriced).
    assert okc_row["edge"] < 0

    lal_row = by_ticker["KXNBAGAME-26MAY05LALOKC-LAL"]
    assert lal_row["yes_team"] == "LAL"
    assert lal_row["model_prob"] == pytest.approx(0.36, abs=1e-6)  # LAL is away → away_win_prob
    assert lal_row["market_prob"] == pytest.approx(0.115, abs=1e-6)
    # LAL contract: model 0.36 vs market 0.115 → positive edge (LAL underpriced).
    assert lal_row["edge"] > 0


def test_match_markets_to_predictions_handles_nan_subtitle():
    """When a market row has a NaN subtitle (e.g., loaded from a CSV where
    empty cells became NaN), the matcher must not raise TypeError."""
    import pandas as pd
    import numpy as np

    predictions = pd.DataFrame([{
        "home_team": "DET", "away_team": "ORL",
        "home_win_prob": 0.78, "away_win_prob": 0.22,
        "predicted_spread": 8.0, "predicted_total": 220.0,
    }])

    markets = pd.DataFrame([{
        "platform": "kalshi",
        "ticker": "KXNBAGAME-26MAY03DETORL-DET",
        "title": "Will the Detroit Pistons beat the Orlando Magic?",
        "subtitle": np.nan,  # the failure mode we're guarding against
        "yes_bid": 0.65, "yes_ask": 0.67, "yes_mid": 0.66,
        "no_bid": 0.33, "no_ask": 0.35,
        "volume": 1000, "open_interest": 500,
        "status": "active", "close_time": "", "url": "",
    }])

    # Should not raise.
    result = ms.match_markets_to_predictions(markets, predictions)
    assert len(result) == 1


def test_match_markets_to_predictions_picks_correct_side_for_underdog_ticker():
    """The companion contract (ORL side of the same game) must use ORL's
    win prob, not DET's."""
    import pandas as pd

    predictions = pd.DataFrame([{
        "home_team": "DET", "away_team": "ORL",
        "home_win_prob": 0.78, "away_win_prob": 0.22,
        "predicted_spread": 8.0, "predicted_total": 220.0,
    }])

    markets = pd.DataFrame([{
        "platform": "kalshi",
        "ticker": "KXNBAGAME-26MAY03DETORL-ORL",
        "title": "Will the Orlando Magic beat the Detroit Pistons?",
        "subtitle": "",
        "yes_bid": 0.33, "yes_ask": 0.35, "yes_mid": 0.34,
        "no_bid": 0.65, "no_ask": 0.67,
        "volume": 800, "open_interest": 400,
        "status": "active", "close_time": "", "url": "",
    }])

    result = ms.match_markets_to_predictions(markets, predictions)
    if result.empty:
        pytest.fail("expected match for ORL-side per-game market")
    row = result.iloc[0]
    assert row["yes_team"] == "ORL"
    assert row["model_prob"] == pytest.approx(0.22, abs=1e-6)
    assert row["market_prob"] == pytest.approx(0.34, abs=1e-6)

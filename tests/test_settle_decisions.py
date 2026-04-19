"""
Tests for settle_decisions.py — the nightly settlement closer.

Covers:
  * compute_settlement: pnl + home_win derivation from bet_side + yes_won
  * fetch_yes_won: Kalshi market status → True/False/None mapping
  * settle_pending: walks ledger, settles fillable rows only
"""

import csv
from unittest.mock import MagicMock

import pytest


def _seed_filled(path, *, bet_side="home", filled_price=0.53, filled_size=10):
    from bet_decisions import record_decision, update_fill

    did = record_decision(
        decisions_path=path,
        ticker="KXNBAGAME-26APR17BOSMIA-BOS",
        home_team="BOS", away_team="MIA",
        bet_side=bet_side, contract_side="YES",
        model_prob_home=0.60, model_prob_away=0.40,
        kalshi_yes_bid=0.51, kalshi_yes_ask=0.53, kalshi_yes_mid=0.52,
        entry_limit_price=0.53,
        edge_pp=7.0, ev=0.07,
        signal_threshold=0.03, base_threshold=0.03, recommended_stake=25.0,
        triangulation_tier=2,
        kalshi_vs_consensus_pp=8.0, model_vs_consensus_pp=-1.0, kalshi_vs_model_pp=9.0,
        would_bet_at_base=True, would_bet_at_adjusted=True,
    )
    update_fill(
        decisions_path=path, decision_id=did,
        order_id="order_xyz", filled_at="2026-04-17T22:00:05Z",
        filled_price=filled_price, filled_size=filled_size,
    )
    return did


class TestComputeSettlement:
    def test_yes_won_bet_home_positive_pnl_and_home_win(self):
        from settle_decisions import compute_settlement

        r = compute_settlement(bet_side="home", filled_price=0.53, filled_size=10, yes_won=True)
        assert r["home_win"] is True
        assert r["realized_pnl"] == pytest.approx(10 * (1 - 0.53))

    def test_yes_lost_bet_home_negative_pnl_and_not_home_win(self):
        from settle_decisions import compute_settlement

        r = compute_settlement(bet_side="home", filled_price=0.53, filled_size=10, yes_won=False)
        assert r["home_win"] is False
        assert r["realized_pnl"] == pytest.approx(-10 * 0.53)

    def test_yes_won_bet_away_positive_pnl_and_not_home_win(self):
        from settle_decisions import compute_settlement

        r = compute_settlement(bet_side="away", filled_price=0.45, filled_size=5, yes_won=True)
        assert r["home_win"] is False
        assert r["realized_pnl"] == pytest.approx(5 * (1 - 0.45))

    def test_yes_lost_bet_away_negative_pnl_and_home_win(self):
        from settle_decisions import compute_settlement

        r = compute_settlement(bet_side="away", filled_price=0.45, filled_size=5, yes_won=False)
        assert r["home_win"] is True
        assert r["realized_pnl"] == pytest.approx(-5 * 0.45)


class TestFetchYesWon:
    def test_returns_true_when_result_yes(self):
        from settle_decisions import fetch_yes_won

        client = MagicMock()
        client.get_market.return_value = {"market": {"status": "settled", "result": "yes"}}
        assert fetch_yes_won(client, "T") is True

    def test_returns_false_when_result_no(self):
        from settle_decisions import fetch_yes_won

        client = MagicMock()
        client.get_market.return_value = {"market": {"status": "settled", "result": "no"}}
        assert fetch_yes_won(client, "T") is False

    def test_returns_none_when_market_not_settled(self):
        from settle_decisions import fetch_yes_won

        client = MagicMock()
        client.get_market.return_value = {"market": {"status": "active"}}
        assert fetch_yes_won(client, "T") is None

    def test_returns_none_when_api_raises(self):
        from settle_decisions import fetch_yes_won

        client = MagicMock()
        client.get_market.side_effect = Exception("network")
        assert fetch_yes_won(client, "T") is None


class TestSettlePending:
    def test_settles_only_filled_unsettled_rows(self, tmp_path):
        from bet_decisions import update_settlement
        from settle_decisions import settle_pending

        path = tmp_path / "bet_decisions.csv"
        did_filled = _seed_filled(path)

        client = MagicMock()
        client.get_market.return_value = {"market": {"status": "settled", "result": "yes"}}

        counts = settle_pending(path, client)
        assert counts["settled"] == 1
        assert counts["skipped"] == 0

        rows = list(csv.DictReader(path.open()))
        row = next(r for r in rows if r["decision_id"] == did_filled)
        assert row["settled_at"] != ""
        assert row["home_win"] == "True"
        assert float(row["realized_pnl"]) == pytest.approx(10 * (1 - 0.53))

    def test_skips_unfilled_rows(self, tmp_path):
        from bet_decisions import record_decision
        from settle_decisions import settle_pending

        path = tmp_path / "bet_decisions.csv"
        record_decision(
            decisions_path=path,
            ticker="KXNBAGAME-26APR17BOSMIA-BOS",
            home_team="BOS", away_team="MIA",
            bet_side="home", contract_side="YES",
            model_prob_home=0.60, model_prob_away=0.40,
            kalshi_yes_bid=0.51, kalshi_yes_ask=0.53, kalshi_yes_mid=0.52,
            entry_limit_price=0.53,
            edge_pp=7.0, ev=0.07,
            signal_threshold=0.03, base_threshold=0.03, recommended_stake=25.0,
            triangulation_tier=2,
            kalshi_vs_consensus_pp=8.0, model_vs_consensus_pp=-1.0, kalshi_vs_model_pp=9.0,
            would_bet_at_base=True, would_bet_at_adjusted=True,
        )

        client = MagicMock()
        client.get_market.return_value = {"market": {"result": "yes"}}

        counts = settle_pending(path, client)
        assert counts["settled"] == 0
        # no filled_at → not even scanned
        assert counts["scanned"] == 0

    def test_skips_already_settled_rows(self, tmp_path):
        from bet_decisions import update_settlement
        from settle_decisions import settle_pending

        path = tmp_path / "bet_decisions.csv"
        did = _seed_filled(path)
        update_settlement(
            decisions_path=path, decision_id=did,
            settled_at="2026-04-18T03:30:00Z", home_win=True,
            realized_pnl=4.7, clv_pp=0.0,
        )

        client = MagicMock()
        # If we accidentally re-settle, we'd see Kalshi called; fail loudly.
        client.get_market.side_effect = AssertionError("should not query Kalshi for settled rows")

        counts = settle_pending(path, client)
        assert counts["settled"] == 0
        assert counts["scanned"] == 0

    def test_skips_when_market_not_yet_settled(self, tmp_path):
        from settle_decisions import settle_pending

        path = tmp_path / "bet_decisions.csv"
        _seed_filled(path)

        client = MagicMock()
        client.get_market.return_value = {"market": {"status": "active"}}

        counts = settle_pending(path, client)
        assert counts["settled"] == 0
        assert counts["skipped"] == 1
        assert counts["scanned"] == 1

        row = list(csv.DictReader(path.open()))[0]
        assert row["settled_at"] == ""

    def test_caches_ticker_lookups(self, tmp_path):
        """Two decisions on the same ticker → one Kalshi call."""
        from settle_decisions import settle_pending

        path = tmp_path / "bet_decisions.csv"
        _seed_filled(path)
        _seed_filled(path)

        client = MagicMock()
        client.get_market.return_value = {"market": {"status": "settled", "result": "yes"}}

        settle_pending(path, client)
        assert client.get_market.call_count == 1

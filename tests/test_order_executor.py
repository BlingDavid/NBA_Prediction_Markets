"""
Tests for order_executor.place_order's ledger integration.

Focuses on the Step-3 wiring: when a decision_id + decisions_path are
supplied, a successful order placement calls bet_decisions.update_fill
with the right fields (order_id, filled_price in dollars, filled_size).
"""

import csv
from unittest.mock import MagicMock

import pytest


def _seed_decision(path):
    from bet_decisions import record_decision

    return record_decision(
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


def _mock_client(order_response=None):
    client = MagicMock()
    client.get_balance.return_value = {"balance": 100_000}  # $1,000 in cents
    client.create_order.return_value = order_response or {
        "order": {
            "order_id": "order_xyz",
            "status": "resting",
            "filled_count": 10,
            "remaining_count": 0,
        }
    }
    return client


class TestPlaceOrderLedgerIntegration:
    def test_updates_fill_when_decision_id_provided(self, tmp_path):
        from order_executor import place_order

        path = tmp_path / "bet_decisions.csv"
        did = _seed_decision(path)
        client = _mock_client()

        place_order(
            client=client,
            ticker="KXNBAGAME-26APR17BOSMIA-BOS",
            action="buy",
            side="yes",
            price_cents=53,
            quantity=10,
            skip_confirm=True,
            decision_id=did,
            decisions_path=path,
        )

        row = list(csv.DictReader(path.open()))[0]
        assert row["order_id"] == "order_xyz"
        # price_cents / 100 → dollars
        assert row["filled_price"] == "0.53"
        assert row["filled_size"] == "10.0"
        assert row["filled_at"] != ""

    def test_skips_update_when_no_decision_id(self, tmp_path):
        from order_executor import place_order

        path = tmp_path / "bet_decisions.csv"
        did = _seed_decision(path)
        client = _mock_client()

        place_order(
            client=client,
            ticker="KXNBAGAME-26APR17BOSMIA-BOS",
            action="buy",
            side="yes",
            price_cents=53,
            quantity=10,
            skip_confirm=True,
        )

        row = list(csv.DictReader(path.open()))[0]
        assert row["order_id"] == ""
        assert row["filled_price"] == ""

    def test_update_fill_miss_does_not_raise(self, tmp_path, capsys):
        """If decision_id isn't in the ledger, the order still succeeds
        and we surface a warning rather than crashing."""
        from order_executor import place_order

        path = tmp_path / "bet_decisions.csv"
        _seed_decision(path)  # seed a different decision_id
        client = _mock_client()

        result = place_order(
            client=client,
            ticker="KXNBAGAME-26APR17BOSMIA-BOS",
            action="buy",
            side="yes",
            price_cents=53,
            quantity=10,
            skip_confirm=True,
            decision_id="does_not_exist",
            decisions_path=path,
        )

        # order still placed
        assert result is not None
        # warning visible to user
        out = capsys.readouterr().out
        assert "decision_id" in out.lower() or "ledger" in out.lower()

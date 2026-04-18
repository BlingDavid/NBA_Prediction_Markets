"""
Tests for bet_decisions.py — the unified bet-lifecycle ledger.

Covers:
  * decision_id generation (format + uniqueness)
  * record_decision (header-on-first-call, append, returns id)
  * update_fill (match by decision_id, returns True/False)
  * update_settlement (match by decision_id, returns True/False)
  * round-trip: record → update_fill → update_settlement → read back
"""

import csv
from datetime import datetime, timezone

import pytest


class TestMakeDecisionId:
    def test_format_is_ticker_underscore_timestamp(self):
        from bet_decisions import make_decision_id

        ts = datetime(2026, 4, 17, 22, 0, 47, 123456, tzinfo=timezone.utc)
        did = make_decision_id("KXNBAGAME-26APR17BOSMIA-BOS", ts)
        assert did == "KXNBAGAME-26APR17BOSMIA-BOS_20260417T220047123456Z"

    def test_two_ids_one_microsecond_apart_are_distinct(self):
        from bet_decisions import make_decision_id

        ts1 = datetime(2026, 4, 17, 22, 0, 0, 1, tzinfo=timezone.utc)
        ts2 = datetime(2026, 4, 17, 22, 0, 0, 2, tzinfo=timezone.utc)
        assert make_decision_id("T", ts1) != make_decision_id("T", ts2)


class TestRecordDecision:
    def _base_row(self):
        return dict(
            decided_at="2026-04-17T22:00:00Z",
            ticker="KXNBAGAME-26APR17BOSMIA-BOS",
            home_team="BOS",
            away_team="MIA",
            bet_side="home",
            contract_side="YES",
            model_prob_home=0.60,
            model_prob_away=0.40,
            kalshi_yes_bid=0.51,
            kalshi_yes_ask=0.53,
            kalshi_yes_mid=0.52,
            entry_limit_price=0.53,
            edge_pp=7.0,
            ev=0.07,
            signal_threshold=0.03,
            base_threshold=0.03,
            recommended_stake=25.0,
            triangulation_tier=2,
            kalshi_vs_consensus_pp=8.0,
            model_vs_consensus_pp=-1.0,
            kalshi_vs_model_pp=9.0,
            would_bet_at_base=True,
            would_bet_at_adjusted=True,
        )

    def test_creates_file_with_full_header_on_first_call(self, tmp_path):
        from bet_decisions import record_decision, DECISION_COLUMNS

        path = tmp_path / "bet_decisions.csv"
        did = record_decision(decisions_path=path, **self._base_row())

        assert path.exists()
        assert isinstance(did, str) and did.startswith("KXNBAGAME-26APR17BOSMIA-BOS_")

        lines = path.read_text().splitlines()
        header = lines[0].split(",")
        assert header == DECISION_COLUMNS

    def test_fill_and_settlement_columns_are_blank_on_record(self, tmp_path):
        from bet_decisions import record_decision

        path = tmp_path / "bet_decisions.csv"
        record_decision(decisions_path=path, **self._base_row())

        row = list(csv.DictReader(path.open()))[0]
        for blank_col in (
            "order_id", "filled_at", "filled_price", "filled_size",
            "settled_at", "home_win", "realized_pnl", "clv_pp",
        ):
            assert row[blank_col] == "", f"{blank_col} should be blank on record, got {row[blank_col]!r}"

    def test_appends_across_calls(self, tmp_path):
        from bet_decisions import record_decision

        path = tmp_path / "bet_decisions.csv"
        for _ in range(3):
            record_decision(decisions_path=path, **self._base_row())

        lines = path.read_text().splitlines()
        # 1 header + 3 rows
        assert len(lines) == 4

    def test_returns_unique_decision_ids(self, tmp_path):
        from bet_decisions import record_decision

        path = tmp_path / "bet_decisions.csv"
        row = self._base_row()
        # Drop the fixed decided_at so record_decision stamps each call with
        # its own microsecond-precise now() — this is the realistic caller path.
        row.pop("decided_at")
        ids = [record_decision(decisions_path=path, **row) for _ in range(5)]
        assert len(set(ids)) == 5


class TestUpdateFill:
    def _seed(self, tmp_path):
        from bet_decisions import record_decision

        path = tmp_path / "bet_decisions.csv"
        did = record_decision(
            decisions_path=path,
            decided_at="2026-04-17T22:00:00Z",
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
        return path, did

    def test_updates_fill_fields_when_decision_id_found(self, tmp_path):
        from bet_decisions import update_fill

        path, did = self._seed(tmp_path)
        ok = update_fill(
            decisions_path=path,
            decision_id=did,
            order_id="order_xyz",
            filled_at="2026-04-17T22:00:05Z",
            filled_price=0.53,
            filled_size=47.0,
        )
        assert ok is True

        row = list(csv.DictReader(path.open()))[0]
        assert row["order_id"] == "order_xyz"
        assert row["filled_at"] == "2026-04-17T22:00:05Z"
        assert row["filled_price"] == "0.53"
        assert row["filled_size"] == "47.0"

    def test_returns_false_when_decision_id_not_found(self, tmp_path):
        from bet_decisions import update_fill

        path, _ = self._seed(tmp_path)
        ok = update_fill(
            decisions_path=path,
            decision_id="does_not_exist",
            order_id="order_xyz",
            filled_at="2026-04-17T22:00:05Z",
            filled_price=0.53,
            filled_size=47.0,
        )
        assert ok is False


class TestUpdateSettlement:
    def _seed(self, tmp_path):
        from bet_decisions import record_decision, update_fill

        path = tmp_path / "bet_decisions.csv"
        did = record_decision(
            decisions_path=path,
            decided_at="2026-04-17T22:00:00Z",
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
        update_fill(
            decisions_path=path, decision_id=did,
            order_id="order_xyz", filled_at="2026-04-17T22:00:05Z",
            filled_price=0.53, filled_size=47.0,
        )
        return path, did

    def test_updates_settlement_fields(self, tmp_path):
        from bet_decisions import update_settlement

        path, did = self._seed(tmp_path)
        ok = update_settlement(
            decisions_path=path,
            decision_id=did,
            settled_at="2026-04-18T03:30:00Z",
            home_win=True,
            realized_pnl=22.14,
            clv_pp=1.5,
        )
        assert ok is True

        row = list(csv.DictReader(path.open()))[0]
        assert row["settled_at"] == "2026-04-18T03:30:00Z"
        assert row["home_win"] == "True"
        assert row["realized_pnl"] == "22.14"
        assert row["clv_pp"] == "1.5"

    def test_returns_false_when_decision_id_not_found(self, tmp_path):
        from bet_decisions import update_settlement

        path, _ = self._seed(tmp_path)
        ok = update_settlement(
            decisions_path=path,
            decision_id="does_not_exist",
            settled_at="2026-04-18T03:30:00Z",
            home_win=True,
            realized_pnl=0.0,
            clv_pp=0.0,
        )
        assert ok is False


class TestRoundTrip:
    def test_record_fill_settle_then_read_back_all_fields(self, tmp_path):
        from bet_decisions import record_decision, update_fill, update_settlement

        path = tmp_path / "bet_decisions.csv"
        did = record_decision(
            decisions_path=path,
            decided_at="2026-04-17T22:00:00Z",
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
        assert update_fill(
            decisions_path=path, decision_id=did,
            order_id="order_xyz", filled_at="2026-04-17T22:00:05Z",
            filled_price=0.53, filled_size=47.0,
        )
        assert update_settlement(
            decisions_path=path, decision_id=did,
            settled_at="2026-04-18T03:30:00Z", home_win=True,
            realized_pnl=22.14, clv_pp=1.5,
        )

        row = list(csv.DictReader(path.open()))[0]
        assert row["decision_id"] == did
        assert row["ticker"] == "KXNBAGAME-26APR17BOSMIA-BOS"
        assert row["triangulation_tier"] == "2"
        assert row["entry_limit_price"] == "0.53"
        assert row["filled_price"] == "0.53"
        assert row["home_win"] == "True"
        assert row["realized_pnl"] == "22.14"

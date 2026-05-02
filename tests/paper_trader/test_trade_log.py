"""trade_log: atomic JSON state, JSONL event append, CSV append.

The append-only files are the source of truth; the JSON state is a cached
snapshot.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import pytest

from paper_trader.trade_log import (
    write_state,
    read_state,
    emit_event,
    record_trade,
    CLOSED_TRADES_HEADER,
)


def test_write_state_atomic(tmp_path: Path):
    target = tmp_path / "trader_state.json"
    state = {"mode": "A", "trades_closed": 0, "open_positions": []}
    write_state(target, state)
    assert json.loads(target.read_text()) == state
    # No tmp file left behind:
    assert not target.with_suffix(".json.tmp").exists()


def test_read_state_returns_none_when_missing(tmp_path: Path):
    assert read_state(tmp_path / "missing.json") is None


def test_read_state_roundtrip(tmp_path: Path):
    target = tmp_path / "s.json"
    write_state(target, {"k": 1})
    assert read_state(target) == {"k": 1}


def test_emit_event_appends_jsonl(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    emit_event(log, {"event_type": "tick", "ts_utc": "2026-05-01T00:00:00Z"})
    emit_event(log, {"event_type": "entry", "ts_utc": "2026-05-01T00:00:01Z"})
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event_type"] == "tick"
    assert json.loads(lines[1])["event_type"] == "entry"


def test_record_trade_writes_header_once(tmp_path: Path):
    csv_path = tmp_path / "closed_trades.csv"
    row = {col: "" for col in CLOSED_TRADES_HEADER}
    row["trade_id"] = "T1"
    record_trade(csv_path, row)
    record_trade(csv_path, {**row, "trade_id": "T2"})
    text = csv_path.read_text()
    assert text.count("trade_id,") == 1  # header only once
    assert "T1" in text and "T2" in text


def _writer(args):
    csv_path, trade_id = args
    row = {col: "" for col in CLOSED_TRADES_HEADER}
    row["trade_id"] = trade_id
    record_trade(csv_path, row)


def test_record_trade_no_torn_writes_under_concurrency(tmp_path: Path):
    csv_path = tmp_path / "closed_trades.csv"
    # Pre-create header to avoid two writers both creating it.
    record_trade(csv_path, {col: "init" for col in CLOSED_TRADES_HEADER})
    args = [(csv_path, f"T{i}") for i in range(20)]
    with mp.Pool(4) as pool:
        pool.map(_writer, args)
    lines = csv_path.read_text().splitlines()
    # 1 header + 1 init row + 20 concurrent rows
    assert len(lines) == 22
    # Each line must have exactly the right number of fields.
    expected_fields = len(CLOSED_TRADES_HEADER)
    for line in lines:
        assert line.count(",") == expected_fields - 1

"""Disk-side persistence for the paper trader.

Three sinks:
  - trader_state.json   atomic JSON, single source of truth for live state.
  - trader_log.jsonl    append-only event stream, the replay source.
  - closed_trades.csv   append-only closed-position rows.
open_positions.csv is a derived mirror of state.open_positions and is
overwritten atomically each time state is written (handled by callers via
write_state -> mirror_open_positions).
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable


CLOSED_TRADES_HEADER: list[str] = [
    "trade_id", "ticker", "game_key", "home_team", "away_team",
    "entry_captured_at", "exit_captured_at", "hold_seconds",
    "entry_yes_ask", "entry_yes_mid", "exit_yes_bid", "exit_yes_mid",
    "contracts",
    "p_raw", "p_calibrated",
    "trigger_mode",
    "expected_pnl_per_contract",
    "realized_label_home_up_5m",
    "pnl_mid", "pnl_realistic", "fees",
    "exit_basis",
    "bankroll_at_entry",
    "notes",
]


def write_state(path: Path, state: dict[str, Any]) -> None:
    """Atomic write: serialize → write to .tmp → os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(state, indent=2, sort_keys=True, default=str)
    tmp.write_text(payload)
    os.replace(tmp, path)


def read_state(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def emit_event(path: Path, event: dict[str, Any]) -> None:
    """Append one JSON-encoded line to the event log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, default=str, sort_keys=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def record_trade(path: Path, row: dict[str, Any]) -> None:
    """Append one row to closed_trades.csv. Writes header if file is new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CLOSED_TRADES_HEADER, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def mirror_open_positions(path: Path, open_positions: Iterable[dict[str, Any]]) -> None:
    """Overwrite open_positions.csv atomically from in-memory state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    rows = list(open_positions)
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            fh.write("trade_id\n")  # header-only sentinel
    os.replace(tmp, path)

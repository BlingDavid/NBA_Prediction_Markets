"""In-memory position state + open/close lifecycle.

Owns the writes to trader_state.json, open_positions.csv, closed_trades.csv,
and the entry/exit JSONL events. Pure functions live in cost_model and
trigger; this module is where state changes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from paper_trader import trade_log
from paper_trader.cost_model import winner_fee_per_contract


HOLD_SECONDS = 5 * 60
MAX_OPEN = 20
KELLY_CAP_FRACTION = 0.05


def kelly_contracts(mu: float, var: float, bankroll: float, yes_ask: float) -> int:
    """Continuous-payoff Kelly: f = mu / sigma^2, capped at 5% of bankroll, floored at 1.

    Returns integer contract count. yes_ask is the entry price (bankroll
    consumed = contracts * yes_ask).
    """
    cap_dollars = KELLY_CAP_FRACTION * float(bankroll)
    if var <= 0.0:
        # Degenerate variance: spend the cap.
        dollars = cap_dollars
    else:
        f = float(mu) / float(var)
        f = max(0.0, min(f, KELLY_CAP_FRACTION))  # cap at 5% of bankroll
        dollars = f * float(bankroll)
    contracts = int(dollars / float(yes_ask))
    return max(contracts, 1)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


class PositionManager:
    def __init__(
        self,
        state_path: Path,
        log_path: Path,
        closed_csv_path: Path,
        open_csv_path: Path,
        bankroll_initial: float,
        mode_sizing: str,
        live_or_paper: str,
    ):
        self.state_path = Path(state_path)
        self.log_path = Path(log_path)
        self.closed_csv_path = Path(closed_csv_path)
        self.open_csv_path = Path(open_csv_path)
        self.bankroll_initial = float(bankroll_initial)
        self.mode_sizing = mode_sizing
        self.live_or_paper = live_or_paper
        self.cumulative_pnl_mid = 0.0
        self.cumulative_pnl_realistic = 0.0
        self.trades_closed = 0
        self.max_drawdown = 0.0
        self.drawdown_breached = False
        self.open_positions: list[dict[str, Any]] = []
        self.last_processed_captured_at_per_ticker: dict[str, str] = {}
        self.last_closed_trade: dict[str, Any] | None = None

    @property
    def bankroll_current(self) -> float:
        return self.bankroll_initial + self.cumulative_pnl_realistic

    def load_state(self) -> None:
        state = trade_log.read_state(self.state_path)
        if state is None:
            return
        self.cumulative_pnl_mid = float(state.get("cumulative_pnl_mid", 0.0))
        self.cumulative_pnl_realistic = float(state.get("cumulative_pnl_realistic", 0.0))
        self.trades_closed = int(state.get("trades_closed", 0))
        self.max_drawdown = float(state.get("max_drawdown", 0.0))
        self.drawdown_breached = bool(state.get("drawdown_breached", False))
        self.open_positions = list(state.get("open_positions", []))
        self.last_processed_captured_at_per_ticker = dict(
            state.get("last_processed_captured_at_per_ticker", {})
        )

    def write_state(self) -> None:
        state = {
            "mode": self.mode_sizing,
            "live_or_paper": self.live_or_paper,
            "bankroll_initial": self.bankroll_initial,
            "cumulative_pnl_realistic": self.cumulative_pnl_realistic,
            "cumulative_pnl_mid": self.cumulative_pnl_mid,
            "trades_closed": self.trades_closed,
            "max_drawdown": self.max_drawdown,
            "drawdown_breached": self.drawdown_breached,
            "open_positions": self.open_positions,
            "last_processed_captured_at_per_ticker": self.last_processed_captured_at_per_ticker,
            "last_tick_at": _now_iso(),
        }
        trade_log.write_state(self.state_path, state)
        trade_log.mirror_open_positions(self.open_csv_path, self.open_positions)

    def open_position(
        self,
        ticker: str,
        game_key: str,
        home_team: str,
        away_team: str,
        captured_at: str,
        yes_ask: float,
        yes_mid: float,
        yes_bid: float,
        contracts: int,
        p_raw: float,
        p_calibrated: float,
        expected_pnl_per_contract: float,
    ) -> bool:
        if len(self.open_positions) >= MAX_OPEN:
            trade_log.emit_event(self.log_path, {
                "event_type": "signal", "ts_utc": _now_iso(),
                "captured_at": captured_at, "ticker": ticker,
                "accept": False, "reason": "concurrency_cap",
            })
            return False
        if self.live_or_paper == "live" and self.drawdown_breached:
            trade_log.emit_event(self.log_path, {
                "event_type": "halt", "ts_utc": _now_iso(),
                "captured_at": captured_at, "ticker": ticker,
                "reason": "drawdown_halt_live",
            })
            return False
        trade_id = f"{ticker}_{captured_at}"
        position = {
            "trade_id": trade_id,
            "ticker": ticker, "game_key": game_key,
            "home_team": home_team, "away_team": away_team,
            "entry_captured_at": captured_at,
            "entry_yes_ask": float(yes_ask),
            "entry_yes_mid": float(yes_mid),
            "entry_yes_bid": float(yes_bid),
            "contracts": int(contracts),
            "p_raw": float(p_raw),
            "p_calibrated": float(p_calibrated),
            "expected_pnl_per_contract": float(expected_pnl_per_contract),
            "bankroll_at_entry": self.bankroll_current,
        }
        self.open_positions.append(position)
        trade_log.emit_event(self.log_path, {
            "event_type": "entry", "ts_utc": _now_iso(),
            "captured_at": captured_at, **position,
        })
        return True

    def tick_open_positions(
        self,
        now_captured_at: str,
        latest_bid_mid_per_ticker: dict[str, tuple[float, float]],
        exit_basis: str = "5min_timer",
    ) -> None:
        if not self.open_positions:
            return
        now = _parse_iso(now_captured_at)
        still_open: list[dict[str, Any]] = []
        for pos in self.open_positions:
            entry = _parse_iso(pos["entry_captured_at"])
            elapsed = (now - entry).total_seconds()
            if elapsed < HOLD_SECONDS:
                still_open.append(pos)
                continue
            quote = latest_bid_mid_per_ticker.get(pos["ticker"])
            if quote is None:
                # No fresh quote for this ticker yet — keep waiting.
                still_open.append(pos)
                continue
            exit_yes_bid, exit_yes_mid = quote
            self._close_position(pos, now_captured_at, exit_yes_bid, exit_yes_mid, exit_basis)
        self.open_positions = still_open

    def _close_position(
        self,
        pos: dict[str, Any],
        exit_captured_at: str,
        exit_yes_bid: float,
        exit_yes_mid: float,
        exit_basis: str,
    ) -> None:
        contracts = int(pos["contracts"])
        entry_ask = float(pos["entry_yes_ask"])
        entry_mid = float(pos["entry_yes_mid"])
        pnl_mid = (float(exit_yes_mid) - entry_mid) * contracts
        gross_realistic = (float(exit_yes_bid) - entry_ask) * contracts
        realized_rise = 1 if float(exit_yes_mid) > entry_mid else 0
        # Per spec: loser fee = 0; winner pays Kalshi fee at settlement.
        fees = winner_fee_per_contract(pos["p_calibrated"]) * contracts if realized_rise == 1 else 0.0
        pnl_realistic = gross_realistic - fees

        hold_seconds = (_parse_iso(exit_captured_at) - _parse_iso(pos["entry_captured_at"])).total_seconds()
        row = {
            "trade_id": pos["trade_id"],
            "ticker": pos["ticker"], "game_key": pos["game_key"],
            "home_team": pos["home_team"], "away_team": pos["away_team"],
            "entry_captured_at": pos["entry_captured_at"],
            "exit_captured_at": exit_captured_at,
            "hold_seconds": hold_seconds,
            "entry_yes_ask": entry_ask,
            "entry_yes_mid": entry_mid,
            "exit_yes_bid": float(exit_yes_bid),
            "exit_yes_mid": float(exit_yes_mid),
            "contracts": contracts,
            "p_raw": pos["p_raw"],
            "p_calibrated": pos["p_calibrated"],
            "trigger_mode": self.mode_sizing,
            "expected_pnl_per_contract": pos["expected_pnl_per_contract"],
            "realized_label_home_up_5m": realized_rise,
            "pnl_mid": pnl_mid,
            "pnl_realistic": pnl_realistic,
            "fees": fees,
            "exit_basis": exit_basis,
            "bankroll_at_entry": pos["bankroll_at_entry"],
            "notes": "",
        }
        trade_log.record_trade(self.closed_csv_path, row)
        trade_log.emit_event(self.log_path, {
            "event_type": "exit", "ts_utc": _now_iso(),
            "captured_at": exit_captured_at, **row,
        })
        self.cumulative_pnl_mid += pnl_mid
        self.cumulative_pnl_realistic += pnl_realistic
        self.trades_closed += 1
        # Drawdown tracking (max peak-to-trough on cumulative_pnl_realistic).
        if self.cumulative_pnl_realistic < -self.max_drawdown:
            self.max_drawdown = -self.cumulative_pnl_realistic
        self.last_closed_trade = row

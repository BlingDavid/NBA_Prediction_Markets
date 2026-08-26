"""Edge-to-Action paper trader entry point.

CLI:
    python run_paper_trader.py --mode paper --bankroll 1000 [--interval 15]

Run forever, polling data/live/features/latest_features.csv every --interval
seconds, scoring eligible rows, opening EV-positive positions, and exiting
each on a hard 5-minute timer keyed off captured_at.

State and logs land in --output-dir (default: outputs/paper_trades/).
"""
from __future__ import annotations

import argparse
import math
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paper_trader import gate_filters, scorer, trigger, trade_log
from paper_trader.feed_reader import read_unprocessed
from paper_trader.move_size_estimator import (
    Store, expected_moves, load_pooled_means,
)
from paper_trader.position_manager import (
    HOLD_SECONDS, MAX_OPEN, PositionManager, kelly_contracts,
)
from paper_trader.cost_model import winner_fee_per_contract


DRAWDOWN_GATE_DOLLARS = 100.0
STALE_WARN_SECONDS = 300
STALE_HALT_SECONDS = 600
LIVE_PROMOTION_TRADES = 1000
LIVE_PROMOTION_SHARPE = 1.0
LIVE_PROMOTION_MAX_DD_FRACTION = 0.10


def replay_from_log(log_path: Path, output_dir: Path, bankroll_initial: float) -> None:
    """Reconstruct closed_trades.csv and trader_state.json from trader_log.jsonl.

    Used by the disaster-recovery story and the replay invariant test.
    Reads only `entry`/`exit`/`tick` events (the rest are diagnostic).
    """
    import json as _json
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    closed_csv = output_dir / "closed_trades.csv"
    state_path = output_dir / "trader_state.json"
    open_csv = output_dir / "open_positions.csv"
    if closed_csv.exists():
        closed_csv.unlink()
    pm = PositionManager(
        state_path=state_path, log_path=output_dir / "_replay_unused.jsonl",
        closed_csv_path=closed_csv, open_csv_path=open_csv,
        bankroll_initial=bankroll_initial, mode_sizing="A",
        live_or_paper="paper",
    )
    open_by_id: dict[str, dict[str, Any]] = {}
    with Path(log_path).open() as fh:
        for line in fh:
            event = _json.loads(line)
            etype = event.get("event_type")
            if etype == "entry":
                position = {k: event[k] for k in (
                    "trade_id", "ticker", "game_key", "home_team", "away_team",
                    "entry_captured_at", "entry_yes_ask", "entry_yes_mid",
                    "entry_yes_bid", "contracts", "p_raw", "p_calibrated",
                    "expected_pnl_per_contract", "bankroll_at_entry",
                )}
                open_by_id[event["trade_id"]] = position
                pm.open_positions.append(position)
            elif etype == "exit":
                tid = event["trade_id"]
                if tid in open_by_id:
                    pos = open_by_id.pop(tid)
                    pm.open_positions = [p for p in pm.open_positions if p["trade_id"] != tid]
                    pm._close_position(
                        pos=pos,
                        exit_captured_at=event["exit_captured_at"],
                        exit_yes_bid=event["exit_yes_bid"],
                        exit_yes_mid=event["exit_yes_mid"],
                        exit_basis=event["exit_basis"],
                    )
    pm.write_state()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EngineContext:
    bundle: dict[str, Any]
    pooled_store: Store
    feed_path: Path
    output_dir: Path
    state_path: Path
    log_path: Path
    closed_csv_path: Path
    open_csv_path: Path
    position_manager: PositionManager
    mode_sizing: str
    live_or_paper: str
    bankroll_initial: float
    interval_seconds: int = 15


def build_engine_context(
    model_path: Path,
    pooled_path: Path,
    feed_path: Path,
    output_dir: Path,
    bankroll: float,
    mode_sizing: str,
    live_or_paper: str,
    interval_seconds: int = 15,
) -> EngineContext:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = scorer.load_model(model_path)
    pooled_store = load_pooled_means(pooled_path)
    state_path = output_dir / "trader_state.json"
    log_path = output_dir / "trader_log.jsonl"
    closed_csv_path = output_dir / "closed_trades.csv"
    open_csv_path = output_dir / "open_positions.csv"
    pm = PositionManager(
        state_path=state_path, log_path=log_path,
        closed_csv_path=closed_csv_path, open_csv_path=open_csv_path,
        bankroll_initial=bankroll, mode_sizing=mode_sizing,
        live_or_paper=live_or_paper,
    )
    pm.load_state()
    return EngineContext(
        bundle=bundle, pooled_store=pooled_store,
        feed_path=Path(feed_path), output_dir=output_dir,
        state_path=state_path, log_path=log_path,
        closed_csv_path=closed_csv_path, open_csv_path=open_csv_path,
        position_manager=pm, mode_sizing=mode_sizing,
        live_or_paper=live_or_paper, bankroll_initial=bankroll,
        interval_seconds=interval_seconds,
    )


def check_live_boot_gates(output_dir: Path, bankroll_initial: float) -> tuple[bool, list[str]]:
    """The four gates per spec §Risk controls / Mode transitions."""
    output_dir = Path(output_dir)
    closed_csv = output_dir / "closed_trades.csv"
    reasons: list[str] = []
    if not closed_csv.exists():
        return False, ["trades_closed=0 (need 1000)"]
    df = pd.read_csv(closed_csv)
    n = len(df)
    if n < LIVE_PROMOTION_TRADES:
        reasons.append(f"trades_closed={n} (need {LIVE_PROMOTION_TRADES})")
    pnl = df["pnl_realistic"].astype(float)
    if len(pnl) > 1 and pnl.std(ddof=0) > 0:
        sharpe = float(pnl.mean() / pnl.std(ddof=0))
    else:
        sharpe = 0.0
    if sharpe <= LIVE_PROMOTION_SHARPE:
        reasons.append(f"sharpe={sharpe:.3f} (need > {LIVE_PROMOTION_SHARPE})")
    cum = pnl.cumsum()
    max_dd = float(-cum.min()) if (cum < 0).any() else 0.0
    max_dd_fraction = max_dd / bankroll_initial if bankroll_initial > 0 else math.inf
    if max_dd_fraction >= LIVE_PROMOTION_MAX_DD_FRACTION:
        reasons.append(f"max_dd={max_dd_fraction:.3f} (need < {LIVE_PROMOTION_MAX_DD_FRACTION})")
    return (len(reasons) == 0), reasons


def _detect_stale(now_utc: datetime, latest_captured_at: datetime | None) -> str | None:
    if latest_captured_at is None:
        return None
    age = (now_utc - latest_captured_at).total_seconds()
    if age >= STALE_HALT_SECONDS:
        return "halt"
    if age >= STALE_WARN_SECONDS:
        return "warn"
    return None


def process_tick(ctx: EngineContext) -> None:
    tick_start = time.monotonic()
    now_utc = datetime.now(timezone.utc)
    pm = ctx.position_manager
    rows = read_unprocessed(ctx.feed_path, pm.last_processed_captured_at_per_ticker)

    latest_capture: datetime | None = None
    quote_table: dict[str, tuple[float, float]] = {}

    # First pass: collect latest_capture + quote_table from every row so we
    # can decide stale state and so existing positions can be exited even if
    # we end up halting new entries.
    if not rows.empty:
        for _, row in rows.iterrows():
            ticker = str(row["ticker"])
            pm.last_processed_captured_at_per_ticker[ticker] = str(row["captured_at"])
            try:
                quote_table[ticker] = (float(row["yes_bid"]), float(row["yes_mid"]))
            except (TypeError, ValueError):
                pass
            ts = pd.to_datetime(row["captured_at"], utc=True, errors="coerce")
            if pd.notna(ts) and (latest_capture is None or ts.to_pydatetime() > latest_capture):
                latest_capture = ts.to_pydatetime()

    stale = _detect_stale(now_utc, latest_capture)
    if stale == "warn":
        trade_log.emit_event(ctx.log_path, {
            "event_type": "halt", "ts_utc": _now_iso(),
            "reason": "stale_feed_warn", "latest_captured_at": str(latest_capture),
        })
    elif stale == "halt":
        trade_log.emit_event(ctx.log_path, {
            "event_type": "halt", "ts_utc": _now_iso(),
            "reason": "stale_feed_halt", "latest_captured_at": str(latest_capture),
        })

    # Second pass: evaluate gates/scoring/trigger only when feed is healthy.
    # On stale-halt we skip new entries per spec §Risk controls, but still
    # exit existing positions below.
    if not rows.empty and stale != "halt":
        for _, row in rows.iterrows():
            try:
                _process_one_row(ctx, row.to_dict())
            except Exception as exc:
                trade_log.emit_event(ctx.log_path, {
                    "event_type": "error", "ts_utc": _now_iso(),
                    "captured_at": str(row.get("captured_at")),
                    "ticker": str(row.get("ticker")),
                    "error": repr(exc), "trace": traceback.format_exc(),
                })
                continue

    # Exit any positions whose 5-min mark has passed. Per spec, this happens
    # regardless of stale state — existing positions are evaluated against
    # captured_at of whatever next valid row arrives.
    if latest_capture is not None:
        pm.tick_open_positions(now_captured_at=latest_capture.isoformat(),
                               latest_bid_mid_per_ticker=quote_table)

    # Drawdown gate (paper: log once + continue; live: also halt new entries).
    if pm.cumulative_pnl_realistic <= -DRAWDOWN_GATE_DOLLARS and not pm.drawdown_breached:
        pm.drawdown_breached = True
        trade_log.emit_event(ctx.log_path, {
            "event_type": "halt", "ts_utc": _now_iso(),
            "reason": "drawdown_breached",
            "cumulative_pnl_realistic": pm.cumulative_pnl_realistic,
        })

    pm.write_state()
    tick_to_decision_ms = (time.monotonic() - tick_start) * 1000
    trade_log.emit_event(ctx.log_path, {
        "event_type": "tick", "ts_utc": _now_iso(),
        "captured_at": str(latest_capture) if latest_capture else None,
        "tick_to_decision_ms": tick_to_decision_ms,
        "rows_processed": int(len(rows)),
        "open_positions": len(pm.open_positions),
    })


def _process_one_row(ctx: EngineContext, row: dict[str, Any]) -> None:
    """One row → gate → score → trigger → maybe-open."""
    pm = ctx.position_manager
    captured_at = str(row["captured_at"])
    ticker = str(row["ticker"])
    passed, reason = gate_filters.evaluate(row)
    if not passed:
        trade_log.emit_event(ctx.log_path, {
            "event_type": "gate_blocked", "ts_utc": _now_iso(),
            "captured_at": captured_at, "ticker": ticker, "reason": reason,
        })
        return

    df = pd.DataFrame([row])
    p_raw, p_cal = scorer.score(ctx.bundle, df)
    p_raw_v = float(p_raw[0])
    p_cal_v = float(p_cal[0])
    e_up, e_down, var = expected_moves(p_cal_v, ctx.mode_sizing, ctx.pooled_store)
    fee = winner_fee_per_contract(p_cal_v)
    accept, ev, trigger_reason = trigger.evaluate(
        p_calibrated=p_cal_v, e_up=e_up, e_down=e_down,
        yes_ask=float(row["yes_ask"]), yes_mid=float(row["yes_mid"]),
        fee_per_contract_if_win=fee,
    )
    trade_log.emit_event(ctx.log_path, {
        "event_type": "signal", "ts_utc": _now_iso(),
        "captured_at": captured_at, "ticker": ticker,
        "p_raw": p_raw_v, "p_calibrated": p_cal_v,
        "e_up": e_up, "e_down": e_down,
        "expected_pnl_per_contract": ev,
        "accept": accept, "reason": trigger_reason,
    })
    if not accept:
        return
    # Sizing: continuous Kelly on (μ=ev, σ²=var).
    mu = ev
    contracts = kelly_contracts(mu=mu, var=var, bankroll=pm.bankroll_current,
                                yes_ask=float(row["yes_ask"]))
    pm.open_position(
        ticker=ticker, game_key=str(row["game_key"]),
        home_team=str(row["home_team"]), away_team=str(row["away_team"]),
        captured_at=captured_at,
        yes_ask=float(row["yes_ask"]), yes_mid=float(row["yes_mid"]),
        yes_bid=float(row["yes_bid"]),
        contracts=contracts, p_raw=p_raw_v, p_calibrated=p_cal_v,
        expected_pnl_per_contract=ev,
    )


def _run_forever(ctx: EngineContext) -> None:
    while True:
        try:
            process_tick(ctx)
        except Exception as exc:
            trade_log.emit_event(ctx.log_path, {
                "event_type": "error", "ts_utc": _now_iso(),
                "error": repr(exc), "trace": traceback.format_exc(),
            })
            time.sleep(5)
            continue
        time.sleep(ctx.interval_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--features-path", type=Path,
                        default=Path("data/live/features/latest_features.csv"))
    parser.add_argument("--model-path", type=Path,
                        default=Path("models/live_home_up_5m_bootstrap.pkl"))
    parser.add_argument("--pooled-path", type=Path,
                        default=Path("outputs/paper_trades/pooled_means.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/paper_trades"))
    parser.add_argument("--mode-sizing", choices=["A", "B"], default="B")
    parser.add_argument("--once", action="store_true",
                        help="Run a single tick and exit (smoke / debug).")
    args = parser.parse_args(argv)

    if args.mode == "live":
        ok, reasons = check_live_boot_gates(args.output_dir, args.bankroll)
        if not ok:
            print("Refusing to start in live mode. Failing gates:")
            for r in reasons:
                print(f"  - {r}")
            return 2

    ctx = build_engine_context(
        model_path=args.model_path, pooled_path=args.pooled_path,
        feed_path=args.features_path, output_dir=args.output_dir,
        bankroll=args.bankroll, mode_sizing=args.mode_sizing,
        live_or_paper=args.mode, interval_seconds=args.interval,
    )

    if args.once:
        process_tick(ctx)
        return 0
    _run_forever(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

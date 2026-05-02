"""Read-only paper trader status. Safe under `watch -n 30 python tools/trader_status.py`."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _read_state(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _summary(closed: pd.DataFrame) -> dict:
    if closed.empty:
        return {"trades_closed": 0, "wins": 0, "win_rate": 0.0, "sharpe": 0.0}
    wins = int((closed["realized_label_home_up_5m"] == 1).sum())
    n = len(closed)
    pnl = closed["pnl_realistic"].astype(float)
    sharpe = float(pnl.mean() / pnl.std(ddof=0)) if pnl.std(ddof=0) > 0 else 0.0
    return {
        "trades_closed": n,
        "wins": wins,
        "win_rate": wins / n if n else 0.0,
        "sharpe": sharpe,
        "pnl_realistic": float(pnl.sum()),
        "pnl_mid": float(closed["pnl_mid"].astype(float).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_trades"))
    args = parser.parse_args(argv)
    state = _read_state(args.output_dir / "trader_state.json")
    closed_path = args.output_dir / "closed_trades.csv"
    closed = pd.read_csv(closed_path) if closed_path.exists() else pd.DataFrame()
    summary = _summary(closed)
    last_tick = state.get("last_tick_at")
    if last_tick:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(last_tick.replace("Z", "+00:00"))).total_seconds()
    else:
        age = float("inf")
    print(f"Mode             : {state.get('mode', '?')} (sizing)  {state.get('live_or_paper', '?')}")
    print(f"Bankroll         : ${state.get('bankroll_initial', 0):.2f}")
    print(f"Cumulative PnL   : realistic ${summary.get('pnl_realistic', 0):+.2f}  "
          f"|  mid ${summary.get('pnl_mid', 0):+.2f}  "
          f"|  alpha capture ${summary.get('pnl_mid', 0) - summary.get('pnl_realistic', 0):+.2f}")
    print(f"Trades closed    : {summary['trades_closed']}   "
          f"wins {summary['wins']} ({summary['win_rate']*100:.1f}%)   "
          f"sharpe(per-trade) {summary['sharpe']:.2f}")
    print(f"Open positions   : {len(state.get('open_positions', []))} / 20")
    print(f"Last tick        : {last_tick}  ({age:.0f}s ago)")
    print(f"Drawdown         : current ${state.get('cumulative_pnl_realistic', 0):+.2f}   "
          f"max ${-state.get('max_drawdown', 0):.2f}")
    if not closed.empty:
        print("Recent 5 trades  :")
        cols = ["trade_id", "pnl_realistic", "pnl_mid", "realized_label_home_up_5m", "exit_captured_at"]
        print(closed.tail(5)[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Nightly settlement closer for the bet_decisions ledger.

Scans bet_decisions.csv for rows that have been filled but not yet
settled, queries Kalshi for each market's final result, and writes
home_win + realized_pnl via bet_decisions.update_settlement.

Run once per night after all NBA games have resolved:
  python settle_decisions.py

PnL semantics: all decisions are YES-contract buys. For a fill at
filled_price (dollars, 0–1):
  * YES wins → pnl_per_contract = 1.0 - filled_price
  * NO wins  → pnl_per_contract = -filled_price
  * realized_pnl = filled_size * pnl_per_contract  (dollars)

home_win is derived from the bet_side column (which side the ticker
referred to) together with Kalshi's yes/no outcome.

clv_pp is written as 0.0 for now — computing it requires the final
closing line and is deferred to a follow-up.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from bet_decisions import update_settlement


def compute_settlement(
    *,
    bet_side: str,
    filled_price: float,
    filled_size: float,
    yes_won: bool,
) -> dict:
    pnl_per = (1.0 - filled_price) if yes_won else -filled_price
    realized_pnl = filled_size * pnl_per
    home_win = yes_won if bet_side == "home" else (not yes_won)
    return {"home_win": home_win, "realized_pnl": realized_pnl}


def fetch_yes_won(client, ticker: str) -> bool | None:
    """Return True if YES won, False if NO won, None if still open."""
    try:
        data = client.get_market(ticker)
    except Exception:
        return None
    market = data.get("market", data) if isinstance(data, dict) else {}
    result = str(market.get("result", "")).lower()
    if result == "yes":
        return True
    if result == "no":
        return False
    return None


def settle_pending(decisions_path, client) -> dict:
    path = Path(decisions_path)
    if not path.exists():
        return {"scanned": 0, "settled": 0, "skipped": 0}

    with path.open() as f:
        rows = list(csv.DictReader(f))

    ticker_cache: dict[str, bool | None] = {}
    scanned = 0
    settled = 0
    skipped = 0

    for row in rows:
        if not row.get("filled_at"):
            continue
        if row.get("settled_at"):
            continue
        if not row.get("decision_id"):
            continue
        scanned += 1

        ticker = row["ticker"]
        if ticker not in ticker_cache:
            ticker_cache[ticker] = fetch_yes_won(client, ticker)
        yes_won = ticker_cache[ticker]

        if yes_won is None:
            skipped += 1
            continue

        try:
            filled_price = float(row["filled_price"])
            filled_size = float(row["filled_size"])
        except (ValueError, KeyError):
            skipped += 1
            continue

        s = compute_settlement(
            bet_side=row["bet_side"],
            filled_price=filled_price,
            filled_size=filled_size,
            yes_won=yes_won,
        )

        update_settlement(
            decisions_path=path,
            decision_id=row["decision_id"],
            settled_at=datetime.now(tz=timezone.utc).isoformat(),
            home_win=s["home_win"],
            realized_pnl=s["realized_pnl"],
            clv_pp=0.0,
        )
        settled += 1

    return {"scanned": scanned, "settled": settled, "skipped": skipped}


def main():
    from config import OUTPUTS_DIR
    from kalshi_auth import KalshiAuthClient

    try:
        client = KalshiAuthClient()
    except (ValueError, FileNotFoundError) as e:
        print(f"  Authentication error: {e}")
        return

    path = OUTPUTS_DIR / "bet_decisions.csv"
    print(f"Scanning {path}...")
    counts = settle_pending(path, client)
    print(f"  Scanned: {counts['scanned']}  Settled: {counts['settled']}  Skipped: {counts['skipped']}")


if __name__ == "__main__":
    main()

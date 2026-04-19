"""
Order Executor — Manual Order Placement for Kalshi
────────────────────────────────────────────────────
Place, monitor, and cancel orders on Kalshi through the authenticated API.

Every action requires explicit confirmation before executing.

Usage:
  python order_executor.py balance                          # Check account balance
  python order_executor.py positions                        # View open positions
  python order_executor.py orders                           # View open orders
  python order_executor.py buy TICKER --side yes --price 45 --qty 10
  python order_executor.py sell TICKER --side yes --price 55 --qty 10
  python order_executor.py cancel ORDER_ID
  python order_executor.py cancel-all

Docs: https://docs.kalshi.com/getting_started/quick_start_create_order
"""

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from bet_decisions import update_fill
from kalshi_auth import KalshiAuthClient, KalshiAPIError
from config import BANKROLL, OUTPUTS_DIR, TEAM_ABBREV_MAP


# ═══════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════

def fmt_dollars(cents: int | float) -> str:
    """Convert cents to formatted dollar string."""
    return f"${cents / 100:,.2f}"


def print_header(title: str):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


# ═══════════════════════════════════════════════════════════════════════
# BALANCE
# ═══════════════════════════════════════════════════════════════════════

def show_balance(client: KalshiAuthClient):
    """Display account balance and portfolio value."""
    print_header("ACCOUNT BALANCE")

    try:
        data = client.get_balance()
    except KalshiAPIError as e:
        print(f"\n  ERROR: {e}")
        return

    balance = data.get("balance", 0)
    portfolio = data.get("portfolio_value", 0)
    total = balance + portfolio

    print(f"\n  Available cash:     {fmt_dollars(balance)}")
    print(f"  Portfolio value:    {fmt_dollars(portfolio)}")
    print(f"  ─────────────────────────")
    print(f"  Total equity:       {fmt_dollars(total)}")
    print()


# ═══════════════════════════════════════════════════════════════════════
# POSITIONS
# ═══════════════════════════════════════════════════════════════════════

def show_positions(client: KalshiAuthClient, ticker: str | None = None):
    """Display current positions."""
    print_header("OPEN POSITIONS")

    try:
        data = client.get_positions(ticker=ticker, settlement_status="unsettled")
    except KalshiAPIError as e:
        print(f"\n  ERROR: {e}")
        return

    positions = data.get("market_positions", data.get("positions", []))

    if not positions:
        print("\n  No open positions.")
        return

    print(f"\n  {'Ticker':<45} {'Side':<6} {'Qty':>6} {'Avg $':>8} {'Value':>10}")
    print(f"  {'─' * 45} {'─' * 6} {'─' * 6} {'─' * 8} {'─' * 10}")

    for pos in positions:
        t = pos.get("ticker", "?")
        # Positions may have yes and no quantities
        yes_qty = pos.get("total_traded", pos.get("yes_quantity", 0))
        no_qty = pos.get("no_quantity", 0)

        if yes_qty > 0:
            avg = pos.get("average_price", pos.get("yes_avg_price", 0))
            value = yes_qty * avg
            print(f"  {t:<45} {'YES':<6} {yes_qty:>6} {fmt_dollars(avg):>8} {fmt_dollars(value):>10}")
        if no_qty > 0:
            avg = pos.get("no_avg_price", 0)
            value = no_qty * avg
            print(f"  {t:<45} {'NO':<6} {no_qty:>6} {fmt_dollars(avg):>8} {fmt_dollars(value):>10}")

    print()


# ═══════════════════════════════════════════════════════════════════════
# ORDERS
# ═══════════════════════════════════════════════════════════════════════

def show_orders(client: KalshiAuthClient, ticker: str | None = None, status: str = "resting"):
    """Display open/resting orders."""
    print_header(f"ORDERS (status: {status})")

    try:
        data = client.get_orders(ticker=ticker, status=status)
    except KalshiAPIError as e:
        print(f"\n  ERROR: {e}")
        return

    orders = data.get("orders", [])

    if not orders:
        print(f"\n  No {status} orders.")
        return

    print(f"\n  {'Order ID':<38} {'Ticker':<35} {'Act':<5} {'Side':<5} {'Qty':>5} {'Price':>7}")
    print(f"  {'─' * 38} {'─' * 35} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 7}")

    for o in orders:
        oid = o.get("order_id", "?")[:36]
        t = o.get("ticker", "?")
        action = o.get("action", "?")
        side = o.get("side", "?")
        qty = o.get("remaining_count", o.get("count", 0))
        price = o.get("yes_price", o.get("no_price", 0))
        print(f"  {oid:<38} {t:<35} {action:<5} {side:<5} {qty:>5} {fmt_dollars(price):>7}")

    print()


# ═══════════════════════════════════════════════════════════════════════
# ORDER PLACEMENT
# ═══════════════════════════════════════════════════════════════════════

def place_order(
    client: KalshiAuthClient,
    ticker: str,
    action: str,
    side: str,
    price_cents: int,
    quantity: int,
    time_in_force: str = "gtc",
    post_only: bool = False,
    skip_confirm: bool = False,
    decision_id: str | None = None,
    decisions_path: Path | str | None = None,
):
    """
    Place an order with confirmation prompt.

    price_cents: price in cents (1-99).
      For a YES buy at $0.45 → price_cents=45
    """
    # Validate inputs
    if side not in ("yes", "no"):
        print(f"  ERROR: side must be 'yes' or 'no', got '{side}'")
        return None
    if action not in ("buy", "sell"):
        print(f"  ERROR: action must be 'buy' or 'sell', got '{action}'")
        return None
    if price_cents < 1 or price_cents > 99:
        print(f"  ERROR: price must be 1-99 cents, got {price_cents}")
        return None
    if quantity < 1:
        print(f"  ERROR: quantity must be >= 1, got {quantity}")
        return None

    cost_per = price_cents  # cents per contract
    total_cost_cents = cost_per * quantity
    potential_profit_cents = (100 - price_cents) * quantity

    print_header(f"ORDER PREVIEW — {action.upper()} {side.upper()}")
    print(f"\n  Ticker:             {ticker}")
    print(f"  Action:             {action.upper()}")
    print(f"  Side:               {side.upper()}")
    print(f"  Quantity:           {quantity} contracts")
    print(f"  Price:              {fmt_dollars(price_cents)} per contract ({price_cents}¢)")
    print(f"  Time in force:      {time_in_force}")
    if post_only:
        print(f"  Post only:          Yes (maker only)")
    print(f"")
    print(f"  Total cost:         {fmt_dollars(total_cost_cents)}")
    print(f"  Potential profit:   {fmt_dollars(potential_profit_cents)} (if contract settles YES)")
    print(f"  Max loss:           {fmt_dollars(total_cost_cents)} (if contract settles NO)")

    # Check balance
    try:
        balance_data = client.get_balance()
        available = balance_data.get("balance", 0)
        print(f"\n  Available balance:  {fmt_dollars(available)}")
        if total_cost_cents > available:
            print(f"  ⚠  WARNING: Order cost ({fmt_dollars(total_cost_cents)}) exceeds available balance!")
    except KalshiAPIError:
        print(f"\n  (Could not verify balance)")

    if not skip_confirm:
        print(f"\n  ┌───────────────────────────────────────────────┐")
        print(f"  │  Do you want to place this order?             │")
        print(f"  │  Type 'yes' to confirm, anything else cancels │")
        print(f"  └───────────────────────────────────────────────┘")
        confirm = input("\n  Confirm → ").strip().lower()
        if confirm != "yes":
            print("\n  Order CANCELED. Nothing was placed.")
            return None

    # Place the order
    print("\n  Placing order...")

    try:
        if side == "yes":
            result = client.create_order(
                ticker=ticker,
                side=side,
                action=action,
                count=quantity,
                yes_price_cents=price_cents,
                time_in_force=time_in_force,
                post_only=post_only,
            )
        else:
            result = client.create_order(
                ticker=ticker,
                side=side,
                action=action,
                count=quantity,
                no_price_cents=price_cents,
                time_in_force=time_in_force,
                post_only=post_only,
            )
    except KalshiAPIError as e:
        print(f"\n  ORDER FAILED: {e}")
        return None

    order = result.get("order", result)
    order_id = order.get("order_id", "unknown")
    status = order.get("status", "unknown")

    print(f"\n  ✓ ORDER PLACED SUCCESSFULLY")
    print(f"    Order ID:  {order_id}")
    print(f"    Status:    {status}")

    # Show fill info if immediate
    filled = order.get("filled_count", order.get("total_matched", 0))
    remaining = order.get("remaining_count", quantity - filled)
    if filled > 0:
        print(f"    Filled:    {filled} contracts")
    if remaining > 0 and status == "resting":
        print(f"    Resting:   {remaining} contracts (waiting for fill)")

    # Link this fill back to the ev_analyzer decision via bet_decisions ledger.
    if decision_id is not None:
        ledger_path = Path(decisions_path) if decisions_path else OUTPUTS_DIR / "bet_decisions.csv"
        try:
            matched = update_fill(
                decisions_path=ledger_path,
                decision_id=decision_id,
                order_id=order_id,
                filled_at=datetime.now(tz=timezone.utc).isoformat(),
                filled_price=price_cents / 100.0,
                filled_size=float(filled),
            )
            if matched:
                print(f"    Ledger:    linked to decision_id={decision_id}")
            else:
                print(f"    Ledger:    decision_id not found ({decision_id}); fill NOT linked")
        except Exception as e:
            print(f"    Ledger:    update failed ({e})")

    print()
    return order


# ═══════════════════════════════════════════════════════════════════════
# CANCEL
# ═══════════════════════════════════════════════════════════════════════

def cancel_order(client: KalshiAuthClient, order_id: str, skip_confirm: bool = False):
    """Cancel a specific order."""
    print_header("CANCEL ORDER")

    # Show the order first
    try:
        order_data = client.get_order(order_id)
        order = order_data.get("order", order_data)
        print(f"\n  Order:   {order.get('order_id', order_id)}")
        print(f"  Ticker:  {order.get('ticker', '?')}")
        print(f"  Side:    {order.get('side', '?')}")
        print(f"  Action:  {order.get('action', '?')}")
        print(f"  Status:  {order.get('status', '?')}")
    except KalshiAPIError:
        print(f"\n  Order ID: {order_id}")

    if not skip_confirm:
        confirm = input("\n  Cancel this order? (yes/no) → ").strip().lower()
        if confirm != "yes":
            print("  Not canceled.")
            return

    try:
        result = client.cancel_order(order_id)
        print(f"\n  ✓ Order {order_id} CANCELED.")
        return result
    except KalshiAPIError as e:
        print(f"\n  CANCEL FAILED: {e}")


def cancel_all_orders(client: KalshiAuthClient, skip_confirm: bool = False):
    """Cancel all resting orders."""
    print_header("CANCEL ALL ORDERS")

    try:
        data = client.get_orders(status="resting")
    except KalshiAPIError as e:
        print(f"\n  ERROR: {e}")
        return

    orders = data.get("orders", [])
    if not orders:
        print("\n  No resting orders to cancel.")
        return

    print(f"\n  Found {len(orders)} resting order(s):")
    for o in orders:
        print(f"    {o.get('order_id', '?')[:36]}  {o.get('ticker', '?')}  {o.get('side', '?')}  {o.get('action', '?')}")

    if not skip_confirm:
        confirm = input(f"\n  Cancel ALL {len(orders)} orders? (yes/no) → ").strip().lower()
        if confirm != "yes":
            print("  Not canceled.")
            return

    canceled = client.cancel_all_orders()
    print(f"\n  ✓ Canceled {len(canceled)} order(s).")


# ═══════════════════════════════════════════════════════════════════════
# ORDER FROM EV ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def order_from_ev(
    client: KalshiAuthClient,
    ticker: str,
    side: str,
    model_prob: float,
    market_price_cents: int,
    kelly_fraction: float,
    bankroll: float = BANKROLL,
):
    """
    Place an order based on EV analyzer output.

    This is the bridge between ev_analyzer.py and order execution.
    Computes quantity from Kelly sizing and places a limit order.
    """
    # Kelly-based position size
    bet_amount_dollars = kelly_fraction * bankroll
    quantity = int(bet_amount_dollars / (market_price_cents / 100))

    if quantity < 1:
        print("  Kelly sizing results in 0 contracts. No order placed.")
        return None

    print(f"\n  EV-Based Order:")
    print(f"    Model probability:  {model_prob:.1%}")
    print(f"    Market price:       {fmt_dollars(market_price_cents)}")
    print(f"    Edge:               {model_prob - market_price_cents / 100:+.1%}")
    print(f"    Kelly bet size:     ${bet_amount_dollars:,.2f}")
    print(f"    → Contracts:        {quantity}")

    return place_order(
        client=client,
        ticker=ticker,
        action="buy",
        side=side,
        price_cents=market_price_cents,
        quantity=quantity,
    )


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Kalshi Order Executor — Place and manage orders",
        epilog="""
Examples:
  python order_executor.py balance
  python order_executor.py positions
  python order_executor.py orders
  python order_executor.py buy KXNBAGAME-26APR12ATLMIA-MIA --side yes --price 45 --qty 10
  python order_executor.py sell KXNBAGAME-26APR12ATLMIA-MIA --side yes --price 55 --qty 5
  python order_executor.py cancel abc123-order-id
  python order_executor.py cancel-all
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # balance
    subparsers.add_parser("balance", help="Show account balance")

    # positions
    pos_parser = subparsers.add_parser("positions", help="Show open positions")
    pos_parser.add_argument("--ticker", help="Filter by ticker")

    # orders
    ord_parser = subparsers.add_parser("orders", help="Show orders")
    ord_parser.add_argument("--ticker", help="Filter by ticker")
    ord_parser.add_argument("--status", default="resting", help="Order status filter (default: resting)")

    # buy
    buy_parser = subparsers.add_parser("buy", help="Buy contracts")
    buy_parser.add_argument("ticker", help="Market ticker")
    buy_parser.add_argument("--side", required=True, choices=["yes", "no"], help="Contract side")
    buy_parser.add_argument("--price", required=True, type=int, help="Limit price in cents (1-99)")
    buy_parser.add_argument("--qty", required=True, type=int, help="Number of contracts")
    buy_parser.add_argument("--tif", default="gtc", choices=["gtc", "ioc", "fill_or_kill", "day"],
                            help="Time in force (default: gtc)")
    buy_parser.add_argument("--post-only", action="store_true", help="Maker-only order")
    buy_parser.add_argument("--decision-id", help="bet_decisions ledger key from ev_analyzer")

    # sell
    sell_parser = subparsers.add_parser("sell", help="Sell contracts")
    sell_parser.add_argument("ticker", help="Market ticker")
    sell_parser.add_argument("--side", required=True, choices=["yes", "no"], help="Contract side")
    sell_parser.add_argument("--price", required=True, type=int, help="Limit price in cents (1-99)")
    sell_parser.add_argument("--qty", required=True, type=int, help="Number of contracts")
    sell_parser.add_argument("--tif", default="gtc", choices=["gtc", "ioc", "fill_or_kill", "day"],
                            help="Time in force (default: gtc)")
    sell_parser.add_argument("--post-only", action="store_true", help="Maker-only order")
    sell_parser.add_argument("--decision-id", help="bet_decisions ledger key from ev_analyzer")

    # cancel
    cancel_parser = subparsers.add_parser("cancel", help="Cancel an order")
    cancel_parser.add_argument("order_id", help="Order ID to cancel")

    # cancel-all
    subparsers.add_parser("cancel-all", help="Cancel all resting orders")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize authenticated client
    try:
        client = KalshiAuthClient()
    except (ValueError, FileNotFoundError) as e:
        print(f"\n  Authentication error: {e}")
        print("\n  Make sure your .env file has:")
        print("    KALSHI_API_KEY=your_api_key")
        print("    KALSHI_RSA_PRIVATE_KEY_PATH=/path/to/private_key.pem")
        return

    # Execute command
    if args.command == "balance":
        show_balance(client)

    elif args.command == "positions":
        show_positions(client, ticker=getattr(args, "ticker", None))

    elif args.command == "orders":
        show_orders(client, ticker=args.ticker, status=args.status)

    elif args.command == "buy":
        place_order(
            client=client,
            ticker=args.ticker,
            action="buy",
            side=args.side,
            price_cents=args.price,
            quantity=args.qty,
            time_in_force=args.tif,
            post_only=args.post_only,
            decision_id=args.decision_id,
        )

    elif args.command == "sell":
        place_order(
            client=client,
            ticker=args.ticker,
            action="sell",
            side=args.side,
            price_cents=args.price,
            quantity=args.qty,
            time_in_force=args.tif,
            post_only=args.post_only,
            decision_id=args.decision_id,
        )

    elif args.command == "cancel":
        cancel_order(client, args.order_id)

    elif args.command == "cancel-all":
        cancel_all_orders(client)


if __name__ == "__main__":
    main()

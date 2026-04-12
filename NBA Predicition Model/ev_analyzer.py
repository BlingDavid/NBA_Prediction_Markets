"""
Expected Value Analyzer
────────────────────────
Combines the trained NBA prediction model with live Kalshi orderbook data
to produce a full EV breakdown for any specific market.

Includes live adjustments for injuries and back-to-back schedules.

Usage:
  python ev_analyzer.py KXNBAGAME-26APR12ATLMIA-MIA
  python ev_analyzer.py KXNBAGAME-26APR12ATLMIA-MIA --bankroll 5000
  python ev_analyzer.py KXNBAGAME-26APR12ATLMIA-MIA --all   # Show both YES and NO side
  python ev_analyzer.py KXNBAGAME-26APR12ATLMIA-MIA --injuries "MIA:Jimmy Butler:OUT,ATL:Trae Young:OUT"
  python ev_analyzer.py KXNBAGAME-26APR12ATLMIA-MIA --no-adjust  # Skip live adjustments
  python ev_analyzer.py KXNBAGAME-26APR12ATLMIA-MIA --execute    # Analyze + place order if +EV

What it does:
  1. Fetches the live Kalshi market data + full orderbook
  2. Parses the ticker to identify the teams involved
  3. Loads the trained model and generates a base win probability
  4. Fetches injury reports + schedule data and adjusts the probability
  5. Computes EV, Kelly sizing, and fill simulation against the orderbook
  6. Gives a clear BUY / NO BUY recommendation
  7. (Optional) Places an order on Kalshi if +EV and --execute flag is set
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BANKROLL,
    KELLY_FRACTION,
    MAX_BET_FRACTION,
    MIN_EDGE_THRESHOLD,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    TEAM_ABBREV_MAP,
)
from market_scanner import KalshiClient
from live_adjustments import (
    compute_adjusted_probability,
    fetch_injuries,
    parse_manual_injuries,
    print_adjustment_report,
)
from live_data import (
    fetch_full_game_context,
    get_odds_comparison,
    print_game_context,
    print_odds_comparison,
)


# ═══════════════════════════════════════════════════════════════════════
# TICKER PARSER
# ═══════════════════════════════════════════════════════════════════════

# Kalshi NBA game tickers follow this pattern:
# KXNBAGAME-YYMMMDDAWAYTEAMHOMETEAM-YOURPICK
# e.g. KXNBAGAME-26APR12ATLMIA-MIA
#   → Date: 2026-APR-12, Away: ATL, Home: MIA, Betting on: MIA

# Map Kalshi's 2-3 letter abbreviations to NBA API abbreviations
KALSHI_TO_NBA = {
    "ATL": "ATL", "BOS": "BOS", "BKN": "BKN", "CHA": "CHA",
    "CHI": "CHI", "CLE": "CLE", "DAL": "DAL", "DEN": "DEN",
    "DET": "DET", "GSW": "GSW", "GS": "GSW",
    "HOU": "HOU", "IND": "IND",
    "LAC": "LAC", "LAL": "LAL",
    "MEM": "MEM", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN",
    "NOP": "NOP", "NO": "NOP",
    "NYK": "NYK", "NY": "NYK",
    "OKC": "OKC", "ORL": "ORL",
    "PHI": "PHI", "PHX": "PHX",
    "POR": "POR", "SAC": "SAC", "SAS": "SAS", "SA": "SAS",
    "TOR": "TOR", "UTA": "UTA",
    "WAS": "WAS",
}


def parse_nba_ticker(ticker: str) -> dict | None:
    """
    Parse a Kalshi NBA game ticker to extract teams and bet side.
    Returns dict with away_team, home_team, bet_team, date_str.
    """
    # Try to extract from the event portion: KXNBAGAME-26APR12ATLMIA
    parts = ticker.split("-")
    if len(parts) < 2:
        return None

    event_part = parts[1]  # e.g., "26APR12ATLMIA"
    bet_team = parts[2] if len(parts) >= 3 else None

    # Extract date and teams from event part
    # Pattern: YYMMMDD followed by two team abbreviations (2-3 chars each)
    match = re.match(r"(\d{2})([A-Z]{3})(\d{2})([A-Z]{2,3})([A-Z]{2,3})", event_part)
    if not match:
        return None

    year, month, day = match.group(1), match.group(2), match.group(3)
    away_kalshi = match.group(4)
    home_kalshi = match.group(5)

    away_nba = KALSHI_TO_NBA.get(away_kalshi, away_kalshi)
    home_nba = KALSHI_TO_NBA.get(home_kalshi, home_kalshi)
    bet_nba = KALSHI_TO_NBA.get(bet_team, bet_team) if bet_team else None

    return {
        "away_team": away_nba,
        "home_team": home_nba,
        "bet_team": bet_nba,
        "date_str": f"20{year}-{month}-{day}",
        "away_kalshi": away_kalshi,
        "home_kalshi": home_kalshi,
    }


# ═══════════════════════════════════════════════════════════════════════
# MODEL PREDICTION LOADER
# ═══════════════════════════════════════════════════════════════════════

def get_model_prediction(home_team: str, away_team: str) -> dict | None:
    """
    Load the trained model and get a win probability for this matchup.
    Uses the most recent feature data for each team.
    """
    model_path = MODELS_DIR / "nba_predictor.pkl"
    features_path = PROCESSED_DIR / "features.csv"

    if not model_path.exists():
        print("  WARNING: No trained model found.")
        print("  Run: python run_pipeline.py  (to train the model first)")
        return None

    if not features_path.exists():
        print("  WARNING: No feature data found.")
        print("  Run: python run_pipeline.py  (to ingest data first)")
        return None

    from model import NBAPredictor

    predictor = NBAPredictor()
    predictor.load()

    games = pd.read_csv(features_path, parse_dates=["date"])

    # Find the most recent game involving both teams to get their current form
    # We need the latest feature row for each team
    home_games = games[
        (games["home_team"] == home_team) | (games["away_team"] == home_team)
    ].tail(1)

    away_games = games[
        (games["home_team"] == away_team) | (games["away_team"] == away_team)
    ].tail(1)

    # Find the most recent direct matchup or create a synthetic row
    # using the most recent features for each team
    matchup = games[
        ((games["home_team"] == home_team) & (games["away_team"] == away_team))
    ].tail(1)

    if matchup.empty:
        # Try reverse matchup
        matchup = games[
            ((games["home_team"] == away_team) & (games["away_team"] == home_team))
        ].tail(1)
        if not matchup.empty:
            # Flip perspective — model will give away_team as "home" prob
            preds = predictor.predict(matchup)
            return {
                "home_win_prob": float(preds.iloc[0]["away_win_prob"]),
                "away_win_prob": float(preds.iloc[0]["home_win_prob"]),
                "predicted_spread": -float(preds.iloc[0]["predicted_spread"]),
                "predicted_total": float(preds.iloc[0]["predicted_total"]),
                "home_team": home_team,
                "away_team": away_team,
                "data_date": str(matchup.iloc[0]["date"]),
                "note": "Using most recent reverse matchup features",
            }

    if matchup.empty:
        # No direct matchup — use the most recent game with this home team
        matchup = games[games["home_team"] == home_team].tail(1)
        if matchup.empty:
            print(f"  WARNING: No recent data for {home_team} as home team.")
            return None

    preds = predictor.predict(matchup)
    return {
        "home_win_prob": float(preds.iloc[0]["home_win_prob"]),
        "away_win_prob": float(preds.iloc[0]["away_win_prob"]),
        "predicted_spread": float(preds.iloc[0]["predicted_spread"]),
        "predicted_total": float(preds.iloc[0]["predicted_total"]),
        "home_team": home_team,
        "away_team": away_team,
        "data_date": str(matchup.iloc[0]["date"]),
    }


# ═══════════════════════════════════════════════════════════════════════
# EV CALCULATION
# ═══════════════════════════════════════════════════════════════════════

def compute_ev(
    model_prob: float,
    market_price: float,
    quantity: float = 1.0,
) -> dict:
    """
    Compute expected value for a Kalshi YES contract.

    On Kalshi:
      - You buy YES at the ask price (e.g., $0.33)
      - If you win, you get $1.00 (profit = $1.00 - price)
      - If you lose, you lose your price

    EV = (model_prob × profit_if_win) - ((1 - model_prob) × loss_if_lose)
    """
    cost = market_price * quantity
    profit_if_win = (1.0 - market_price) * quantity
    loss_if_lose = market_price * quantity

    ev = (model_prob * profit_if_win) - ((1 - model_prob) * loss_if_lose)
    ev_pct = ev / cost if cost > 0 else 0

    # Implied probability from the market price
    implied_prob = market_price

    # Edge
    edge = model_prob - implied_prob

    # Kelly criterion for Kalshi (binary contract)
    # Decimal odds equivalent: 1 / market_price
    decimal_odds = 1.0 / market_price if market_price > 0 else 1.0
    b = decimal_odds - 1
    q = 1 - model_prob
    full_kelly = (model_prob * b - q) / b if b > 0 else 0
    kelly = max(0, full_kelly * KELLY_FRACTION)
    kelly = min(kelly, MAX_BET_FRACTION)

    return {
        "model_prob": model_prob,
        "market_price": market_price,
        "implied_prob": implied_prob,
        "edge": edge,
        "ev_per_contract": round(ev, 4),
        "ev_pct": round(ev_pct, 4),
        "profit_if_win": round(profit_if_win, 4),
        "loss_if_lose": round(loss_if_lose, 4),
        "decimal_odds": round(decimal_odds, 3),
        "full_kelly": round(full_kelly, 4),
        "kelly_fraction": round(kelly, 4),
        "recommended_bet": round(kelly * BANKROLL, 2),
    }


def simulate_orderbook_fill(
    orderbook_bids: list[tuple[float, float]],
    side: str,
    bet_amount: float,
) -> dict:
    """
    Simulate filling an order against the live orderbook.

    For buying YES: you're taking the YES ask side (buying from sellers).
    But Kalshi's orderbook only shows bids. A NO bid at $0.70 is equivalent
    to a YES ask at $0.30 (because YES + NO = $1.00).

    side: 'yes' or 'no'
    Returns fill details: avg price, contracts filled, total cost.
    """
    if not orderbook_bids:
        return {"filled": False, "reason": "Empty orderbook"}

    # Sort by price descending (best bids first for the opposing side)
    sorted_bids = sorted(orderbook_bids, key=lambda x: x[0], reverse=True)

    contracts_filled = 0
    total_cost = 0
    remaining = bet_amount
    fills = []

    for price, qty in sorted_bids:
        if remaining <= 0:
            break

        # For YES buyer: you pay (1 - NO_bid_price) per contract
        # For NO buyer: you pay (1 - YES_bid_price) per contract
        if side == "yes":
            fill_price = 1.0 - price  # Convert NO bids to YES cost
        else:
            fill_price = 1.0 - price  # Convert YES bids to NO cost

        affordable = remaining / fill_price if fill_price > 0 else 0
        fill_qty = min(qty, affordable)

        cost = fill_qty * fill_price
        total_cost += cost
        contracts_filled += fill_qty
        remaining -= cost

        fills.append({
            "price": round(fill_price, 4),
            "quantity": round(fill_qty, 0),
            "cost": round(cost, 2),
        })

    avg_price = total_cost / contracts_filled if contracts_filled > 0 else 0

    return {
        "filled": contracts_filled > 0,
        "contracts": round(contracts_filled, 0),
        "avg_price": round(avg_price, 4),
        "total_cost": round(total_cost, 2),
        "unfilled_amount": round(remaining, 2),
        "fills": fills[:5],  # Top 5 fill levels
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN ANALYZER
# ═══════════════════════════════════════════════════════════════════════

def analyze_market(
    ticker: str,
    bankroll: float = BANKROLL,
    show_both: bool = False,
    injuries_str: str | None = None,
    skip_adjustments: bool = False,
    execute_order: bool = False,
):
    """
    Full EV analysis for a Kalshi market ticker.

    Steps:
      1. Parse ticker → identify teams
      2. Fetch live market data + orderbook from Kalshi
      3. Load model → generate base win probability
      4. Fetch injuries + schedule → adjust probability
      5. Compute EV, edge, Kelly sizing
      6. Simulate orderbook fill
      7. Print recommendation
    """
    print("\n" + "═" * 70)
    print("  NBA EXPECTED VALUE ANALYZER")
    print("═" * 70)

    # ── Step 1: Parse ticker ──
    parsed = parse_nba_ticker(ticker)
    if not parsed:
        print(f"\n  Could not parse ticker: {ticker}")
        print("  Expected format: KXNBAGAME-YYMMMDDAWAYHMOE-TEAM")
        print("  Example: KXNBAGAME-26APR12ATLMIA-MIA")
        return

    home = parsed["home_team"]
    away = parsed["away_team"]
    bet_team = parsed["bet_team"]
    bet_side = "home" if bet_team == home else "away"
    home_name = TEAM_ABBREV_MAP.get(home, home)
    away_name = TEAM_ABBREV_MAP.get(away, away)
    bet_name = TEAM_ABBREV_MAP.get(bet_team, bet_team)

    print(f"\n  Game:       {away_name} @ {home_name}")
    print(f"  Bet on:     {bet_name} ({bet_team}) — {'HOME' if bet_side == 'home' else 'AWAY'}")
    print(f"  Ticker:     {ticker}")

    # ── Step 2: Fetch live market data ──
    print("\n  Fetching live Kalshi data...")
    client = KalshiClient()

    try:
        market = client.get_market(ticker)
    except Exception as e:
        print(f"  ERROR: Could not fetch market: {e}")
        return

    yes_bid = float(market.get("yes_bid_dollars", 0) or 0)
    yes_ask = float(market.get("yes_ask_dollars", 0) or 0)
    last_price = float(market.get("last_price_dollars", 0) or 0)
    volume = float(market.get("volume_fp", market.get("volume", 0)) or 0)
    oi = float(market.get("open_interest_fp", market.get("open_interest", 0)) or 0)

    print(f"\n  Market Data (live):")
    print(f"    YES bid / ask:  ${yes_bid:.2f} / ${yes_ask:.2f}")
    print(f"    Last trade:     ${last_price:.2f}")
    print(f"    Volume:         {volume:,.0f} contracts")
    print(f"    Open interest:  {oi:,.0f}")
    print(f"    Implied prob:   {yes_ask:.1%} (from ask)")

    # ── Step 3: Fetch orderbook ──
    try:
        ob = client.get_orderbook(ticker)
        ob_data = ob.get("orderbook_fp") or ob.get("orderbook", {})
        yes_orders = ob_data.get("yes_dollars") or ob_data.get("yes", [])
        no_orders = ob_data.get("no_dollars") or ob_data.get("no", [])

        yes_parsed = [(float(e[0]), float(e[1])) for e in (yes_orders or []) if len(e) >= 2]
        no_parsed = [(float(e[0]), float(e[1])) for e in (no_orders or []) if len(e) >= 2]

        total_yes_depth = sum(p * q for p, q in yes_parsed)
        total_no_depth = sum(p * q for p, q in no_parsed)
        print(f"    YES book depth: ${total_yes_depth:,.0f}")
        print(f"    NO book depth:  ${total_no_depth:,.0f}")
    except Exception as e:
        print(f"    Orderbook error: {e}")
        yes_parsed, no_parsed = [], []

    # ── Step 3b: Fetch live game context (team stats, players, H2H, odds) ──
    print("\n  Fetching live game context...")
    try:
        game_ctx = fetch_full_game_context(home, away)
        print_game_context(game_ctx)
    except Exception as e:
        print(f"    Could not fetch full game context: {e}")
        game_ctx = None

    # ── Step 4: Model prediction ──
    print("\n  Loading model prediction...")
    prediction = get_model_prediction(home, away)

    if prediction is None:
        print("  Cannot compute EV without model prediction.")
        print("  Run the full pipeline first: python run_pipeline.py")
        return

    base_home_prob = prediction["home_win_prob"]
    base_away_prob = prediction["away_win_prob"]

    print(f"\n  Base Model Prediction:")
    print(f"    {home} (home) win prob:  {base_home_prob:.1%}")
    print(f"    {away} (away) win prob:  {base_away_prob:.1%}")
    print(f"    Predicted spread:        {prediction['predicted_spread']:+.1f} (home perspective)")
    print(f"    Predicted total:         {prediction['predicted_total']:.1f}")
    if prediction.get("note"):
        print(f"    Note: {prediction['note']}")

    # ── Step 4b: Live adjustments (injuries + schedule) ──
    if not skip_adjustments:
        print("\n  Fetching live adjustments (injuries + schedule)...")

        # Get injuries: manual input takes priority, then try live fetch
        if injuries_str:
            injuries = parse_manual_injuries(injuries_str)
            print(f"    Using {len(injuries)} manually specified injuries")
        else:
            injuries = fetch_injuries()
            if not injuries.empty:
                # Filter to only the two teams in this game
                injuries = injuries[injuries["team"].isin([home, away])]
                print(f"    Found {len(injuries)} injury report(s) for {home}/{away}")
            else:
                print("    No injury data available (use --injuries for manual input)")

        # Extract game date from parsed ticker for B2B detection
        game_date = parsed.get("date_str")

        adj = compute_adjusted_probability(
            base_home_prob=base_home_prob,
            home_team=home,
            away_team=away,
            injuries=injuries,
            game_date=game_date,
        )

        print_adjustment_report(adj, home, away)

        # Use adjusted probabilities for EV calculation
        model_home_prob = adj["adjusted_home_prob"]
        model_away_prob = adj["adjusted_away_prob"]
    else:
        print("\n  (Live adjustments skipped — using raw model probabilities)")
        model_home_prob = base_home_prob
        model_away_prob = base_away_prob

    model_prob = model_home_prob if bet_side == "home" else model_away_prob

    # ── Step 5: EV calculation ──
    # Use the ask price (what you'd actually pay to buy YES)
    buy_price = yes_ask if yes_ask > 0 else last_price

    ev = compute_ev(model_prob, buy_price)

    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  EV ANALYSIS: BUY YES on {bet_team:<5}                          │")
    print(f"  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │  Model probability:     {ev['model_prob']:>8.1%}                       │")
    print(f"  │  Market price (ask):    ${ev['market_price']:>7.2f}  (implied: {ev['implied_prob']:.1%})    │")
    print(f"  │  Edge:                  {ev['edge']:>+8.2%}                       │")
    print(f"  │                                                             │")
    print(f"  │  If WIN:  +${ev['profit_if_win']:.2f} per contract                      │")
    print(f"  │  If LOSE: -${ev['loss_if_lose']:.2f} per contract                      │")
    print(f"  │  EV per contract:       ${ev['ev_per_contract']:>+7.4f}                     │")
    print(f"  │  EV %:                  {ev['ev_pct']:>+8.2%}                       │")
    print(f"  │                                                             │")
    print(f"  │  Kelly (full):          {ev['full_kelly']:>8.2%}                       │")
    print(f"  │  Kelly (quarter):       {ev['kelly_fraction']:>8.2%}                       │")
    print(f"  │  Recommended bet:       ${ev['recommended_bet']:>8,.0f}  (of ${bankroll:,.0f})       │")

    # ── Verdict ──
    if ev["edge"] > MIN_EDGE_THRESHOLD and ev["ev_per_contract"] > 0:
        verdict = "+EV — MODEL SAYS BUY"
        verdict_detail = f"Edge of {ev['edge']:.1%} exceeds {MIN_EDGE_THRESHOLD:.0%} threshold"
    elif ev["edge"] > 0 and ev["ev_per_contract"] > 0:
        verdict = "MARGINAL +EV — THIN EDGE"
        verdict_detail = f"Edge of {ev['edge']:.1%} is below {MIN_EDGE_THRESHOLD:.0%} threshold"
    else:
        verdict = "-EV — DO NOT BUY"
        verdict_detail = f"Model prob ({ev['model_prob']:.1%}) < market ({ev['implied_prob']:.1%})"

    print(f"  │                                                             │")
    print(f"  │  VERDICT: {verdict:<49}│")
    print(f"  │  {verdict_detail:<57}│")
    print(f"  └─────────────────────────────────────────────────────────────┘")

    # ── Step 5b: Odds comparison (Kalshi vs ESPN vs sportsbooks) ──
    try:
        if game_ctx and game_ctx.get("odds"):
            odds_comparison = game_ctx["odds"]
        else:
            odds_comparison = get_odds_comparison(home, away)

        # Add Kalshi data to comparison
        odds_comparison["kalshi"] = {
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "implied_home": buy_price if bet_side == "home" else 1 - buy_price,
        }

        # For the comparison, we need home team implied from Kalshi
        kalshi_home_implied = buy_price if bet_side == "home" else (1 - buy_price)
        print_odds_comparison(odds_comparison, model_home_prob, kalshi_home_implied)
    except Exception as e:
        print(f"\n  (Could not fetch odds comparison: {e})")

    # ── Step 6: Orderbook fill simulation ──
    if ev["recommended_bet"] > 0 and no_parsed:
        print(f"\n  Orderbook Fill Simulation (${ev['recommended_bet']:,.0f} bet):")
        # To buy YES, we're buying against NO sellers
        fill = simulate_orderbook_fill(no_parsed, "yes", ev["recommended_bet"])
        if fill["filled"]:
            print(f"    Contracts filled:  {fill['contracts']:,.0f}")
            print(f"    Avg fill price:    ${fill['avg_price']:.4f}")
            print(f"    Total cost:        ${fill['total_cost']:,.2f}")
            if fill["unfilled_amount"] > 0:
                print(f"    Unfilled:          ${fill['unfilled_amount']:,.2f} (insufficient liquidity)")
            print(f"    Fill levels:")
            for f in fill["fills"]:
                print(f"      ${f['price']:.2f} × {f['quantity']:,.0f} = ${f['cost']:,.2f}")
        else:
            print(f"    Could not fill: {fill.get('reason', 'no liquidity')}")

    # ── Show opposite side if requested ──
    if show_both:
        opp_team = away if bet_side == "home" else home
        opp_prob = model_away_prob if bet_side == "home" else model_home_prob  # Already adjusted
        opp_price = 1.0 - yes_bid if yes_bid > 0 else 1.0 - last_price  # NO price

        opp_ev = compute_ev(opp_prob, opp_price)
        print(f"\n  ── Opposite side: BUY NO (bet on {opp_team}) ──")
        print(f"    Model prob:      {opp_ev['model_prob']:.1%}")
        print(f"    NO price:        ${opp_ev['market_price']:.2f}")
        print(f"    Edge:            {opp_ev['edge']:+.2%}")
        print(f"    EV per contract: ${opp_ev['ev_per_contract']:+.4f}")
        if opp_ev["edge"] > MIN_EDGE_THRESHOLD:
            print(f"    VERDICT: +EV — Consider buying NO instead")
        else:
            print(f"    VERDICT: Not +EV on this side either")

    # ── Save analysis ──
    analysis = {
        "ticker": ticker,
        "home_team": home,
        "away_team": away,
        "bet_team": bet_team,
        "bet_side": bet_side,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "base_home_prob": base_home_prob,
        "base_away_prob": base_away_prob,
        "adj_home_prob": model_home_prob,
        "adj_away_prob": model_away_prob,
        "model_bet_prob": model_prob,
        "market_implied": buy_price,
        "edge": ev["edge"],
        "ev_per_contract": ev["ev_per_contract"],
        "ev_pct": ev["ev_pct"],
        "kelly_fraction": ev["kelly_fraction"],
        "recommended_bet": ev["recommended_bet"],
        "verdict": verdict,
        "volume": volume,
        "open_interest": oi,
        "injuries_applied": not skip_adjustments,
    }

    analysis_df = pd.DataFrame([analysis])
    analysis_df.to_csv(OUTPUTS_DIR / "last_ev_analysis.csv", index=False)

    print(f"\n  Analysis saved to {OUTPUTS_DIR / 'last_ev_analysis.csv'}")

    # ── Step 7 (optional): Execute order ──
    if execute_order and ev["edge"] > MIN_EDGE_THRESHOLD and ev["ev_per_contract"] > 0:
        print(f"\n  ── ORDER EXECUTION MODE ──")
        try:
            from kalshi_auth import KalshiAuthClient
            from order_executor import order_from_ev

            auth_client = KalshiAuthClient()
            order_from_ev(
                client=auth_client,
                ticker=ticker,
                side="yes" if bet_side == "home" else "yes",  # Ticker already encodes the team
                model_prob=model_prob,
                market_price_cents=int(buy_price * 100),
                kelly_fraction=ev["kelly_fraction"],
                bankroll=bankroll,
            )
        except ImportError as e:
            print(f"  Could not load order module: {e}")
        except Exception as e:
            print(f"  Order execution error: {e}")
    elif execute_order:
        print(f"\n  Not +EV — skipping order execution.")

    print("═" * 70)

    return analysis


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NBA EV Analyzer — Evaluate any Kalshi NBA market against our model",
        epilog="""
Examples:
  python ev_analyzer.py KXNBAGAME-26APR12ATLMIA-MIA
  python ev_analyzer.py KXNBAGAME-26APR12ATLMIA-MIA --all
  python ev_analyzer.py KXNBAGAME-26APR12ATLMIA-MIA --bankroll 5000
  python ev_analyzer.py KXNBAGAME-26APR12ATLMIA-MIA --injuries "MIA:Jimmy Butler:OUT,ATL:Trae Young:OUT"
  python ev_analyzer.py KXNBAGAME-26APR12ATLMIA-MIA --no-adjust
  python ev_analyzer.py KXNBAGAME-26APR12ATLMIA-MIA --execute

Find tickers with:
  python market_scanner.py --series KXNBAGAME
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("ticker", help="Kalshi market ticker (e.g., KXNBAGAME-26APR12ATLMIA-MIA)")
    parser.add_argument("--bankroll", type=float, default=BANKROLL, help=f"Bankroll for sizing (default: ${BANKROLL:,})")
    parser.add_argument("--all", action="store_true", help="Show EV for both YES and NO sides")
    parser.add_argument(
        "--injuries", type=str, default=None,
        help='Manual injury input: "TEAM:Player:STATUS,TEAM:Player:STATUS" (e.g., "MIA:Jimmy Butler:OUT,ATL:Trae Young:OUT")'
    )
    parser.add_argument("--no-adjust", action="store_true", help="Skip live adjustments (injuries + schedule)")
    parser.add_argument("--execute", action="store_true",
                        help="Place order on Kalshi if analysis is +EV (requires API key in .env)")

    args = parser.parse_args()
    analyze_market(
        args.ticker,
        bankroll=args.bankroll,
        show_both=args.all,
        injuries_str=args.injuries,
        skip_adjustments=args.no_adjust,
        execute_order=args.execute,
    )

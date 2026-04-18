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
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import (
    BANKROLL,
    CONSENSUS_TIER_MODE,
    KELLY_FRACTION,
    MAX_BET_FRACTION,
    MIN_EDGE_THRESHOLD,
    OUTPUTS_DIR,
    TEAM_ABBREV_MAP,
)
from consensus_divergence import apply_consensus_tier, record_divergence_decision
from market_scanner import KalshiClient
from nba_market_utils import parse_nba_ticker
from prediction_utils import get_model_prediction
from live_adjustments import (
    compute_adjusted_probability,
    fetch_injuries,
    parse_manual_injuries,
    print_adjustment_report,
)
from late_lineups import (
    get_late_lineup_update,
    print_late_report,
    snapshot_injuries,
)
from venue_edge import apply_venue_edge
from referee_bias import get_ref_adjustment
from calibration_monitor import current_threshold, record_prediction
from player_usage import print_replacement_analysis, print_team_depth_report
from live_data import (
    fetch_full_game_context,
    get_odds_comparison,
    print_game_context,
    print_odds_comparison,
)


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
    late_check: bool = False,
    extra_signals: bool = False,
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
    prediction = get_model_prediction(home, away, game_date=parsed.get("date_str"))

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

        # Show replacement analysis (#2) for injured star players
        for label in ["home", "away"]:
            injury_data = adj[f"{label}_injury"]
            repls = injury_data.get("replacements", [])
            for repl in repls:
                if repl and repl.get("replacement_name") and repl.get("projected_impact", 0) > 0.03:
                    print_replacement_analysis(repl)

        # Use adjusted probabilities for EV calculation
        model_home_prob = adj["adjusted_home_prob"]
        model_away_prob = adj["adjusted_away_prob"]

        # ── Step 4c: Late lineup check (#6) ──
        # Diffs the ESPN feed against today's cached snapshot and applies a
        # late-scratch premium for new OUTs inside the late window.
        if late_check:
            # Cache the baseline report on first call of the day so subsequent
            # invocations can diff against it.
            if injuries is not None and not injuries.empty:
                snapshot_injuries(injuries)
            update = get_late_lineup_update(
                home_team=home,
                away_team=away,
                tip_time=None,  # TODO: plumb real tip time from market.close_time
                base_injuries=injuries,
            )
            print_late_report(update, home, away)
            boost = update.get("impact_boost", 0.0)
            if abs(boost) > 0.0005:
                model_home_prob = float(np.clip(model_home_prob + boost, 0.02, 0.98))
                model_away_prob = 1 - model_home_prob
                print(f"    Applied late-check boost: {boost:+.2%} → new home prob {model_home_prob:.1%}")
    else:
        print("\n  (Live adjustments skipped — using raw model probabilities)")
        model_home_prob = base_home_prob
        model_away_prob = base_away_prob

    # ── Step 4d: Extra signals (venue edge #3 + ref bias + calibration) ──
    signal_threshold = MIN_EDGE_THRESHOLD
    if extra_signals:
        print("\n  Extra signals:")

        # Venue edge (#3)
        try:
            venue = apply_venue_edge(home, away, model_home_prob)
            if abs(venue["shift"]) > 0.0005:
                model_home_prob = venue["adjusted_home_prob"]
                model_away_prob = 1 - model_home_prob
                print(f"    Venue edge:      {venue['breakdown']['detail']} → shift {venue['shift']:+.2%}")
            else:
                print(f"    Venue edge:      {venue['breakdown'].get('detail', 'no data')}")
        except Exception as e:
            print(f"    Venue edge:      (unavailable: {e})")

        # Referee bias
        try:
            game_date = parsed.get("date_str", "")
            ref_adj = get_ref_adjustment(home, away, game_date)
            shift = ref_adj.get("home_shift", 0.0)
            if abs(shift) > 0.0005:
                model_home_prob = float(np.clip(model_home_prob + shift, 0.02, 0.98))
                model_away_prob = 1 - model_home_prob
            crew = ref_adj.get("crew")
            print(f"    Ref crew:        {crew or 'not yet assigned'} → shift {shift:+.2%}")
        except Exception as e:
            print(f"    Ref bias:        (unavailable: {e})")

        # Adaptive calibration-driven threshold
        try:
            signal_threshold = current_threshold()
            if signal_threshold > MIN_EDGE_THRESHOLD:
                print(f"    Edge threshold:  {signal_threshold:.2%} (inflated from {MIN_EDGE_THRESHOLD:.2%} due to Brier drift)")
            else:
                print(f"    Edge threshold:  {signal_threshold:.2%} (baseline)")
        except Exception as e:
            print(f"    Edge threshold:  (unavailable: {e})")

        # Consensus-divergence signal (#8): adjust threshold based on Kalshi↔consensus↔model tier.
        tier_result = None
        try:
            comparison = get_odds_comparison(home, away)
            yes_mid = (yes_bid + yes_ask) / 2.0 if (yes_bid > 0 and yes_ask > 0) else None
            tier_result = apply_consensus_tier(
                comparison=comparison,
                model_prob_home=model_home_prob,
                kalshi_yes_mid=yes_mid,
                bet_side=bet_side,
                base_threshold=signal_threshold,
                mode=CONSENSUS_TIER_MODE,
            )
            tier = tier_result["triangulation_tier"]
            print(
                f"    Consensus tier:  {tier:+d} ({tier_result['tier_reason']})"
            )
            if CONSENSUS_TIER_MODE == "active" and tier_result["signal_threshold"] != signal_threshold:
                print(
                    f"    Tier adjusts edge threshold: "
                    f"{signal_threshold:.2%} → {tier_result['signal_threshold']:.2%} (active mode)"
                )
                signal_threshold = tier_result["signal_threshold"]
            elif CONSENSUS_TIER_MODE == "shadow":
                print(
                    f"    (Shadow mode — adjusted threshold would be "
                    f"{tier_result['shadow_signal_threshold']:.2%}, not applied)"
                )
        except Exception as e:
            print(f"    Consensus tier:  (unavailable: {e})")
            tier_result = None

        if tier_result is not None:
            try:
                # Whether the bet would fire at each threshold. Uses model_home_prob
                # as the model's home-win prob; kalshi mid is yes-side probability.
                kalshi_home_prob = yes_mid if bet_side == "home" else (1.0 - yes_mid)
                model_side_prob = model_home_prob if bet_side == "home" else 1.0 - model_home_prob
                model_edge = abs(model_side_prob - (yes_mid if bet_side == "home" else 1.0 - yes_mid))
                would_bet_base = model_edge >= tier_result["base_threshold"]
                would_bet_adjusted = model_edge >= tier_result["signal_threshold"]
                record_divergence_decision(
                    metrics_path=OUTPUTS_DIR / "consensus_divergence_metrics.csv",
                    game_id=ticker,
                    decided_at=datetime.now(tz=timezone.utc).isoformat(),
                    mode=tier_result["mode"],
                    tier=tier_result["triangulation_tier"],
                    kalshi_vs_consensus_pp=tier_result["kalshi_vs_consensus_pp"],
                    model_vs_consensus_pp=tier_result["model_vs_consensus_pp"],
                    kalshi_vs_model_pp=tier_result["kalshi_vs_model_pp"],
                    bet_taken=False,
                    stake=0.0,
                    base_threshold=tier_result["base_threshold"],
                    adjusted_threshold=tier_result["signal_threshold"],
                    would_have_bet_at_base=would_bet_base,
                    would_have_bet_at_adjusted=would_bet_adjusted,
                )
            except Exception as e:
                print(f"    Metrics log:     (failed: {e})")

    model_prob = model_home_prob if bet_side == "home" else model_away_prob

    # Log this prediction for calibration tracking (outcome filled in later)
    try:
        record_prediction(
            game_id=ticker,
            predicted_home_prob=model_home_prob,
            game_date=parsed.get("date_str", ""),
        )
    except Exception:
        pass

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
    if ev["edge"] > signal_threshold and ev["ev_per_contract"] > 0:
        verdict = "+EV — MODEL SAYS BUY"
        verdict_detail = f"Edge of {ev['edge']:.1%} exceeds {signal_threshold:.2%} threshold"
    elif ev["edge"] > 0 and ev["ev_per_contract"] > 0:
        verdict = "MARGINAL +EV — THIN EDGE"
        verdict_detail = f"Edge of {ev['edge']:.1%} is below {signal_threshold:.2%} threshold"
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
# SCAN-ALL MODE
# ═══════════════════════════════════════════════════════════════════════

def scan_all_nba_markets(
    bankroll: float = BANKROLL,
    injuries_str: str | None = None,
    skip_adjustments: bool = False,
    min_edge: float = MIN_EDGE_THRESHOLD,
    series_ticker: str = "KXNBAGAME",
    late_check: bool = False,
    extra_signals: bool = False,
):
    """
    Pull every open Kalshi NBA market, run analyze_market() on each,
    and print a ranked summary of +EV opportunities.
    """
    print(f"\n  Fetching open markets in series '{series_ticker}'...")
    client = KalshiClient()
    try:
        result = client.get_markets(series_ticker=series_ticker, status="open")
        markets = result.get("markets", [])
    except Exception as e:
        print(f"  Failed to fetch markets: {e}")
        return

    if not markets:
        print("  No open NBA markets right now.")
        return

    print(f"  Found {len(markets)} open markets. Running EV analysis on each...\n")

    results = []
    for i, m in enumerate(markets, 1):
        ticker = m.get("ticker", "")
        print(f"\n  [{i}/{len(markets)}] {ticker}")
        print("  " + "─" * 68)
        try:
            analysis = analyze_market(
                ticker,
                bankroll=bankroll,
                show_both=False,
                injuries_str=injuries_str,
                skip_adjustments=skip_adjustments,
                execute_order=False,
                late_check=late_check,
                extra_signals=extra_signals,
            )
            if analysis:
                results.append(analysis)
        except SystemExit:
            # analyze_market calls sys.exit on parse failures — keep scanning
            print(f"    Skipped (ticker parse/model failure)")
            continue
        except Exception as e:
            print(f"    Error analyzing {ticker}: {e}")
            continue

    if not results:
        print("\n  No markets could be analyzed.")
        return

    # Rank by EV per contract
    df = pd.DataFrame(results)
    df = df.sort_values("ev_per_contract", ascending=False).reset_index(drop=True)
    df.to_csv(OUTPUTS_DIR / "scan_all_ev.csv", index=False)

    print("\n" + "═" * 80)
    print("  SCAN-ALL SUMMARY — ranked by EV per contract")
    print("═" * 80)
    print(f"  {'#':<4}{'Ticker':<36}{'Model':<9}{'Market':<9}{'Edge':<9}{'EV/ct':<10}{'Verdict'}")
    print("  " + "─" * 78)
    for i, row in df.iterrows():
        print(
            f"  {i+1:<4}{str(row.get('ticker',''))[:34]:<36}"
            f"{row.get('model_prob', 0):<9.1%}{row.get('market_implied', 0):<9.1%}"
            f"{row.get('edge', 0):<+9.1%}${row.get('ev_per_contract', 0):<9.3f}"
            f"{row.get('verdict','')}"
        )

    positive = df[df["ev_per_contract"] > 0]
    print(f"\n  +EV markets: {len(positive)} of {len(df)}")
    print(f"  Saved to {OUTPUTS_DIR / 'scan_all_ev.csv'}")
    print("═" * 80)


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
  python ev_analyzer.py --scan-all --bankroll 10000      # scan every open NBA market

Find tickers with:
  python market_scanner.py --series KXNBAGAME
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("ticker", nargs="?", default=None,
                        help="Kalshi market ticker (e.g., KXNBAGAME-26APR12ATLMIA-MIA). Omit with --scan-all.")
    parser.add_argument("--bankroll", type=float, default=BANKROLL, help=f"Bankroll for sizing (default: ${BANKROLL:,})")
    parser.add_argument("--all", action="store_true", help="Show EV for both YES and NO sides (single ticker)")
    parser.add_argument("--scan-all", action="store_true",
                        help="Scan every open NBA market and rank by EV")
    parser.add_argument("--series", type=str, default="KXNBAGAME",
                        help="Series to scan when using --scan-all (default: KXNBAGAME)")
    parser.add_argument(
        "--injuries", type=str, default=None,
        help='Manual injury input: "TEAM:Player:STATUS,TEAM:Player:STATUS" (e.g., "MIA:Jimmy Butler:OUT,ATL:Trae Young:OUT")'
    )
    parser.add_argument("--no-adjust", action="store_true", help="Skip live adjustments (injuries + schedule)")
    parser.add_argument("--execute", action="store_true",
                        help="Place order on Kalshi if analysis is +EV (requires API key in .env)")
    parser.add_argument("--late-check", action="store_true", dest="late_check",
                        help="Re-fetch ESPN injury feed and apply late-scratch premium (#6)")
    parser.add_argument("--extra-signals", action="store_true", dest="extra_signals",
                        help="Apply venue edge (#3), referee bias, and adaptive calibration threshold")

    args = parser.parse_args()

    if args.scan_all:
        scan_all_nba_markets(
            bankroll=args.bankroll,
            injuries_str=args.injuries,
            skip_adjustments=args.no_adjust,
            series_ticker=args.series,
            late_check=args.late_check,
            extra_signals=args.extra_signals,
        )
    elif args.ticker:
        analyze_market(
            args.ticker,
            bankroll=args.bankroll,
            show_both=args.all,
            injuries_str=args.injuries,
            skip_adjustments=args.no_adjust,
            execute_order=args.execute,
            late_check=args.late_check,
            extra_signals=args.extra_signals,
        )
    else:
        parser.error("Provide a ticker, or use --scan-all to analyze every open NBA market.")

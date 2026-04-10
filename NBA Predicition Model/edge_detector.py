"""
Edge Detection & Betting Module
─────────────────────────────────
Compares model probabilities against market odds to find +EV bets.
Uses fractional Kelly criterion for position sizing.

Key concepts:
  - Edge = model_prob - implied_prob (positive = value bet)
  - Kelly fraction = (edge * decimal_odds - 1) / (decimal_odds - 1)
  - Quarter-Kelly for safety (full Kelly is too aggressive for real bankrolls)
"""

import numpy as np
import pandas as pd

from config import (
    BANKROLL,
    KELLY_FRACTION,
    MAX_BET_FRACTION,
    MIN_EDGE_THRESHOLD,
    OUTPUTS_DIR,
)
from data_ingest import american_to_decimal, american_to_implied_prob, implied_prob_to_fair


# ═══════════════════════════════════════════════════════════════════════
# EDGE DETECTION
# ═══════════════════════════════════════════════════════════════════════

def find_edges(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    min_edge: float = MIN_EDGE_THRESHOLD,
) -> pd.DataFrame:
    """
    Match model predictions against current market odds.
    Returns bets where model edge exceeds threshold.

    Parameters
    ----------
    predictions : Model output with home_win_prob, predicted_spread, predicted_total
    odds : Current odds with ml_home, ml_away, spread_home, total_line
    min_edge : Minimum edge required to generate a bet signal
    """
    # Match games by home_team + away_team
    merged = predictions.merge(
        odds,
        on=["home_team", "away_team"],
        how="inner",
        suffixes=("_pred", "_odds"),
    )

    if merged.empty:
        print("  No matching games found between predictions and odds.")
        return pd.DataFrame()

    bets = []

    for _, row in merged.iterrows():
        game_bets = []

        # ── Moneyline edges ──
        if pd.notna(row.get("ml_home")) and pd.notna(row.get("ml_away")):
            impl_home = american_to_implied_prob(row["ml_home"])
            impl_away = american_to_implied_prob(row["ml_away"])
            fair_home, fair_away = implied_prob_to_fair(impl_home, impl_away)

            # Home ML edge
            home_edge = row["home_win_prob"] - fair_home
            if home_edge > min_edge:
                game_bets.append(_build_bet(
                    row, "moneyline", "home", row["home_team"],
                    model_prob=row["home_win_prob"],
                    fair_prob=fair_home,
                    implied_prob=impl_home,
                    edge=home_edge,
                    american_odds=row["ml_home"],
                ))

            # Away ML edge
            away_edge = row["away_win_prob"] - fair_away
            if away_edge > min_edge:
                game_bets.append(_build_bet(
                    row, "moneyline", "away", row["away_team"],
                    model_prob=row["away_win_prob"],
                    fair_prob=fair_away,
                    implied_prob=impl_away,
                    edge=away_edge,
                    american_odds=row["ml_away"],
                ))

        # ── Spread edges ──
        if pd.notna(row.get("spread_home")) and pd.notna(row.get("predicted_spread")):
            spread_line = row["spread_home"]
            model_spread = row["predicted_spread"]

            # If model thinks home team covers (model spread > line)
            # Standard spread assumption: -110 both sides → implied ~52.4%
            spread_impl = 0.524  # -110 standard
            model_cover_prob = _spread_cover_prob(model_spread, spread_line)

            spread_edge = model_cover_prob - spread_impl
            if spread_edge > min_edge:
                game_bets.append(_build_bet(
                    row, "spread", "home", f"{row['home_team']} {spread_line:+.1f}",
                    model_prob=model_cover_prob,
                    fair_prob=spread_impl,
                    implied_prob=spread_impl,
                    edge=spread_edge,
                    american_odds=row.get("spread_home_price", -110),
                ))
            elif -spread_edge > min_edge:
                game_bets.append(_build_bet(
                    row, "spread", "away", f"{row['away_team']} {-spread_line:+.1f}",
                    model_prob=1 - model_cover_prob,
                    fair_prob=1 - spread_impl,
                    implied_prob=1 - spread_impl,
                    edge=-spread_edge,
                    american_odds=row.get("spread_home_price", -110),
                ))

        # ── Total edges ──
        if pd.notna(row.get("total_line")) and pd.notna(row.get("predicted_total")):
            total_line = row["total_line"]
            model_total = row["predicted_total"]
            total_impl = 0.524  # -110 standard

            over_prob = _total_over_prob(model_total, total_line)

            if over_prob - total_impl > min_edge:
                game_bets.append(_build_bet(
                    row, "total", "over", f"O {total_line}",
                    model_prob=over_prob,
                    fair_prob=total_impl,
                    implied_prob=total_impl,
                    edge=over_prob - total_impl,
                    american_odds=row.get("total_over_price", -110),
                ))
            elif (1 - over_prob) - total_impl > min_edge:
                game_bets.append(_build_bet(
                    row, "total", "under", f"U {total_line}",
                    model_prob=1 - over_prob,
                    fair_prob=1 - total_impl,
                    implied_prob=1 - total_impl,
                    edge=(1 - over_prob) - total_impl,
                    american_odds=row.get("total_over_price", -110),
                ))

        bets.extend(game_bets)

    if not bets:
        print("  No edges found above threshold.")
        return pd.DataFrame()

    bets_df = pd.DataFrame(bets)
    bets_df = bets_df.sort_values("edge", ascending=False).reset_index(drop=True)

    return bets_df


def _build_bet(row, market: str, side: str, selection: str, **kwargs) -> dict:
    """Build a single bet record."""
    decimal_odds = american_to_decimal(kwargs["american_odds"]) if kwargs.get("american_odds") else 2.0

    bet = {
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "market": market,
        "side": side,
        "selection": selection,
        "model_prob": round(kwargs["model_prob"], 4),
        "fair_prob": round(kwargs["fair_prob"], 4),
        "implied_prob": round(kwargs["implied_prob"], 4),
        "edge": round(kwargs["edge"], 4),
        "american_odds": kwargs.get("american_odds"),
        "decimal_odds": round(decimal_odds, 3),
    }

    # Kelly sizing
    kelly = kelly_criterion(kwargs["model_prob"], decimal_odds)
    bet["kelly_fraction"] = round(kelly, 4)
    bet["bet_size"] = round(kelly * BANKROLL, 2)
    bet["expected_value"] = round(kwargs["model_prob"] * decimal_odds - 1, 4)

    return bet


def _spread_cover_prob(model_spread: float, line: float, std: float = 10.5) -> float:
    """
    Estimate probability of covering the spread.
    Uses a normal distribution with std ≈ 10.5 points (historical NBA std dev).
    P(actual_spread > line) where model_spread is our expected value.
    """
    from scipy.stats import norm
    # Home covers if actual_spread > -line (line is already from home perspective)
    # model_spread is our expected home margin
    z = (model_spread - line) / std
    return float(norm.cdf(z))


def _total_over_prob(model_total: float, line: float, std: float = 18.0) -> float:
    """
    Estimate probability of going over the total.
    Uses a normal distribution with std ≈ 18 points (historical NBA total std dev).
    """
    from scipy.stats import norm
    z = (model_total - line) / std
    return float(norm.cdf(z))


# ═══════════════════════════════════════════════════════════════════════
# KELLY CRITERION
# ═══════════════════════════════════════════════════════════════════════

def kelly_criterion(prob: float, decimal_odds: float) -> float:
    """
    Fractional Kelly criterion for position sizing.

    Full Kelly: f* = (p * b - q) / b
      where p = win probability, q = 1-p, b = decimal_odds - 1

    We use KELLY_FRACTION (default 0.25 = quarter-Kelly) for safety.
    Also caps at MAX_BET_FRACTION.
    """
    if prob <= 0 or prob >= 1 or decimal_odds <= 1:
        return 0.0

    b = decimal_odds - 1  # net payout per dollar wagered
    q = 1 - prob

    full_kelly = (prob * b - q) / b

    if full_kelly <= 0:
        return 0.0

    # Apply fraction and cap
    sized = full_kelly * KELLY_FRACTION
    return min(sized, MAX_BET_FRACTION)


# ═══════════════════════════════════════════════════════════════════════
# DISPLAY & OUTPUT
# ═══════════════════════════════════════════════════════════════════════

def print_bet_card(bets: pd.DataFrame, bankroll: float = BANKROLL):
    """Print a formatted betting card."""
    if bets.empty:
        print("\nNo bets to display.")
        return

    print("\n" + "=" * 80)
    print(f"  NBA EDGE DETECTOR — {len(bets)} bets found")
    print(f"  Bankroll: ${bankroll:,.0f} | Min edge: {MIN_EDGE_THRESHOLD:.1%}")
    print("=" * 80)

    total_risk = 0
    total_ev = 0

    for _, bet in bets.iterrows():
        edge_stars = "★" * min(5, int(bet["edge"] * 50))
        print(f"\n  {bet['home_team']} vs {bet['away_team']}")
        print(f"  ├─ Market:    {bet['market'].upper()} — {bet['selection']}")
        print(f"  ├─ Model:     {bet['model_prob']:.1%} vs Market: {bet['fair_prob']:.1%}")
        print(f"  ├─ Edge:      {bet['edge']:+.1%} {edge_stars}")
        print(f"  ├─ Odds:      {bet['american_odds']:+.0f} ({bet['decimal_odds']:.3f})")
        print(f"  ├─ Kelly:     {bet['kelly_fraction']:.2%}")
        print(f"  ├─ Bet size:  ${bet['bet_size']:,.0f}")
        print(f"  └─ +EV:       {bet['expected_value']:+.2%}")

        total_risk += bet["bet_size"]
        total_ev += bet["bet_size"] * bet["expected_value"]

    print(f"\n{'─' * 80}")
    print(f"  Total risk: ${total_risk:,.0f} ({total_risk/bankroll:.1%} of bankroll)")
    print(f"  Expected profit: ${total_ev:,.0f}")
    print(f"{'─' * 80}\n")


def save_bet_card(bets: pd.DataFrame):
    """Save betting card to CSV."""
    if not bets.empty:
        path = OUTPUTS_DIR / "bet_card.csv"
        bets.to_csv(path, index=False)
        print(f"  Bet card saved to {path}")


if __name__ == "__main__":
    # Example: load predictions and odds, find edges
    preds = pd.read_csv(OUTPUTS_DIR / "predictions.csv") if (OUTPUTS_DIR / "predictions.csv").exists() else None
    odds = pd.read_csv(PROCESSED_DIR / "current_odds.csv") if (PROCESSED_DIR / "current_odds.csv").exists() else None

    if preds is not None and odds is not None:
        bets = find_edges(preds, odds)
        print_bet_card(bets)
        save_bet_card(bets)
    else:
        print("Run the full pipeline first: python run_pipeline.py")

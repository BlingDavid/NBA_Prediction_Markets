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
    apply_shrinkage: bool = True,
) -> pd.DataFrame:
    """
    Match model predictions against current market odds.
    Returns bets where model edge exceeds threshold.

    Parameters
    ----------
    predictions : Model output with home_win_prob, predicted_spread, predicted_total
    odds : Current odds with ml_home, ml_away, spread_home, total_line
    min_edge : Minimum edge required to generate a bet signal
    apply_shrinkage : If True (default), shrink model_prob toward fair_prob for
        |raw edge| > 0.10, capped at 50% shrinkage by |edge| ≥ 0.30. Discounts
        extreme calls where the model historically over-predicts (per-team
        biases like BKN/NOP/SAC; playoff distribution shift; series effects).
        Set False to recover raw model output for backtesting.
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

    def _candidate(model_prob: float, fair_prob: float):
        """Return (shrunk_model_prob, shrunk_edge, raw_model_prob, raw_edge)
        for a bet candidate. Filtering downstream uses the shrunk edge so
        bets that shrink below min_edge are correctly dropped."""
        raw_edge = model_prob - fair_prob
        shrunk = _shrink_edge(model_prob, fair_prob) if apply_shrinkage else model_prob
        return shrunk, shrunk - fair_prob, model_prob, raw_edge

    bets = []

    for _, row in merged.iterrows():
        game_bets = []

        # ── Moneyline edges ──
        if pd.notna(row.get("ml_home")) and pd.notna(row.get("ml_away")):
            impl_home = american_to_implied_prob(row["ml_home"])
            impl_away = american_to_implied_prob(row["ml_away"])
            fair_home, fair_away = implied_prob_to_fair(impl_home, impl_away)

            home_mp, home_edge, home_mp_raw, home_edge_raw = _candidate(row["home_win_prob"], fair_home)
            if home_edge > min_edge:
                game_bets.append(_build_bet(
                    row, "moneyline", "home", row["home_team"],
                    model_prob=home_mp, model_prob_raw=home_mp_raw,
                    fair_prob=fair_home, implied_prob=impl_home,
                    edge=home_edge, edge_raw=home_edge_raw,
                    american_odds=row["ml_home"],
                ))

            away_mp, away_edge, away_mp_raw, away_edge_raw = _candidate(row["away_win_prob"], fair_away)
            if away_edge > min_edge:
                game_bets.append(_build_bet(
                    row, "moneyline", "away", row["away_team"],
                    model_prob=away_mp, model_prob_raw=away_mp_raw,
                    fair_prob=fair_away, implied_prob=impl_away,
                    edge=away_edge, edge_raw=away_edge_raw,
                    american_odds=row["ml_away"],
                ))

        # ── Spread edges ──
        if pd.notna(row.get("spread_home")) and pd.notna(row.get("predicted_spread")):
            spread_line = row["spread_home"]
            model_spread = row["predicted_spread"]

            # The line itself is set so each side is ~50/50 — that's the
            # bookmaker's no-vig fair. The vigged price differs by sportsbook;
            # use the actual price for display + EV calc, but compare the
            # model to the no-vig fair (0.5) to get the true edge.
            spread_price = row.get("spread_home_price", -110)
            spread_impl = american_to_implied_prob(spread_price)
            spread_fair = 0.5
            model_cover_prob = _spread_cover_prob(model_spread, spread_line)

            home_mp, home_edge, home_mp_raw, home_edge_raw = _candidate(model_cover_prob, spread_fair)
            away_mp, away_edge, away_mp_raw, away_edge_raw = _candidate(1 - model_cover_prob, 1 - spread_fair)

            if home_edge > min_edge:
                game_bets.append(_build_bet(
                    row, "spread", "home", f"{row['home_team']} {spread_line:+.1f}",
                    model_prob=home_mp, model_prob_raw=home_mp_raw,
                    fair_prob=spread_fair, implied_prob=spread_impl,
                    edge=home_edge, edge_raw=home_edge_raw,
                    american_odds=spread_price,
                ))
            elif away_edge > min_edge:
                game_bets.append(_build_bet(
                    row, "spread", "away", f"{row['away_team']} {-spread_line:+.1f}",
                    model_prob=away_mp, model_prob_raw=away_mp_raw,
                    fair_prob=1 - spread_fair, implied_prob=1 - spread_impl,
                    edge=away_edge, edge_raw=away_edge_raw,
                    american_odds=spread_price,
                ))

        # ── Total edges ──
        if pd.notna(row.get("total_line")) and pd.notna(row.get("predicted_total")):
            total_line = row["total_line"]
            model_total = row["predicted_total"]
            total_price = row.get("total_over_price", -110)
            total_impl = american_to_implied_prob(total_price)
            total_fair = 0.5

            over_prob = _total_over_prob(model_total, total_line)

            over_mp, over_edge, over_mp_raw, over_edge_raw = _candidate(over_prob, total_fair)
            under_mp, under_edge, under_mp_raw, under_edge_raw = _candidate(1 - over_prob, 1 - total_fair)

            if over_edge > min_edge:
                game_bets.append(_build_bet(
                    row, "total", "over", f"O {total_line}",
                    model_prob=over_mp, model_prob_raw=over_mp_raw,
                    fair_prob=total_fair, implied_prob=total_impl,
                    edge=over_edge, edge_raw=over_edge_raw,
                    american_odds=total_price,
                ))
            elif under_edge > min_edge:
                game_bets.append(_build_bet(
                    row, "total", "under", f"U {total_line}",
                    model_prob=under_mp, model_prob_raw=under_mp_raw,
                    fair_prob=1 - total_fair, implied_prob=1 - total_impl,
                    edge=under_edge, edge_raw=under_edge_raw,
                    american_odds=total_price,
                ))

        bets.extend(game_bets)

    if not bets:
        print("  No edges found above threshold.")
        return pd.DataFrame()

    bets_df = pd.DataFrame(bets)
    bets_df = bets_df.sort_values("edge", ascending=False).reset_index(drop=True)

    return bets_df


def _build_bet(row, market: str, side: str, selection: str, **kwargs) -> dict:
    """Build a single bet record.

    Caller passes `model_prob` (post-shrinkage, used for sizing/EV/display) and
    `model_prob_raw` (pre-shrinkage, surfaced for transparency). Same split for
    `edge` / `edge_raw`. When shrinkage is disabled the two are equal.
    """
    decimal_odds = american_to_decimal(kwargs["american_odds"]) if kwargs.get("american_odds") else 2.0
    model_prob = float(kwargs["model_prob"])
    model_prob_raw = float(kwargs.get("model_prob_raw", model_prob))
    edge = float(kwargs["edge"])
    edge_raw = float(kwargs.get("edge_raw", edge))

    bet = {
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "market": market,
        "side": side,
        "selection": selection,
        "model_prob": round(model_prob, 4),
        "model_prob_raw": round(model_prob_raw, 4),
        "fair_prob": round(kwargs["fair_prob"], 4),
        "implied_prob": round(kwargs["implied_prob"], 4),
        "edge": round(edge, 4),
        "edge_raw": round(edge_raw, 4),
        "american_odds": kwargs.get("american_odds"),
        "decimal_odds": round(decimal_odds, 3),
    }

    # Kelly + EV use the post-shrinkage probability so sized bets reflect the
    # actionable edge, not the inflated one.
    kelly = kelly_criterion(model_prob, decimal_odds)
    bet["kelly_fraction"] = round(kelly, 4)
    bet["bet_size"] = round(kelly * BANKROLL, 2)
    bet["expected_value"] = round(model_prob * decimal_odds - 1, 4)

    return bet


def _shrink_edge(
    model_prob: float,
    fair_prob: float,
    free_threshold: float = 0.10,
    max_threshold: float = 0.30,
    min_factor: float = 0.50,
) -> float:
    """Pull `model_prob` toward `fair_prob` for large raw edges.

    Why this exists: backtest analysis showed the model is well-calibrated up
    to about |edge| = 0.10 (within Wilson 95% CI of actuals). Beyond that there
    are real per-team biases (BKN went 0/13 with model_prob ≈ 0.41) and a
    distribution shift into playoff games the regular-season training data
    doesn't cover. Shrinking large edges by up to 50% trades a little EV on
    real edges for a lot of robustness on overconfident extrapolations.

    Curve:
      |raw_edge| ≤ free_threshold (0.10)  → factor 1.0  (no shrinkage)
      free_threshold < |raw_edge| < max   → factor linear from 1.0 → min_factor
      |raw_edge| ≥ max_threshold (0.30)   → factor min_factor (0.50)

    Direction is preserved: a positive raw edge stays positive (just smaller).
    """
    raw_edge = model_prob - fair_prob
    abs_edge = abs(raw_edge)
    if abs_edge <= free_threshold:
        return float(model_prob)
    if abs_edge >= max_threshold:
        factor = min_factor
    else:
        t = (abs_edge - free_threshold) / (max_threshold - free_threshold)
        factor = 1.0 - t * (1.0 - min_factor)
    shrunk = fair_prob + raw_edge * factor
    return float(shrunk)


def _spread_cover_prob(model_spread: float, line: float, std: float = 10.5) -> float:
    """
    Estimate probability the home team covers the spread.

    `line` follows the standard sportsbook convention: negative when the
    home team is favored (e.g. spread_home = -4.6 for "BOS -4.6"), positive
    when the home team is the underdog. Home covers iff actual home margin
    > -line, so

        P(margin > -line | margin ~ Normal(model_spread, std))
            = norm.cdf((model_spread - (-line)) / std)
            = norm.cdf((model_spread + line) / std)

    NBA std is ~10.5 historically; std is a conservative single-game spread.
    """
    from scipy.stats import norm
    z = (model_spread + line) / std
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

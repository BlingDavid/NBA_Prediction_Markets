"""
Backtesting Framework
──────────────────────
Simulates model performance over historical data using walk-forward analysis.

Walk-forward approach:
  1. Train model on seasons 1..N
  2. Predict season N+1 games one-by-one
  3. Simulate bets using Kelly sizing against synthetic/historical odds
  4. Track bankroll, win rate, ROI, drawdown, Sharpe ratio

This is the most critical module — if the backtest doesn't show edge,
the model shouldn't be trusted with real money.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    BANKROLL,
    KELLY_FRACTION,
    MAX_BET_FRACTION,
    MIN_EDGE_THRESHOLD,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
)
from edge_detector import kelly_criterion
from features import EloTracker, build_features, get_feature_columns
from model import NBAPredictor


# ═══════════════════════════════════════════════════════════════════════
# WALK-FORWARD BACKTESTER
# ═══════════════════════════════════════════════════════════════════════

class Backtester:
    """
    Walk-forward backtesting engine.

    For each game in the test period:
      1. Uses only data available BEFORE that game (no look-ahead bias)
      2. Generates a prediction
      3. Compares against synthetic closing line odds
      4. Sizes bet via Kelly criterion
      5. Resolves the bet and updates bankroll
    """

    def __init__(
        self,
        bankroll: float = BANKROLL,
        min_edge: float = MIN_EDGE_THRESHOLD,
        kelly_frac: float = KELLY_FRACTION,
        max_bet_frac: float = MAX_BET_FRACTION,
    ):
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.min_edge = min_edge
        self.kelly_frac = kelly_frac
        self.max_bet_frac = max_bet_frac
        self.bet_log: list[dict] = []
        self.bankroll_history: list[float] = [bankroll]

    def run(
        self,
        games: pd.DataFrame,
        train_seasons: list[str],
        test_seasons: list[str],
    ) -> dict:
        """
        Run walk-forward backtest.

        Parameters
        ----------
        games : Full featured DataFrame (output of build_features)
        train_seasons : Season IDs to train on initially
        test_seasons : Season IDs to test on
        """
        print("\n" + "=" * 70)
        print("  WALK-FORWARD BACKTEST")
        print(f"  Train: {train_seasons}")
        print(f"  Test:  {test_seasons}")
        print(f"  Bankroll: ${self.initial_bankroll:,.0f}")
        print("=" * 70)

        # Split data
        train_mask = games["season"].isin(train_seasons)
        test_mask = games["season"].isin(test_seasons)

        train_data = games[train_mask].copy()
        test_data = games[test_mask].copy().sort_values("date").reset_index(drop=True)

        if train_data.empty or test_data.empty:
            print("  ERROR: No data in train or test split.")
            return {}

        feature_cols = get_feature_columns(games)
        print(f"\n  Training on {len(train_data)} games...")
        print(f"  Testing on {len(test_data)} games...")

        # Train model on training data
        predictor = NBAPredictor()
        predictor.train(train_data)

        # Generate predictions for test period
        print("\n  Running predictions on test period...")
        predictions = predictor.predict(test_data)

        # Simulate betting against synthetic closing lines
        print("  Simulating bets...\n")
        self._simulate_bets(test_data, predictions)

        # Compute summary statistics
        results = self._compute_results()
        self._print_results(results)

        # Save detailed log
        self._save_results()

        return results

    def _simulate_bets(self, test_data: pd.DataFrame, predictions: pd.DataFrame):
        """Simulate bets for each game in the test period."""
        for idx in range(len(test_data)):
            row = test_data.iloc[idx]
            pred = predictions.iloc[idx]

            # Generate synthetic closing line from actual result + noise
            # In production, you'd use real historical odds
            synthetic_odds = self._generate_synthetic_odds(row)

            if synthetic_odds is None:
                continue

            # Check for moneyline edge
            model_home_prob = pred["home_win_prob"]
            market_home_prob = synthetic_odds["implied_home_prob"]

            home_edge = model_home_prob - market_home_prob
            away_edge = (1 - model_home_prob) - (1 - market_home_prob)

            # Bet on the side with edge
            if abs(home_edge) > self.min_edge:
                if home_edge > 0:
                    # Bet home
                    self._place_bet(
                        row, pred, "home", model_home_prob,
                        market_home_prob, synthetic_odds["home_decimal"],
                    )
                else:
                    # Bet away
                    self._place_bet(
                        row, pred, "away", 1 - model_home_prob,
                        1 - market_home_prob, synthetic_odds["away_decimal"],
                    )

    def _generate_synthetic_odds(self, row: pd.DataFrame) -> dict | None:
        """
        Generate synthetic closing line odds based on the game result.
        Uses historical spread data to create realistic implied probabilities.

        In a real backtest, you'd use actual historical closing odds.
        This synthetic approach adds noise around the true probability to
        simulate market efficiency.
        """
        # Use the ELO probability as a proxy for "market" consensus
        if "elo_home_win_prob" not in row.index:
            return None

        elo_prob = row["elo_home_win_prob"]

        # Add noise to simulate market deviation from true probability
        # Markets are efficient but not perfect — std dev ~3-5%
        noise = np.random.normal(0, 0.04)
        market_prob = np.clip(elo_prob + noise, 0.05, 0.95)

        # Convert to decimal odds (include ~4.5% vig)
        vig = 1.045
        home_decimal = vig / market_prob
        away_decimal = vig / (1 - market_prob)

        return {
            "implied_home_prob": market_prob,
            "home_decimal": home_decimal,
            "away_decimal": away_decimal,
        }

    def _place_bet(
        self,
        game_row,
        pred_row,
        side: str,
        model_prob: float,
        market_prob: float,
        decimal_odds: float,
    ):
        """Place a bet and record the result."""
        edge = model_prob - market_prob
        kelly = kelly_criterion(model_prob, decimal_odds)

        if kelly <= 0:
            return

        bet_amount = min(kelly * self.bankroll, self.max_bet_frac * self.bankroll)
        bet_amount = round(bet_amount, 2)

        if bet_amount < 1:  # Minimum bet
            return

        # Resolve bet
        home_won = game_row["home_win"] == 1
        bet_won = (side == "home" and home_won) or (side == "away" and not home_won)

        if bet_won:
            profit = bet_amount * (decimal_odds - 1)
        else:
            profit = -bet_amount

        self.bankroll += profit
        self.bankroll_history.append(self.bankroll)

        self.bet_log.append({
            "date": game_row["date"],
            "home_team": game_row["home_team"],
            "away_team": game_row["away_team"],
            "side": side,
            "model_prob": round(model_prob, 4),
            "market_prob": round(market_prob, 4),
            "edge": round(edge, 4),
            "decimal_odds": round(decimal_odds, 3),
            "kelly": round(kelly, 4),
            "bet_amount": bet_amount,
            "won": bet_won,
            "profit": round(profit, 2),
            "bankroll": round(self.bankroll, 2),
        })

    def _compute_results(self) -> dict:
        """Compute backtest summary statistics."""
        if not self.bet_log:
            return {"error": "No bets placed"}

        df = pd.DataFrame(self.bet_log)

        total_bets = len(df)
        wins = df["won"].sum()
        losses = total_bets - wins
        win_rate = wins / total_bets

        total_wagered = df["bet_amount"].sum()
        total_profit = df["profit"].sum()
        roi = total_profit / total_wagered if total_wagered > 0 else 0

        # Bankroll metrics
        peak_bankroll = max(self.bankroll_history)
        min_bankroll = min(self.bankroll_history)
        max_drawdown = self._max_drawdown()

        # Risk-adjusted returns
        daily_returns = df.groupby("date")["profit"].sum()
        sharpe = self._sharpe_ratio(daily_returns)

        # Average edge on winning vs losing bets
        avg_edge_win = df[df["won"]]["edge"].mean() if wins > 0 else 0
        avg_edge_loss = df[~df["won"]]["edge"].mean() if losses > 0 else 0

        # CLV (Closing Line Value) — how often model beat the closing line
        # Approximated by edge positivity rate
        clv_rate = (df["edge"] > 0).mean()

        return {
            "total_bets": total_bets,
            "wins": int(wins),
            "losses": int(losses),
            "win_rate": round(win_rate, 4),
            "total_wagered": round(total_wagered, 2),
            "total_profit": round(total_profit, 2),
            "roi": round(roi, 4),
            "final_bankroll": round(self.bankroll, 2),
            "peak_bankroll": round(peak_bankroll, 2),
            "min_bankroll": round(min_bankroll, 2),
            "max_drawdown": round(max_drawdown, 4),
            "sharpe_ratio": round(sharpe, 4),
            "avg_edge_winners": round(avg_edge_win, 4),
            "avg_edge_losers": round(avg_edge_loss, 4),
            "clv_rate": round(clv_rate, 4),
            "avg_bet_size": round(df["bet_amount"].mean(), 2),
            "avg_kelly": round(df["kelly"].mean(), 4),
        }

    def _max_drawdown(self) -> float:
        """Compute maximum drawdown from peak."""
        peak = self.bankroll_history[0]
        max_dd = 0

        for val in self.bankroll_history:
            peak = max(peak, val)
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)

        return max_dd

    def _sharpe_ratio(self, daily_returns: pd.Series, annualize: bool = True) -> float:
        """Compute Sharpe ratio of daily P&L."""
        if daily_returns.std() == 0 or len(daily_returns) < 10:
            return 0.0

        sr = daily_returns.mean() / daily_returns.std()

        if annualize:
            # ~180 betting days per NBA season
            sr *= np.sqrt(180)

        return sr

    def _print_results(self, results: dict):
        """Print formatted backtest results."""
        print("\n" + "=" * 70)
        print("  BACKTEST RESULTS")
        print("=" * 70)

        if "error" in results:
            print(f"  {results['error']}")
            return

        profit_color = "+" if results["total_profit"] >= 0 else ""

        print(f"""
  Bets:           {results['total_bets']} ({results['wins']}W - {results['losses']}L)
  Win Rate:       {results['win_rate']:.1%}
  Total Wagered:  ${results['total_wagered']:,.0f}
  Total Profit:   {profit_color}${results['total_profit']:,.0f}
  ROI:            {results['roi']:+.2%}

  Starting Bank:  ${self.initial_bankroll:,.0f}
  Final Bank:     ${results['final_bankroll']:,.0f}
  Peak Bank:      ${results['peak_bankroll']:,.0f}
  Max Drawdown:   {results['max_drawdown']:.1%}

  Sharpe Ratio:   {results['sharpe_ratio']:.2f}
  CLV Rate:       {results['clv_rate']:.1%}
  Avg Edge (W):   {results['avg_edge_winners']:+.2%}
  Avg Edge (L):   {results['avg_edge_losers']:+.2%}
  Avg Bet Size:   ${results['avg_bet_size']:,.0f}
  Avg Kelly:      {results['avg_kelly']:.2%}
        """)

        # Quick health check
        print("  Health Check:")
        if results["roi"] > 0:
            print("    ✓ Positive ROI")
        else:
            print("    ✗ Negative ROI — model may not have real edge")

        if results["win_rate"] > 0.52:
            print("    ✓ Win rate above break-even (~52.4% at -110)")
        else:
            print("    ✗ Win rate below break-even")

        if results["max_drawdown"] < 0.20:
            print("    ✓ Drawdown under 20%")
        else:
            print("    ✗ Drawdown exceeds 20% — reduce Kelly fraction")

        if results["sharpe_ratio"] > 1.0:
            print("    ✓ Sharpe > 1.0 (good risk-adjusted returns)")
        elif results["sharpe_ratio"] > 0.5:
            print("    ~ Sharpe 0.5-1.0 (moderate)")
        else:
            print("    ✗ Sharpe < 0.5 (poor risk-adjusted returns)")

        if results["clv_rate"] > 0.50:
            print("    ✓ Beating closing lines >50% of the time")
        else:
            print("    ✗ Not consistently beating closing lines")

        print("=" * 70)

    def _save_results(self):
        """Save bet log and bankroll history."""
        if self.bet_log:
            df = pd.DataFrame(self.bet_log)
            df.to_csv(OUTPUTS_DIR / "backtest_bets.csv", index=False)

            # Bankroll curve
            pd.DataFrame({
                "bet_number": range(len(self.bankroll_history)),
                "bankroll": self.bankroll_history,
            }).to_csv(OUTPUTS_DIR / "bankroll_curve.csv", index=False)

            print(f"\n  Bet log saved to {OUTPUTS_DIR / 'backtest_bets.csv'}")
            print(f"  Bankroll curve saved to {OUTPUTS_DIR / 'bankroll_curve.csv'}")


# ═══════════════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION
# ═══════════════════════════════════════════════════════════════════════

def monte_carlo_bankroll(
    bet_log: pd.DataFrame,
    n_simulations: int = 1000,
    n_bets: int = 500,
    bankroll: float = BANKROLL,
) -> dict:
    """
    Run Monte Carlo simulations using historical bet distribution.
    Answers: "Given my edge, what's my probability of ruin?"

    Resamples from actual bet results to simulate future paths.
    """
    if bet_log.empty:
        return {}

    np.random.seed(RANDOM_SEED)

    profits = bet_log["profit"].values
    bet_sizes = bet_log["bet_amount"].values

    final_bankrolls = []
    ruin_count = 0
    max_drawdowns = []

    for _ in range(n_simulations):
        br = bankroll
        peak = bankroll
        max_dd = 0

        for _ in range(n_bets):
            # Resample a random historical bet outcome
            idx = np.random.randint(len(profits))
            # Scale bet size proportionally to current bankroll
            scale = br / bankroll
            profit = profits[idx] * scale

            br += profit
            peak = max(peak, br)
            dd = (peak - br) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

            if br <= 0:
                ruin_count += 1
                break

        final_bankrolls.append(br)
        max_drawdowns.append(max_dd)

    final_arr = np.array(final_bankrolls)

    return {
        "simulations": n_simulations,
        "bets_per_sim": n_bets,
        "median_final": round(float(np.median(final_arr)), 2),
        "mean_final": round(float(np.mean(final_arr)), 2),
        "p5_final": round(float(np.percentile(final_arr, 5)), 2),
        "p95_final": round(float(np.percentile(final_arr, 95)), 2),
        "prob_profit": round(float((final_arr > bankroll).mean()), 4),
        "prob_ruin": round(ruin_count / n_simulations, 4),
        "prob_double": round(float((final_arr > bankroll * 2).mean()), 4),
        "median_max_dd": round(float(np.median(max_drawdowns)), 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading feature data...")
    games = pd.read_csv(PROCESSED_DIR / "features.csv", parse_dates=["date"])

    # Get unique seasons
    seasons = sorted(games["season"].unique())
    print(f"Available seasons: {seasons}")

    if len(seasons) < 2:
        print("Need at least 2 seasons for walk-forward backtest.")
    else:
        # Train on all but last season, test on last
        train_seasons = seasons[:-1]
        test_seasons = [seasons[-1]]

        bt = Backtester()
        results = bt.run(games, train_seasons, test_seasons)

        # Monte Carlo
        if bt.bet_log:
            print("\nRunning Monte Carlo simulation (1000 paths)...")
            mc = monte_carlo_bankroll(pd.DataFrame(bt.bet_log))
            print(f"\n  Monte Carlo Results:")
            print(f"    Median final bankroll: ${mc['median_final']:,.0f}")
            print(f"    P(profit):  {mc['prob_profit']:.1%}")
            print(f"    P(ruin):    {mc['prob_ruin']:.1%}")
            print(f"    P(2x):      {mc['prob_double']:.1%}")
            print(f"    5th pctile: ${mc['p5_final']:,.0f}")
            print(f"    95th pctile: ${mc['p95_final']:,.0f}")

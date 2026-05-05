#!/usr/bin/env python3
"""
NBA Quant Prediction Model — Main Pipeline
════════════════════════════════════════════

Usage:
  python run_pipeline.py              # Full pipeline: ingest → features → train → predict → scan
  python run_pipeline.py --backtest   # Run walk-forward backtest
  python run_pipeline.py --predict    # Predict today's games (uses saved model)
  python run_pipeline.py --scan       # Scan prediction markets only
  python run_pipeline.py --retrain    # Re-ingest data and retrain model

Pipeline stages:
  1. Data Ingestion   — pull NBA game data + current odds
  2. Feature Eng.     — compute ELO, rolling stats, rest days, etc.
  3. Model Training   — train ensemble (XGBoost + LightGBM + LogReg)
  4. Prediction       — generate win prob, spread, total for upcoming games
  5. Edge Detection   — compare model vs. odds, size bets via Kelly
  6. Market Scan      — scan Kalshi + Polymarket for mispriced contracts
  7. Backtest         — walk-forward simulation with bankroll tracking
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR, TRAINING_SEASONS


def main():
    parser = argparse.ArgumentParser(description="NBA Quant Prediction Model")
    parser.add_argument("--backtest", action="store_true", help="Run walk-forward backtest")
    parser.add_argument("--predict", action="store_true", help="Predict today's games only")
    parser.add_argument("--scan", action="store_true", help="Scan prediction markets only")
    parser.add_argument("--retrain", action="store_true", help="Re-ingest and retrain")
    parser.add_argument("--min-edge", type=float, default=None, help="Override min edge threshold")
    parser.add_argument("--bankroll", type=float, default=None, help="Override bankroll amount")
    parser.add_argument("--no-shrink", action="store_true",
                        help="Disable extreme-edge shrinkage (raw model output, useful for backtesting).")
    args = parser.parse_args()

    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║         NBA QUANT PREDICTION MODEL                      ║
    ║         Game Outcomes · Spread · Totals                  ║
    ║         Kalshi · Polymarket · Sportsbook Edge            ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Scan-only mode ──
    if args.scan:
        _run_scan_only()
        return

    # ── Predict-only mode (uses saved model) ──
    if args.predict:
        _run_predict_only(args)
        return

    # ── Backtest mode ──
    if args.backtest:
        _run_backtest(args)
        return

    # ── Full pipeline ──
    _run_full_pipeline(args)


def _run_full_pipeline(args):
    """Run the complete pipeline: ingest → features → train → predict → scan."""
    from data_ingest import run_full_ingest
    from features import build_features, build_upcoming_features
    from model import NBAPredictor, cross_validate_model
    from edge_detector import find_edges, print_bet_card, save_bet_card
    from market_scanner import scan_all_markets

    # Stage 1: Data Ingestion
    print("\n" + "━" * 60)
    print("  STAGE 1: DATA INGESTION")
    print("━" * 60)
    games, odds = run_full_ingest()

    # Stage 2: Feature Engineering
    print("\n" + "━" * 60)
    print("  STAGE 2: FEATURE ENGINEERING")
    print("━" * 60)
    featured_games = build_features(games)

    # Stage 3: Cross-Validation
    print("\n" + "━" * 60)
    print("  STAGE 3: CROSS-VALIDATION")
    print("━" * 60)
    cv_results = cross_validate_model(featured_games)
    print(f"\n  CV Mean Accuracy: {cv_results['accuracy'].mean():.4f} ± {cv_results['accuracy'].std():.4f}")
    print(f"  CV Mean AUC:      {cv_results['auc'].mean():.4f} ± {cv_results['auc'].std():.4f}")

    # Stage 4: Model Training
    print("\n" + "━" * 60)
    print("  STAGE 4: MODEL TRAINING")
    print("━" * 60)
    predictor = NBAPredictor()
    metrics = predictor.train(featured_games)

    # Stage 5: Generate Predictions
    print("\n" + "━" * 60)
    print("  STAGE 5: PREDICTIONS")
    print("━" * 60)
    historical_predictions = predictor.predict(featured_games)
    historical_path = OUTPUTS_DIR / "historical_predictions.csv"
    historical_predictions.tail(250).to_csv(historical_path, index=False)
    print(f"  Saved historical validation predictions to {historical_path}")

    predictions = pd.DataFrame()
    if not odds.empty:
        upcoming_features = build_upcoming_features(
            games,
            odds[["odds_game_id", "commence_time", "home_team", "away_team"]],
        )
        predictions = predictor.predict(upcoming_features)
        predictions.to_csv(OUTPUTS_DIR / "predictions.csv", index=False)
        print(f"  Saved {len(predictions)} upcoming game predictions")
    else:
        print("  No current odds available, so no upcoming predictions were generated.")

    # Stage 6: Edge Detection (against sportsbook odds)
    print("\n" + "━" * 60)
    print("  STAGE 6: EDGE DETECTION")
    print("━" * 60)
    if not odds.empty and not predictions.empty:
        bets = find_edges(predictions, odds, apply_shrinkage=not args.no_shrink)
        print_bet_card(bets)
        save_bet_card(bets)
    else:
        print("  Skipping — no odds data available (set ODDS_API_KEY in .env)")

    # Stage 7: Market Scanner
    print("\n" + "━" * 60)
    print("  STAGE 7: PREDICTION MARKET SCAN")
    print("━" * 60)
    try:
        market_edges = scan_all_markets(predictions)
    except Exception as e:
        print(f"  Market scan error: {e}")

    print("\n" + "━" * 60)
    print("  PIPELINE COMPLETE")
    print("━" * 60)
    print(f"\n  Output files in: {OUTPUTS_DIR}")
    print("  Key files:")
    print("    predictions.csv     — model predictions for upcoming games")
    print("    historical_predictions.csv — recent historical model outputs")
    print("    bet_card.csv        — sportsbook edge bets")
    print("    all_markets.csv     — prediction market listings")
    print("    market_edges.csv    — prediction market edges")


def _run_predict_only(args):
    """Load saved model and predict upcoming games."""
    from data_ingest import fetch_current_odds
    from features import build_upcoming_features
    from model import NBAPredictor
    from edge_detector import find_edges, print_bet_card, save_bet_card

    model_path = MODELS_DIR / "nba_predictor.pkl"
    if not model_path.exists():
        print("  ERROR: No trained model found. Run full pipeline first.")
        print("  Usage: python run_pipeline.py")
        sys.exit(1)

    print("  Loading saved model...")
    predictor = NBAPredictor()
    predictor.load()

    # Fetch current odds
    print("  Fetching current odds...")
    odds = fetch_current_odds()

    if not odds.empty:
        print(f"  Found {len(odds)} upcoming games with odds")

        games = _load_history_frame()
        upcoming_features = build_upcoming_features(
            games,
            odds[["odds_game_id", "commence_time", "home_team", "away_team"]],
        )
        predictions = predictor.predict(upcoming_features)
        predictions.to_csv(OUTPUTS_DIR / "predictions.csv", index=False)

        bets = find_edges(predictions, odds, apply_shrinkage=not args.no_shrink)
        print_bet_card(bets)
        save_bet_card(bets)
    else:
        print("  No odds data available.")


def _run_backtest(args):
    """Run walk-forward backtest."""
    from features import build_features
    from backtester import Backtester, monte_carlo_bankroll

    features_path = PROCESSED_DIR / "features.csv"
    if not features_path.exists():
        print("  No feature data found. Running ingestion first...")
        raw_path = PROCESSED_DIR / "all_games.csv"
        if raw_path.exists():
            print("  Found raw game data, building features...")
            games = pd.read_csv(raw_path, parse_dates=["date"])
        else:
            from data_ingest import fetch_all_historical_games
            games = fetch_all_historical_games()
        featured_games = build_features(games)
    else:
        print("  Loading feature data...")
        featured_games = pd.read_csv(features_path, parse_dates=["date"])

    seasons = sorted(featured_games["season"].unique())
    print(f"  Available seasons: {seasons}")

    if len(seasons) < 2:
        print("  ERROR: Need at least 2 seasons for backtest.")
        sys.exit(1)

    # Walk-forward: train on all but last, test on last
    train_seasons = list(seasons[:-1])
    test_seasons = [seasons[-1]]

    bankroll = args.bankroll if args.bankroll else None
    bt_kwargs = {}
    if bankroll:
        bt_kwargs["bankroll"] = bankroll
    if args.min_edge:
        bt_kwargs["min_edge"] = args.min_edge

    bt = Backtester(**bt_kwargs)
    results = bt.run(featured_games, train_seasons, test_seasons)

    # Monte Carlo simulation
    if bt.bet_log:
        print("\n  Running Monte Carlo simulation (1,000 paths × 500 bets)...")
        mc = monte_carlo_bankroll(pd.DataFrame(bt.bet_log))
        print(f"""
  Monte Carlo Results:
    Median final:   ${mc['median_final']:,.0f}
    Mean final:     ${mc['mean_final']:,.0f}
    P(profit):      {mc['prob_profit']:.1%}
    P(ruin):        {mc['prob_ruin']:.1%}
    P(2x bankroll): {mc['prob_double']:.1%}
    5th percentile: ${mc['p5_final']:,.0f}
    95th percentile: ${mc['p95_final']:,.0f}
    Median max DD:  {mc['median_max_dd']:.1%}
        """)


def _run_scan_only():
    """Scan prediction markets without model matching."""
    from market_scanner import scan_all_markets

    # Try to load predictions for matching
    pred_path = OUTPUTS_DIR / "predictions.csv"
    predictions = None
    if pred_path.exists():
        predictions = pd.read_csv(pred_path)
        print(f"  Loaded {len(predictions)} predictions for market matching")

    scan_all_markets(predictions)


def _load_history_frame() -> pd.DataFrame:
    """Load the richest completed-game history available for matchup synthesis."""
    raw_path = PROCESSED_DIR / "all_games.csv"
    features_path = PROCESSED_DIR / "features.csv"

    if raw_path.exists():
        return pd.read_csv(raw_path, parse_dates=["date"])
    if features_path.exists():
        return pd.read_csv(features_path, parse_dates=["date"])

    print("  ERROR: No historical game data found. Run the full pipeline first.")
    sys.exit(1)


if __name__ == "__main__":
    main()

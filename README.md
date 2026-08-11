# NBA Prediction Markets

Quantitative model for finding edge in NBA event contracts (moneyline, spreads, totals) on Kalshi, priced against a calibrated win-probability ensemble and sized with fractional Kelly.

## What it does

The system estimates a fair home-win probability for each game, compares it to the market's implied probability from live Kalshi orderbooks, and flags contracts where the modeled edge clears a threshold. It runs both as a historical backtester and as a live scanner against real-time markets.

## Approach

**Model.** An ensemble of XGBoost, LightGBM, and logistic regression, each producing a home-win probability that is then probability-calibrated (`CalibratedClassifierCV`) and averaged. Logistic regression anchors a well-calibrated baseline; the gradient-boosted models capture non-linear feature interactions.

**Validation.** Trained across four seasons (2021-22 through 2024-25) with `TimeSeriesSplit` cross-validation and a held-out final 20% of each season — no lookahead leakage from future games into past predictions. Evaluated on Brier score, log loss, and ROC-AUC rather than raw accuracy, because calibration matters more than hit rate when the output feeds a betting decision.

**Features.** A team-strength ELO system (home-advantage and season-reversion priors) combined with pregame market consensus, plus a live in-game feature matrix that ingests Kalshi orderbook state — bid/ask, mid, depth-weighted notional at multiple levels, and order-book imbalance — to track how market-implied probability moves against model probability during a game.

**Sizing.** Quarter-Kelly position sizing with a hard 5% max-bet cap on bankroll, and a minimum edge threshold below which no bet is placed.

## Results

Measured on the committed backtest output (`outputs/backtest_bets.csv`): **1,076 bets placed over the 2024-25 season, 2024-10-22 through 2025-04-13**, from a starting bankroll of $10,000.

| Metric | Model | Market-implied baseline |
|---|---|---|
| Brier score | 0.2187 | **0.2163** |
| Log loss | 0.6277 | **0.6226** |

Calibration of the model's probability against realized outcomes, on the games it chose to bet:

| Model probability | Bets | Actual win rate |
|---|---|---|
| 0.0 – 0.2 | 31 | 0.226 |
| 0.2 – 0.4 | 453 | 0.265 |
| 0.4 – 0.6 | 505 | 0.410 |
| 0.6 – 0.8 | 81 | 0.642 |
| 0.8 – 1.0 | 6 | 0.667 |

**Key finding: the model does not beat the market on the contracts it selects.** Its Brier score and log loss are both marginally *worse* than simply taking the market's implied probability, and the calibration table shows it running consistently hot — the 0.4–0.6 bucket wins 41% of the time, and the 0.2–0.4 bucket wins 26.5%. Since the bet-selection rule fires precisely where model probability most exceeds market probability, that overconfidence is concentrated in exactly the population being staked. The average flagged "edge" is 13.6 percentage points, which is not a plausible standing mispricing in a market this liquid; it is a signal that the model is miscalibrated relative to the price, not that the price is wrong.

The headline bankroll number from this run is not a real result and should not be read as one. See below.

## Notes & limitations

- **The backtest's bankroll curve is not credible and is retained only as a diagnostic.** The run compounds to roughly $5.65B from $10,000 across 1,076 sequential bets. The per-bet arithmetic is internally consistent — a realized +22% return per dollar staked, compounded at a 2–5% stake fraction over 1,076 bets, does produce that figure — but the premises don't survive contact with reality. It assumes bets settle strictly sequentially with immediate reinvestment (real NBA slates run concurrently, so capital cannot recycle that fast), and it assumes unlimited fill at the quoted price: by April the sizing rule is staking over $250M on a single NBA moneyline, orders of magnitude beyond what those Kalshi markets can absorb. A realistic version needs a per-market liquidity cap and same-day bankroll locking.
- **Fills are assumed at the displayed price.** Slippage, the bid/ask spread, and Kalshi fees are not deducted. On thin markets these costs are large relative to any genuine edge.
- **Selection bias in the reported metrics.** Brier and log loss above are computed on the ~1,076 games the strategy chose to bet, not on the full four-season universe, so they measure the model where it is most confident rather than on average.
- **Coverage is uneven.** Moneyline is the best-supported market; spreads and totals are partial.
- **Realized win rate is 36.3% at average decimal odds of 5.44** (≈18% implied). A 2x standing discrepancy of that size across a full season is far more likely to indicate a join or side-assignment issue between the game records and the odds records than a genuine inefficiency; that reconciliation is the next thing to fix.

## Repository layout

| File | Purpose |
|---|---|
| `config.py` | Central config: paths, ELO params, Kelly settings, API keys (via env vars) |
| `data_ingest.py` | Pulls historical games (nba_api) and odds (The-Odds-API) |
| `features.py` | ELO engine and feature construction |
| `model.py` | Ensemble training, calibration, evaluation |
| `edge_detector.py` | Edge = model prob − implied prob; fractional Kelly sizing |
| `backtester.py` | Historical backtest and bankroll curve |
| `market_scanner.py` | Live Kalshi market search and orderbook pulls |
| `live_data.py` / `live_adjustments.py` | Real-time odds and in-game state |
| `ev_analyzer.py` | Scans active markets for +EV opportunities |
| `order_executor.py` | Account balance, positions, order placement |
| `run_pipeline.py` | End-to-end driver: ingest → features → train → backtest |

Project code lives in `nba_prediction_model/`. `in_game_training_matrix.csv` at the repo root is a captured sample of the live orderbook feature matrix.

## Setup

```bash
cd nba_prediction_model
pip install -r requirements.txt
```

Create a `.env` file (not committed) with your keys:

```
ODDS_API_KEY=...
KALSHI_API_KEY=...
KALSHI_API_SECRET=...
```

## Usage

All commands run from inside `nba_prediction_model/`.

```bash
# Train and backtest
python backtester.py

# Scan all active NBA markets for +EV opportunities
python ev_analyzer.py --all --bankroll 10000

# Search Kalshi markets / pull an orderbook
python market_scanner.py --series KXNBAGAME
python market_scanner.py --market KXNBAGAME-26APR12ATLMIA-MIA

# Account
python order_executor.py balance
python order_executor.py positions
```

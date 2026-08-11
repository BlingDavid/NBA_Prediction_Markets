# NBA Prediction Markets

Quantitative model for finding edge in NBA event contracts (moneyline, spreads, totals) on Kalshi, priced against a calibrated win-probability ensemble and sized with fractional Kelly.

## What it does

The system estimates a fair home-win probability for each game, compares it to an implied probability, and flags contracts where the modeled edge clears a threshold. It runs in two modes, and the distinction matters: the **live scanner** (`market_scanner.py`, `ev_analyzer.py`) prices against real Kalshi orderbooks, while the **historical backtester** currently prices against a synthetic line rather than recorded odds — see Results.

## Approach

**Model.** An ensemble of XGBoost, LightGBM, and logistic regression, each producing a home-win probability that is then probability-calibrated (`CalibratedClassifierCV`) and averaged. Logistic regression anchors a well-calibrated baseline; the gradient-boosted models capture non-linear feature interactions.

**Validation.** Trained across four seasons (2021-22 through 2024-25) with `TimeSeriesSplit` cross-validation and a held-out final 20% of each season — no lookahead leakage from future games into past predictions. Evaluated on Brier score, log loss, and ROC-AUC rather than raw accuracy, because calibration matters more than hit rate when the output feeds a betting decision.

**Features.** A team-strength ELO system (home-advantage and season-reversion priors) combined with pregame market consensus, plus a live in-game feature matrix that ingests Kalshi orderbook state — bid/ask, mid, depth-weighted notional at multiple levels, and order-book imbalance — to track how market-implied probability moves against model probability during a game.

**Sizing.** Quarter-Kelly position sizing with a hard 5% max-bet cap on bankroll, and a minimum edge threshold below which no bet is placed.

## Results

**The backtest does not evaluate the model against a real market, and its profit figures should not be read as evidence of edge.** `backtester.py::_generate_synthetic_odds` constructs the counterparty price synthetically, as `elo_home_win_prob + N(0, 0.04)` clipped to [0.05, 0.95] with 4.5% vig applied. Since `elo_home_win_prob` is itself one of the features the ensemble trains on (`features.py::get_feature_columns`), the "edge" being traded is the ensemble's disagreement with one of its own inputs. Details in Notes & limitations below. The honest summary of this repo today: the feature pipeline, the ensemble, and the live Kalshi integration are real; the backtest that scores them is not.

The classification metrics below are meaningful — they are measured against actual game outcomes. Held-out test split, 2024-25 season:

| Metric | Ensemble | ELO baseline |
|---|---|---|
| Accuracy | 0.6447 | — |
| ROC-AUC | 0.7142 | — |
| Brier score | 0.2213 | — |

Against the *synthetic* line, over 1,056 bets in the 2024-25 season from a $10,000 bankroll, the model posts a Brier score of 0.2187 versus 0.2163 for that line — i.e. it does not even outperform ELO-plus-noise on the contracts it selects.

Calibration of the model's probability against realized outcomes, on the games it chose to bet:

| Model probability | Bets | Actual win rate |
|---|---|---|
| 0.0 – 0.2 | 31 | 0.226 |
| 0.2 – 0.4 | 453 | 0.265 |
| 0.4 – 0.6 | 505 | 0.410 |
| 0.6 – 0.8 | 81 | 0.642 |
| 0.8 – 1.0 | 6 | 0.667 |

**Key finding: the model runs consistently overconfident.** The 0.4–0.6 bucket wins 41% of the time and the 0.2–0.4 bucket wins 26.5%. Because the bet-selection rule fires precisely where model probability most exceeds the reference price, that overconfidence is concentrated in exactly the population being staked. The average flagged "edge" is 13.6 percentage points — against a real market that would be implausible on its face, and here it simply measures how far the ensemble departs from its own ELO input.

## Notes & limitations

- **The backtest prices against a synthetic line, not a market — this is the top item to fix.** `_generate_synthetic_odds` derives the counterparty price from `elo_home_win_prob`, a feature the ensemble is trained on, plus `N(0, 0.04)` noise and 4.5% vig. The consequences: the reported "edge" is self-referential; the 100% CLV rate in the health check is tautological, since the bet trigger *is* deviation from that line; and the maximum decimal odds in the output is exactly 20.9, which is `1.045 / (1 − 0.95)` — the vig over the clip bound, not anything a market quoted. The fix is to wire the historical odds that `data_ingest.py` already knows how to pull into the backtest path and re-run. Until then, no profit number from `backtester.py` means anything.
- **The bankroll curve is not credible even on its own terms.** The run compounds to roughly $7.6B from $10,000. The per-bet arithmetic is internally consistent, but it assumes bets settle strictly sequentially with immediate reinvestment (real NBA slates run concurrently, so capital cannot recycle that fast), and unlimited fill at the quoted price: by April the sizing rule stakes over $250M on a single NBA moneyline, orders of magnitude beyond what those Kalshi markets can absorb. A realistic version needs a per-market liquidity cap and same-day bankroll locking.
- **The backtest is not reproducible run to run.** The synthetic noise draw is never seeded, so bet count and ROI move between identical invocations — 1,056 and 1,076 bets on two runs of the same code and data.
- **The Monte Carlo is broken.** `monte_carlo_bankroll` resamples historical profits without rescaling stake to the current bankroll, so paths run negative: it reports a median final bankroll of −$2.1M and P(ruin) of 100%, on the same bet log that the main backtest reports as profitable.
- **Fills are assumed at the displayed price.** Slippage, the bid/ask spread, and Kalshi fees are not deducted. On thin markets these costs are large relative to any genuine edge.
- **Selection bias in the reported metrics.** Brier and log loss over the bet population are computed on the ~1,056 games the strategy chose, not the full four-season universe, so they measure the model where it is most confident rather than on average.
- **Coverage is uneven.** Moneyline is the best-supported market; spreads and totals are partial.

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

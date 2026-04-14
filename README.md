# NBA_Prediction_Markets
Finding edge within prediction markets for NBA games mainly for Money line, Spreads, and Total Points.


Built out a quantitative model that runs backtests based on historical NBA game data. This also has access to connect to The-Odds-API as well as the Kalshi API to pull in real-time orderbook data.



Important commands to search data and execute:

1. **Search Kalshi by keyword:**

python market_scanner.py --search "FED" --limit 20
python market_scanner.py --search "WEATHER-NYC" --limit 10

2. **Pull orderbook for a specific market ticker:**

python -c "from market_scanner import KalshiClient; import json; print(json.dumps(KalshiClient().get_orderbook('KXNBAGAME-26APR13LALBOS-LAL'), indent=2))"

3. **To pull up information on every available NBA game:** 

python market_scanner.py --series KXNBAGAME

4. **markets:** KXNBAGAME-YYMMMDD{AWAY}{HOME}-{TEAM}: example:...

python market_scanner.py --market KXNBAGAME-26APR12ATLMIA-MIA

5. **Run a backtest:**

python backtester.py

6. **Account Balance:**

python order_executor.py balance

7. **Open Positions:**

python order_executor.py positions

8. **Scan all active NBA markets for +EV opportunities**:

python ev_analyzer.py --all --bankroll 10000

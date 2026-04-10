"""
Prediction Market Scanner
──────────────────────────
Scans Kalshi and Polymarket for NBA-related markets.
Compares prediction market prices against our model's probabilities
to find mispriced contracts.

Kalshi: event contracts priced 0-100 cents (= probability in %)
Polymarket: binary outcome tokens priced 0-$1 (= probability)
"""

import json
from datetime import datetime

import pandas as pd
import requests

from config import (
    KALSHI_API_KEY,
    KALSHI_API_SECRET,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    TEAM_ABBREV_MAP,
    TEAM_NAME_TO_ABBREV,
    MIN_EDGE_THRESHOLD,
    BANKROLL,
)
from edge_detector import kelly_criterion


# ═══════════════════════════════════════════════════════════════════════
# KALSHI SCANNER
# ═══════════════════════════════════════════════════════════════════════

class KalshiScanner:
    """
    Scan Kalshi for NBA markets.
    Public endpoints don't require auth for reading market data.
    """

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

    def search_nba_markets(self) -> pd.DataFrame:
        """Search for active NBA-related markets on Kalshi."""
        try:
            # Search for NBA events
            resp = self.session.get(
                f"{self.BASE_URL}/events",
                params={
                    "status": "open",
                    "series_ticker": "NBA",  # Try series-based search
                    "limit": 100,
                },
                timeout=15,
            )

            if resp.status_code == 200:
                events = resp.json().get("events", [])
                return self._parse_kalshi_events(events)

            # Fallback: search by keyword
            resp = self.session.get(
                f"{self.BASE_URL}/events",
                params={
                    "status": "open",
                    "limit": 200,
                },
                timeout=15,
            )

            if resp.status_code == 200:
                all_events = resp.json().get("events", [])
                nba_events = [
                    e for e in all_events
                    if any(kw in (e.get("title", "") + e.get("category", "")).lower()
                           for kw in ["nba", "basketball", "lakers", "celtics", "warriors"])
                ]
                return self._parse_kalshi_events(nba_events)

        except Exception as e:
            print(f"  Kalshi API error: {e}")

        return pd.DataFrame()

    def get_market_orderbook(self, ticker: str) -> dict:
        """Get the order book for a specific market."""
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/markets/{ticker}/orderbook",
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"  Kalshi orderbook error for {ticker}: {e}")
        return {}

    def _parse_kalshi_events(self, events: list[dict]) -> pd.DataFrame:
        """Parse Kalshi events into a clean DataFrame."""
        rows = []
        for event in events:
            for market in event.get("markets", []):
                rows.append({
                    "platform": "kalshi",
                    "event_title": event.get("title", ""),
                    "market_ticker": market.get("ticker", ""),
                    "market_title": market.get("title", market.get("subtitle", "")),
                    "yes_price": market.get("yes_price", 0) / 100,  # cents to probability
                    "no_price": market.get("no_price", 0) / 100,
                    "volume": market.get("volume", 0),
                    "open_interest": market.get("open_interest", 0),
                    "close_time": market.get("close_time", ""),
                    "status": market.get("status", ""),
                })
        return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# POLYMARKET SCANNER
# ═══════════════════════════════════════════════════════════════════════

class PolymarketScanner:
    """
    Scan Polymarket for NBA markets.
    Uses the Gamma Markets API (public, no auth required).
    """

    GAMMA_URL = "https://gamma-api.polymarket.com"

    def __init__(self):
        self.session = requests.Session()

    def search_nba_markets(self) -> pd.DataFrame:
        """Search for active NBA markets on Polymarket."""
        try:
            # Search for NBA-related markets
            resp = self.session.get(
                f"{self.GAMMA_URL}/markets",
                params={
                    "tag": "nba",
                    "active": True,
                    "closed": False,
                    "limit": 100,
                },
                timeout=15,
            )

            if resp.status_code == 200:
                markets = resp.json()
                if isinstance(markets, list):
                    return self._parse_poly_markets(markets)

            # Fallback: broader search
            resp = self.session.get(
                f"{self.GAMMA_URL}/markets",
                params={
                    "active": True,
                    "closed": False,
                    "limit": 200,
                    "tag": "sports",
                },
                timeout=15,
            )

            if resp.status_code == 200:
                all_markets = resp.json()
                if isinstance(all_markets, list):
                    nba_markets = [
                        m for m in all_markets
                        if any(kw in m.get("question", "").lower()
                               for kw in ["nba", "basketball", "lakers", "celtics",
                                           "warriors", "knicks", "bucks"])
                    ]
                    return self._parse_poly_markets(nba_markets)

        except Exception as e:
            print(f"  Polymarket API error: {e}")

        return pd.DataFrame()

    def _parse_poly_markets(self, markets: list[dict]) -> pd.DataFrame:
        """Parse Polymarket markets into a clean DataFrame."""
        rows = []
        for m in markets:
            # Polymarket uses outcome prices that sum to ~$1
            outcomes = m.get("outcomes", [])
            prices = m.get("outcomePrices", [])

            if len(outcomes) >= 2 and len(prices) >= 2:
                try:
                    yes_price = float(prices[0])
                    no_price = float(prices[1])
                except (ValueError, TypeError):
                    yes_price, no_price = 0.5, 0.5
            else:
                yes_price, no_price = 0.5, 0.5

            rows.append({
                "platform": "polymarket",
                "event_title": m.get("groupSlug", m.get("question", "")),
                "market_ticker": m.get("conditionId", m.get("id", "")),
                "market_title": m.get("question", ""),
                "yes_price": yes_price,
                "no_price": no_price,
                "volume": m.get("volume", 0),
                "open_interest": m.get("liquidityNum", 0),
                "close_time": m.get("endDate", ""),
                "status": "active" if m.get("active") else "closed",
            })

        return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# MARKET MATCHING & EDGE FINDING
# ═══════════════════════════════════════════════════════════════════════

def match_markets_to_predictions(
    markets: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attempt to match prediction market contracts to our model's game predictions.
    Uses fuzzy matching on team names in market titles.
    """
    if markets.empty or predictions.empty:
        return pd.DataFrame()

    matches = []

    for _, market in markets.iterrows():
        title = market["market_title"].lower()

        # Try to find team names in the market title
        home_match, away_match = None, None

        for abbrev, full_name in TEAM_ABBREV_MAP.items():
            name_parts = full_name.lower().split()
            # Match on city or team name
            if any(part in title for part in name_parts):
                if home_match is None:
                    home_match = abbrev
                elif away_match is None and abbrev != home_match:
                    away_match = abbrev

        if home_match and away_match:
            # Find matching prediction
            pred = predictions[
                ((predictions["home_team"] == home_match) & (predictions["away_team"] == away_match)) |
                ((predictions["home_team"] == away_match) & (predictions["away_team"] == home_match))
            ]

            if not pred.empty:
                pred_row = pred.iloc[0]

                # Determine which team is "yes" in the market
                # Heuristic: first team mentioned is usually the "yes" outcome
                first_team = home_match  # default
                for abbrev in [home_match, away_match]:
                    full = TEAM_ABBREV_MAP.get(abbrev, "").lower()
                    idx = min(
                        (title.find(part) for part in full.split() if part in title),
                        default=999,
                    )
                    if idx < title.find(TEAM_ABBREV_MAP.get(
                        away_match if abbrev == home_match else home_match, ""
                    ).lower().split()[-1]):
                        first_team = abbrev
                        break

                # Model probability for the "yes" team
                if first_team == pred_row["home_team"]:
                    model_prob = pred_row["home_win_prob"]
                else:
                    model_prob = pred_row["away_win_prob"]

                market_prob = market["yes_price"]
                edge = model_prob - market_prob

                matches.append({
                    "platform": market["platform"],
                    "market_title": market["market_title"],
                    "market_ticker": market["market_ticker"],
                    "yes_team": first_team,
                    "market_prob": round(market_prob, 4),
                    "model_prob": round(model_prob, 4),
                    "edge": round(edge, 4),
                    "volume": market["volume"],
                    "kelly": round(kelly_criterion(model_prob, 1 / market_prob if market_prob > 0 else 2), 4),
                    "bet_size": round(kelly_criterion(model_prob, 1 / market_prob if market_prob > 0 else 2) * BANKROLL, 2),
                    "close_time": market["close_time"],
                })

    result = pd.DataFrame(matches)
    if not result.empty:
        result = result[result["edge"].abs() > MIN_EDGE_THRESHOLD]
        result = result.sort_values("edge", ascending=False).reset_index(drop=True)

    return result


# ═══════════════════════════════════════════════════════════════════════
# SCAN ALL MARKETS
# ═══════════════════════════════════════════════════════════════════════

def scan_all_markets(predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Scan Kalshi and Polymarket for NBA opportunities.
    Optionally match against model predictions.
    """
    print("\n[Market Scanner]")

    all_markets = []

    # Kalshi
    print("  Scanning Kalshi...")
    kalshi = KalshiScanner()
    kalshi_markets = kalshi.search_nba_markets()
    if not kalshi_markets.empty:
        all_markets.append(kalshi_markets)
        print(f"    Found {len(kalshi_markets)} NBA markets on Kalshi")
    else:
        print("    No NBA markets found on Kalshi")

    # Polymarket
    print("  Scanning Polymarket...")
    poly = PolymarketScanner()
    poly_markets = poly.search_nba_markets()
    if not poly_markets.empty:
        all_markets.append(poly_markets)
        print(f"    Found {len(poly_markets)} NBA markets on Polymarket")
    else:
        print("    No NBA markets found on Polymarket")

    if not all_markets:
        print("  No markets found on any platform.")
        return pd.DataFrame()

    combined = pd.concat(all_markets, ignore_index=True)
    combined.to_csv(OUTPUTS_DIR / "all_markets.csv", index=False)

    # Match against predictions if available
    if predictions is not None and not predictions.empty:
        print("\n  Matching markets to model predictions...")
        edges = match_markets_to_predictions(combined, predictions)
        if not edges.empty:
            edges.to_csv(OUTPUTS_DIR / "market_edges.csv", index=False)
            print(f"    Found {len(edges)} actionable edges")
            _print_market_edges(edges)
        return edges

    return combined


def _print_market_edges(edges: pd.DataFrame):
    """Print formatted market edge opportunities."""
    print("\n" + "=" * 80)
    print("  PREDICTION MARKET EDGES")
    print("=" * 80)

    for _, row in edges.iterrows():
        direction = "BUY YES" if row["edge"] > 0 else "BUY NO"
        print(f"\n  [{row['platform'].upper()}] {row['market_title']}")
        print(f"  ├─ Action:     {direction} ({row['yes_team']})")
        print(f"  ├─ Market:     {row['market_prob']:.1%}")
        print(f"  ├─ Model:      {row['model_prob']:.1%}")
        print(f"  ├─ Edge:       {row['edge']:+.1%}")
        print(f"  ├─ Kelly:      {row['kelly']:.2%}")
        print(f"  ├─ Bet size:   ${row['bet_size']:,.0f}")
        print(f"  └─ Volume:     {row['volume']:,.0f}")

    print(f"\n{'─' * 80}\n")


if __name__ == "__main__":
    # Standalone scan (no model matching)
    markets = scan_all_markets()
    if not markets.empty:
        print(f"\nTotal markets found: {len(markets)}")

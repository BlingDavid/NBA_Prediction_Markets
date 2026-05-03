"""
Prediction Market Scanner
──────────────────────────
Scans Kalshi and Polymarket for NBA-related markets.
Compares prediction market prices against our model's probabilities
to find mispriced contracts.

Kalshi API: Public endpoints at api.elections.kalshi.com/trade-api/v2
  - No auth required for market data (series, events, markets, orderbook)
  - Despite "elections" subdomain, covers ALL Kalshi markets

Usage:
  python market_scanner.py                       # Scan all NBA markets
  python market_scanner.py --browse              # Interactive browser: list series & pick markets
  python market_scanner.py --series KXNBA        # List all markets in a series
  python market_scanner.py --market KXNBA-25APR10-BOS  # Get details + orderbook for one market
  python market_scanner.py --search "lakers"     # Search markets by keyword
"""

import argparse
import json
import re
from datetime import datetime

import pandas as pd
import requests

from config import (
    OUTPUTS_DIR,
    PROCESSED_DIR,
    TEAM_ABBREV_MAP,
    TEAM_NAME_TO_ABBREV,
    MIN_EDGE_THRESHOLD,
    BANKROLL,
)
from edge_detector import kelly_criterion


# Kalshi series tickers we want to scan for NBA markets.
#
# - KXNBA: futures-style championship markets (KXNBA-26-{TEAM} = "Will <team>
#   win the 2026 Finals?"). One per team. Useful for season-long edges.
# - KXNBAPLAYOFFS / KXNBAFINALS: round-specific futures markets.
# - KXNBAGAME: per-game H2H markets (KXNBAGAME-26MAY03DETORL-DET = "Will DET
#   beat ORL on May 3?"). Two per game (one contract per team). These are the
#   markets the per-game pipeline (run_pipeline --scan, ev_analyzer) needs to
#   match against the model's per-game predictions.
NBA_SERIES_TICKERS = ["KXNBAGAME", "KXNBA", "KXNBAPLAYOFFS", "KXNBAFINALS", "NBA"]


# ═══════════════════════════════════════════════════════════════════════
# KALSHI API CLIENT
# ═══════════════════════════════════════════════════════════════════════

class KalshiClient:
    """
    Client for Kalshi's public market data API.
    No authentication needed — all endpoints are read-only and public.

    Key concepts:
      - Series: a recurring theme (e.g., "NBA Games" → KXNBA)
      - Event: a specific occurrence within a series (e.g., "NBA Games Apr 10")
      - Market: a tradeable contract within an event (e.g., "Lakers vs Celtics")
      - Orderbook: live bids for YES and NO on a specific market
    """

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
        })

    # ── Series ─────────────────────────────────────────────────────

    def get_series(self, series_ticker: str) -> dict:
        """Get info about a specific series (e.g., KXNBA)."""
        resp = self.session.get(
            f"{self.BASE_URL}/series/{series_ticker}",
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("series", {})

    # ── Events ─────────────────────────────────────────────────────

    def get_events(
        self,
        series_ticker: str | None = None,
        status: str = "open",
        limit: int = 100,
    ) -> list[dict]:
        """
        Get events, optionally filtered by series.
        Status: 'open', 'closed', 'settled', or omit for all.
        """
        params = {"limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status

        resp = self.session.get(
            f"{self.BASE_URL}/events",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("events", [])

    # ── Markets ────────────────────────────────────────────────────

    def get_markets(
        self,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        status: str = "open",
        limit: int = 200,
        cursor: str | None = None,
    ) -> dict:
        """
        Get markets with optional filters.
        Returns dict with 'markets' list and 'cursor' for pagination.
        """
        params = {"limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor

        resp = self.session.get(
            f"{self.BASE_URL}/markets",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def get_market(self, ticker: str) -> dict:
        """Get details for a single market by ticker."""
        resp = self.session.get(
            f"{self.BASE_URL}/markets/{ticker}",
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("market", {})

    def get_orderbook(self, ticker: str) -> dict:
        """Get the live orderbook for a market."""
        resp = self.session.get(
            f"{self.BASE_URL}/markets/{ticker}/orderbook",
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    # ── Search ─────────────────────────────────────────────────────

    def search_markets(
        self,
        keywords: list[str],
        status: str = "open",
        limit: int = 200,
        max_pages: int = 3,
    ) -> list[dict]:
        """
        Search for markets matching keywords in title.
        Paginates through Kalshi markets with rate limiting.
        Caps at max_pages to avoid hitting API rate limits.
        """
        import time

        all_markets = []
        cursor = None
        page = 0

        while page < max_pages:
            try:
                result = self.get_markets(status=status, limit=limit, cursor=cursor)
            except requests.exceptions.HTTPError as e:
                if "429" in str(e):
                    print(f"    Rate limited — pausing 5s (fetched {len(all_markets)} markets so far)...")
                    time.sleep(5)
                    try:
                        result = self.get_markets(status=status, limit=limit, cursor=cursor)
                    except Exception:
                        break
                else:
                    print(f"    API error: {e}")
                    break

            markets = result.get("markets", [])
            all_markets.extend(markets)
            cursor = result.get("cursor")
            page += 1

            if not cursor or not markets:
                break

            # Rate limit: pause between pages
            time.sleep(0.5)

        print(f"    Scanned {len(all_markets)} markets across {page} page(s)")

        # Filter by keywords
        keywords_lower = [kw.lower() for kw in keywords]
        matched = []
        for m in all_markets:
            text = (
                m.get("title", "") + " " +
                m.get("subtitle", "") + " " +
                m.get("event_ticker", "") + " " +
                m.get("ticker", "")
            ).lower()
            if any(kw in text for kw in keywords_lower):
                matched.append(m)

        return matched

    def search_nba_markets(self) -> list[dict]:
        """Search specifically for NBA-related markets.

        Pulls every series in NBA_SERIES_TICKERS so the per-game H2H series
        (KXNBAGAME) is included alongside the futures series (KXNBA et al).
        Without KXNBAGAME, run_pipeline --scan only sees Finals-winner
        contracts and never matches the model's per-game predictions.
        """
        markets = []
        seen_tickers: set[str] = set()

        for ticker in NBA_SERIES_TICKERS:
            try:
                result = self.get_markets(series_ticker=ticker, status="open")
                for m in result.get("markets", []):
                    t = m.get("ticker", "")
                    if t and t not in seen_tickers:
                        markets.append(m)
                        seen_tickers.add(t)
            except Exception:
                continue

        if markets:
            return markets

        # Fallback: keyword search across all markets
        nba_keywords = [
            "nba", "basketball", "lakers", "celtics", "warriors",
            "knicks", "bucks", "nuggets", "76ers", "heat",
            "cavaliers", "thunder", "timberwolves", "mavericks",
        ]
        return self.search_markets(nba_keywords)


# ═══════════════════════════════════════════════════════════════════════
# KALSHI DATA PARSER
# ═══════════════════════════════════════════════════════════════════════

def parse_kalshi_markets(markets: list[dict]) -> pd.DataFrame:
    """Parse raw Kalshi market dicts into a clean DataFrame."""
    rows = []
    for m in markets:
        # Prices: yes_bid/yes_ask are in dollars (0.00-1.00 range)
        # Some fields use cents, some dollars — handle both
        yes_bid = m.get("yes_bid_dollars") or m.get("yes_bid", 0)
        yes_ask = m.get("yes_ask_dollars") or m.get("yes_ask", 0)
        no_bid = m.get("no_bid_dollars") or m.get("no_bid", 0)
        no_ask = m.get("no_ask_dollars") or m.get("no_ask", 0)

        # If values look like cents (>1), convert to probability
        if isinstance(yes_bid, (int, float)) and yes_bid > 1:
            yes_bid = yes_bid / 100
        if isinstance(yes_ask, (int, float)) and yes_ask > 1:
            yes_ask = yes_ask / 100
        if isinstance(no_bid, (int, float)) and no_bid > 1:
            no_bid = no_bid / 100
        if isinstance(no_ask, (int, float)) and no_ask > 1:
            no_ask = no_ask / 100

        # Midpoint as best estimate of market price
        yes_mid = (float(yes_bid or 0) + float(yes_ask or 0)) / 2 if yes_bid and yes_ask else float(yes_bid or yes_ask or 0)

        rows.append({
            "platform": "kalshi",
            "ticker": m.get("ticker", ""),
            "event_ticker": m.get("event_ticker", ""),
            "series_ticker": m.get("series_ticker", ""),
            "title": m.get("title", ""),
            "subtitle": m.get("subtitle", ""),
            "yes_bid": round(float(yes_bid or 0), 4),
            "yes_ask": round(float(yes_ask or 0), 4),
            "yes_mid": round(yes_mid, 4),
            "no_bid": round(float(no_bid or 0), 4),
            "no_ask": round(float(no_ask or 0), 4),
            "volume": m.get("volume", 0) or m.get("volume_fp", 0),
            "open_interest": m.get("open_interest", 0) or m.get("open_interest_fp", 0),
            "status": m.get("status", ""),
            "close_time": m.get("close_time", m.get("expiration_time", "")),
            "result": m.get("result", ""),
            "url": f"https://kalshi.com/markets/{m.get('ticker', '')}",
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# POLYMARKET SCANNER (unchanged)
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
            resp = self.session.get(
                f"{self.GAMMA_URL}/markets",
                params={"tag": "nba", "active": True, "closed": False, "limit": 100},
                timeout=15,
            )
            if resp.status_code == 200:
                markets = resp.json()
                if isinstance(markets, list) and markets:
                    return self._parse(markets)

            # Fallback: sports tag
            resp = self.session.get(
                f"{self.GAMMA_URL}/markets",
                params={"active": True, "closed": False, "limit": 200, "tag": "sports"},
                timeout=15,
            )
            if resp.status_code == 200:
                all_markets = resp.json()
                if isinstance(all_markets, list):
                    nba = [m for m in all_markets
                           if any(kw in m.get("question", "").lower()
                                  for kw in ["nba", "basketball", "lakers", "celtics",
                                             "warriors", "knicks", "bucks"])]
                    return self._parse(nba)
        except Exception as e:
            print(f"  Polymarket API error: {e}")

        return pd.DataFrame()

    def _parse(self, markets: list[dict]) -> pd.DataFrame:
        rows = []
        for m in markets:
            outcomes = m.get("outcomes", [])
            prices = m.get("outcomePrices", [])
            try:
                yes_price = float(prices[0]) if len(prices) >= 1 else 0.5
                no_price = float(prices[1]) if len(prices) >= 2 else 0.5
            except (ValueError, TypeError):
                yes_price, no_price = 0.5, 0.5

            rows.append({
                "platform": "polymarket",
                "ticker": m.get("conditionId", m.get("id", "")),
                "event_ticker": m.get("groupSlug", ""),
                "series_ticker": "",
                "title": m.get("question", ""),
                "subtitle": "",
                "yes_bid": yes_price,
                "yes_ask": yes_price,
                "yes_mid": yes_price,
                "no_bid": no_price,
                "no_ask": no_price,
                "volume": m.get("volume", 0),
                "open_interest": m.get("liquidityNum", 0),
                "status": "active" if m.get("active") else "closed",
                "close_time": m.get("endDate", ""),
                "result": "",
                "url": f"https://polymarket.com/event/{m.get('groupSlug', '')}",
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
    Match prediction market contracts to our model's game predictions.

    Uses fuzzy matching on team names in market titles to find the matching
    game, then determines which side the contract pays out on:

    - Kalshi per-game tickers (KXNBAGAME-YYMMMDDAWAYHOME-TEAM) carry the
      YES team in the ticker suffix; the title text is the same string for
      both contracts of a given game and so is unreliable as a side cue.
    - All other markets use first-team-in-title as the YES side.
    """
    if markets.empty or predictions.empty:
        return pd.DataFrame()

    matches = []

    for _, market in markets.iterrows():
        ticker = _safe_text(market.get("ticker", ""))

        # KXNBAGAME tickers encode both teams in the AWAYHOME segment, which
        # is more reliable than parsing the truncated title text Kalshi uses
        # in its UI (e.g. "Los Angeles L at Oklahoma City Winner?").
        kx_pair = _kxnbagame_team_pair(ticker)
        if kx_pair is not None:
            away_abbrev, home_abbrev = kx_pair
            first_team, second_team = away_abbrev, home_abbrev
        else:
            title_text = _safe_text(market.get("title", ""))
            subtitle_text = _safe_text(market.get("subtitle", ""))
            title = (title_text + " " + subtitle_text).lower()

            found_teams = _extract_title_teams(title)
            if len(found_teams) < 2:
                continue
            first_team, second_team = found_teams[0], found_teams[1]

        # Find matching prediction
        pred = predictions[
            ((predictions["home_team"] == first_team) & (predictions["away_team"] == second_team)) |
            ((predictions["home_team"] == second_team) & (predictions["away_team"] == first_team))
        ]

        if pred.empty:
            continue

        pred_row = pred.iloc[0]

        # Determine the YES team for this contract.
        yes_team = _kxnbagame_yes_team(ticker, {first_team, second_team})
        if yes_team is None:
            yes_team = first_team

        market_prob = market["yes_mid"] if market["yes_mid"] > 0 else market["yes_bid"]

        if yes_team == pred_row["home_team"]:
            model_prob = pred_row["home_win_prob"]
        else:
            model_prob = pred_row["away_win_prob"]

        edge = model_prob - market_prob
        decimal_odds = 1 / market_prob if market_prob > 0 else 2.0

        matches.append({
            "platform": market["platform"],
            "ticker": market["ticker"],
            "title": market.get("title", ""),
            "yes_team": yes_team,
            "market_prob": round(market_prob, 4),
            "model_prob": round(model_prob, 4),
            "edge": round(edge, 4),
            "volume": market["volume"],
            "kelly": round(kelly_criterion(model_prob, decimal_odds), 4),
            "bet_size": round(kelly_criterion(model_prob, decimal_odds) * BANKROLL, 2),
            "close_time": market["close_time"],
            "url": market.get("url", ""),
        })

    result = pd.DataFrame(matches)
    if not result.empty:
        result = result[result["edge"].abs() > MIN_EDGE_THRESHOLD]
        result = result.sort_values("edge", ascending=False).reset_index(drop=True)
    return result


def _safe_text(value) -> str:
    """Coerce a possibly-NaN/None field to string. pandas reads empty CSV
    cells as float NaN; concatenating those with strings raises TypeError."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def _kxnbagame_yes_team(ticker: str, valid_teams: set[str]) -> str | None:
    """For a Kalshi per-game H2H ticker (KXNBAGAME-YYMMMDDAWAYHOME-TEAM),
    return the team abbreviation in the suffix iff it's in `valid_teams`.
    Otherwise None — caller falls back to the title-order heuristic."""
    if not ticker.startswith("KXNBAGAME-"):
        return None
    parts = ticker.split("-")
    if len(parts) < 3:
        return None
    suffix = parts[-1].upper()
    return suffix if suffix in valid_teams else None


def _kxnbagame_team_pair(ticker: str) -> tuple[str, str] | None:
    """Parse the (away, home) team pair from a KXNBAGAME ticker.

    Format is KXNBAGAME-YYMMMDDAWAYHOME-TEAM, where the middle segment is
    YY (2) + MMM (3) + DD (2) + AWAY (3) + HOME (3) = 13 chars. The last
    six characters of that segment are the team abbreviations.
    """
    if not ticker.startswith("KXNBAGAME-"):
        return None
    parts = ticker.split("-")
    if len(parts) < 3:
        return None
    middle = parts[1]
    if len(middle) < 13:
        return None
    away = middle[7:10].upper()
    home = middle[10:13].upper()
    if not (away.isalpha() and home.isalpha()):
        return None
    return away, home


def _extract_title_teams(title: str) -> list[str]:
    """Find teams in market text and preserve the order they appear in the title."""
    matches: list[tuple[int, str]] = []

    for abbrev, full_name in TEAM_ABBREV_MAP.items():
        aliases = {
            full_name.lower(),
            full_name.lower().split()[-1],
            abbrev.lower(),
        }

        best_pos = None
        for alias in aliases:
            pattern = rf"(?<![a-z]){re.escape(alias)}(?![a-z])"
            found = re.search(pattern, title)
            if found and (best_pos is None or found.start() < best_pos):
                best_pos = found.start()

        if best_pos is not None:
            matches.append((best_pos, abbrev))

    ordered = []
    seen = set()
    for _, abbrev in sorted(matches, key=lambda item: item[0]):
        if abbrev not in seen:
            ordered.append(abbrev)
            seen.add(abbrev)

    return ordered


# ═══════════════════════════════════════════════════════════════════════
# SCAN ALL MARKETS
# ═══════════════════════════════════════════════════════════════════════

def scan_all_markets(predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Scan Kalshi and Polymarket for NBA opportunities."""
    print("\n[Market Scanner]")
    all_markets = []

    # Kalshi
    print("  Scanning Kalshi...")
    try:
        kalshi = KalshiClient()
        raw_markets = kalshi.search_nba_markets()
        if raw_markets:
            kalshi_df = parse_kalshi_markets(raw_markets)
            all_markets.append(kalshi_df)
            print(f"    Found {len(kalshi_df)} NBA markets on Kalshi")
        else:
            print("    No NBA markets found on Kalshi")
    except Exception as e:
        print(f"    Kalshi error: {e}")

    # Polymarket
    print("  Scanning Polymarket...")
    try:
        poly = PolymarketScanner()
        poly_df = poly.search_nba_markets()
        if not poly_df.empty:
            all_markets.append(poly_df)
            print(f"    Found {len(poly_df)} NBA markets on Polymarket")
        else:
            print("    No NBA markets found on Polymarket")
    except Exception as e:
        print(f"    Polymarket error: {e}")

    if not all_markets:
        print("  No markets found on any platform.")
        return pd.DataFrame()

    combined = pd.concat(all_markets, ignore_index=True)
    combined.to_csv(OUTPUTS_DIR / "all_markets.csv", index=False)

    if predictions is not None and not predictions.empty:
        print("\n  Matching markets to model predictions...")
        edges = match_markets_to_predictions(combined, predictions)
        if not edges.empty:
            edges.to_csv(OUTPUTS_DIR / "market_edges.csv", index=False)
            print(f"    Found {len(edges)} actionable edges")
            _print_market_edges(edges)
        return edges

    return combined


# ═══════════════════════════════════════════════════════════════════════
# INTERACTIVE MARKET BROWSER
# ═══════════════════════════════════════════════════════════════════════

def browse_kalshi():
    """Interactive Kalshi market browser."""
    client = KalshiClient()

    print("\n" + "=" * 70)
    print("  KALSHI MARKET BROWSER")
    print("  Searching for NBA markets...")
    print("=" * 70)

    markets = client.search_nba_markets()

    if not markets:
        print("\n  No open NBA markets found. Trying broader search...")
        markets = client.search_markets(["nba", "basketball"])

    if not markets:
        print("  No NBA markets currently available on Kalshi.")
        print("  NBA markets typically appear on game days.")
        print("\n  Try searching for other markets:")
        print("    python market_scanner.py --search 'your keyword'")
        return

    df = parse_kalshi_markets(markets)
    _print_market_table(df)

    # Save for reference
    df.to_csv(OUTPUTS_DIR / "kalshi_nba_markets.csv", index=False)
    print(f"\n  Saved to {OUTPUTS_DIR / 'kalshi_nba_markets.csv'}")


def inspect_market(ticker: str):
    """Get detailed info + orderbook for a specific market."""
    client = KalshiClient()

    print(f"\n  Fetching market: {ticker}")
    try:
        market = client.get_market(ticker)
    except Exception as e:
        print(f"  Error: {e}")
        return

    print("\n" + "=" * 70)
    print(f"  MARKET: {market.get('title', ticker)}")
    print(f"  Subtitle: {market.get('subtitle', '')}")
    print("=" * 70)
    print(f"  Ticker:        {market.get('ticker')}")
    print(f"  Event:         {market.get('event_ticker')}")
    print(f"  Series:        {market.get('series_ticker')}")
    print(f"  Status:        {market.get('status')}")
    print(f"  Close time:    {market.get('close_time', market.get('expiration_time', ''))}")

    yes_bid = float(market.get('yes_bid_dollars', market.get('yes_bid', 0)) or 0)
    yes_ask = float(market.get('yes_ask_dollars', market.get('yes_ask', 0)) or 0)
    print(f"  YES bid/ask:   ${yes_bid:.4f} / ${yes_ask:.4f}")
    vol = float(market.get('volume', market.get('volume_fp', 0)) or 0)
    oi = float(market.get('open_interest', market.get('open_interest_fp', 0)) or 0)
    print(f"  Volume:        {vol:,.0f}")
    print(f"  Open interest: {oi:,.0f}")
    print(f"  URL:           https://kalshi.com/markets/{ticker}")

    # Orderbook
    print("\n  Orderbook (sorted by price, highest first):")
    try:
        ob = client.get_orderbook(ticker)
        ob_data = ob.get("orderbook_fp") or ob.get("orderbook", {})

        yes_orders = ob_data.get("yes_dollars") or ob_data.get("yes", [])
        no_orders = ob_data.get("no_dollars") or ob_data.get("no", [])

        # Parse and sort by price descending (most relevant bids first)
        def parse_orders(orders):
            parsed = []
            for entry in (orders or []):
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    parsed.append((float(entry[0]), float(entry[1])))
            return sorted(parsed, key=lambda x: x[0], reverse=True)

        yes_parsed = parse_orders(yes_orders)
        no_parsed = parse_orders(no_orders)

        print(f"\n    YES bids ({len(yes_parsed)} price levels):")
        print(f"    {'Price':<10} {'Quantity':<15} {'Notional'}")
        print(f"    {'─'*10} {'─'*15} {'─'*12}")
        for price, qty in yes_parsed:
            notional = price * qty
            print(f"    ${price:<9.2f} {qty:<15,.0f} ${notional:,.0f}")

        print(f"\n    NO bids ({len(no_parsed)} price levels):")
        print(f"    {'Price':<10} {'Quantity':<15} {'Notional'}")
        print(f"    {'─'*10} {'─'*15} {'─'*12}")
        for price, qty in no_parsed:
            notional = price * qty
            print(f"    ${price:<9.2f} {qty:<15,.0f} ${notional:,.0f}")

        # Summary
        if yes_parsed:
            best_yes = yes_parsed[0]
            total_yes_liq = sum(p * q for p, q in yes_parsed)
            print(f"\n    Summary:")
            print(f"      Best YES bid:     ${best_yes[0]:.2f} × {best_yes[1]:,.0f}")
            if no_parsed:
                best_no = no_parsed[0]
                print(f"      Best NO bid:      ${best_no[0]:.2f} × {best_no[1]:,.0f}")
                # YES bid + NO bid should be close to $1.00
                # The gap is the spread/vig
                print(f"      Spread:           ${1.0 - best_yes[0] - best_no[0]:.2f}")
            print(f"      Total YES depth:  ${total_yes_liq:,.0f}")
            total_no_liq = sum(p * q for p, q in no_parsed)
            print(f"      Total NO depth:   ${total_no_liq:,.0f}")

    except Exception as e:
        print(f"    Could not fetch orderbook: {e}")


def search_kalshi(keyword: str, limit: int | None = None):
    """Search Kalshi markets by keyword."""
    client = KalshiClient()

    print(f"\n  Searching Kalshi for '{keyword}'...")
    markets = client.search_markets([keyword])

    if not markets:
        print(f"  No open markets found matching '{keyword}'.")
        return

    df = parse_kalshi_markets(markets)
    if limit:
        df = df.head(limit)
    _print_market_table(df)
    df.to_csv(OUTPUTS_DIR / f"kalshi_search_{keyword}.csv", index=False)


def list_series_markets(series_ticker: str, limit: int | None = None):
    """List all open markets in a Kalshi series."""
    client = KalshiClient()

    print(f"\n  Fetching series: {series_ticker}")
    try:
        series = client.get_series(series_ticker)
        print(f"  Series: {series.get('title', series_ticker)}")
        print(f"  Category: {series.get('category', 'N/A')}")
    except Exception:
        pass

    result = client.get_markets(series_ticker=series_ticker, status="open")
    markets = result.get("markets", [])

    if not markets:
        print(f"  No open markets in series {series_ticker}.")
        # Try all statuses
        result = client.get_markets(series_ticker=series_ticker, status="")
        markets = result.get("markets", [])
        if markets:
            print(f"  Found {len(markets)} markets (including closed) — showing recent:")
            markets = markets[:20]

    if markets:
        df = parse_kalshi_markets(markets)
        if limit:
            df = df.head(limit)
        _print_market_table(df)


# ═══════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _print_market_table(df: pd.DataFrame):
    """Print a formatted table of markets."""
    if df.empty:
        print("  No markets to display.")
        return

    print(f"\n  {'#':<4} {'Ticker':<30} {'YES Bid':<10} {'Volume':<10} {'Title'}")
    print(f"  {'─'*4} {'─'*30} {'─'*10} {'─'*10} {'─'*40}")

    for i, (_, row) in enumerate(df.iterrows(), 1):
        title = row.get("title", "")
        if len(title) > 50:
            title = title[:47] + "..."
        try:
            vol = float(row.get("volume", 0) or 0)
            vol_str = f"${vol:,.0f}" if vol else "—"
        except (ValueError, TypeError):
            vol_str = "—"

        try:
            yes_bid = float(row.get("yes_bid", 0) or 0)
        except (ValueError, TypeError):
            yes_bid = 0.0

        print(f"  {i:<4} {row['ticker']:<30} {yes_bid:<10.2f} {vol_str:<10} {title}")

    print(f"\n  To inspect a market:  python market_scanner.py --market <TICKER>")
    print(f"  To see orderbook:     python market_scanner.py --market <TICKER>")


def _print_market_edges(edges: pd.DataFrame):
    """Print formatted market edge opportunities."""
    print("\n" + "=" * 80)
    print("  PREDICTION MARKET EDGES")
    print("=" * 80)

    for _, row in edges.iterrows():
        direction = "BUY YES" if row["edge"] > 0 else "BUY NO"
        print(f"\n  [{row['platform'].upper()}] {row['title']}")
        print(f"  ├─ Ticker:     {row['ticker']}")
        print(f"  ├─ Action:     {direction} ({row['yes_team']})")
        print(f"  ├─ Market:     {row['market_prob']:.1%}")
        print(f"  ├─ Model:      {row['model_prob']:.1%}")
        print(f"  ├─ Edge:       {row['edge']:+.1%}")
        print(f"  ├─ Kelly:      {row['kelly']:.2%}")
        print(f"  ├─ Bet size:   ${row['bet_size']:,.0f}")
        try:
            volume_num = float(row["volume"]) if row.get("volume") not in (None, "", float("nan")) else 0.0
        except (TypeError, ValueError):
            volume_num = 0.0
        print(f"  ├─ Volume:     {volume_num:,.0f}")
        print(f"  └─ URL:        {row.get('url', '')}")

    print(f"\n{'─' * 80}\n")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalshi & Polymarket NBA Scanner")
    parser.add_argument("--browse", action="store_true", help="Browse all NBA markets on Kalshi")
    parser.add_argument("--series", type=str, help="List markets in a Kalshi series (e.g., KXNBA)")
    parser.add_argument("--market", type=str, help="Inspect a specific market ticker + orderbook")
    parser.add_argument("--search", type=str, help="Search markets by keyword")
    parser.add_argument("--scan", action="store_true", help="Scan all platforms + match to model")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of results displayed")
    args = parser.parse_args()

    if args.market:
        inspect_market(args.market)
    elif args.series:
        list_series_markets(args.series, limit=args.limit)
    elif args.search:
        search_kalshi(args.search, limit=args.limit)
    elif args.browse:
        browse_kalshi()
    elif args.scan:
        pred_path = OUTPUTS_DIR / "predictions.csv"
        predictions = pd.read_csv(pred_path) if pred_path.exists() else None
        scan_all_markets(predictions)
    else:
        # Default: browse NBA markets
        browse_kalshi()

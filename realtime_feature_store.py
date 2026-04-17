"""
Real-time feature store and event logger for live NBA states + market snapshots.

This module captures three append-only datasets:
  1. game state snapshots   — one row per ESPN game state poll
  2. market snapshots       — one row per Kalshi market + orderbook poll
  3. live feature rows      — one joined row per market, ready for in-game model training

It also emits a JSONL event log when important state changes occur:
  - score changes
  - game status / period changes
  - price moves
  - orderbook liquidity moves
  - market appearance / disappearance
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from config import (
    LIVE_EVENTS_DIR,
    LIVE_FEATURES_DIR,
    LIVE_GAMES_DIR,
    LIVE_MARKETS_DIR,
    LIVE_STATE_DIR,
)
from consensus_divergence import compute_divergence, consensus_implied_prob
from live_data import fetch_odds_api_lines
from market_scanner import KalshiClient
from nba_market_utils import normalize_nba_abbrev, parse_nba_ticker
from prediction_utils import get_model_prediction


ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
EASTERN = ZoneInfo("America/New_York")
STATE_PATH = LIVE_STATE_DIR / "latest_state.json"
PRICE_EPSILON = 0.005
DEPTH_EPSILON = 100.0


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_price(value) -> float | None:
    price = _safe_float(value)
    if price is None:
        return None
    return price / 100 if price > 1 else price


def _parse_clock_to_seconds(clock: str | None) -> int | None:
    if not clock or not isinstance(clock, str) or ":" not in clock:
        return None
    try:
        minutes, seconds = clock.split(":", 1)
        return int(minutes) * 60 + int(seconds)
    except (TypeError, ValueError):
        return None


def _estimate_seconds_elapsed(period: int | None, seconds_left: int | None, is_final: bool) -> int | None:
    if is_final:
        if period is None or period <= 4:
            return 48 * 60
        return 48 * 60 + max(period - 4, 0) * 5 * 60

    if period is None or seconds_left is None or period <= 0:
        return None

    if period <= 4:
        return (period - 1) * 12 * 60 + (12 * 60 - seconds_left)

    return 48 * 60 + (period - 5) * 5 * 60 + (5 * 60 - seconds_left)


def _to_local_game_date(value) -> str | None:
    if value is None or value == "":
        return None

    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None

    if getattr(ts, "tzinfo", None) is None:
        return pd.Timestamp(ts).strftime("%Y-%m-%d")

    return pd.Timestamp(ts).tz_convert(EASTERN).strftime("%Y-%m-%d")


def _make_game_key(game_date: str | None, home_team: str | None, away_team: str | None) -> str | None:
    if not game_date or not home_team or not away_team:
        return None
    return f"{game_date}_{away_team}_{home_team}"


def _american_to_implied_prob(american_odds: float | int | None) -> float | None:
    odds = _safe_float(american_odds)
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return round(100 / (odds + 100), 4)
    return round(abs(odds) / (abs(odds) + 100), 4)


def _append_csv(path: Path, df: pd.DataFrame):
    if df.empty:
        return
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _append_jsonl(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")


def _normalize_record(record: dict) -> dict:
    normalized = {}
    for key, value in record.items():
        if isinstance(value, pd.Timestamp):
            normalized[key] = value.isoformat()
        elif isinstance(value, np.generic):
            normalized[key] = value.item()
        elif pd.isna(value) if isinstance(value, (float, np.floating)) else False:
            normalized[key] = None
        else:
            normalized[key] = value
    return normalized


def _parse_orderbook_levels(levels) -> list[tuple[float, float]]:
    parsed = []
    for entry in levels or []:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            price = _normalize_price(entry[0])
            qty = _safe_float(entry[1], 0.0)
            if price is not None and qty is not None:
                parsed.append((price, qty))
    return sorted(parsed, key=lambda item: item[0], reverse=True)


def _summarize_levels(levels: list[tuple[float, float]], top_n: int) -> dict:
    subset = levels[:top_n]
    qty = sum(qty for _, qty in subset)
    notional = sum(price * qty for price, qty in subset)
    weighted = notional / qty if qty else None
    best_price = subset[0][0] if subset else None
    best_qty = subset[0][1] if subset else None
    return {
        "best_price": best_price,
        "best_qty": best_qty,
        "qty": round(qty, 2),
        "notional": round(notional, 2),
        "weighted_price": round(weighted, 4) if weighted is not None else None,
        "levels_json": json.dumps([[round(price, 4), round(qty, 2)] for price, qty in subset]),
    }


def _market_home_implied(contract_prob: float | None, bet_side: str | None) -> float | None:
    if contract_prob is None or bet_side not in {"home", "away"}:
        return None
    return contract_prob if bet_side == "home" else round(1 - contract_prob, 4)


def _load_partitioned_csv(directory: Path, prefix: str) -> pd.DataFrame:
    files = sorted(directory.glob(f"{prefix}_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def load_feature_history() -> pd.DataFrame:
    return _load_partitioned_csv(LIVE_FEATURES_DIR, "live_features")


def load_game_history() -> pd.DataFrame:
    return _load_partitioned_csv(LIVE_GAMES_DIR, "game_states")


def load_market_history() -> pd.DataFrame:
    return _load_partitioned_csv(LIVE_MARKETS_DIR, "market_snapshots")


def load_event_history() -> pd.DataFrame:
    files = sorted(LIVE_EVENTS_DIR.glob("live_events_*.jsonl"))
    if not files:
        return pd.DataFrame()

    rows = []
    for path in files:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _prepare_game_snapshot_frame(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame()

    games = games_df.copy()
    games["captured_at"] = pd.to_datetime(games["captured_at"], utc=True, errors="coerce")

    for col in ["home_team", "away_team"]:
        if col in games.columns:
            games[col] = games[col].apply(normalize_nba_abbrev)

    if "scheduled_start" in games.columns:
        missing_game_date = games.get("game_date").isna() if "game_date" in games.columns else pd.Series(True, index=games.index)
        if "game_date" not in games.columns:
            games["game_date"] = None
        games.loc[missing_game_date, "game_date"] = games.loc[missing_game_date, "scheduled_start"].apply(_to_local_game_date)

    games["game_key"] = games.apply(
        lambda row: _make_game_key(row.get("game_date"), row.get("home_team"), row.get("away_team")),
        axis=1,
    )
    return (
        games.dropna(subset=["captured_at", "game_key"])
        .sort_values(["game_key", "captured_at"])
        .reset_index(drop=True)
    )


def _prepare_market_snapshot_frame(markets_df: pd.DataFrame) -> pd.DataFrame:
    if markets_df.empty:
        return pd.DataFrame()

    markets = markets_df.copy()
    markets["captured_at"] = pd.to_datetime(markets["captured_at"], utc=True, errors="coerce")

    parsed_cache: dict[str, dict] = {}

    def parsed_value(ticker: str, field: str):
        if ticker not in parsed_cache:
            parsed_cache[ticker] = parse_nba_ticker(ticker) or {}
        return parsed_cache[ticker].get(field)

    for col in ["home_team", "away_team", "bet_team"]:
        if col not in markets.columns:
            markets[col] = None

    for idx, row in markets.iterrows():
        ticker = row.get("ticker", "")
        if not ticker:
            continue

        if pd.isna(row.get("game_date")):
            markets.at[idx, "game_date"] = parsed_value(ticker, "game_date")

        for col, field in [
            ("home_team", "home_team"),
            ("away_team", "away_team"),
            ("bet_team", "bet_team"),
            ("bet_side", "bet_side"),
        ]:
            if col not in markets.columns or pd.isna(row.get(col)):
                markets.at[idx, col] = parsed_value(ticker, field)

    for col in ["home_team", "away_team", "bet_team"]:
        if col in markets.columns:
            markets[col] = markets[col].apply(normalize_nba_abbrev)

    markets["game_key"] = markets.apply(
        lambda row: _make_game_key(row.get("game_date"), row.get("home_team"), row.get("away_team")),
        axis=1,
    )
    return (
        markets.dropna(subset=["captured_at", "game_key", "ticker"])
        .sort_values(["game_key", "captured_at", "ticker"])
        .reset_index(drop=True)
    )


class LiveFeatureStore:
    def __init__(
        self,
        series_ticker: str = "KXNBAGAME",
        tickers: list[str] | None = None,
        max_markets: int | None = None,
    ):
        self.series_ticker = series_ticker
        self.tickers = tickers or []
        self.max_markets = max_markets
        self.kalshi = KalshiClient()
        self.http = requests.Session()
        self.http.headers.update({"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if not STATE_PATH.exists():
            return {"games": {}, "markets": {}}
        try:
            return json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            return {"games": {}, "markets": {}}

    def _save_state(self, games_df: pd.DataFrame, markets_df: pd.DataFrame):
        games = self.state.get("games", {})
        markets = self.state.get("markets", {})

        if not games_df.empty:
            games = {
                row["game_key"]: _normalize_record(row.to_dict())
                for _, row in games_df.iterrows()
                if row.get("game_key")
            }

        if not markets_df.empty:
            markets = {
                row["ticker"]: _normalize_record(row.to_dict())
                for _, row in markets_df.iterrows()
                if row.get("ticker")
            }

        # Persist latest divergence tier per game_key for next-poll diffing.
        divergence_state = {}
        last_features = getattr(self, "_last_features_df", None)
        if last_features is not None and not last_features.empty and "game_key" in last_features.columns:
            latest = (
                last_features.dropna(subset=["game_key"])
                .sort_values("captured_at")
                .groupby("game_key", as_index=False)
                .tail(1)
            )
            for _, row in latest.iterrows():
                game_key = row.get("game_key")
                divergence_state[game_key] = {
                    "triangulation_tier": _safe_int(row.get("triangulation_tier")),
                    "kalshi_vs_consensus_pp": _safe_float(row.get("kalshi_vs_consensus_pp")),
                    "model_vs_consensus_pp": _safe_float(row.get("model_vs_consensus_pp")),
                    "kalshi_vs_model_pp": _safe_float(row.get("kalshi_vs_model_pp")),
                }

        state = {"games": games, "markets": markets, "divergence": divergence_state}
        STATE_PATH.write_text(json.dumps(state, indent=2, default=str))
        self.state = state

    def fetch_game_states(self, game_date: str | None = None) -> pd.DataFrame:
        captured_at = _now_utc()
        params = {}
        if game_date:
            params["dates"] = pd.to_datetime(game_date).strftime("%Y%m%d")

        try:
            resp = self.http.get(ESPN_SCOREBOARD_URL, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            print(f"  ESPN scoreboard fetch failed: {exc}")
            return pd.DataFrame()

        rows = []
        for event in payload.get("events", []):
            competitions = event.get("competitions", [])
            if not competitions:
                continue
            comp = competitions[0]
            status_blob = comp.get("status", {})
            status_type = status_blob.get("type", {})

            teams = {}
            for competitor in comp.get("competitors", []):
                side = competitor.get("homeAway")
                teams[side] = competitor

            home = teams.get("home", {})
            away = teams.get("away", {})
            home_team = normalize_nba_abbrev(home.get("team", {}).get("abbreviation"))
            away_team = normalize_nba_abbrev(away.get("team", {}).get("abbreviation"))
            game_local_date = _to_local_game_date(event.get("date"))

            odds = (comp.get("odds") or [{}])[0]
            home_ml = _safe_int(odds.get("homeTeamOdds", {}).get("moneyLine"))
            away_ml = _safe_int(odds.get("awayTeamOdds", {}).get("moneyLine"))
            display_clock = status_blob.get("displayClock") or ""
            seconds_left = _parse_clock_to_seconds(display_clock)
            period = _safe_int(status_blob.get("period"), 0)
            is_final = bool(status_type.get("completed"))

            row = {
                "captured_at": captured_at.isoformat(),
                "espn_event_id": event.get("id"),
                "game_date": game_local_date,
                "game_key": _make_game_key(game_local_date, home_team, away_team),
                "scheduled_start": event.get("date"),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": _safe_int(home.get("score"), 0),
                "away_score": _safe_int(away.get("score"), 0),
                "score_margin_home": _safe_int(home.get("score"), 0) - _safe_int(away.get("score"), 0),
                "total_points": _safe_int(home.get("score"), 0) + _safe_int(away.get("score"), 0),
                "game_status": status_type.get("description") or status_type.get("name"),
                "status_detail": status_type.get("detail") or status_type.get("shortDetail"),
                "status_state": status_type.get("state"),
                "is_live": status_type.get("state") == "in",
                "is_final": is_final,
                "period": period,
                "display_clock": display_clock,
                "seconds_left_in_period": seconds_left,
                "seconds_elapsed": _estimate_seconds_elapsed(period, seconds_left, is_final),
                "espn_provider": odds.get("provider", {}).get("name"),
                "espn_home_moneyline": home_ml,
                "espn_away_moneyline": away_ml,
                "espn_home_implied": _american_to_implied_prob(home_ml),
                "espn_away_implied": _american_to_implied_prob(away_ml),
                "espn_spread": odds.get("details"),
                "espn_total": _safe_float(odds.get("overUnder")),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def fetch_market_snapshots(self) -> pd.DataFrame:
        captured_at = _now_utc().isoformat()
        market_map: dict[str, dict] = {}

        if self.tickers:
            for ticker in self.tickers:
                try:
                    market = self.kalshi.get_market(ticker)
                    if market:
                        market_map[market.get("ticker", ticker)] = market
                except Exception as exc:
                    print(f"  Market fetch failed for {ticker}: {exc}")
        else:
            try:
                result = self.kalshi.get_markets(series_ticker=self.series_ticker, status="open", limit=200)
                raw_markets = result.get("markets", [])
            except Exception as exc:
                print(f"  Kalshi market fetch failed: {exc}")
                return pd.DataFrame()

            if self.max_markets:
                raw_markets = raw_markets[: self.max_markets]

            for market in raw_markets:
                ticker = market.get("ticker")
                if ticker:
                    market_map[ticker] = market

            # Keep following previously seen markets so we can resolve
            # terminal status and a better close proxy for label generation.
            tracked_tickers = set(self.state.get("markets", {})) - set(market_map)
            for ticker in sorted(tracked_tickers):
                try:
                    market = self.kalshi.get_market(ticker)
                    if market:
                        market_map[ticker] = market
                except Exception:
                    continue

        raw_markets = list(market_map.values())

        rows = []
        for market in raw_markets:
            ticker = market.get("ticker", "")
            parsed = parse_nba_ticker(ticker)
            if not parsed:
                continue

            try:
                orderbook = self.kalshi.get_orderbook(ticker)
            except Exception as exc:
                print(f"  Orderbook fetch failed for {ticker}: {exc}")
                orderbook = {}

            ob_data = orderbook.get("orderbook_fp") or orderbook.get("orderbook", {})
            yes_levels = _parse_orderbook_levels(ob_data.get("yes_dollars") or ob_data.get("yes", []))
            no_levels = _parse_orderbook_levels(ob_data.get("no_dollars") or ob_data.get("no", []))

            yes3 = _summarize_levels(yes_levels, 3)
            yes5 = _summarize_levels(yes_levels, 5)
            no3 = _summarize_levels(no_levels, 3)
            no5 = _summarize_levels(no_levels, 5)

            yes_bid = _normalize_price(market.get("yes_bid_dollars", market.get("yes_bid")))
            yes_ask = _normalize_price(market.get("yes_ask_dollars", market.get("yes_ask")))
            no_bid = _normalize_price(market.get("no_bid_dollars", market.get("no_bid")))
            no_ask = _normalize_price(market.get("no_ask_dollars", market.get("no_ask")))
            last_price = _normalize_price(market.get("last_price_dollars", market.get("last_price")))

            contract_prob = yes_ask if yes_ask is not None else yes_bid
            yes_mid = (
                round((yes_bid + yes_ask) / 2, 4)
                if yes_bid is not None and yes_ask is not None
                else contract_prob
            )

            row = {
                "captured_at": captured_at,
                "ticker": ticker,
                "event_ticker": market.get("event_ticker"),
                "series_ticker": market.get("series_ticker"),
                "title": market.get("title"),
                "subtitle": market.get("subtitle"),
                "status": market.get("status"),
                "close_time": market.get("close_time", market.get("expiration_time")),
                "game_date": parsed.get("game_date"),
                "game_key": _make_game_key(parsed.get("game_date"), parsed.get("home_team"), parsed.get("away_team")),
                "home_team": parsed.get("home_team"),
                "away_team": parsed.get("away_team"),
                "bet_team": parsed.get("bet_team"),
                "bet_side": parsed.get("bet_side"),
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "yes_mid": yes_mid,
                "no_bid": no_bid,
                "no_ask": no_ask,
                "last_price": last_price,
                "volume": _safe_float(market.get("volume_fp", market.get("volume")), 0.0),
                "open_interest": _safe_float(market.get("open_interest_fp", market.get("open_interest")), 0.0),
                "market_home_implied": _market_home_implied(contract_prob, parsed.get("bet_side")),
                "best_yes_bid_qty": yes3["best_qty"],
                "best_no_bid_qty": no3["best_qty"],
                "yes_depth_qty_3": yes3["qty"],
                "yes_depth_notional_3": yes3["notional"],
                "yes_weighted_price_3": yes3["weighted_price"],
                "yes_depth_qty_5": yes5["qty"],
                "yes_depth_notional_5": yes5["notional"],
                "yes_weighted_price_5": yes5["weighted_price"],
                "no_depth_qty_3": no3["qty"],
                "no_depth_notional_3": no3["notional"],
                "no_weighted_price_3": no3["weighted_price"],
                "no_depth_qty_5": no5["qty"],
                "no_depth_notional_5": no5["notional"],
                "no_weighted_price_5": no5["weighted_price"],
                "yes_levels_top5": yes5["levels_json"],
                "no_levels_top5": no5["levels_json"],
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def _build_consensus_map(self) -> dict[tuple[str, str], dict]:
        """
        Build per-matchup de-vigged consensus from The Odds API.

        Uses consensus_divergence.consensus_implied_prob so behavior matches
        ev_analyzer and the CLI (proportional de-vig, median across books,
        staleness filter, min_books gate).
        """
        consensus_map: dict[tuple[str, str], dict] = {}
        odds_games = fetch_odds_api_lines()
        for game in odds_games:
            home = game.get("home")
            away = game.get("away")
            if not home or not away:
                continue

            books_input = []
            for book in game.get("books", []):
                h2h = book.get("markets", {}).get("h2h", {})
                books_input.append({
                    "name": book.get("name", ""),
                    "last_update": book.get("last_update"),
                    "home_moneyline": h2h.get(home, {}).get("price"),
                    "away_moneyline": h2h.get(away, {}).get("price"),
                })

            consensus = consensus_implied_prob(books_input)
            if consensus is None:
                continue
            consensus_map[(home, away)] = {
                "home_prob": round(consensus["home_prob"], 4),
                "away_prob": round(consensus["away_prob"], 4),
                "n_books": consensus["n_books"],
                "commence": game.get("commence"),
            }
        return consensus_map

    def _join_market_game_context(
        self,
        games_df: pd.DataFrame,
        markets_df: pd.DataFrame,
        join_tolerance_minutes: int = 30,
    ) -> pd.DataFrame:
        markets = _prepare_market_snapshot_frame(markets_df)
        if markets.empty:
            return pd.DataFrame()

        games = _prepare_game_snapshot_frame(games_df)
        if games.empty:
            return markets

        tolerance = pd.Timedelta(minutes=join_tolerance_minutes)
        games = games.rename(columns={"captured_at": "game_captured_at"})
        game_groups = {
            game_key: frame.sort_values("game_captured_at")
            for game_key, frame in games.groupby("game_key", sort=False)
        }
        game_only_cols = [col for col in games.columns if col not in {"game_key"} and col not in markets.columns]

        joined_parts = []
        for game_key, market_group in markets.groupby("game_key", sort=False):
            market_group = market_group.sort_values("captured_at")
            game_group = game_groups.get(game_key)

            if game_group is None or game_group.empty:
                part = market_group.copy()
                for col in game_only_cols:
                    part[col] = np.nan
            else:
                part = pd.merge_asof(
                    market_group,
                    game_group,
                    left_on="captured_at",
                    right_on="game_captured_at",
                    direction="backward",
                    tolerance=tolerance,
                    suffixes=("", "_game"),
                )

            joined_parts.append(part)

        joined = pd.concat(joined_parts, ignore_index=True) if joined_parts else pd.DataFrame()
        return joined.sort_values(["captured_at", "ticker"]).reset_index(drop=True)

    def build_feature_rows(
        self,
        games_df: pd.DataFrame,
        markets_df: pd.DataFrame,
        include_consensus: bool = True,
        join_tolerance_minutes: int = 30,
    ) -> pd.DataFrame:
        joined = self._join_market_game_context(
            games_df,
            markets_df,
            join_tolerance_minutes=join_tolerance_minutes,
        )
        if joined.empty:
            return pd.DataFrame()

        consensus_map = self._build_consensus_map() if include_consensus else {}
        pregame_cache: dict[tuple[str, str, str | None], dict | None] = {}

        rows = []
        for _, market in joined.iterrows():
            market_row = market.to_dict()
            game_key = market_row.get("game_key")

            pregame_key = (
                market_row.get("home_team"),
                market_row.get("away_team"),
                market_row.get("game_date"),
            )
            if pregame_key not in pregame_cache:
                pregame_cache[pregame_key] = get_model_prediction(
                    market_row.get("home_team"),
                    market_row.get("away_team"),
                    game_date=market_row.get("game_date"),
                )
            pregame = pregame_cache[pregame_key] or {}

            consensus = consensus_map.get(
                (market_row.get("home_team"), market_row.get("away_team")),
                {},
            )

            market_home_implied = market_row.get("market_home_implied")
            consensus_candidates = [
                market_row.get("espn_home_implied"),
                consensus.get("home_prob"),
            ]
            consensus_candidates = [value for value in consensus_candidates if value is not None]
            consensus_home = round(float(np.mean(consensus_candidates)), 4) if consensus_candidates else None

            # Divergence signal (IMPROVEMENTS #8)
            consensus_for_div = (
                {"home_prob": consensus.get("home_prob"),
                 "away_prob": consensus.get("away_prob"),
                 "n_books": consensus.get("n_books", 0)}
                if consensus and consensus.get("home_prob") is not None
                else None
            )
            div = compute_divergence(
                kalshi_yes_mid=market_home_implied,
                model_prob_home=pregame.get("home_win_prob"),
                consensus=consensus_for_div,
            )

            feature_row = {
                "captured_at": market_row.get("captured_at"),
                "ticker": market_row.get("ticker"),
                "event_ticker": market_row.get("event_ticker"),
                "game_key": game_key,
                "game_date": market_row.get("game_date"),
                "home_team": market_row.get("home_team"),
                "away_team": market_row.get("away_team"),
                "bet_team": market_row.get("bet_team"),
                "bet_side": market_row.get("bet_side"),
                "game_status": market_row.get("game_status"),
                "status_state": market_row.get("status_state"),
                "is_live": market_row.get("is_live"),
                "is_final": market_row.get("is_final"),
                "period": market_row.get("period"),
                "display_clock": market_row.get("display_clock"),
                "seconds_left_in_period": market_row.get("seconds_left_in_period"),
                "seconds_elapsed": market_row.get("seconds_elapsed"),
                "home_score": market_row.get("home_score"),
                "away_score": market_row.get("away_score"),
                "score_margin_home": market_row.get("score_margin_home"),
                "total_points": market_row.get("total_points"),
                "yes_bid": market_row.get("yes_bid"),
                "yes_ask": market_row.get("yes_ask"),
                "yes_mid": market_row.get("yes_mid"),
                "no_bid": market_row.get("no_bid"),
                "no_ask": market_row.get("no_ask"),
                "last_price": market_row.get("last_price"),
                "market_home_implied": market_home_implied,
                "volume": market_row.get("volume"),
                "open_interest": market_row.get("open_interest"),
                "yes_depth_notional_3": market_row.get("yes_depth_notional_3"),
                "yes_depth_notional_5": market_row.get("yes_depth_notional_5"),
                "no_depth_notional_3": market_row.get("no_depth_notional_3"),
                "no_depth_notional_5": market_row.get("no_depth_notional_5"),
                "yes_weighted_price_3": market_row.get("yes_weighted_price_3"),
                "no_weighted_price_3": market_row.get("no_weighted_price_3"),
                "espn_home_implied": market_row.get("espn_home_implied"),
                "espn_away_implied": market_row.get("espn_away_implied"),
                "oddsapi_home_consensus": consensus.get("home_prob"),
                "oddsapi_away_consensus": consensus.get("away_prob"),
                "oddsapi_books": consensus.get("n_books"),
                "market_consensus_home": consensus_home,
                "pregame_home_win_prob": pregame.get("home_win_prob"),
                "pregame_away_win_prob": pregame.get("away_win_prob"),
                "pregame_spread": pregame.get("predicted_spread"),
                "pregame_total": pregame.get("predicted_total"),
                "pregame_data_date": pregame.get("data_date"),
                "kalshi_vs_consensus_pp": div["kalshi_vs_consensus_pp"],
                "model_vs_consensus_pp": div["model_vs_consensus_pp"],
                "kalshi_vs_model_pp": div["kalshi_vs_model_pp"],
                "triangulation_tier": div["triangulation_tier"],
                "consensus_n_books": consensus.get("n_books", 0) if consensus else 0,
            }

            if feature_row["pregame_home_win_prob"] is not None and market_home_implied is not None:
                feature_row["pregame_edge_home"] = round(
                    feature_row["pregame_home_win_prob"] - market_home_implied, 4
                )
            else:
                feature_row["pregame_edge_home"] = None

            if consensus_home is not None and market_home_implied is not None:
                feature_row["consensus_gap_home"] = round(
                    market_home_implied - consensus_home, 4
                )
            else:
                feature_row["consensus_gap_home"] = None

            rows.append(feature_row)

        return pd.DataFrame(rows)

    def rebuild_feature_history(
        self,
        include_consensus: bool = False,
        join_tolerance_minutes: int = 30,
    ) -> dict:
        games_history = load_game_history()
        markets_history = load_market_history()
        features_df = self.build_feature_rows(
            games_history,
            markets_history,
            include_consensus=include_consensus,
            join_tolerance_minutes=join_tolerance_minutes,
        )

        if features_df.empty:
            return {
                "games": int(len(games_history)),
                "markets": int(len(markets_history)),
                "features": 0,
                "rows_with_game_state": 0,
                "rows_with_espn_implied": 0,
                "partitions": [],
            }

        features_df["captured_at"] = pd.to_datetime(features_df["captured_at"], utc=True, errors="coerce")
        features_df["capture_day"] = features_df["captured_at"].dt.tz_convert(EASTERN).dt.strftime("%Y-%m-%d")

        written = []
        for capture_day, partition in features_df.groupby("capture_day", sort=True):
            path = LIVE_FEATURES_DIR / f"live_features_{capture_day}.csv"
            out = partition.drop(columns=["capture_day"]).sort_values(["captured_at", "ticker"])
            out.to_csv(path, index=False)
            written.append(str(path))

        latest_rows = (
            features_df.sort_values(["captured_at", "ticker"])
            .groupby("ticker", as_index=False)
            .tail(1)
            .drop(columns=["capture_day"])
            .sort_values(["home_team", "away_team", "ticker"])
        )
        latest_rows.to_csv(LIVE_FEATURES_DIR / "latest_features.csv", index=False)

        summary = {
            "games": int(len(games_history)),
            "markets": int(len(markets_history)),
            "features": int(len(features_df)),
            "rows_with_game_state": int(features_df["game_status"].notna().sum()),
            "rows_with_espn_implied": int(features_df["espn_home_implied"].notna().sum()),
            "partitions": written,
            "include_consensus": include_consensus,
            "join_tolerance_minutes": join_tolerance_minutes,
        }

        summary_path = LIVE_FEATURES_DIR / "rebuild_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        summary["summary_path"] = str(summary_path)
        return summary

    def _detect_game_events(self, games_df: pd.DataFrame) -> list[dict]:
        if games_df.empty:
            return []

        events = []
        previous = self.state.get("games", {})
        current = {
            row["game_key"]: _normalize_record(row.to_dict())
            for _, row in games_df.iterrows()
            if row.get("game_key")
        }

        captured_at = _now_utc().isoformat()

        for game_key, game in current.items():
            prev = previous.get(game_key)
            if prev is None:
                events.append({
                    "captured_at": captured_at,
                    "entity_type": "game",
                    "event_type": "game_observed",
                    "entity_key": game_key,
                    "home_team": game.get("home_team"),
                    "away_team": game.get("away_team"),
                    "current": game,
                })
                continue

            if (
                prev.get("home_score") != game.get("home_score")
                or prev.get("away_score") != game.get("away_score")
            ):
                events.append({
                    "captured_at": captured_at,
                    "entity_type": "game",
                    "event_type": "score_change",
                    "entity_key": game_key,
                    "home_team": game.get("home_team"),
                    "away_team": game.get("away_team"),
                    "period": game.get("period"),
                    "display_clock": game.get("display_clock"),
                    "previous": {
                        "home_score": prev.get("home_score"),
                        "away_score": prev.get("away_score"),
                    },
                    "current": {
                        "home_score": game.get("home_score"),
                        "away_score": game.get("away_score"),
                    },
                    "delta_home": _safe_int(game.get("home_score"), 0) - _safe_int(prev.get("home_score"), 0),
                    "delta_away": _safe_int(game.get("away_score"), 0) - _safe_int(prev.get("away_score"), 0),
                })

            if prev.get("game_status") != game.get("game_status"):
                events.append({
                    "captured_at": captured_at,
                    "entity_type": "game",
                    "event_type": "game_status_change",
                    "entity_key": game_key,
                    "home_team": game.get("home_team"),
                    "away_team": game.get("away_team"),
                    "previous": prev.get("game_status"),
                    "current": game.get("game_status"),
                })

            if prev.get("period") != game.get("period"):
                events.append({
                    "captured_at": captured_at,
                    "entity_type": "game",
                    "event_type": "period_change",
                    "entity_key": game_key,
                    "home_team": game.get("home_team"),
                    "away_team": game.get("away_team"),
                    "previous": prev.get("period"),
                    "current": game.get("period"),
                })

        for game_key, prev in previous.items():
            if game_key not in current:
                events.append({
                    "captured_at": captured_at,
                    "entity_type": "game",
                    "event_type": "game_missing",
                    "entity_key": game_key,
                    "home_team": prev.get("home_team"),
                    "away_team": prev.get("away_team"),
                    "previous": prev,
                })

        return events

    def _detect_market_events(self, markets_df: pd.DataFrame) -> list[dict]:
        if markets_df.empty:
            return []

        events = []
        previous = self.state.get("markets", {})
        current = {
            row["ticker"]: _normalize_record(row.to_dict())
            for _, row in markets_df.iterrows()
            if row.get("ticker")
        }
        captured_at = _now_utc().isoformat()

        for ticker, market in current.items():
            prev = previous.get(ticker)
            if prev is None:
                events.append({
                    "captured_at": captured_at,
                    "entity_type": "market",
                    "event_type": "market_observed",
                    "entity_key": ticker,
                    "ticker": ticker,
                    "home_team": market.get("home_team"),
                    "away_team": market.get("away_team"),
                    "current": market,
                })
                continue

            if prev.get("status") != market.get("status"):
                events.append({
                    "captured_at": captured_at,
                    "entity_type": "market",
                    "event_type": "market_status_change",
                    "entity_key": ticker,
                    "ticker": ticker,
                    "previous": prev.get("status"),
                    "current": market.get("status"),
                })

            changed_prices = {}
            for field in ["yes_bid", "yes_ask", "yes_mid", "no_bid", "no_ask", "market_home_implied"]:
                old_val = _safe_float(prev.get(field))
                new_val = _safe_float(market.get(field))
                if old_val is None or new_val is None:
                    continue
                if abs(new_val - old_val) >= PRICE_EPSILON:
                    changed_prices[field] = {"previous": old_val, "current": new_val, "delta": round(new_val - old_val, 4)}
            if changed_prices:
                events.append({
                    "captured_at": captured_at,
                    "entity_type": "market",
                    "event_type": "price_change",
                    "entity_key": ticker,
                    "ticker": ticker,
                    "home_team": market.get("home_team"),
                    "away_team": market.get("away_team"),
                    "changes": changed_prices,
                })

            changed_depth = {}
            for field in ["yes_depth_notional_3", "yes_depth_notional_5", "no_depth_notional_3", "no_depth_notional_5"]:
                old_val = _safe_float(prev.get(field), 0.0)
                new_val = _safe_float(market.get(field), 0.0)
                if abs(new_val - old_val) >= DEPTH_EPSILON:
                    changed_depth[field] = {"previous": old_val, "current": new_val, "delta": round(new_val - old_val, 2)}
            if changed_depth:
                events.append({
                    "captured_at": captured_at,
                    "entity_type": "market",
                    "event_type": "liquidity_change",
                    "entity_key": ticker,
                    "ticker": ticker,
                    "home_team": market.get("home_team"),
                    "away_team": market.get("away_team"),
                    "changes": changed_depth,
                })

            prev_volume = _safe_float(prev.get("volume"), 0.0)
            new_volume = _safe_float(market.get("volume"), 0.0)
            if new_volume > prev_volume:
                events.append({
                    "captured_at": captured_at,
                    "entity_type": "market",
                    "event_type": "volume_change",
                    "entity_key": ticker,
                    "ticker": ticker,
                    "previous": prev_volume,
                    "current": new_volume,
                    "delta": round(new_volume - prev_volume, 2),
                })

        for ticker, prev in previous.items():
            if ticker not in current:
                events.append({
                    "captured_at": captured_at,
                    "entity_type": "market",
                    "event_type": "market_missing",
                    "entity_key": ticker,
                    "ticker": ticker,
                    "home_team": prev.get("home_team"),
                    "away_team": prev.get("away_team"),
                    "previous": prev,
                })

        return events

    def _detect_divergence_events(self, features_df: pd.DataFrame) -> list[dict]:
        """
        Emit divergence_change events when triangulation_tier flips between
        polls, and divergence_observed events on first sighting.
        """
        if features_df.empty or "triangulation_tier" not in features_df.columns:
            return []

        events = []
        previous = self.state.get("divergence", {})
        captured_at = _now_utc().isoformat()

        # Take the latest row per game_key to avoid duplicate events for
        # multi-ticker matchups in the same poll cycle.
        latest = (
            features_df.dropna(subset=["game_key"])
            .sort_values("captured_at")
            .groupby("game_key", as_index=False)
            .tail(1)
        )

        for _, row in latest.iterrows():
            game_key = row.get("game_key")
            tier = _safe_int(row.get("triangulation_tier"))
            current = {
                "triangulation_tier": tier,
                "kalshi_vs_consensus_pp": _safe_float(row.get("kalshi_vs_consensus_pp")),
                "model_vs_consensus_pp": _safe_float(row.get("model_vs_consensus_pp")),
                "kalshi_vs_model_pp": _safe_float(row.get("kalshi_vs_model_pp")),
            }
            prev = previous.get(game_key)

            if prev is None:
                events.append({
                    "captured_at": captured_at,
                    "entity_type": "divergence",
                    "event_type": "divergence_observed",
                    "entity_key": game_key,
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "current": current,
                })
            elif prev.get("triangulation_tier") != tier:
                events.append({
                    "captured_at": captured_at,
                    "entity_type": "divergence",
                    "event_type": "divergence_change",
                    "entity_key": game_key,
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "previous": prev,
                    "current": current,
                })

        return events

    def capture_once(self, game_date: str | None = None) -> dict:
        games_df = self.fetch_game_states(game_date=game_date)
        markets_df = self.fetch_market_snapshots()
        features_df = self.build_feature_rows(games_df, markets_df)

        game_events = self._detect_game_events(games_df)
        market_events = self._detect_market_events(markets_df)
        divergence_events = self._detect_divergence_events(features_df)
        events = game_events + market_events + divergence_events

        captured_day = _to_local_game_date(_now_utc()) or pd.Timestamp.today().strftime("%Y-%m-%d")
        games_path = LIVE_GAMES_DIR / f"game_states_{captured_day}.csv"
        markets_path = LIVE_MARKETS_DIR / f"market_snapshots_{captured_day}.csv"
        features_path = LIVE_FEATURES_DIR / f"live_features_{captured_day}.csv"
        events_path = LIVE_EVENTS_DIR / f"live_events_{captured_day}.jsonl"

        _append_csv(games_path, games_df)
        _append_csv(markets_path, markets_df)
        _append_csv(features_path, features_df)
        _append_jsonl(events_path, events)

        if not features_df.empty:
            latest_path = LIVE_FEATURES_DIR / "latest_features.csv"
            features_df.sort_values(["home_team", "away_team", "ticker"]).to_csv(latest_path, index=False)

        self._last_features_df = features_df
        self._save_state(games_df, markets_df)

        return {
            "games": len(games_df),
            "markets": len(markets_df),
            "features": len(features_df),
            "events": len(events),
            "games_path": str(games_path),
            "markets_path": str(markets_path),
            "features_path": str(features_path),
            "events_path": str(events_path),
        }

    def run_loop(self, interval_seconds: int = 15, game_date: str | None = None):
        while True:
            summary = self.capture_once(game_date=game_date)
            print(
                "  captured "
                f"{summary['games']} games, {summary['markets']} markets, "
                f"{summary['features']} features, {summary['events']} events"
            )
            time.sleep(interval_seconds)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time NBA feature store + event logger for live states and market snapshots.",
    )
    parser.add_argument(
        "--rebuild-history",
        action="store_true",
        help="Rebuild data/live/features from raw game and market snapshots.",
    )
    parser.add_argument("--loop", action="store_true", help="Poll continuously instead of a single capture.")
    parser.add_argument("--interval", type=int, default=15, help="Polling interval in seconds for --loop.")
    parser.add_argument("--series", type=str, default="KXNBAGAME", help="Kalshi series ticker to monitor.")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated Kalshi tickers to monitor.")
    parser.add_argument("--max-markets", type=int, default=None, help="Limit markets per capture.")
    parser.add_argument("--date", type=str, default=None, help="Override ESPN scoreboard date (YYYY-MM-DD).")
    parser.add_argument(
        "--include-consensus",
        action="store_true",
        help="Refresh Odds API consensus when rebuilding history.",
    )
    parser.add_argument(
        "--join-tolerance-minutes",
        type=int,
        default=30,
        help="Maximum market-to-game snapshot gap for joining raw history.",
    )
    args = parser.parse_args()

    tickers = [ticker.strip() for ticker in args.tickers.split(",")] if args.tickers else None
    store = LiveFeatureStore(
        series_ticker=args.series,
        tickers=tickers,
        max_markets=args.max_markets,
    )

    if args.rebuild_history:
        summary = store.rebuild_feature_history(
            include_consensus=args.include_consensus,
            join_tolerance_minutes=args.join_tolerance_minutes,
        )
        print(f"  Raw game rows:            {summary['games']}")
        print(f"  Raw market rows:          {summary['markets']}")
        print(f"  Rebuilt feature rows:     {summary['features']}")
        print(f"  Rows with game state:     {summary['rows_with_game_state']}")
        print(f"  Rows with ESPN implied:   {summary['rows_with_espn_implied']}")
        print(f"  Partitions written:       {len(summary['partitions'])}")
        print(f"  Summary:                  {summary['summary_path']}")
        return

    if args.loop:
        store.run_loop(interval_seconds=args.interval, game_date=args.date)
        return

    summary = store.capture_once(game_date=args.date)
    print(f"  Games captured:    {summary['games']}")
    print(f"  Markets captured:  {summary['markets']}")
    print(f"  Feature rows:      {summary['features']}")
    print(f"  Events emitted:    {summary['events']}")
    print(f"  Game snapshots:    {summary['games_path']}")
    print(f"  Market snapshots:  {summary['markets_path']}")
    print(f"  Feature store:     {summary['features_path']}")
    print(f"  Event log:         {summary['events_path']}")


if __name__ == "__main__":
    main()

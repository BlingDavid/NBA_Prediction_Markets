"""
Live Data Module
─────────────────
Fetches real-time NBA data for EV analysis:
  1. Current season team stats (offensive/defensive rating, pace, etc.)
  2. Current season player stats (key players per team)
  3. Head-to-head matchup history
  4. Live injury reports
  5. ESPN sportsbook odds for comparison
  6. The Odds API sportsbook lines

All data is cached briefly to avoid hammering APIs during a session.
"""

import json
import time
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

from config import (
    CURRENT_SEASON,
    ODDS_API_BASE,
    ODDS_API_KEY,
    ODDS_SPORT,
    TEAM_ABBREV_MAP,
)


# ═══════════════════════════════════════════════════════════════════════
# TEAM STATS — Current Season
# ═══════════════════════════════════════════════════════════════════════

def fetch_team_stats(season: str = CURRENT_SEASON) -> pd.DataFrame:
    """
    Fetch current season team stats: offensive rating, defensive rating,
    net rating, pace, record, etc.

    Uses nba_api LeagueDashTeamStats with Advanced measure type.
    """
    try:
        from nba_api.stats.endpoints import leaguedashteamstats

        time.sleep(0.6)
        # Base stats (W, L, PTS, etc.)
        base = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Base",
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
        )
        base_df = base.get_data_frames()[0]

        # Try advanced stats — but don't fail if it errors
        try:
            time.sleep(0.6)
            adv = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="PerGame",
                season_type_all_star="Regular Season",
            )
            adv_df = adv.get_data_frames()[0]

            # Only select columns that actually exist in the advanced df
            adv_cols_wanted = ["TEAM_ID", "OFF_RATING", "DEF_RATING", "NET_RATING",
                               "PACE", "AST_PCT", "AST_TO", "OREB_PCT", "DREB_PCT",
                               "EFG_PCT", "TS_PCT", "PIE"]
            adv_cols_available = [c for c in adv_cols_wanted if c in adv_df.columns]

            if "TEAM_ID" in adv_cols_available and len(adv_cols_available) > 1:
                merged = base_df.merge(
                    adv_df[adv_cols_available],
                    on="TEAM_ID",
                    how="left",
                    suffixes=("", "_adv"),
                )
            else:
                merged = base_df
        except Exception as adv_e:
            print(f"    (Advanced stats unavailable: {adv_e})")
            merged = base_df

        print(f"    Fetched stats for {len(merged)} teams ({season})")
        # Debug: print available columns so we can adapt
        # print(f"    Columns: {list(merged.columns)}")
        return merged

    except Exception as e:
        print(f"    Could not fetch team stats: {e}")
        return pd.DataFrame()


def get_team_profile(team_abbrev: str, stats_df: pd.DataFrame = None) -> dict:
    """
    Get a formatted profile for a specific team from the stats DataFrame.
    Handles varying column names from different nba_api versions.
    """
    if stats_df is None or stats_df.empty:
        stats_df = fetch_team_stats()

    if stats_df.empty:
        return {"error": "No stats available"}

    # Find the team abbreviation column (varies by nba_api version)
    abbrev_col = None
    for candidate in ["TEAM_ABBREVIATION", "TeamAbbreviation", "TEAM_ABBR", "TEAM"]:
        if candidate in stats_df.columns:
            abbrev_col = candidate
            break

    if abbrev_col is None:
        # Try to find any column containing 'abbreviation' or 'abbr'
        for col in stats_df.columns:
            if "abbr" in col.lower():
                abbrev_col = col
                break

    if abbrev_col is None:
        return {"error": f"No team abbreviation column found. Columns: {list(stats_df.columns)[:10]}"}

    row = stats_df[stats_df[abbrev_col] == team_abbrev]
    if row.empty:
        return {"error": f"Team {team_abbrev} not found in {abbrev_col}"}

    r = row.iloc[0]

    def safe_get(key, default=0):
        """Get a value from the row, trying multiple possible column names."""
        if key in r.index:
            val = r[key]
            return val if pd.notna(val) else default
        return default

    # Find team name column
    team_name = team_abbrev
    for name_col in ["TEAM_NAME", "TeamName", "TEAM"]:
        if name_col in r.index:
            team_name = r[name_col]
            break

    return {
        "team": team_abbrev,
        "name": team_name,
        "record": f"{int(safe_get('W', 0))}-{int(safe_get('L', 0))}",
        "win_pct": round(safe_get("W_PCT", 0), 3),
        "pts_per_game": round(safe_get("PTS", 0), 1),
        "opp_pts": round(safe_get("PLUS_MINUS", 0) * -1 + safe_get("PTS", 0), 1) if "PLUS_MINUS" in r.index else None,
        "off_rating": round(safe_get("OFF_RATING", 0), 1),
        "def_rating": round(safe_get("DEF_RATING", 0), 1),
        "net_rating": round(safe_get("NET_RATING", 0), 1),
        "pace": round(safe_get("PACE", 0), 1),
        "efg_pct": round(safe_get("EFG_PCT", 0), 3),
        "ts_pct": round(safe_get("TS_PCT", 0), 3),
    }


# ═══════════════════════════════════════════════════════════════════════
# PLAYER STATS — Key Players Per Team
# ═══════════════════════════════════════════════════════════════════════

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    # Try case-insensitive partial match
    for c in candidates:
        for col in df.columns:
            if c.lower() in col.lower():
                return col
    return None


def fetch_player_stats(season: str = CURRENT_SEASON, top_n: int = 8) -> pd.DataFrame:
    """
    Fetch current season player stats.
    Returns top N players per team by minutes played.
    """
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats

        time.sleep(0.6)
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
        )
        df = stats.get_data_frames()[0]

        # Find the minutes and team abbreviation columns
        min_col = _find_col(df, ["MIN", "Minutes", "MPG"])
        team_col = _find_col(df, ["TEAM_ABBREVIATION", "TeamAbbreviation", "TEAM_ABBR", "TEAM"])

        if min_col is None or team_col is None:
            print(f"    Player stats columns not as expected: {list(df.columns)[:15]}")
            return df

        # Normalize column names for downstream use
        df = df.rename(columns={team_col: "TEAM_ABBREVIATION", min_col: "MIN"})

        # Filter to players with meaningful minutes
        df = df[df["MIN"] >= 10.0].copy()

        # Rank within team by minutes
        df["team_rank"] = df.groupby("TEAM_ABBREVIATION")["MIN"].rank(
            ascending=False, method="first"
        )
        df = df[df["team_rank"] <= top_n]

        print(f"    Fetched stats for {len(df)} players ({season})")
        return df

    except Exception as e:
        print(f"    Could not fetch player stats: {e}")
        return pd.DataFrame()


def get_team_key_players(team_abbrev: str, player_df: pd.DataFrame = None) -> list[dict]:
    """Get key players and their stats for a team."""
    if player_df is None or player_df.empty:
        player_df = fetch_player_stats()

    if player_df.empty:
        return []

    # Find team column
    team_col = _find_col(player_df, ["TEAM_ABBREVIATION", "TeamAbbreviation", "TEAM_ABBR"])
    if team_col is None:
        return []

    min_col = _find_col(player_df, ["MIN", "Minutes", "MPG"]) or "MIN"

    team_players = player_df[
        player_df[team_col] == team_abbrev
    ].sort_values(min_col, ascending=False)

    def pget(row, candidates, default=0):
        """Get value from row trying multiple column names."""
        for c in candidates:
            if c in row.index and pd.notna(row[c]):
                return row[c]
        return default

    players = []
    for _, p in team_players.iterrows():
        players.append({
            "name": pget(p, ["PLAYER_NAME", "PlayerName", "PLAYER"], "Unknown"),
            "ppg": round(pget(p, ["PTS", "Points"]), 1),
            "rpg": round(pget(p, ["REB", "Rebounds"]), 1),
            "apg": round(pget(p, ["AST", "Assists"]), 1),
            "mpg": round(pget(p, ["MIN", "Minutes"]), 1),
            "fg_pct": round(pget(p, ["FG_PCT", "FGPct"]), 3),
            "fg3_pct": round(pget(p, ["FG3_PCT", "FG3Pct"]), 3),
            "plus_minus": round(pget(p, ["PLUS_MINUS", "PlusMinus"]), 1),
        })

    return players


# ═══════════════════════════════════════════════════════════════════════
# HEAD-TO-HEAD MATCHUP HISTORY
# ═══════════════════════════════════════════════════════════════════════

def fetch_h2h_history(
    team1: str,
    team2: str,
    last_n_seasons: int = 3,
) -> dict:
    """
    Fetch head-to-head matchup history between two teams.
    Returns win/loss record, average score, recent results.
    """
    try:
        from nba_api.stats.endpoints import leaguegamefinder
        from nba_api.stats.static import teams as nba_teams

        # Get team IDs
        all_teams = nba_teams.get_teams()
        t1_info = [t for t in all_teams if t["abbreviation"] == team1]
        t2_info = [t for t in all_teams if t["abbreviation"] == team2]

        if not t1_info or not t2_info:
            return {"error": f"Team not found: {team1 if not t1_info else team2}"}

        t1_id = t1_info[0]["id"]
        t2_id = t2_info[0]["id"]

        # Fetch games for team1 vs team2
        games = []
        current_year = datetime.now().year
        current_month = datetime.now().month

        for offset in range(last_n_seasons):
            if current_month >= 10:
                season_year = current_year - offset
            else:
                season_year = current_year - 1 - offset
            season = f"{season_year}-{str(season_year + 1)[-2:]}"

            time.sleep(0.6)
            finder = leaguegamefinder.LeagueGameFinder(
                team_id_nullable=t1_id,
                vs_team_id_nullable=t2_id,
                season_nullable=season,
                season_type_nullable="Regular Season",
                league_id_nullable="00",
            )
            df = finder.get_data_frames()[0]
            if not df.empty:
                games.append(df)

        if not games:
            return {"total_games": 0, "error": "No matchup history found"}

        all_games = pd.concat(games, ignore_index=True)
        all_games["GAME_DATE"] = pd.to_datetime(all_games["GAME_DATE"])
        all_games = all_games.sort_values("GAME_DATE", ascending=False)

        wins = len(all_games[all_games["WL"] == "W"])
        losses = len(all_games[all_games["WL"] == "L"])
        avg_pts = all_games["PTS"].mean()
        avg_margin = all_games["PLUS_MINUS"].mean()

        # Recent games (last 5)
        recent = []
        for _, g in all_games.head(5).iterrows():
            recent.append({
                "date": str(g["GAME_DATE"].date()),
                "matchup": g.get("MATCHUP", ""),
                "result": g.get("WL", ""),
                "score": f"{int(g['PTS'])}",
                "margin": f"{int(g['PLUS_MINUS']):+d}",
            })

        return {
            "team": team1,
            "opponent": team2,
            "total_games": wins + losses,
            "wins": wins,
            "losses": losses,
            "record": f"{wins}-{losses}",
            "avg_pts": round(avg_pts, 1),
            "avg_margin": round(avg_margin, 1),
            "recent_games": recent,
        }

    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# LIVE INJURY REPORTS
# ═══════════════════════════════════════════════════════════════════════

def fetch_live_injuries() -> pd.DataFrame:
    """
    Fetch current injury reports from multiple sources.
    Tries:
      1. NBA official injury report CDN
      2. ESPN scoreboard data
      3. nbainjuries package (if installed)
    """
    # Try NBA CDN first
    injuries = _fetch_nba_cdn_injuries()
    if injuries is not None and not injuries.empty:
        return injuries

    # Try ESPN scoreboard
    injuries = _fetch_espn_injuries()
    if injuries is not None and not injuries.empty:
        return injuries

    print("    No live injury data available. Use --injuries for manual input.")
    return pd.DataFrame(columns=["team", "player", "status", "reason"])


def _fetch_nba_cdn_injuries() -> pd.DataFrame | None:
    """Fetch from NBA's official injury report endpoint."""
    try:
        # NBA official injury report
        url = "https://official.nba.com/wp-json/api/v1/injury-report"
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://official.nba.com/",
        })
        if resp.status_code == 200:
            data = resp.json()
            rows = []
            for entry in data if isinstance(data, list) else data.get("data", []):
                rows.append({
                    "team": entry.get("team_abbreviation", entry.get("Team", "")),
                    "player": entry.get("player_name", entry.get("Player", "")),
                    "status": entry.get("current_status", entry.get("Status", "")),
                    "reason": entry.get("reason", entry.get("Reason", "")),
                })
            if rows:
                df = pd.DataFrame(rows)
                print(f"    Fetched {len(df)} injury reports from NBA.com")
                return df
    except Exception:
        pass
    return None


def _fetch_espn_injuries() -> pd.DataFrame | None:
    """Fetch injury info from ESPN's scoreboard API."""
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None

        data = resp.json()
        rows = []
        for event in data.get("events", []):
            for comp in event.get("competitions", []):
                for team in comp.get("competitors", []):
                    team_abbr = team.get("team", {}).get("abbreviation", "")
                    for player in team.get("injuries", []):
                        rows.append({
                            "team": team_abbr,
                            "player": player.get("athlete", {}).get("displayName", ""),
                            "status": player.get("status", ""),
                            "reason": player.get("description", player.get("type", "")),
                        })

        if rows:
            df = pd.DataFrame(rows)
            print(f"    Fetched {len(df)} injury reports from ESPN")
            return df
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════
# ESPN ODDS — Live Sportsbook Lines
# ═══════════════════════════════════════════════════════════════════════

def fetch_espn_odds(home_team: str = None, away_team: str = None) -> list[dict]:
    """
    Fetch live NBA odds from ESPN's public scoreboard API.
    Returns odds for today's games. Filters to specific matchup if teams given.
    """
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []

        data = resp.json()
        games_odds = []

        for event in data.get("events", []):
            game_info = {
                "name": event.get("name", ""),
                "date": event.get("date", ""),
                "status": event.get("status", {}).get("type", {}).get("description", ""),
            }

            # Get teams
            comps = event.get("competitions", [{}])
            if not comps:
                continue
            comp = comps[0]

            teams = {}
            for c in comp.get("competitors", []):
                hoa = c.get("homeAway", "")
                teams[hoa] = {
                    "abbrev": c.get("team", {}).get("abbreviation", ""),
                    "name": c.get("team", {}).get("displayName", ""),
                    "score": c.get("score", ""),
                }

            game_info["home"] = teams.get("home", {}).get("abbrev", "")
            game_info["away"] = teams.get("away", {}).get("abbrev", "")

            # Filter to specific matchup if requested
            if home_team and away_team:
                if game_info["home"] != home_team or game_info["away"] != away_team:
                    # Also try reverse
                    if game_info["home"] != away_team or game_info["away"] != home_team:
                        continue

            # Extract odds
            odds_list = comp.get("odds", [])
            if odds_list:
                odds = odds_list[0]  # Primary provider (usually DraftKings)
                game_info["provider"] = odds.get("provider", {}).get("name", "Unknown")
                game_info["spread"] = odds.get("details", "")
                game_info["overUnder"] = odds.get("overUnder", "")

                # Home/away spread and moneyline
                home_odds = odds.get("homeTeamOdds", {})
                away_odds = odds.get("awayTeamOdds", {})

                game_info["home_spread"] = home_odds.get("spreadOdds", "")
                game_info["home_moneyline"] = home_odds.get("moneyLine", "")
                game_info["away_spread"] = away_odds.get("spreadOdds", "")
                game_info["away_moneyline"] = away_odds.get("moneyLine", "")

                # Implied probability from moneyline
                for side in ["home", "away"]:
                    ml = game_info.get(f"{side}_moneyline")
                    if ml and ml != "":
                        try:
                            ml_int = int(ml)
                            if ml_int > 0:
                                game_info[f"{side}_implied_prob"] = round(100 / (ml_int + 100), 4)
                            else:
                                game_info[f"{side}_implied_prob"] = round(abs(ml_int) / (abs(ml_int) + 100), 4)
                        except (ValueError, ZeroDivisionError):
                            pass

            games_odds.append(game_info)

        return games_odds

    except Exception as e:
        print(f"    ESPN odds fetch error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════
# THE ODDS API — Multiple Sportsbooks
# ═══════════════════════════════════════════════════════════════════════

def fetch_odds_api_lines(home_team: str = None, away_team: str = None) -> list[dict]:
    """
    Fetch current NBA moneyline odds from The Odds API.
    Returns odds from multiple sportsbooks for comparison.
    """
    if not ODDS_API_KEY or ODDS_API_KEY.startswith("your_"):
        return []

    try:
        url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/odds"
        resp = requests.get(url, params={
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
        }, timeout=10)

        if resp.status_code != 200:
            return []

        data = resp.json()
        results = []

        # Map full team names to abbreviations for matching
        name_to_abbrev = {}
        for abbr, name in TEAM_ABBREV_MAP.items():
            name_to_abbrev[name] = abbr
            # Also add short forms
            parts = name.split()
            if len(parts) > 1:
                name_to_abbrev[parts[-1]] = abbr  # "Hawks", "Celtics", etc.

        for game in data:
            h_team = game.get("home_team", "")
            a_team = game.get("away_team", "")
            h_abbr = name_to_abbrev.get(h_team, h_team)
            a_abbr = name_to_abbrev.get(a_team, a_team)

            # Filter if specific teams requested
            if home_team and away_team:
                if not ((h_abbr == home_team and a_abbr == away_team) or
                        (h_abbr == away_team and a_abbr == home_team)):
                    continue

            game_odds = {
                "home": h_abbr,
                "away": a_abbr,
                "commence": game.get("commence_time", ""),
                "books": [],
            }

            for book in game.get("bookmakers", []):
                book_data = {
                    "name": book.get("title", ""),
                    "markets": {},
                }

                for market in book.get("markets", []):
                    mkey = market.get("key", "")
                    outcomes = {}
                    for outcome in market.get("outcomes", []):
                        team_name = outcome.get("name", "")
                        abbr = name_to_abbrev.get(team_name, team_name)
                        outcomes[abbr] = {
                            "price": outcome.get("price", 0),
                            "point": outcome.get("point", None),
                        }
                    book_data["markets"][mkey] = outcomes

                game_odds["books"].append(book_data)

            results.append(game_odds)

        return results

    except Exception as e:
        print(f"    Odds API error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════
# COMBINED ODDS COMPARISON
# ═══════════════════════════════════════════════════════════════════════

def get_odds_comparison(home_team: str, away_team: str) -> dict:
    """
    Fetch and combine odds from Kalshi, ESPN, and The Odds API
    into a single comparison dict for display.
    """
    comparison = {
        "home_team": home_team,
        "away_team": away_team,
        "kalshi": {},
        "espn": {},
        "sportsbooks": [],
    }

    # ESPN odds
    espn = fetch_espn_odds(home_team, away_team)
    if espn:
        e = espn[0]
        comparison["espn"] = {
            "provider": e.get("provider", "ESPN/DraftKings"),
            "home_moneyline": e.get("home_moneyline", ""),
            "away_moneyline": e.get("away_moneyline", ""),
            "home_implied": e.get("home_implied_prob"),
            "away_implied": e.get("away_implied_prob"),
            "spread": e.get("spread", ""),
            "over_under": e.get("overUnder", ""),
        }

    # The Odds API
    odds_api = fetch_odds_api_lines(home_team, away_team)
    if odds_api:
        game = odds_api[0]
        for book in game.get("books", []):
            h2h = book["markets"].get("h2h", {})
            spreads = book["markets"].get("spreads", {})
            totals = book["markets"].get("totals", {})

            home_ml = h2h.get(home_team, {}).get("price")
            away_ml = h2h.get(away_team, {}).get("price")

            book_entry = {
                "name": book["name"],
                "home_moneyline": home_ml,
                "away_moneyline": away_ml,
            }

            # Implied probabilities from moneyline
            if home_ml:
                if home_ml > 0:
                    book_entry["home_implied"] = round(100 / (home_ml + 100), 4)
                else:
                    book_entry["home_implied"] = round(abs(home_ml) / (abs(home_ml) + 100), 4)
            if away_ml:
                if away_ml > 0:
                    book_entry["away_implied"] = round(100 / (away_ml + 100), 4)
                else:
                    book_entry["away_implied"] = round(abs(away_ml) / (abs(away_ml) + 100), 4)

            # Spread
            home_spread = spreads.get(home_team, {})
            if home_spread.get("point") is not None:
                book_entry["spread"] = home_spread["point"]

            # Total
            for key, val in totals.items():
                if val.get("point") is not None:
                    book_entry["over_under"] = val["point"]
                    break

            comparison["sportsbooks"].append(book_entry)

    return comparison


def print_odds_comparison(comparison: dict, model_prob_home: float, kalshi_price: float):
    """Print a formatted odds comparison table."""
    home = comparison["home_team"]
    away = comparison["away_team"]

    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  ODDS COMPARISON: {away} @ {home:<35}│")
    print(f"  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │  {'Source':<20} {'Home ML':>9} {'Away ML':>9} {'Home %':>8} {'Away %':>8}│")
    print(f"  │  {'─' * 20} {'─' * 9} {'─' * 9} {'─' * 8} {'─' * 8}│")

    # Our model
    print(f"  │  {'Our Model':<20} {'':>9} {'':>9} {model_prob_home:>7.1%} {1 - model_prob_home:>7.1%}│")

    # Kalshi
    kalshi_implied = kalshi_price
    print(f"  │  {'Kalshi (ask)':<20} {'':>9} {'':>9} {kalshi_implied:>7.1%} {1 - kalshi_implied:>7.1%}│")

    # ESPN
    espn = comparison.get("espn", {})
    if espn:
        h_ml = espn.get("home_moneyline", "")
        a_ml = espn.get("away_moneyline", "")
        h_imp = espn.get("home_implied")
        a_imp = espn.get("away_implied")
        provider = espn.get("provider", "ESPN")[:20]
        h_ml_str = f"{h_ml:>+d}" if isinstance(h_ml, (int, float)) and h_ml else str(h_ml)[:9]
        a_ml_str = f"{a_ml:>+d}" if isinstance(a_ml, (int, float)) and a_ml else str(a_ml)[:9]
        h_imp_str = f"{h_imp:>7.1%}" if h_imp else "    N/A "
        a_imp_str = f"{a_imp:>7.1%}" if a_imp else "    N/A "
        print(f"  │  {provider:<20} {h_ml_str:>9} {a_ml_str:>9} {h_imp_str} {a_imp_str}│")

        if espn.get("spread"):
            print(f"  │    Spread: {espn['spread']:<48}│")
        if espn.get("over_under"):
            print(f"  │    O/U:    {espn['over_under']:<48}│")

    # Sportsbooks from Odds API
    for book in comparison.get("sportsbooks", [])[:5]:
        name = book["name"][:20]
        h_ml = book.get("home_moneyline")
        a_ml = book.get("away_moneyline")
        h_imp = book.get("home_implied")
        a_imp = book.get("away_implied")
        h_ml_str = f"{h_ml:>+d}" if isinstance(h_ml, (int, float)) else "     N/A"
        a_ml_str = f"{a_ml:>+d}" if isinstance(a_ml, (int, float)) else "     N/A"
        h_imp_str = f"{h_imp:>7.1%}" if h_imp else "    N/A "
        a_imp_str = f"{a_imp:>7.1%}" if a_imp else "    N/A "
        print(f"  │  {name:<20} {h_ml_str:>9} {a_ml_str:>9} {h_imp_str} {a_imp_str}│")

    # Average implied across all sources
    all_home_implied = []
    if espn.get("home_implied"):
        all_home_implied.append(espn["home_implied"])
    for book in comparison.get("sportsbooks", []):
        if book.get("home_implied"):
            all_home_implied.append(book["home_implied"])

    if all_home_implied:
        avg_market = np.mean(all_home_implied)
        our_edge = model_prob_home - avg_market
        print(f"  │  {'─' * 57}│")
        print(f"  │  {'Market consensus':<20} {'':>9} {'':>9} {avg_market:>7.1%} {1 - avg_market:>7.1%}│")
        edge_label = f"Model edge: {our_edge:+.1%}"
        print(f"  │  {edge_label:<57}│")

    print(f"  └─────────────────────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════════════════════
# FULL LIVE CONTEXT (combines everything)
# ═══════════════════════════════════════════════════════════════════════

def fetch_full_game_context(home_team: str, away_team: str) -> dict:
    """
    Fetch all available live context for a matchup:
    team stats, key players, H2H history, injuries, odds.
    """
    print("  Fetching live game context...")

    context = {
        "home_team": home_team,
        "away_team": away_team,
    }

    # Team stats
    print("    [1/5] Team stats...")
    stats_df = fetch_team_stats()
    context["home_profile"] = get_team_profile(home_team, stats_df)
    context["away_profile"] = get_team_profile(away_team, stats_df)

    # Player stats
    print("    [2/5] Player stats...")
    player_df = fetch_player_stats()
    context["home_players"] = get_team_key_players(home_team, player_df)
    context["away_players"] = get_team_key_players(away_team, player_df)

    # H2H history
    print("    [3/5] Head-to-head history...")
    context["h2h"] = fetch_h2h_history(home_team, away_team)

    # Injuries
    print("    [4/5] Injury reports...")
    context["injuries"] = fetch_live_injuries()

    # Odds comparison
    print("    [5/5] Sportsbook odds...")
    context["odds"] = get_odds_comparison(home_team, away_team)

    return context


def print_game_context(ctx: dict):
    """Print formatted game context summary."""
    home = ctx["home_team"]
    away = ctx["away_team"]
    h = ctx.get("home_profile", {})
    a = ctx.get("away_profile", {})

    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  LIVE GAME CONTEXT                                          │")
    print(f"  ├─────────────────────────────────────────────────────────────┤")

    # Team comparison
    if h.get("record") and a.get("record"):
        print(f"  │                   {home:<12}         {away:<12}       │")
        print(f"  │  Record:         {h['record']:<12}         {a['record']:<12}       │")
        print(f"  │  Off Rating:     {h.get('off_rating', 'N/A'):<12}         {a.get('off_rating', 'N/A'):<12}       │")
        print(f"  │  Def Rating:     {h.get('def_rating', 'N/A'):<12}         {a.get('def_rating', 'N/A'):<12}       │")
        print(f"  │  Net Rating:     {h.get('net_rating', 'N/A'):<12}         {a.get('net_rating', 'N/A'):<12}       │")
        print(f"  │  Pace:           {h.get('pace', 'N/A'):<12}         {a.get('pace', 'N/A'):<12}       │")
        print(f"  │  PPG:            {h.get('pts_per_game', 'N/A'):<12}         {a.get('pts_per_game', 'N/A'):<12}       │")

    # Key players
    for team, label, players in [(home, "HOME", ctx.get("home_players", [])),
                                  (away, "AWAY", ctx.get("away_players", []))]:
        if players:
            print(f"  │                                                             │")
            print(f"  │  {team} Key Players ({label}):                              │")
            for p in players[:5]:
                line = f"{p['name']:<22} {p['ppg']:>5.1f}p {p['rpg']:>4.1f}r {p['apg']:>4.1f}a {p['mpg']:>4.1f}m"
                print(f"  │    {line:<55}│")

    # H2H
    h2h = ctx.get("h2h", {})
    if h2h.get("total_games", 0) > 0:
        print(f"  │                                                             │")
        print(f"  │  H2H (last 3 seasons): {home} {h2h['record']} vs {away:<20}│")
        print(f"  │  Avg margin: {h2h['avg_margin']:+.1f} pts                                  │")
        for g in h2h.get("recent_games", [])[:3]:
            print(f"  │    {g['date']}  {g['matchup']:<20} {g['result']} ({g['margin']})     │")

    print(f"  └─────────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        home = sys.argv[1].upper()
        away = sys.argv[2].upper()
    else:
        home, away = "MIA", "ATL"

    print(f"\nFetching live data for {away} @ {home}...")
    ctx = fetch_full_game_context(home, away)
    print_game_context(ctx)

    odds = ctx.get("odds", {})
    if odds:
        print_odds_comparison(odds, 0.55, 0.50)  # Placeholder model/kalshi probs

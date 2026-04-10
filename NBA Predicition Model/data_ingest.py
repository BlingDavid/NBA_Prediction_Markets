"""
Data Ingestion Pipeline
─────────────────────────
Pulls NBA game data from nba_api and odds from The Odds API.
Stores raw data as CSV/JSON in data/raw/, cleaned data in data/processed/.
"""

import json
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    CURRENT_SEASON,
    ODDS_API_BASE,
    ODDS_API_KEY,
    ODDS_FORMAT,
    ODDS_MARKETS,
    ODDS_REGIONS,
    ODDS_SPORT,
    PROCESSED_DIR,
    RAW_DIR,
    TEAM_ABBREV_MAP,
    TEAM_NAME_TO_ABBREV,
    TRAINING_SEASONS,
)


# ═══════════════════════════════════════════════════════════════════════
# NBA GAME DATA (via nba_api)
# ═══════════════════════════════════════════════════════════════════════

def fetch_season_games(season: str) -> pd.DataFrame:
    """
    Fetch all regular-season games for a given season using nba_api.
    Returns a DataFrame with one row per game.
    """
    from nba_api.stats.endpoints import leaguegamefinder

    print(f"  Fetching games for {season}...")
    time.sleep(1)  # Rate limiting

    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
    )
    games_df = finder.get_data_frames()[0]

    if games_df.empty:
        print(f"  WARNING: No games found for {season}")
        return pd.DataFrame()

    # Each game appears twice (once per team). Pivot to one row per game.
    games_df["GAME_DATE"] = pd.to_datetime(games_df["GAME_DATE"])
    games_df = games_df.sort_values("GAME_DATE")

    return games_df


def build_game_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raw LeagueGameFinder output (2 rows per game) into
    a single-row-per-game format with home/away splits.
    """
    if raw_df.empty:
        return pd.DataFrame()

    # Identify home vs away from the MATCHUP column
    # Home games: "BOS vs. LAL", Away games: "LAL @ BOS"
    raw_df = raw_df.copy()
    raw_df["IS_HOME"] = raw_df["MATCHUP"].str.contains("vs.", na=False)

    home = raw_df[raw_df["IS_HOME"]].copy()
    away = raw_df[~raw_df["IS_HOME"]].copy()

    # Merge on GAME_ID
    merged = home.merge(
        away,
        on="GAME_ID",
        suffixes=("_home", "_away"),
    )

    games = pd.DataFrame({
        "game_id": merged["GAME_ID"],
        "date": merged["GAME_DATE_home"],
        "season": merged["SEASON_ID_home"],
        "home_team": merged["TEAM_ABBREVIATION_home"],
        "away_team": merged["TEAM_ABBREVIATION_away"],
        "home_pts": merged["PTS_home"],
        "away_pts": merged["PTS_away"],
        "home_fgm": merged["FGM_home"],
        "home_fga": merged["FGA_home"],
        "home_fg3m": merged["FG3M_home"],
        "home_fg3a": merged["FG3A_home"],
        "home_ftm": merged["FTM_home"],
        "home_fta": merged["FTA_home"],
        "home_oreb": merged["OREB_home"],
        "home_dreb": merged["DREB_home"],
        "home_reb": merged["REB_home"],
        "home_ast": merged["AST_home"],
        "home_stl": merged["STL_home"],
        "home_blk": merged["BLK_home"],
        "home_tov": merged["TOV_home"],
        "away_fgm": merged["FGM_away"],
        "away_fga": merged["FGA_away"],
        "away_fg3m": merged["FG3M_away"],
        "away_fg3a": merged["FG3A_away"],
        "away_ftm": merged["FTM_away"],
        "away_fta": merged["FTA_away"],
        "away_oreb": merged["OREB_away"],
        "away_dreb": merged["DREB_away"],
        "away_reb": merged["REB_away"],
        "away_ast": merged["AST_away"],
        "away_stl": merged["STL_away"],
        "away_blk": merged["BLK_away"],
        "away_tov": merged["TOV_away"],
        "home_win": (merged["PTS_home"] > merged["PTS_away"]).astype(int),
        "total_pts": merged["PTS_home"] + merged["PTS_away"],
        "spread": merged["PTS_home"] - merged["PTS_away"],
    })

    return games.sort_values("date").reset_index(drop=True)


def fetch_all_historical_games() -> pd.DataFrame:
    """Fetch and process games for all training seasons."""
    all_games = []

    for season in TRAINING_SEASONS:
        raw = fetch_season_games(season)
        games = build_game_rows(raw)
        if not games.empty:
            all_games.append(games)
            # Save raw data
            raw.to_csv(RAW_DIR / f"raw_games_{season}.csv", index=False)

    if not all_games:
        raise ValueError("No game data retrieved for any season!")

    combined = pd.concat(all_games, ignore_index=True)
    combined = combined.drop_duplicates(subset="game_id")
    combined.to_csv(PROCESSED_DIR / "all_games.csv", index=False)
    print(f"  Saved {len(combined)} games to {PROCESSED_DIR / 'all_games.csv'}")

    return combined


# ═══════════════════════════════════════════════════════════════════════
# ODDS DATA (via The Odds API)
# ═══════════════════════════════════════════════════════════════════════

def fetch_current_odds() -> pd.DataFrame:
    """
    Fetch current NBA odds from The Odds API.
    Returns moneyline, spread, and totals for upcoming games.
    """
    if not ODDS_API_KEY or ODDS_API_KEY.startswith("your_"):
        print("  WARNING: No ODDS_API_KEY set. Skipping odds fetch.")
        print("  Get a free key at https://the-odds-api.com/ and add it to .env")
        return pd.DataFrame()

    url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": ODDS_REGIONS,
        "markets": ODDS_MARKETS,
        "oddsFormat": ODDS_FORMAT,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"  WARNING: Odds API returned error: {e}")
        print("  Continuing without odds data.")
        return pd.DataFrame()

    data = resp.json()
    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"  Fetched {len(data)} upcoming games. API requests remaining: {remaining}")

    # Save raw JSON
    with open(RAW_DIR / "current_odds.json", "w") as f:
        json.dump(data, f, indent=2)

    return _parse_odds(data)


def _parse_odds(data: list[dict]) -> pd.DataFrame:
    """Parse The Odds API response into a clean DataFrame."""
    rows = []

    for game in data:
        game_id = game["id"]
        sport = game["sport_key"]
        commence = game["commence_time"]
        home = game["home_team"]
        away = game["away_team"]

        # Aggregate odds across bookmakers (take consensus/average)
        h2h_home, h2h_away = [], []
        spread_home, spread_home_point = [], []
        total_over, total_over_point = [], []

        for book in game.get("bookmakers", []):
            for market in book.get("markets", []):
                outcomes = {o["name"]: o for o in market["outcomes"]}

                if market["key"] == "h2h":
                    if home in outcomes:
                        h2h_home.append(outcomes[home]["price"])
                    if away in outcomes:
                        h2h_away.append(outcomes[away]["price"])

                elif market["key"] == "spreads":
                    if home in outcomes:
                        spread_home.append(outcomes[home]["price"])
                        spread_home_point.append(outcomes[home].get("point", 0))

                elif market["key"] == "totals":
                    if "Over" in outcomes:
                        total_over.append(outcomes["Over"]["price"])
                        total_over_point.append(outcomes["Over"].get("point", 0))

        row = {
            "odds_game_id": game_id,
            "commence_time": commence,
            "home_team_full": home,
            "away_team_full": away,
            "home_team": TEAM_NAME_TO_ABBREV.get(home, home),
            "away_team": TEAM_NAME_TO_ABBREV.get(away, away),
            "ml_home": _mean_or_none(h2h_home),
            "ml_away": _mean_or_none(h2h_away),
            "spread_home": _mean_or_none(spread_home_point),
            "spread_home_price": _mean_or_none(spread_home),
            "total_line": _mean_or_none(total_over_point),
            "total_over_price": _mean_or_none(total_over),
            "n_books": len(game.get("bookmakers", [])),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(PROCESSED_DIR / "current_odds.csv", index=False)
    return df


def _mean_or_none(vals: list) -> float | None:
    return round(sum(vals) / len(vals), 1) if vals else None


# ═══════════════════════════════════════════════════════════════════════
# ODDS CONVERSION UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def american_to_implied_prob(american_odds: float) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def implied_prob_to_fair(home_prob: float, away_prob: float) -> tuple[float, float]:
    """Remove vig from implied probabilities to get fair probabilities."""
    total = home_prob + away_prob
    return home_prob / total, away_prob / total


# ═══════════════════════════════════════════════════════════════════════
# HISTORICAL ODDS (for backtesting)
# ═══════════════════════════════════════════════════════════════════════

def fetch_historical_odds(date_str: str) -> pd.DataFrame:
    """
    Fetch historical odds for a specific date.
    Requires The Odds API historical endpoint (paid tier).
    Falls back to generating synthetic odds from closing lines if unavailable.
    """
    if not ODDS_API_KEY:
        return pd.DataFrame()

    url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/odds-history/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": ODDS_REGIONS,
        "markets": "h2h,spreads,totals",
        "oddsFormat": ODDS_FORMAT,
        "date": date_str,  # ISO format: 2024-01-15T00:00:00Z
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 422:
            # Historical endpoint may not be available on free tier
            return pd.DataFrame()
        resp.raise_for_status()
        return _parse_odds(resp.json().get("data", []))
    except Exception as e:
        print(f"  Could not fetch historical odds for {date_str}: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def run_full_ingest():
    """Run the complete data ingestion pipeline."""
    print("=" * 60)
    print("NBA Quant Model — Data Ingestion")
    print("=" * 60)

    print("\n[1/2] Fetching historical NBA game data...")
    games = fetch_all_historical_games()
    print(f"  Total games: {len(games)}")
    print(f"  Date range: {games['date'].min()} to {games['date'].max()}")
    print(f"  Seasons: {games['season'].nunique()}")

    print("\n[2/2] Fetching current odds...")
    odds = fetch_current_odds()
    if not odds.empty:
        print(f"  Upcoming games with odds: {len(odds)}")
    else:
        print("  No odds data (set ODDS_API_KEY in .env)")

    print("\nData ingestion complete.")
    return games, odds


if __name__ == "__main__":
    run_full_ingest()

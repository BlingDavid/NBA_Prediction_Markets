"""
Live Game Adjustments Module
──────────────────────────────
Fetches real-time injury reports and schedule data to adjust
model predictions for upcoming games.

Two key adjustments:
  1. Injuries — star players out can shift win probability 5-15%
  2. Back-to-back / rest — teams on B2B lose ~2-4% win probability

Data sources:
  - NBA API (nba_api): scoreboard, injury reports, team schedule
  - Fallback: manual injury input if API is unavailable
"""

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import CURRENT_SEASON, TEAM_ABBREV_MAP


# ═══════════════════════════════════════════════════════════════════════
# INJURY DATA
# ═══════════════════════════════════════════════════════════════════════

def fetch_injuries() -> pd.DataFrame:
    """
    Fetch current NBA injury report.
    Tries multiple sources in order of reliability.
    """
    # Try NBA API first
    injuries = _fetch_injuries_nba_api()
    if injuries is not None and not injuries.empty:
        return injuries

    # Fallback: empty DataFrame (user can input manually)
    print("    Could not fetch live injury data.")
    print("    You can manually specify injuries with --injuries flag.")
    return pd.DataFrame(columns=[
        "team", "player", "status", "reason", "impact_rating"
    ])


def _fetch_injuries_nba_api() -> pd.DataFrame | None:
    """Fetch injuries from NBA API endpoints."""
    try:
        from nba_api.stats.endpoints import playerindex
        from nba_api.stats.static import teams as nba_teams

        # The NBA API doesn't have a direct injury endpoint,
        # but we can get player status from the daily scoreboard
        # and league game log endpoints

        # Try the cumulative player stats to find players with
        # recent inactivity (proxy for injury)
        pass
    except ImportError:
        pass

    # Use the NBA injury report page data
    try:
        from nba_api.stats.endpoints import scoreboardv2

        time.sleep(0.6)
        scoreboard = scoreboardv2.ScoreboardV2(
            game_date=datetime.now().strftime("%Y-%m-%d"),
            league_id="00",
        )

        # Check if there's game-level inactive players data
        dfs = scoreboard.get_data_frames()
        # Scoreboard returns multiple DataFrames; look for player info
        for df in dfs:
            if "PLAYER_NAME" in df.columns or "INACTIVE" in str(df.columns):
                return _parse_scoreboard_injuries(df)

    except Exception as e:
        pass

    # Fallback: try fetching from the NBA injury report
    try:
        import requests

        # NBA's official injury JSON feed
        url = "https://cdn.nba.com/static/json/liveData/odds/odds_todaysGames.json"
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nba.com/",
        })
        if resp.status_code == 200:
            data = resp.json()
            # Parse if injury data is available
            return _parse_nba_cdn_injuries(data)
    except Exception:
        pass

    return None


def _parse_scoreboard_injuries(df: pd.DataFrame) -> pd.DataFrame | None:
    """Parse scoreboard data for inactive/injured players."""
    if df.empty:
        return None

    rows = []
    for _, row in df.iterrows():
        player = row.get("PLAYER_NAME", row.get("PLAYER", ""))
        team = row.get("TEAM_ABBREVIATION", row.get("TEAM", ""))
        status = row.get("COMMENT", row.get("DESCRIPTION", ""))

        if player and team:
            rows.append({
                "team": team,
                "player": player,
                "status": _classify_status(str(status)),
                "reason": str(status)[:100] if status else "Unknown",
                "impact_rating": 0,  # Will be estimated below
            })

    if not rows:
        return None

    result = pd.DataFrame(rows)
    result["impact_rating"] = result.apply(_estimate_player_impact, axis=1)
    return result


def _parse_nba_cdn_injuries(data: dict) -> pd.DataFrame | None:
    """Parse NBA CDN injury/odds data."""
    # Structure varies — attempt to extract
    games = data.get("games", [])
    rows = []
    for game in games:
        for team_key in ["homeTeam", "awayTeam"]:
            team_data = game.get(team_key, {})
            team_abbrev = team_data.get("teamTricode", "")
            for player in team_data.get("injuredPlayers", team_data.get("players", [])):
                status = player.get("injuryStatus", player.get("status", ""))
                if status and status.lower() != "active":
                    rows.append({
                        "team": team_abbrev,
                        "player": player.get("name", player.get("firstName", "") + " " + player.get("lastName", "")),
                        "status": _classify_status(status),
                        "reason": player.get("injuryDescription", player.get("reason", status)),
                        "impact_rating": 0,
                    })

    if not rows:
        return None

    result = pd.DataFrame(rows)
    result["impact_rating"] = result.apply(_estimate_player_impact, axis=1)
    return result


def _classify_status(status_str: str) -> str:
    """Classify injury status into standard categories."""
    s = status_str.lower()
    if any(w in s for w in ["out", "ruled out", "inactive"]):
        return "OUT"
    elif any(w in s for w in ["doubtful"]):
        return "DOUBTFUL"
    elif any(w in s for w in ["questionable", "game time", "game-time"]):
        return "QUESTIONABLE"
    elif any(w in s for w in ["probable", "available", "likely"]):
        return "PROBABLE"
    elif any(w in s for w in ["day-to-day", "day to day"]):
        return "DAY-TO-DAY"
    return "UNKNOWN"


# ═══════════════════════════════════════════════════════════════════════
# PLAYER IMPACT ESTIMATION
# ═══════════════════════════════════════════════════════════════════════

# Star players missing shifts win probability significantly.
# This is a simplified model — a real system would use RAPTOR, RPM,
# or win shares to compute exact impact per player.

# Approximate win probability impact when a player is OUT
# Tier 1 (MVP candidates): ~8-15% swing
# Tier 2 (All-Stars): ~4-8% swing
# Tier 3 (Quality starters): ~2-4% swing
# Tier 4 (Role players): ~0.5-2% swing

STAR_PLAYERS = {
    # Tier 1: MVP-caliber (impact: 0.10 - 0.15)
    "Nikola Jokic": 0.14, "Luka Doncic": 0.13, "Shai Gilgeous-Alexander": 0.13,
    "Giannis Antetokounmpo": 0.13, "Jayson Tatum": 0.12, "Anthony Edwards": 0.11,
    "LeBron James": 0.10, "Kevin Durant": 0.10, "Stephen Curry": 0.10,
    "Joel Embiid": 0.12, "Anthony Davis": 0.10, "Donovan Mitchell": 0.09,
    "Jalen Brunson": 0.10, "Victor Wembanyama": 0.11,

    # Tier 2: All-Stars (impact: 0.05 - 0.09)
    "Jaylen Brown": 0.07, "Devin Booker": 0.08, "Trae Young": 0.08,
    "Damian Lillard": 0.07, "De'Aaron Fox": 0.07, "Tyrese Haliburton": 0.07,
    "Ja Morant": 0.08, "Bam Adebayo": 0.06, "Jimmy Butler": 0.07,
    "Paul George": 0.06, "Karl-Anthony Towns": 0.06, "Domantas Sabonis": 0.07,
    "Paolo Banchero": 0.07, "LaMelo Ball": 0.07, "Chet Holmgren": 0.07,
    "Tyrese Maxey": 0.07, "Scottie Barnes": 0.07, "Evan Mobley": 0.06,
    "Franz Wagner": 0.07, "Mikal Bridges": 0.05,

    # Tier 3: Quality starters (impact: 0.03 - 0.05)
    "Derrick White": 0.04, "Jrue Holiday": 0.04, "Khris Middleton": 0.04,
    "Kristaps Porzingis": 0.05, "Zion Williamson": 0.05, "Lauri Markkanen": 0.05,
    "Desmond Bane": 0.04, "Tyler Herro": 0.04, "CJ McCollum": 0.04,
    "Dejounte Murray": 0.04, "OG Anunoby": 0.04, "Jarrett Allen": 0.04,
}


def _estimate_player_impact(row: pd.Series) -> float:
    """Estimate the win probability impact of a player being out."""
    player = row.get("player", "")
    status = row.get("status", "")

    # Look up player impact
    base_impact = STAR_PLAYERS.get(player, 0.02)  # Default: role player

    # Scale by status certainty
    status_multiplier = {
        "OUT": 1.0,
        "DOUBTFUL": 0.85,
        "QUESTIONABLE": 0.50,
        "DAY-TO-DAY": 0.50,
        "PROBABLE": 0.15,
        "UNKNOWN": 0.30,
    }
    multiplier = status_multiplier.get(status, 0.30)

    return round(base_impact * multiplier, 4)


def get_team_injury_impact(injuries: pd.DataFrame, team: str) -> dict:
    """
    Calculate total injury impact for a specific team.
    Returns dict with total impact, list of injured players, and details.
    """
    if injuries.empty:
        return {"total_impact": 0, "players": [], "details": []}

    team_injuries = injuries[injuries["team"] == team]

    if team_injuries.empty:
        return {"total_impact": 0, "players": [], "details": []}

    # Sum up impacts (with diminishing returns for multiple injuries)
    impacts = sorted(team_injuries["impact_rating"].values, reverse=True)
    total = 0
    for i, imp in enumerate(impacts):
        # Diminishing returns: each additional injury has slightly less marginal impact
        # because you can only lose so much from a depleted roster
        total += imp * (0.85 ** i)

    # Cap total impact at 25% — even a gutted roster has some chance
    total = min(total, 0.25)

    details = []
    for _, row in team_injuries.iterrows():
        details.append({
            "player": row["player"],
            "status": row["status"],
            "reason": row["reason"],
            "impact": row["impact_rating"],
        })

    return {
        "total_impact": round(total, 4),
        "players": team_injuries["player"].tolist(),
        "details": details,
    }


# ═══════════════════════════════════════════════════════════════════════
# SCHEDULE / BACK-TO-BACK DETECTION
# ═══════════════════════════════════════════════════════════════════════

def fetch_team_schedule(team: str, season: str = CURRENT_SEASON) -> pd.DataFrame:
    """Fetch a team's schedule to detect back-to-backs."""
    try:
        from nba_api.stats.endpoints import teamgamelog

        # Get team ID
        from nba_api.stats.static import teams as nba_teams
        all_teams = nba_teams.get_teams()
        team_info = [t for t in all_teams if t["abbreviation"] == team]

        if not team_info:
            return pd.DataFrame()

        team_id = team_info[0]["id"]
        time.sleep(0.6)

        log = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
            season_type_all_star="Regular Season",
        )
        df = log.get_data_frames()[0]

        if not df.empty:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
            df = df.sort_values("GAME_DATE")

        return df

    except Exception as e:
        return pd.DataFrame()


def detect_back_to_back(team: str, game_date: str | None = None) -> dict:
    """
    Detect if a team is on a back-to-back for a given date.

    Returns:
      is_b2b: bool
      rest_days: int (days since last game)
      games_in_7d: int (games in last 7 days)
      last_game_date: str
      schedule_note: str
    """
    if game_date:
        target_date = pd.Timestamp(game_date)
    else:
        target_date = pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))

    schedule = fetch_team_schedule(team)

    if schedule.empty:
        # Fallback: try to get from our processed data
        return _detect_b2b_from_features(team, target_date)

    # Find the most recent game before target_date
    past_games = schedule[schedule["GAME_DATE"] < target_date]
    if past_games.empty:
        return {
            "is_b2b": False,
            "rest_days": 3,
            "games_in_7d": 0,
            "last_game_date": None,
            "schedule_note": "No recent games found",
        }

    last_game = past_games.iloc[-1]
    last_date = last_game["GAME_DATE"]
    rest_days = (target_date - last_date).days

    # Games in last 7 days
    week_ago = target_date - timedelta(days=7)
    games_7d = len(past_games[past_games["GAME_DATE"] > week_ago])

    is_b2b = rest_days == 1

    # Also check if they played 2 days ago (so last night was rest)
    schedule_note = ""
    if is_b2b:
        schedule_note = f"BACK-TO-BACK: played yesterday ({last_date.strftime('%b %d')})"
    elif rest_days == 0:
        schedule_note = "Playing second game today (same day B2B)"
    elif rest_days == 2:
        schedule_note = f"1 day rest (last played {last_date.strftime('%b %d')})"
    elif rest_days >= 4:
        schedule_note = f"Well rested: {rest_days} days off"
    else:
        schedule_note = f"{rest_days} days rest"

    if games_7d >= 4:
        schedule_note += f" | Heavy schedule: {games_7d} games in 7 days"

    return {
        "is_b2b": is_b2b,
        "rest_days": rest_days,
        "games_in_7d": games_7d,
        "last_game_date": str(last_date.date()),
        "schedule_note": schedule_note,
    }


def _detect_b2b_from_features(team: str, target_date: pd.Timestamp) -> dict:
    """Fallback: detect B2B from our processed game data."""
    from config import PROCESSED_DIR

    games_path = PROCESSED_DIR / "all_games.csv"
    if not games_path.exists():
        return {
            "is_b2b": False, "rest_days": -1, "games_in_7d": -1,
            "last_game_date": None,
            "schedule_note": "No schedule data available",
        }

    games = pd.read_csv(games_path, parse_dates=["date"])

    # Find games for this team
    team_games = games[
        (games["home_team"] == team) | (games["away_team"] == team)
    ].sort_values("date")

    past = team_games[team_games["date"] < target_date]
    if past.empty:
        return {
            "is_b2b": False, "rest_days": -1, "games_in_7d": -1,
            "last_game_date": None,
            "schedule_note": "No recent games in data",
        }

    last_date = past.iloc[-1]["date"]
    rest_days = (target_date - last_date).days

    week_ago = target_date - timedelta(days=7)
    games_7d = len(past[past["date"] > week_ago])

    is_b2b = rest_days == 1

    note = "BACK-TO-BACK" if is_b2b else f"{rest_days} days rest"
    if games_7d >= 4:
        note += f" | {games_7d} games in 7 days"

    return {
        "is_b2b": is_b2b,
        "rest_days": rest_days,
        "games_in_7d": games_7d,
        "last_game_date": str(last_date.date()),
        "schedule_note": note,
    }


# ═══════════════════════════════════════════════════════════════════════
# PROBABILITY ADJUSTMENT
# ═══════════════════════════════════════════════════════════════════════

# Back-to-back impact on win probability (based on historical data):
# Teams on B2B lose about 2-4% win probability
B2B_PENALTY = 0.03  # 3% reduction in win prob for B2B team
HEAVY_SCHEDULE_PENALTY = 0.015  # 1.5% for 4+ games in 7 days
REST_ADVANTAGE_BONUS = 0.02  # 2% bonus for 3+ days rest vs opponent on B2B


def compute_adjusted_probability(
    base_home_prob: float,
    home_team: str,
    away_team: str,
    injuries: pd.DataFrame,
    game_date: str | None = None,
) -> dict:
    """
    Adjust model probability based on injuries and schedule.

    Returns dict with adjusted probabilities and breakdown of adjustments.
    """
    adjustments = []
    home_adj = 0.0
    away_adj = 0.0

    # ── Injury adjustments ──
    home_injury = get_team_injury_impact(injuries, home_team)
    away_injury = get_team_injury_impact(injuries, away_team)

    if home_injury["total_impact"] > 0:
        home_adj -= home_injury["total_impact"]
        adjustments.append({
            "type": "injury",
            "team": home_team,
            "impact": -home_injury["total_impact"],
            "detail": f"{len(home_injury['players'])} player(s) affected",
        })

    if away_injury["total_impact"] > 0:
        away_adj -= away_injury["total_impact"]
        adjustments.append({
            "type": "injury",
            "team": away_team,
            "impact": -away_injury["total_impact"],
            "detail": f"{len(away_injury['players'])} player(s) affected",
        })

    # ── Schedule adjustments ──
    home_sched = detect_back_to_back(home_team, game_date)
    away_sched = detect_back_to_back(away_team, game_date)

    if home_sched["is_b2b"]:
        home_adj -= B2B_PENALTY
        adjustments.append({
            "type": "schedule",
            "team": home_team,
            "impact": -B2B_PENALTY,
            "detail": home_sched["schedule_note"],
        })

    if away_sched["is_b2b"]:
        away_adj -= B2B_PENALTY
        adjustments.append({
            "type": "schedule",
            "team": away_team,
            "impact": -B2B_PENALTY,
            "detail": away_sched["schedule_note"],
        })

    # Heavy schedule penalty
    if home_sched.get("games_in_7d", 0) >= 4 and not home_sched["is_b2b"]:
        home_adj -= HEAVY_SCHEDULE_PENALTY
        adjustments.append({
            "type": "schedule",
            "team": home_team,
            "impact": -HEAVY_SCHEDULE_PENALTY,
            "detail": f"Heavy: {home_sched['games_in_7d']} games in 7 days",
        })

    if away_sched.get("games_in_7d", 0) >= 4 and not away_sched["is_b2b"]:
        away_adj -= HEAVY_SCHEDULE_PENALTY
        adjustments.append({
            "type": "schedule",
            "team": away_team,
            "impact": -HEAVY_SCHEDULE_PENALTY,
            "detail": f"Heavy: {away_sched['games_in_7d']} games in 7 days",
        })

    # Rest advantage: well-rested team vs B2B opponent
    if home_sched.get("rest_days", 0) >= 3 and away_sched["is_b2b"]:
        home_adj += REST_ADVANTAGE_BONUS
        adjustments.append({
            "type": "schedule",
            "team": home_team,
            "impact": REST_ADVANTAGE_BONUS,
            "detail": f"Rest advantage: {home_sched['rest_days']}d rest vs B2B opponent",
        })
    elif away_sched.get("rest_days", 0) >= 3 and home_sched["is_b2b"]:
        away_adj += REST_ADVANTAGE_BONUS
        adjustments.append({
            "type": "schedule",
            "team": away_team,
            "impact": REST_ADVANTAGE_BONUS,
            "detail": f"Rest advantage: {away_sched['rest_days']}d rest vs B2B opponent",
        })

    # ── Apply adjustments ──
    # Convert adjustments to home probability shift
    # home_adj improves home team → increases home prob
    # away_adj improves away team → decreases home prob
    total_shift = home_adj - away_adj

    adjusted_home_prob = np.clip(base_home_prob + total_shift, 0.02, 0.98)
    adjusted_away_prob = 1 - adjusted_home_prob

    return {
        "base_home_prob": base_home_prob,
        "base_away_prob": 1 - base_home_prob,
        "adjusted_home_prob": round(adjusted_home_prob, 4),
        "adjusted_away_prob": round(adjusted_away_prob, 4),
        "total_shift": round(total_shift, 4),
        "adjustments": adjustments,
        "home_injury": home_injury,
        "away_injury": away_injury,
        "home_schedule": home_sched,
        "away_schedule": away_sched,
    }


# ═══════════════════════════════════════════════════════════════════════
# MANUAL INJURY INPUT
# ═══════════════════════════════════════════════════════════════════════

def parse_manual_injuries(injury_str: str) -> pd.DataFrame:
    """
    Parse manually specified injuries from command line.

    Format: "TEAM:Player Name:STATUS,TEAM:Player Name:STATUS"
    Example: "MIA:Jimmy Butler:OUT,MIA:Tyler Herro:QUESTIONABLE,ATL:Trae Young:OUT"
    """
    if not injury_str:
        return pd.DataFrame(columns=["team", "player", "status", "reason", "impact_rating"])

    rows = []
    for entry in injury_str.split(","):
        parts = entry.strip().split(":")
        if len(parts) >= 2:
            team = parts[0].strip().upper()
            player = parts[1].strip()
            status = parts[2].strip().upper() if len(parts) >= 3 else "OUT"

            impact = STAR_PLAYERS.get(player, 0.02)
            status_mult = {"OUT": 1.0, "DOUBTFUL": 0.85, "QUESTIONABLE": 0.5, "PROBABLE": 0.15}
            impact *= status_mult.get(status, 0.5)

            rows.append({
                "team": team,
                "player": player,
                "status": status,
                "reason": "Manual input",
                "impact_rating": round(impact, 4),
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════

def print_adjustment_report(adj: dict, home_team: str, away_team: str):
    """Print a formatted adjustment report."""
    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  LIVE ADJUSTMENTS                                           │")
    print(f"  ├─────────────────────────────────────────────────────────────┤")

    # Injuries
    for team, label in [(home_team, "HOME"), (away_team, "AWAY")]:
        injury = adj[f"{label.lower()}_injury"]
        sched = adj[f"{label.lower()}_schedule"]

        if injury["details"] or sched.get("schedule_note"):
            print(f"  │                                                             │")
            print(f"  │  {team} ({label}):                                          │")

            if injury["details"]:
                for d in injury["details"]:
                    status_icon = {"OUT": "X", "DOUBTFUL": "?", "QUESTIONABLE": "~", "PROBABLE": "+"}
                    icon = status_icon.get(d["status"], "?")
                    impact_str = f"-{d['impact']:.1%}" if d["impact"] > 0 else ""
                    print(f"  │    [{icon}] {d['player']:<25} {d['status']:<12} {impact_str:<8}│")

            if sched.get("schedule_note"):
                b2b_icon = "!!" if sched["is_b2b"] else "  "
                print(f"  │    {b2b_icon} {sched['schedule_note']:<55}│")

    # Summary
    print(f"  │                                                             │")
    print(f"  │  Base model:    {home_team} {adj['base_home_prob']:.1%}  /  {away_team} {adj['base_away_prob']:.1%}       │")

    shift = adj["total_shift"]
    shift_dir = f"→ {home_team}" if shift > 0 else f"→ {away_team}" if shift < 0 else "none"
    print(f"  │  Adjustment:    {shift:+.1%} ({shift_dir})                          │")
    print(f"  │  Adjusted:      {home_team} {adj['adjusted_home_prob']:.1%}  /  {away_team} {adj['adjusted_away_prob']:.1%}       │")
    print(f"  └─────────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    # Quick test
    print("Testing injury parsing...")
    injuries = parse_manual_injuries("MIA:Jimmy Butler:OUT,MIA:Tyler Herro:QUESTIONABLE,ATL:Trae Young:OUT")
    print(injuries)

    print("\nTesting adjustment calculation...")
    adj = compute_adjusted_probability(
        base_home_prob=0.55,
        home_team="MIA",
        away_team="ATL",
        injuries=injuries,
    )
    print_adjustment_report(adj, "MIA", "ATL")

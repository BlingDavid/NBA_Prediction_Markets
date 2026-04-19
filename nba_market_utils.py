"""
Shared helpers for NBA prediction-market identifiers.
"""

from __future__ import annotations

import re

import pandas as pd


KALSHI_TO_NBA = {
    "ATL": "ATL", "BOS": "BOS", "BKN": "BKN", "BRK": "BKN",
    "CHA": "CHA", "CHI": "CHI", "CLE": "CLE", "DAL": "DAL",
    "DEN": "DEN", "DET": "DET",
    "GSW": "GSW", "GS": "GSW",
    "HOU": "HOU", "IND": "IND",
    "LAC": "LAC", "LAL": "LAL",
    "MEM": "MEM", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN",
    "NOP": "NOP", "NO": "NOP",
    "NYK": "NYK", "NY": "NYK",
    "OKC": "OKC", "ORL": "ORL",
    "PHI": "PHI", "PHX": "PHX", "PHO": "PHX",
    "POR": "POR", "SAC": "SAC",
    "SAS": "SAS", "SA": "SAS",
    "TOR": "TOR",
    "UTA": "UTA", "UTAH": "UTA",
    "WAS": "WAS", "WSH": "WAS",
}


def normalize_nba_abbrev(team: str | None) -> str | None:
    """Normalize external 2-3 letter team codes to repo-standard NBA abbreviations."""
    if not team:
        return team
    return KALSHI_TO_NBA.get(str(team).upper(), str(team).upper())


def parse_nba_ticker(ticker: str) -> dict | None:
    """
    Parse a Kalshi NBA game ticker into a normalized matchup payload.

    Expected format:
      KXNBAGAME-YYMMMDDAWAYHOME-BETTEAM
    """
    parts = ticker.split("-")
    if len(parts) < 2:
        return None

    event_part = parts[1]
    bet_team = parts[2] if len(parts) >= 3 else None

    match = re.match(r"(\d{2})([A-Z]{3})(\d{2})([A-Z]{2,3})([A-Z]{2,3})", event_part)
    if not match:
        return None

    year, month, day, away_kalshi, home_kalshi = match.groups()

    away_nba = normalize_nba_abbrev(away_kalshi)
    home_nba = normalize_nba_abbrev(home_kalshi)
    bet_nba = normalize_nba_abbrev(bet_team) if bet_team else None
    game_date = pd.to_datetime(f"20{year}-{month}-{day}").strftime("%Y-%m-%d")

    return {
        "away_team": away_nba,
        "home_team": home_nba,
        "bet_team": bet_nba,
        "bet_side": (
            "home" if bet_nba == home_nba else "away" if bet_nba == away_nba else None
        ),
        "date_str": game_date,
        "game_date": game_date,
        "away_kalshi": away_kalshi,
        "home_kalshi": home_kalshi,
    }

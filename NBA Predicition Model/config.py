"""
Central configuration for the NBA Quant Model.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
KALSHI_API_KEY = os.getenv("KALSHI_API_KEY", "")
KALSHI_API_SECRET = os.getenv("KALSHI_API_SECRET", "")

# ── NBA Seasons ────────────────────────────────────────────────────────
# Seasons to pull for training data (NBA API format: "2023-24")
TRAINING_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
CURRENT_SEASON = "2024-25"

# ── ELO Settings ───────────────────────────────────────────────────────
ELO_INITIAL = 1500
ELO_K_FACTOR = 20          # Standard K-factor
ELO_HOME_ADVANTAGE = 65    # ~65 ELO points ≈ 3.5 point home advantage
ELO_SEASON_REVERT = 0.25   # Revert 25% toward mean each season

# ── Model Settings ─────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.2             # Hold out last 20% of each season for validation
CV_FOLDS = 5

# ── Betting Settings ──────────────────────────────────────────────────
KELLY_FRACTION = 0.25       # Quarter-Kelly for safety
MIN_EDGE_THRESHOLD = 0.03   # Only bet when model edge > 3%
MAX_BET_FRACTION = 0.05     # Never risk more than 5% of bankroll
BANKROLL = 10_000           # Starting bankroll for backtests

# ── Odds API ───────────────────────────────────────────────────────────
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "basketball_nba"
ODDS_REGIONS = "us"
ODDS_MARKETS = "h2h,spreads,totals"
ODDS_FORMAT = "american"

# ── Team Abbreviation Mapping ─────────────────────────────────────────
# Maps NBA API team abbreviations to common forms used by odds APIs
TEAM_ABBREV_MAP = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets", "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons", "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

# Reverse map: full name -> abbreviation
TEAM_NAME_TO_ABBREV = {v: k for k, v in TEAM_ABBREV_MAP.items()}

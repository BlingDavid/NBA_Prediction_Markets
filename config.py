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
LIVE_DIR = DATA_DIR / "live"
LIVE_GAMES_DIR = LIVE_DIR / "games"
LIVE_MARKETS_DIR = LIVE_DIR / "markets"
LIVE_FEATURES_DIR = LIVE_DIR / "features"
LIVE_EVENTS_DIR = LIVE_DIR / "events"
LIVE_LABELS_DIR = LIVE_DIR / "labels"
LIVE_STATE_DIR = LIVE_DIR / "state"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for d in [
    RAW_DIR,
    PROCESSED_DIR,
    LIVE_GAMES_DIR,
    LIVE_MARKETS_DIR,
    LIVE_FEATURES_DIR,
    LIVE_EVENTS_DIR,
    LIVE_LABELS_DIR,
    LIVE_STATE_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
KALSHI_API_KEY = os.getenv("KALSHI_API_KEY", "")
KALSHI_RSA_PRIVATE_KEY_PATH = os.getenv("KALSHI_RSA_PRIVATE_KEY_PATH", "")
KALSHI_API_SECRET = os.getenv("KALSHI_API_SECRET", "")  # Legacy, not needed for RSA auth

# ── NBA Seasons ────────────────────────────────────────────────────────
# Seasons to pull for training data (NBA API format: "2023-24")
TRAINING_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
CURRENT_SEASON = "2025-26"

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

# ── Consensus Divergence Signal (IMPROVEMENTS #8) ─────────────────────
# Minimum number of non-stale sportsbooks required to compute a consensus.
# Below this count, consensus is None and tier defaults to 1 (neutral).
CONSENSUS_MIN_BOOKS = 3

# Drop book quotes older than this many seconds (uses Odds API last_update).
CONSENSUS_STALE_SECONDS = 300

# Tier-2 / Tier -1 trigger: minimum |Kalshi - consensus| AND |Kalshi - model|
# disagreement in percentage points. See spec for tier assignment rules.
CONSENSUS_DIVERGENCE_THRESHOLD_PP = 3.5

# Multiplier applied to MIN_EDGE_THRESHOLD per tier:
#   2  → multiplier < 1  (easier to bet; high-conviction triangulation)
#   1  → multiplier = 1  (neutral)
#  -1  → multiplier > 1  (stricter; model ↔ consensus contradiction)
TIER_THRESHOLD_MULTIPLIERS = {2: 0.7, 1: 1.0, -1: 1.5}

# Rollout gate: "shadow" logs tiers but does NOT modify bet decisions;
# "active" applies TIER_THRESHOLD_MULTIPLIERS; "off" pins tier to 1 (disabled).
CONSENSUS_TIER_MODE = "shadow"

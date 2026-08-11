"""
Feature Engineering Module
───────────────────────────
Computes predictive features for each game from raw box score data.

Features include:
  - ELO ratings (with home advantage, season reversion)
  - Rolling team stats (offense, defense, pace, efficiency)
  - Rest days & schedule density
  - Strength of schedule
  - Recent form (win streaks, last N performance)
  - Four Factors (eFG%, TOV%, OREB%, FT rate)
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from config import (
    ELO_HOME_ADVANTAGE,
    ELO_INITIAL,
    ELO_K_FACTOR,
    ELO_SEASON_REVERT,
    PROCESSED_DIR,
)


# ═══════════════════════════════════════════════════════════════════════
# ELO RATINGS
# ═══════════════════════════════════════════════════════════════════════

class EloTracker:
    """
    Maintains ELO ratings for all teams across seasons.
    Key design choices:
      - Home advantage baked into expected score calc
      - Margin of victory multiplier (rewards blowouts less at extremes)
      - Season-start reversion toward the mean
    """

    def __init__(self):
        self.ratings: dict[str, float] = {}
        self.history: list[dict] = []

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, ELO_INITIAL)

    def expected_score(self, home_elo: float, away_elo: float) -> float:
        """Expected win probability for home team (includes home advantage)."""
        diff = home_elo - away_elo + ELO_HOME_ADVANTAGE
        return 1 / (1 + 10 ** (-diff / 400))

    def mov_multiplier(self, margin: int, elo_diff: float) -> float:
        """
        Margin-of-victory multiplier.
        Dampens extreme blowouts and accounts for auto-correlation
        (good teams blow out bad teams → ELO already reflects this).
        """
        return np.log(abs(margin) + 1) * (2.2 / ((elo_diff * 0.001) + 2.2))

    def update(self, home_team: str, away_team: str, home_pts: int, away_pts: int):
        """Update ELO after a game. Returns pre-game ratings."""
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)

        # Pre-game expected scores
        exp_home = self.expected_score(home_elo, away_elo)
        exp_away = 1 - exp_home

        # Actual outcome
        if home_pts > away_pts:
            actual_home, actual_away = 1.0, 0.0
        elif home_pts < away_pts:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        # Margin-of-victory multiplier
        margin = home_pts - away_pts
        elo_diff = abs(home_elo - away_elo)
        mov_mult = self.mov_multiplier(margin, elo_diff)

        # Update ratings
        k = ELO_K_FACTOR * mov_mult
        new_home_elo = home_elo + k * (actual_home - exp_home)
        new_away_elo = away_elo + k * (actual_away - exp_away)

        pre_game = {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "elo_home_win_prob": exp_home,
        }

        self.ratings[home_team] = new_home_elo
        self.ratings[away_team] = new_away_elo

        return pre_game

    def season_reset(self):
        """Revert all ratings toward the mean at season start."""
        for team in self.ratings:
            self.ratings[team] = (
                self.ratings[team] * (1 - ELO_SEASON_REVERT)
                + ELO_INITIAL * ELO_SEASON_REVERT
            )


# ═══════════════════════════════════════════════════════════════════════
# ROLLING TEAM STATISTICS
# ═══════════════════════════════════════════════════════════════════════

def compute_rolling_stats(games: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Compute rolling averages for each team over the last `window` games.
    Returns a DataFrame indexed by (date, team) with rolling features.
    """
    # Build per-team game logs from the merged game data
    home_logs = games.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_pts": "pts_for", "away_pts": "pts_against",
        "home_fgm": "fgm", "home_fga": "fga",
        "home_fg3m": "fg3m", "home_fg3a": "fg3a",
        "home_ftm": "ftm", "home_fta": "fta",
        "home_oreb": "oreb", "home_dreb": "dreb",
        "home_reb": "reb", "home_ast": "ast",
        "home_stl": "stl", "home_blk": "blk",
        "home_tov": "tov",
    }).assign(is_home=1)

    away_logs = games.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_pts": "pts_for", "home_pts": "pts_against",
        "away_fgm": "fgm", "away_fga": "fga",
        "away_fg3m": "fg3m", "away_fg3a": "fg3a",
        "away_ftm": "ftm", "away_fta": "fta",
        "away_oreb": "oreb", "away_dreb": "dreb",
        "away_reb": "reb", "away_ast": "ast",
        "away_stl": "stl", "away_blk": "blk",
        "away_tov": "tov",
    }).assign(is_home=0)

    cols = ["game_id", "date", "season", "team", "opponent", "is_home",
            "pts_for", "pts_against", "fgm", "fga", "fg3m", "fg3a",
            "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov"]

    team_logs = pd.concat([home_logs[cols], away_logs[cols]], ignore_index=True)
    team_logs = team_logs.sort_values(["team", "date"]).reset_index(drop=True)

    # Compute derived stats
    team_logs["win"] = (team_logs["pts_for"] > team_logs["pts_against"]).astype(int)
    team_logs["margin"] = team_logs["pts_for"] - team_logs["pts_against"]
    team_logs["possessions"] = _estimate_possessions(team_logs)
    team_logs["off_rtg"] = 100 * team_logs["pts_for"] / team_logs["possessions"]
    team_logs["def_rtg"] = 100 * team_logs["pts_against"] / team_logs["possessions"]
    team_logs["net_rtg"] = team_logs["off_rtg"] - team_logs["def_rtg"]
    team_logs["pace"] = team_logs["possessions"]

    # Four Factors
    team_logs["efg_pct"] = (team_logs["fgm"] + 0.5 * team_logs["fg3m"]) / team_logs["fga"].clip(lower=1)
    team_logs["tov_pct"] = team_logs["tov"] / team_logs["possessions"]
    team_logs["oreb_pct"] = team_logs["oreb"] / (team_logs["oreb"] + team_logs["dreb"]).clip(lower=1)
    team_logs["ft_rate"] = team_logs["ftm"] / team_logs["fga"].clip(lower=1)

    # Rolling averages per team (shift by 1 so we don't leak current game)
    roll_cols = [
        "pts_for", "pts_against", "win", "margin",
        "off_rtg", "def_rtg", "net_rtg", "pace",
        "efg_pct", "tov_pct", "oreb_pct", "ft_rate",
        "fg3m", "fg3a", "ast", "stl", "blk", "tov", "reb",
    ]

    rolled = (
        team_logs.groupby("team")[roll_cols]
        .apply(lambda g: g.shift(1).rolling(window, min_periods=3).mean())
        .reset_index(level=0, drop=True)
    )
    rolled.columns = [f"roll{window}_{c}" for c in rolled.columns]

    team_logs = pd.concat([team_logs, rolled], axis=1)

    return team_logs


def _estimate_possessions(df: pd.DataFrame) -> pd.Series:
    """
    Estimate possessions using the standard formula:
    Poss ≈ FGA - OREB + TOV + 0.44 * FTA
    """
    return df["fga"] - df["oreb"] + df["tov"] + 0.44 * df["fta"]


# ═══════════════════════════════════════════════════════════════════════
# REST & SCHEDULE FEATURES
# ═══════════════════════════════════════════════════════════════════════

def compute_rest_features(games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rest days and schedule density for each team in each game.
    """
    games = games.copy().sort_values("date")

    # Track each team's last game date
    last_game: dict[str, pd.Timestamp] = {}
    games_in_7d: dict[str, list] = {}  # Track recent game dates

    rest_home, rest_away = [], []
    density_home, density_away = [], []
    b2b_home, b2b_away = [], []

    for _, row in games.iterrows():
        date = pd.Timestamp(row["date"])
        ht, at = row["home_team"], row["away_team"]

        # Rest days
        h_rest = (date - last_game[ht]).days if ht in last_game else 3
        a_rest = (date - last_game[at]).days if at in last_game else 3
        rest_home.append(h_rest)
        rest_away.append(a_rest)

        # Back-to-back
        b2b_home.append(1 if h_rest == 1 else 0)
        b2b_away.append(1 if a_rest == 1 else 0)

        # Schedule density: games in last 7 days
        cutoff = date - timedelta(days=7)
        h_recent = [d for d in games_in_7d.get(ht, []) if d > cutoff]
        a_recent = [d for d in games_in_7d.get(at, []) if d > cutoff]
        density_home.append(len(h_recent))
        density_away.append(len(a_recent))

        # Update trackers
        last_game[ht] = date
        last_game[at] = date
        games_in_7d.setdefault(ht, []).append(date)
        games_in_7d.setdefault(at, []).append(date)

    games["home_rest_days"] = rest_home
    games["away_rest_days"] = rest_away
    games["home_b2b"] = b2b_home
    games["away_b2b"] = b2b_away
    games["home_games_7d"] = density_home
    games["away_games_7d"] = density_away
    games["rest_advantage"] = np.array(rest_home) - np.array(rest_away)

    return games


# ═══════════════════════════════════════════════════════════════════════
# WIN STREAK & RECENT FORM
# ═══════════════════════════════════════════════════════════════════════

def compute_streak_features(games: pd.DataFrame) -> pd.DataFrame:
    """Compute current win/loss streak for each team entering each game."""
    games = games.copy().sort_values("date")

    streaks: dict[str, int] = {}  # positive = win streak, negative = loss streak
    last_n_wins: dict[str, list] = {}  # track last 5 results

    home_streak, away_streak = [], []
    home_last5, away_last5 = [], []

    for _, row in games.iterrows():
        ht, at = row["home_team"], row["away_team"]
        hw = row["home_win"]

        # Record pre-game streaks
        home_streak.append(streaks.get(ht, 0))
        away_streak.append(streaks.get(at, 0))

        # Last 5 win pct
        h_hist = last_n_wins.get(ht, [])
        a_hist = last_n_wins.get(at, [])
        home_last5.append(np.mean(h_hist[-5:]) if len(h_hist) >= 3 else 0.5)
        away_last5.append(np.mean(a_hist[-5:]) if len(a_hist) >= 3 else 0.5)

        # Update streaks
        if hw == 1:
            streaks[ht] = max(1, streaks.get(ht, 0) + 1) if streaks.get(ht, 0) >= 0 else 1
            streaks[at] = min(-1, streaks.get(at, 0) - 1) if streaks.get(at, 0) <= 0 else -1
        else:
            streaks[ht] = min(-1, streaks.get(ht, 0) - 1) if streaks.get(ht, 0) <= 0 else -1
            streaks[at] = max(1, streaks.get(at, 0) + 1) if streaks.get(at, 0) >= 0 else 1

        last_n_wins.setdefault(ht, []).append(hw)
        last_n_wins.setdefault(at, []).append(1 - hw)

    games["home_streak"] = home_streak
    games["away_streak"] = away_streak
    games["home_last5_winpct"] = home_last5
    games["away_last5_winpct"] = away_last5

    return games


# ═══════════════════════════════════════════════════════════════════════
# FULL FEATURE PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def build_features(games: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    Takes raw game data and returns a model-ready DataFrame.
    """
    print("\n[Feature Engineering]")
    games = games.copy().sort_values("date").reset_index(drop=True)

    # 1. ELO ratings
    print("  Computing ELO ratings...")
    elo = EloTracker()
    elo_features = []
    current_season = None

    for _, row in games.iterrows():
        season = row["season"]
        if current_season is not None and season != current_season:
            elo.season_reset()
        current_season = season

        pre_game = elo.update(
            row["home_team"], row["away_team"],
            row["home_pts"], row["away_pts"],
        )
        elo_features.append(pre_game)

    elo_df = pd.DataFrame(elo_features)
    games = pd.concat([games, elo_df], axis=1)

    # 2. Rolling team stats
    print("  Computing rolling team stats...")
    team_logs = compute_rolling_stats(games, window=window)

    # Merge rolling stats back to game level
    roll_cols = [c for c in team_logs.columns if c.startswith(f"roll{window}_")]
    home_roll = (
        team_logs[team_logs["is_home"] == 1][["game_id"] + roll_cols]
        .rename(columns={c: f"home_{c}" for c in roll_cols})
    )
    away_roll = (
        team_logs[team_logs["is_home"] == 0][["game_id"] + roll_cols]
        .rename(columns={c: f"away_{c}" for c in roll_cols})
    )
    games = games.merge(home_roll, on="game_id", how="left")
    games = games.merge(away_roll, on="game_id", how="left")

    # 3. Rest & schedule features
    print("  Computing rest/schedule features...")
    games = compute_rest_features(games)

    # 4. Streak & form features
    print("  Computing streak/form features...")
    games = compute_streak_features(games)

    # 5. Derived differential features (model loves these)
    print("  Computing differential features...")
    games = _add_differential_features(games, window)

    # Drop rows with NaN rolling stats (first few games of each season)
    pre_count = len(games)
    feature_cols = [c for c in games.columns if "roll" in c]
    games = games.dropna(subset=feature_cols).reset_index(drop=True)
    print(f"  Dropped {pre_count - len(games)} rows with insufficient history")

    # Save
    games.to_csv(PROCESSED_DIR / "features.csv", index=False)
    print(f"  Saved {len(games)} feature rows to {PROCESSED_DIR / 'features.csv'}")

    return games


def _add_differential_features(games: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add home-minus-away differentials for key rolling stats."""
    prefix = f"roll{window}_"
    diff_pairs = [
        "pts_for", "pts_against", "win", "margin",
        "off_rtg", "def_rtg", "net_rtg", "pace",
        "efg_pct", "tov_pct", "oreb_pct", "ft_rate",
    ]
    for stat in diff_pairs:
        home_col = f"home_{prefix}{stat}"
        away_col = f"away_{prefix}{stat}"
        if home_col in games.columns and away_col in games.columns:
            games[f"diff_{stat}"] = games[home_col] - games[away_col]

    return games


def get_feature_columns(games: pd.DataFrame) -> list[str]:
    """Return the list of feature columns to feed into the model."""
    feature_cols = []

    # ELO features
    feature_cols += ["home_elo", "away_elo", "elo_diff", "elo_home_win_prob"]

    # Rolling stats (home & away)
    feature_cols += [c for c in games.columns if c.startswith("home_roll") or c.startswith("away_roll")]

    # Differentials
    feature_cols += [c for c in games.columns if c.startswith("diff_")]

    # Rest & schedule
    feature_cols += [
        "home_rest_days", "away_rest_days",
        "home_b2b", "away_b2b",
        "home_games_7d", "away_games_7d",
        "rest_advantage",
    ]

    # Streaks & form
    feature_cols += [
        "home_streak", "away_streak",
        "home_last5_winpct", "away_last5_winpct",
    ]

    # Only return columns that actually exist
    return [c for c in feature_cols if c in games.columns]


if __name__ == "__main__":
    # Quick test: load processed games and build features
    games = pd.read_csv(PROCESSED_DIR / "all_games.csv", parse_dates=["date"])
    featured = build_features(games)
    print(f"\nFeature columns ({len(get_feature_columns(featured))}):")
    for col in get_feature_columns(featured):
        print(f"  {col}")

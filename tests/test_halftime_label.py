import pandas as pd
import pytest
from live_labeler import resolve_halftime_leaders, build_labeled_training_set


def _game_history():
    return pd.DataFrame([
        {"game_key": "A", "captured_at": "2026-04-18T00:10:00Z", "period": 1, "home_score": 20, "away_score": 22},
        {"game_key": "A", "captured_at": "2026-04-18T00:40:00Z", "period": 2, "home_score": 55, "away_score": 50},
        {"game_key": "A", "captured_at": "2026-04-18T01:30:00Z", "period": 4, "home_score": 99, "away_score": 90},
        {"game_key": "B", "captured_at": "2026-04-18T00:42:00Z", "period": 2, "home_score": 48, "away_score": 60},
        {"game_key": "C", "captured_at": "2026-04-18T00:43:00Z", "period": 2, "home_score": 51, "away_score": 51},
    ])


def test_resolves_end_of_q2_leader_per_game():
    out = resolve_halftime_leaders(_game_history()).set_index("game_key")
    assert out.loc["A", "label_halftime_home_lead"] == 1
    assert out.loc["A", "label_halftime_margin_home"] == 5
    assert out.loc["B", "label_halftime_home_lead"] == 0


def test_tie_at_halftime_is_flagged_and_label_is_na():
    out = resolve_halftime_leaders(_game_history()).set_index("game_key")
    assert out.loc["C", "label_halftime_is_tie"] == 1
    assert pd.isna(out.loc["C", "label_halftime_home_lead"])


def test_game_without_q2_rows_is_dropped():
    hist = pd.DataFrame([
        {"game_key": "D", "captured_at": "2026-04-18T00:05:00Z", "period": 1, "home_score": 10, "away_score": 8},
    ])
    out = resolve_halftime_leaders(hist)
    assert "D" not in set(out["game_key"])


def test_labeled_set_carries_halftime_label():
    """
    build_labeled_training_set must join halftime labels into the output frame.

    Uses real 2026-04-17 data slices (nrows=200) because constructing fully
    valid synthetic frames for the complex forward-market / close-proxy merges
    in build_labeled_training_set is brittle. The 2026-04-17 game file
    contains period-2 rows for 2026-04-17_GSW_PHX, guaranteeing at least one
    resolved halftime label.
    """
    BASE = "data/live"
    features = pd.read_csv(f"{BASE}/features/live_features_2026-04-17.csv", nrows=200)
    games = pd.read_csv(f"{BASE}/games/game_states_2026-04-17.csv")
    markets = pd.read_csv(f"{BASE}/markets/market_snapshots_2026-04-17.csv", nrows=200)

    result = build_labeled_training_set(
        feature_history=features,
        game_history=games,
        market_history=markets,
    )

    assert not result.empty, "build_labeled_training_set returned empty DataFrame"
    assert "label_halftime_home_lead" in result.columns, (
        "label_halftime_home_lead column missing from labeled training set"
    )
    # At least one row should have a non-null halftime label (GSW_PHX had period-2 rows)
    assert result["label_halftime_home_lead"].notna().any(), (
        "label_halftime_home_lead is null for every row — halftime merge did not attach any labels"
    )

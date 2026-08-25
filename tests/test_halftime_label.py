import pandas as pd
import pytest
from live_labeler import resolve_halftime_leaders


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

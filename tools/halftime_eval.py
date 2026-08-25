"""Honest evaluation of the live halftime-leader predictor: dumb baselines +
a lead-time curve on a temporal hold-out. Research/prediction only."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def current_leader_prob(margin: float, k: float = 0.15) -> float:
    """Probabilistic 'current leader stays' baseline: logistic in current margin."""
    return 1.0 / (1.0 + math.exp(-k * float(margin)))


def elo_prior_prob(pregame_home_win_prob: float) -> float:
    """Pre-game ELO prior held constant through the half (clipped to [0,1])."""
    return float(min(1.0, max(0.0, pregame_home_win_prob)))


def diffusion_prob(margin: float, seconds_left_in_half: float,
                   points_std_per_sec: float = 0.11) -> float:
    """Brownian-bridge baseline: P(home leads at half) = Phi(margin / sigma),
    sigma grows with remaining time."""
    remaining = max(0.0, float(seconds_left_in_half))
    sigma = points_std_per_sec * math.sqrt(remaining) if remaining > 0 else 1e-9
    if sigma <= 1e-9:
        return 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
    z = float(margin) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def lead_time_curve(df: pd.DataFrame, bin_minutes: int = 2) -> pd.DataFrame:
    """Per game-time bin, compare model vs baseline Brier / accuracy.

    df needs columns: minutes_elapsed, y_true, model_prob, baseline_prob.
    Returns one row per bin with n, model_brier, baseline_brier,
    model_acc, baseline_acc.
    """
    d = df.dropna(subset=["y_true", "model_prob", "baseline_prob"]).copy()
    d["minute_bin"] = (d["minutes_elapsed"] // bin_minutes) * bin_minutes
    rows = []
    for b, g in d.groupby("minute_bin"):
        yt = g["y_true"].to_numpy(dtype=float)
        rows.append({
            "minute_bin": int(b),
            "n": int(len(g)),
            "model_brier": float(np.mean((g["model_prob"].to_numpy() - yt) ** 2)),
            "baseline_brier": float(np.mean((g["baseline_prob"].to_numpy() - yt) ** 2)),
            "model_acc": float(np.mean((g["model_prob"].to_numpy() >= 0.5) == (yt == 1))),
            "baseline_acc": float(np.mean((g["baseline_prob"].to_numpy() >= 0.5) == (yt == 1))),
        })
    return pd.DataFrame(rows).sort_values("minute_bin").reset_index(drop=True)

"""Honest evaluation of the live halftime-leader predictor: dumb baselines +
a lead-time curve on a temporal hold-out. Research/prediction only."""
from __future__ import annotations

import math

import numpy as np


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

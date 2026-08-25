import numpy as np
import pandas as pd
import pytest
from tools.halftime_eval import (
    current_leader_prob,
    elo_prior_prob,
    diffusion_prob,
    lead_time_curve,
)


def test_current_leader_prob_monotone_in_margin():
    assert current_leader_prob(10.0) > current_leader_prob(0.0) > current_leader_prob(-10.0)
    assert current_leader_prob(0.0) == pytest.approx(0.5)
    assert 0.0 < current_leader_prob(3.0) < 1.0


def test_elo_prior_passthrough():
    assert elo_prior_prob(0.62) == pytest.approx(0.62)
    assert elo_prior_prob(1.5) == pytest.approx(1.0)
    assert elo_prior_prob(-0.2) == pytest.approx(0.0)


def test_diffusion_prob_tightens_as_time_runs_out():
    near_end = diffusion_prob(margin=4.0, seconds_left_in_half=5.0)
    early = diffusion_prob(margin=4.0, seconds_left_in_half=1200.0)
    assert near_end > early
    assert diffusion_prob(margin=0.0, seconds_left_in_half=600.0) == pytest.approx(0.5)


def test_lead_time_curve_reports_per_bin_brier_for_model_and_baseline():
    df = pd.DataFrame({
        "minutes_elapsed": [1, 1, 13, 13],          # two H1 bins
        "y_true":          [1, 0, 1, 1],
        "model_prob":      [0.6, 0.4, 0.9, 0.8],
        "baseline_prob":   [0.5, 0.5, 0.55, 0.55],
    })
    out = lead_time_curve(df, bin_minutes=6).set_index("minute_bin")
    assert set(["minute_bin", "n", "model_brier", "baseline_brier"]).issubset(out.reset_index().columns)
    assert out.loc[12, "model_brier"] < out.loc[12, "baseline_brier"]

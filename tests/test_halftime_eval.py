import numpy as np
import pytest
from tools.halftime_eval import (
    current_leader_prob,
    elo_prior_prob,
    diffusion_prob,
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

"""Move-size estimator: Phase A pooled, Phase B per-decile, plus the A→B
gate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper_trader.move_size_estimator import (
    Store,
    load_pooled_means,
    expected_moves,
    per_decile_means,
    bootstrap_ci_half_widths,
    ab_gate_passed,
)


@pytest.fixture
def pooled_json(tmp_path: Path) -> Path:
    p = tmp_path / "pooled.json"
    p.write_text(json.dumps({
        "E_delta_given_rises": 0.025,
        "E_delta_given_doesnt": -0.005,
        "var_delta_pooled": 0.0009,
        "rises_count": 1000,
        "doesnt_count": 9000,
    }))
    return p


def test_load_pooled_means_exposes_fields(pooled_json):
    store = load_pooled_means(pooled_json)
    assert isinstance(store, Store)
    assert store.E_rises_pooled == pytest.approx(0.025)
    assert store.E_doesnt_pooled == pytest.approx(-0.005)
    assert store.var_pooled == pytest.approx(0.0009)


def test_expected_moves_phase_a_uses_pooled(pooled_json):
    store = load_pooled_means(pooled_json)
    e_up, e_down, var = expected_moves(p=0.30, mode="A", store=store)
    assert e_up == pytest.approx(0.025)
    assert e_down == pytest.approx(-0.005)
    assert var == pytest.approx(0.0009)


def test_expected_moves_phase_b_uses_decile(pooled_json):
    store = load_pooled_means(pooled_json)
    # Inject a per-decile table covering p=0.30 (decile 3).
    store.decile_table = pd.DataFrame({
        "p_decile": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "E_rises": [0.005, 0.010, 0.015, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090],
        "E_doesnt": [-0.002, -0.003, -0.004, -0.006, -0.008, -0.010, -0.012, -0.014, -0.016, -0.018],
        "var": [0.0001] * 10,
        "rises_n": [50] * 10,
        "doesnt_n": [450] * 10,
    })
    e_up, e_down, var = expected_moves(p=0.30, mode="B", store=store)
    assert e_up == pytest.approx(0.030)
    assert e_down == pytest.approx(-0.006)


def test_phase_b_falls_back_to_pooled_when_decile_missing(pooled_json):
    store = load_pooled_means(pooled_json)
    store.decile_table = None
    e_up, e_down, var = expected_moves(p=0.30, mode="B", store=store)
    assert e_up == pytest.approx(0.025)


def test_load_pooled_means_populates_decile_table_when_present(tmp_path: Path):
    p = tmp_path / "pooled_with_decile.json"
    p.write_text(json.dumps({
        "E_delta_given_rises": 0.025,
        "E_delta_given_doesnt": -0.005,
        "var_delta_pooled": 0.0009,
        "decile_table": [
            {"p_decile": d, "E_rises": 0.01 * (d + 1), "E_doesnt": -0.001 * (d + 1),
             "var": 0.0001, "rises_n": 50, "doesnt_n": 450}
            for d in range(10)
        ],
    }))
    store = load_pooled_means(p)
    assert store.decile_table is not None
    assert len(store.decile_table) == 10
    e_up, e_down, _ = expected_moves(p=0.65, mode="B", store=store)
    # decile 6 → E_rises=0.07, E_doesnt=-0.007
    assert e_up == pytest.approx(0.07)
    assert e_down == pytest.approx(-0.007)


def test_per_decile_means_groups_correctly():
    df = pd.DataFrame({
        "p_calibrated": np.linspace(0.0, 0.99, 100),
        "label_home_up_5m": ([1] * 10 + [0] * 90),
        "yes_mid_delta_realized": np.concatenate([np.full(10, 0.05), np.full(90, -0.01)]),
    })
    table = per_decile_means(df, delta_col="yes_mid_delta_realized")
    assert set(table.columns) >= {"p_decile", "E_rises", "E_doesnt", "var", "rises_n", "doesnt_n"}
    # Decile 0 has all 10 rises in this synthetic setup.
    decile_0 = table[table["p_decile"] == 0].iloc[0]
    assert decile_0["rises_n"] == 10
    assert decile_0["E_rises"] == pytest.approx(0.05)


def test_bootstrap_ci_half_width_shrinks_with_n():
    rng = np.random.default_rng(0)
    small = rng.normal(0.025, 0.03, 50)
    large = rng.normal(0.025, 0.03, 5000)
    hw_small = bootstrap_ci_half_widths(small, n_iter=500, alpha=0.05, seed=1)
    hw_large = bootstrap_ci_half_widths(large, n_iter=500, alpha=0.05, seed=1)
    assert hw_large < hw_small


def test_ab_gate_passes_only_when_all_deciles_tight():
    # All deciles ±0.005 → passes
    rng = np.random.default_rng(0)
    rows = []
    for d in range(10):
        for _ in range(2000):
            rows.append({
                "p_decile": d,
                "is_rise": True,
                "delta": rng.normal(0.025, 0.001),  # very tight
            })
    df = pd.DataFrame(rows)
    assert ab_gate_passed(df, ci_threshold=0.01, n_iter=200, seed=1) is True

    # One decile loose → fails
    df_loose = df.copy()
    mask = df_loose["p_decile"] == 5
    df_loose.loc[mask, "delta"] = rng.normal(0.025, 0.5, mask.sum())  # huge spread
    assert ab_gate_passed(df_loose, ci_threshold=0.01, n_iter=200, seed=1) is False

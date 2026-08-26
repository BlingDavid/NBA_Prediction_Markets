"""build_pooled_means writes a pooled_means.json that also carries a
decile_table — so the trader can use Phase B (per-decile) from boot
without waiting for live closed trades."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_scored(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n_per_decile = 200
    rows = []
    for d in range(10):
        p_low, p_high = d / 10, (d + 1) / 10
        for _ in range(n_per_decile):
            p = float(rng.uniform(p_low, p_high))
            rises = rng.uniform() < (0.05 + 0.05 * d)  # rate climbs with decile
            delta = float(rng.normal(0.01 * (d + 1), 0.005)) if rises else float(rng.normal(-0.001 * (d + 1), 0.005))
            rows.append({
                "bet_side": "home",
                "label_home_up_5m": int(rises),
                "label_yes_mid_move_5m": delta,
                "pred_bootstrap_home_up_5m": p,
            })
    out = tmp_path / "scored.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def test_build_pooled_means_emits_decile_table(synthetic_scored, tmp_path: Path, monkeypatch):
    out_path = tmp_path / "pooled.json"
    monkeypatch.setattr(sys, "argv", [
        "build_pooled_means",
        "--scored-rows", str(synthetic_scored),
        "--out", str(out_path),
    ])
    from tools import build_pooled_means
    rc = build_pooled_means.main()
    assert rc == 0
    payload = json.loads(out_path.read_text())
    assert "decile_table" in payload
    table = payload["decile_table"]
    assert isinstance(table, list)
    assert len(table) == 10
    # Each row carries the contract per_decile_means uses.
    required = {"p_decile", "E_rises", "E_doesnt", "var", "rises_n", "doesnt_n"}
    for row in table:
        assert required.issubset(row.keys())
    # Sanity: deciles are 0..9, no duplicates.
    assert sorted(r["p_decile"] for r in table) == list(range(10))

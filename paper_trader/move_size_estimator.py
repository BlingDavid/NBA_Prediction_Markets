"""Move-size estimator: pooled (Phase A) and per-decile (Phase B) conditional
means of yes_mid Δ over the 5-minute window.

The store object holds both. Phase A is loaded once at engine boot from JSON;
Phase B is recomputed at the end of each tick from realized deltas in the
closed-trades table once trades_closed >= 200 AND every decile passes the CI
gate (see ab_gate_passed).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Store:
    E_rises_pooled: float
    E_doesnt_pooled: float
    var_pooled: float
    decile_table: Optional[pd.DataFrame] = field(default=None)


def load_pooled_means(path: Path) -> Store:
    payload = json.loads(Path(path).read_text())
    decile_rows = payload.get("decile_table")
    decile_table = pd.DataFrame(decile_rows) if decile_rows else None
    return Store(
        E_rises_pooled=float(payload["E_delta_given_rises"]),
        E_doesnt_pooled=float(payload["E_delta_given_doesnt"]),
        var_pooled=float(payload["var_delta_pooled"]),
        decile_table=decile_table,
    )


def expected_moves(p: float, mode: str, store: Store) -> tuple[float, float, float]:
    """Returns (E[Δ|rises], E[Δ|doesn't], var(Δ)) appropriate for the mode.

    Phase B falls back to pooled if no decile_table is loaded yet, or if the
    target decile is empty in the table.
    """
    if mode == "B" and store.decile_table is not None and not store.decile_table.empty:
        decile = min(int(p * 10), 9)
        row = store.decile_table[store.decile_table["p_decile"] == decile]
        if not row.empty and row.iloc[0].get("rises_n", 0) > 0:
            r = row.iloc[0]
            return float(r["E_rises"]), float(r["E_doesnt"]), float(r["var"])
    return store.E_rises_pooled, store.E_doesnt_pooled, store.var_pooled


def per_decile_means(closed: pd.DataFrame, delta_col: str = "yes_mid_delta_realized") -> pd.DataFrame:
    """Build a 10-row table indexed by p_calibrated decile.

    `closed` must have columns: p_calibrated, label_home_up_5m, <delta_col>.
    """
    df = closed.copy()
    df["p_decile"] = (df["p_calibrated"].clip(0, 0.999) * 10).astype(int)
    rows = []
    for d in range(10):
        sub = df[df["p_decile"] == d]
        rises = sub[sub["label_home_up_5m"] == 1][delta_col]
        doesnt = sub[sub["label_home_up_5m"] == 0][delta_col]
        rows.append({
            "p_decile": d,
            "E_rises": float(rises.mean()) if len(rises) else 0.0,
            "E_doesnt": float(doesnt.mean()) if len(doesnt) else 0.0,
            "var": float(sub[delta_col].var(ddof=0)) if len(sub) else 0.0,
            "rises_n": int(len(rises)),
            "doesnt_n": int(len(doesnt)),
        })
    return pd.DataFrame(rows)


def bootstrap_ci_half_widths(values: np.ndarray, n_iter: int = 1000, alpha: float = 0.05, seed: int = 0) -> float:
    """Half-width of the symmetric (1-alpha) percentile bootstrap CI of the mean."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("inf")
    rng = np.random.default_rng(seed)
    means = np.empty(n_iter)
    n = len(values)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        means[i] = values[idx].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float((hi - lo) / 2.0)


def ab_gate_passed(realized: pd.DataFrame, ci_threshold: float = 0.01, n_iter: int = 1000, seed: int = 0) -> bool:
    """For every populated decile in `realized`, the bootstrapped 95% CI
    half-width on E[Δ|rises] must be ≤ ci_threshold (in price units, default 1¢).

    `realized` must have columns: p_decile, is_rise (bool), delta (float).
    Empty deciles disqualify the gate (we want coverage)."""
    deciles = realized["p_decile"].unique()
    if len(deciles) < 10:
        return False
    for d in range(10):
        rises = realized[(realized["p_decile"] == d) & (realized["is_rise"])]["delta"].to_numpy()
        if len(rises) == 0:
            return False
        if bootstrap_ci_half_widths(rises, n_iter=n_iter, seed=seed) > ci_threshold:
            return False
    return True

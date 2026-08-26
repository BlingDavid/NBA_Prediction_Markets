"""Post-hoc isotonic recalibration of the live home-up-5m model.

Fits an IsotonicRegression mapping raw predicted-prob → calibrated-prob
using the OOF predictions (already leakage-safe).  For honest metric
reporting it does a ticker-GROUPED train/eval split first, then fits the
final isotonic on ALL OOF rows and saves a non-destructive candidate
bundle under models/candidates/.

Usage:
    source .venv/bin/activate
    export DYLD_FALLBACK_LIBRARY_PATH="...:opt/homebrew/opt/libomp/lib"
    python tools/recalibrate_live_model.py [--output-dir outputs]

Hard constraints:
    - Does NOT modify models/live_home_up_5m_bootstrap.pkl.
    - Writes ONLY to models/candidates/ and optionally outputs/.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

# ── Repo root ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OOF_CSV = _ROOT / "outputs" / "live_home_up_5m_bootstrap_oof_predictions.csv"
BASE_MODEL_PKL = _ROOT / "models" / "live_home_up_5m_bootstrap.pkl"
CANDIDATE_PKL = _ROOT / "models" / "candidates" / "live_home_up_5m_bootstrap_isotonic.pkl"


# ═══════════════════════════════════════════════════════════════════════════
# Pure helpers (unit-tested in tests/test_recalibrate_live_model.py)
# ═══════════════════════════════════════════════════════════════════════════

def apply_isotonic(
    iso: IsotonicRegression,
    y_prob: np.ndarray,
) -> np.ndarray:
    """Apply a fitted IsotonicRegression to a probability array.

    Clips output to [0, 1] to guard against extrapolation artefacts.
    Returns float32 to stay memory-consistent with OOF conventions.
    """
    calibrated = iso.predict(y_prob.astype(np.float64))
    return np.clip(calibrated, 0.0, 1.0).astype(np.float32)


def grouped_split(
    groups: np.ndarray,
    fit_frac: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices into two halves so no group appears in both halves.

    Parameters
    ----------
    groups:
        1-D array of group labels, one per row.
    fit_frac:
        Fraction of groups to put in the fit half (default 0.5).
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    fit_idx, eval_idx : np.ndarray
        Boolean masks (same length as `groups`).
    """
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    n_fit = max(1, int(len(unique_groups) * fit_frac))
    fit_groups = set(unique_groups[:n_fit])
    fit_mask = np.isin(groups, list(fit_groups))
    eval_mask = ~fit_mask
    return fit_mask, eval_mask


def reliability_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Reliability (calibration) table with columns bin_lo, bin_hi, n, avg_pred, emp_rate, gap."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        avg_pred = float(y_prob[mask].mean())
        emp_rate = float(y_true[mask].mean())
        rows.append({
            "bin_lo": round(lo, 3),
            "bin_hi": round(hi, 3),
            "n": n,
            "avg_pred": avg_pred,
            "emp_rate": emp_rate,
            "gap": emp_rate - avg_pred,
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# EV helpers (mirrored from tools/live_model_eval.py for standalone use)
# ═══════════════════════════════════════════════════════════════════════════

def kalshi_winner_fee(p: float) -> float:
    """ceil(0.07 * p * (1-p) * 100) / 100.  Returns 0 for p<=0 or p>=1."""
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return math.ceil(0.07 * p * (1.0 - p) * 100.0) / 100.0


def ev_per_contract(p: float, e_up: float, e_down: float, half_spread: float) -> float:
    """EV = p*E_up + (1-p)*E_down - half_spread - p*winner_fee(p)."""
    fee = kalshi_winner_fee(p)
    return p * e_up + (1.0 - p) * e_down - half_spread - p * fee


def ev_positive_count(
    y_prob: np.ndarray,
    e_up: float,
    e_down: float,
    half_spread: float = 0.01,
) -> int:
    """Count rows where ev_per_contract > 0 for the given probability array."""
    evs = np.array([ev_per_contract(float(p), e_up, e_down, half_spread) for p in y_prob])
    return int((evs > 0.0).sum())


# ═══════════════════════════════════════════════════════════════════════════
# OOF loader
# ═══════════════════════════════════════════════════════════════════════════

def load_oof(path: Path) -> pd.DataFrame:
    """Load group + label + both prediction columns; use memory-efficient dtypes."""
    usecols = ["group_id", "label_home_up_5m", "oof_prediction_raw", "oof_prediction"]
    dtype = {
        "label_home_up_5m": "int8",
        "oof_prediction_raw": "float32",
        "oof_prediction": "float32",
    }
    return pd.read_csv(path, usecols=usecols, dtype=dtype)


# ═══════════════════════════════════════════════════════════════════════════
# Main calibration pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_recalibration(output_dir: Path | None = None) -> dict:
    """Full recalibration pipeline.  Returns a metrics dict."""
    if output_dir is None:
        output_dir = _ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    CANDIDATE_PKL.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  NBA Live Model — Isotonic Recalibration")
    print("=" * 65)

    # ── 1. Load OOF ─────────────────────────────────────────────────────
    print(f"\n[1] Loading OOF predictions: {OOF_CSV}")
    df = load_oof(OOF_CSV)
    n_total = len(df)
    print(f"    {n_total:,} rows | {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    groups = df["group_id"].to_numpy()
    y_true = df["label_home_up_5m"].to_numpy(dtype=np.int8)
    # Use the alpha=0.95 shrinkage-calibrated prediction (oof_prediction) as input.
    # oof_prediction_raw is the raw LR output for reference.
    y_pred = df["oof_prediction"].to_numpy(dtype=np.float32)
    base_rate = float(y_true.mean())
    n_groups = len(np.unique(groups))
    print(f"    Base rate: {base_rate:.4f} | Groups (tickers): {n_groups}")

    # ── 2. Grouped split ─────────────────────────────────────────────────
    print("\n[2] Ticker-grouped 50/50 split (no ticker appears in both halves)...")
    fit_mask, eval_mask = grouped_split(groups, fit_frac=0.5, seed=42)
    print(f"    Fit half:  {fit_mask.sum():,} rows ({np.unique(groups[fit_mask]).size} groups)")
    print(f"    Eval half: {eval_mask.sum():,} rows ({np.unique(groups[eval_mask]).size} groups)")

    y_true_fit  = y_true[fit_mask]
    y_pred_fit  = y_pred[fit_mask]
    y_true_eval = y_true[eval_mask]
    y_pred_eval = y_pred[eval_mask]

    # ── 3. Fit isotonic on fit half ──────────────────────────────────────
    print("\n[3] Fitting IsotonicRegression on fit half...")
    iso_eval = IsotonicRegression(out_of_bounds="clip")
    iso_eval.fit(y_pred_fit.astype(np.float64), y_true_fit.astype(np.float64))

    # ── 4. Eval on held-out half — raw vs recalibrated ──────────────────
    print("\n[4] Held-out eval metrics (raw vs recalibrated)...")

    y_cal_eval = apply_isotonic(iso_eval, y_pred_eval)

    brier_raw = float(brier_score_loss(y_true_eval, y_pred_eval))
    brier_cal = float(brier_score_loss(y_true_eval, y_cal_eval))
    ll_raw    = float(log_loss(y_true_eval, y_pred_eval))
    ll_cal    = float(log_loss(y_true_eval, y_cal_eval))

    rel_raw = reliability_table(y_true_eval, y_pred_eval, n_bins=10)
    rel_cal = reliability_table(y_true_eval, y_cal_eval,  n_bins=10)

    print(f"\n    {'Metric':<18} {'Raw':>10} {'Recalibrated':>14} {'Delta':>8}")
    print("    " + "-" * 54)
    print(f"    {'Brier score':<18} {brier_raw:>10.6f} {brier_cal:>14.6f} {brier_cal-brier_raw:>+8.6f}")
    print(f"    {'Log-loss':<18} {ll_raw:>10.6f} {ll_cal:>14.6f} {ll_cal-ll_raw:>+8.6f}")

    # Reliability above p=0.20 (where overconfidence lives)
    print("\n    Reliability table (RAW) — eval half:")
    print(rel_raw.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n    Reliability table (RECALIBRATED) — eval half:")
    print(rel_cal.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Reliability gap above p=0.20
    high_raw = rel_raw[rel_raw["bin_lo"] >= 0.20]["gap"].abs()
    high_cal = rel_cal[rel_cal["bin_lo"] >= 0.20]["gap"].abs()
    mean_gap_raw = float(high_raw.mean()) if len(high_raw) > 0 else float("nan")
    mean_gap_cal = float(high_cal.mean()) if len(high_cal) > 0 else float("nan")
    print(f"\n    Mean |gap| for bins p >= 0.20:")
    print(f"      Raw:           {mean_gap_raw:.4f}")
    print(f"      Recalibrated:  {mean_gap_cal:.4f}")

    # ── 5. EV-positive row counts — raw vs recalibrated (ALL OOF) ───────
    print("\n[5] EV-positive row count analysis (full OOF, half_spread=0.01)...")

    # Load pooled means if available, else use fallback
    pooled_json = _ROOT / "outputs" / "paper_trades" / "pooled_means.json"
    if pooled_json.exists():
        pm = json.loads(pooled_json.read_text())
        e_up   = float(pm["E_delta_given_rises"])
        e_down = float(pm["E_delta_given_doesnt"])
        print(f"    Pooled means loaded: E_up={e_up:.5f}, E_down={e_down:.5f}")
    else:
        e_up, e_down = 0.024, -0.005
        print(f"    Pooled means fallback: E_up={e_up:.5f}, E_down={e_down:.5f}")

    # Fit FINAL isotonic on ALL OOF rows for the candidate artifact
    print("\n[6] Fitting FINAL isotonic on ALL OOF rows for candidate bundle...")
    iso_final = IsotonicRegression(out_of_bounds="clip")
    iso_final.fit(y_pred.astype(np.float64), y_true.astype(np.float64))
    y_cal_all = apply_isotonic(iso_final, y_pred)

    n_ev_raw = ev_positive_count(y_pred,    e_up, e_down, half_spread=0.01)
    n_ev_cal = ev_positive_count(y_cal_all, e_up, e_down, half_spread=0.01)

    print(f"\n    EV-positive rows (half_spread=0.01):")
    print(f"      Raw:           {n_ev_raw:,} ({n_ev_raw/n_total:.2%})")
    print(f"      Recalibrated:  {n_ev_cal:,} ({n_ev_cal/n_total:.2%})")
    print(f"      Delta:         {n_ev_cal - n_ev_raw:+,}")

    # Also report at other spreads for completeness
    for hs in [0.0, 0.005, 0.02]:
        nr = ev_positive_count(y_pred,    e_up, e_down, half_spread=hs)
        nc = ev_positive_count(y_cal_all, e_up, e_down, half_spread=hs)
        print(f"      half_spread={hs:.3f}:  raw={nr:,}  cal={nc:,}  delta={nc-nr:+,}")

    # ── 7. Save candidate bundle ─────────────────────────────────────────
    print(f"\n[7] Saving candidate bundle → {CANDIDATE_PKL}")

    # Load original bundle to carry the model reference and key structure
    with open(BASE_MODEL_PKL, "rb") as fh:
        base_bundle = pickle.load(fh)

    candidate = {
        # Reference to original model (same object; not a copy — same memory)
        "model":            base_bundle["model"],
        "feature_cols":     base_bundle["feature_cols"],
        "target":           base_bundle["target"],
        "trained_at":       base_bundle["trained_at"],
        "metadata":         base_bundle["metadata"],
        "calibration":      base_bundle["calibration"],
        "oof_metrics_raw":  base_bundle["oof_metrics_raw"],
        "oof_metrics":      base_bundle["oof_metrics"],
        "baseline_metrics": base_bundle["baseline_metrics"],
        "lift_table":       base_bundle["lift_table"],
        "fold_reports":     base_bundle.get("fold_reports"),
        "top_coefficients": base_bundle.get("top_coefficients"),
        # ── New keys added by this script ──
        "sklearn_version":       sklearn.__version__,
        "isotonic_calibrator":   iso_final,          # fitted IsotonicRegression
        "isotonic_input_col":    "oof_prediction",   # which col was the X for fitting
        "isotonic_recal_report": {
            "n_rows_total":        n_total,
            "n_groups_total":      n_groups,
            "fit_n":               int(fit_mask.sum()),
            "eval_n":              int(eval_mask.sum()),
            "brier_raw_eval":      brier_raw,
            "brier_cal_eval":      brier_cal,
            "logloss_raw_eval":    ll_raw,
            "logloss_cal_eval":    ll_cal,
            "mean_gap_above_p20_raw": mean_gap_raw,
            "mean_gap_above_p20_cal": mean_gap_cal,
            "ev_positive_raw":     n_ev_raw,
            "ev_positive_cal":     n_ev_cal,
            "e_up":                e_up,
            "e_down":              e_down,
            "half_spread":         0.01,
        },
    }

    with open(CANDIDATE_PKL, "wb") as fh:
        pickle.dump(candidate, fh, protocol=5)

    size_mb = CANDIDATE_PKL.stat().st_size / 1e6
    print(f"    Saved  ({size_mb:.1f} MB)")
    print(f"    Keys:  {sorted(candidate.keys())}")

    # Verify round-trip load
    with open(CANDIDATE_PKL, "rb") as fh:
        rt = pickle.load(fh)
    assert "isotonic_calibrator" in rt, "round-trip verification failed"
    print("    Round-trip load verified.")

    metrics = {
        "brier_raw_eval":          brier_raw,
        "brier_cal_eval":          brier_cal,
        "logloss_raw_eval":        ll_raw,
        "logloss_cal_eval":        ll_cal,
        "mean_gap_above_p20_raw":  mean_gap_raw,
        "mean_gap_above_p20_cal":  mean_gap_cal,
        "n_ev_positive_raw":       n_ev_raw,
        "n_ev_positive_cal":       n_ev_cal,
        "n_total":                 n_total,
        "candidate_path":          str(CANDIDATE_PKL),
    }

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Held-out Brier:    raw={brier_raw:.6f}  cal={brier_cal:.6f}  Δ={brier_cal-brier_raw:+.6f}")
    print(f"  Held-out Log-loss: raw={ll_raw:.6f}  cal={ll_cal:.6f}  Δ={ll_cal-ll_raw:+.6f}")
    print(f"  Mean |gap| p≥0.20: raw={mean_gap_raw:.4f}  cal={mean_gap_cal:.4f}")
    print(f"  EV-positive rows:  raw={n_ev_raw:,}  cal={n_ev_cal:,}  Δ={n_ev_cal-n_ev_raw:+,}")
    print(f"  Candidate: {CANDIDATE_PKL}")
    print("=" * 65)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Isotonic recalibration of live model (candidate only)")
    ap.add_argument("--output-dir", type=Path, default=_ROOT / "outputs",
                    help="Dir for reports (default: outputs/)")
    args = ap.parse_args()

    run_recalibration(output_dir=args.output_dir)
    print("\nDone.")

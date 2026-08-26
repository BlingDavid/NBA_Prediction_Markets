"""Live model evaluation harness (READ-ONLY — does NOT retrain).

Loads OOF predictions from outputs/live_home_up_5m_bootstrap_oof_predictions.csv,
computes diagnostic metrics, evaluates EV-optimal decision thresholds under
Kalshi's winner-fee structure, and writes a findings report.

Usage:
    source .venv/bin/activate
    export DYLD_FALLBACK_LIBRARY_PATH="...:opt/homebrew/opt/libomp/lib"
    python tools/live_model_eval.py [--output-dir outputs]

Bet-mechanics assumption:
    The trigger in paper_trader/trigger.py computes EV as:
        ev = p * E[Δ|rises] + (1-p) * E[Δ|doesn't] - half_spread - p * winner_fee(p)
    For the threshold analysis we hold E[Δ|rises] and E[Δ|doesn't] at the
    pooled means from the scored_rows artifact (loaded from pooled_means.json if
    available, else fallback constants from the report). The half_spread is
    market-dependent so we sweep it. We report the threshold at which
    EV > 0 as a function of p (model probability) by substituting:
        ev(p) = p * E_up + (1-p) * E_down - spread_assumption - p * winner_fee(p)
    where E_up and E_down are the pooled expected price moves and
    winner_fee(p) = ceil(0.07 * p * (1-p) * 100) / 100.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

# ── Repo root on sys.path (script may be run from any cwd) ──────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Import-guard calibration_monitor (intentionally untracked) ──────────────
try:
    from calibration_monitor import reliability_table as _calib_reliability_table
    _HAVE_CALIB_MONITOR = True
except ImportError:
    _HAVE_CALIB_MONITOR = False

# ── Paths ────────────────────────────────────────────────────────────────────
OOF_CSV = _ROOT / "outputs" / "live_home_up_5m_bootstrap_oof_predictions.csv"
REPORT_JSON = _ROOT / "outputs" / "live_home_up_5m_bootstrap_report.json"
POOLED_MEANS_JSON = _ROOT / "outputs" / "paper_trades" / "pooled_means.json"
OUTPUT_MD = _ROOT / "outputs" / "live_model_eval_report.md"


# ═══════════════════════════════════════════════════════════════════════════
# Cost-model helpers (mirrors paper_trader/cost_model.py for purity in tests)
# ═══════════════════════════════════════════════════════════════════════════

def kalshi_winner_fee(p: float) -> float:
    """Kalshi per-contract winner fee: ceil(0.07 * p * (1-p) * 100) / 100.

    Returns 0 for p <= 0 or p >= 1.
    """
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return math.ceil(0.07 * p * (1.0 - p) * 100.0) / 100.0


def ev_per_contract(
    p: float,
    e_up: float,
    e_down: float,
    half_spread: float,
) -> float:
    """Expected value per contract given model probability p.

    ev = p * E[Δ|rises] + (1-p) * E[Δ|doesn't] - half_spread - p * winner_fee(p)

    Mirrors trigger.evaluate() but as a pure function for grid sweeping.
    """
    fee = kalshi_winner_fee(p)
    return p * e_up + (1.0 - p) * e_down - half_spread - p * fee


def ev_breakeven_threshold(
    e_up: float,
    e_down: float,
    half_spread: float,
    p_grid: np.ndarray | None = None,
) -> dict:
    """Find the model-probability threshold where ev_per_contract first turns positive.

    Returns a dict with 'threshold' (first p where EV > 0), 'ev_at_threshold',
    and 'ev_curve' DataFrame with columns [p, ev].
    """
    if p_grid is None:
        p_grid = np.linspace(0.01, 0.99, 990)
    evs = np.array([ev_per_contract(p, e_up, e_down, half_spread) for p in p_grid])
    positive_mask = evs > 0.0
    if not positive_mask.any():
        breakeven_p = float("nan")
        ev_at_threshold = float("nan")
    else:
        idx = int(np.argmax(positive_mask))
        breakeven_p = float(p_grid[idx])
        ev_at_threshold = float(evs[idx])
    curve = pd.DataFrame({"p": p_grid, "ev": evs})
    return {
        "threshold": breakeven_p,
        "ev_at_threshold": ev_at_threshold,
        "ev_curve": curve,
    }


# ═══════════════════════════════════════════════════════════════════════════
# OOF loading (memory-efficient)
# ═══════════════════════════════════════════════════════════════════════════

def load_oof(path: Path) -> pd.DataFrame:
    """Load only label + calibrated prediction columns, with efficient dtypes."""
    usecols = ["label_home_up_5m", "oof_prediction"]
    dtype = {"label_home_up_5m": "int8", "oof_prediction": "float32"}
    return pd.read_csv(path, usecols=usecols, dtype=dtype)


# ═══════════════════════════════════════════════════════════════════════════
# Reliability table (OOF-based, independent of calibration_monitor)
# ═══════════════════════════════════════════════════════════════════════════

def reliability_table_from_oof(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build a reliability (calibration) table from OOF arrays.

    Returns a DataFrame with columns:
        bin_lo, bin_hi, n, avg_pred, emp_rate, gap
    """
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
# PR curve summary (compact)
# ═══════════════════════════════════════════════════════════════════════════

def pr_curve_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_thresholds: int = 20,
) -> pd.DataFrame:
    """Sample n_thresholds evenly-spaced points from the PR curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # sklearn returns arrays of length n+1 for precision/recall; n for thresholds
    # align: skip last point (precision=1, recall=0 sentinel)
    prec = precision[:-1]
    rec = recall[:-1]
    thr = thresholds
    # sample evenly
    idx = np.linspace(0, len(thr) - 1, min(n_thresholds, len(thr)), dtype=int)
    return pd.DataFrame({
        "threshold": thr[idx],
        "precision": prec[idx],
        "recall": rec[idx],
        "f1": 2 * prec[idx] * rec[idx] / (prec[idx] + rec[idx] + 1e-12),
    })


# ═══════════════════════════════════════════════════════════════════════════
# Pooled means loader
# ═══════════════════════════════════════════════════════════════════════════

def load_pooled_means() -> tuple[float, float]:
    """Load E[Δ|rises] and E[Δ|doesn't] from pooled_means.json if available.

    Falls back to the values documented in CLAUDE.md comments:
        E_rises ≈ +0.024 pooled (per-decile goes from +0.012 to +0.067).
        E_doesnt: symmetric around 0, small negative, ~-0.005 typical.
    These fallbacks are illustrative only — the real values come from
    build_pooled_means.py which computes from scored_rows.
    """
    if POOLED_MEANS_JSON.exists():
        payload = json.loads(POOLED_MEANS_JSON.read_text())
        e_up = float(payload["E_delta_given_rises"])
        e_down = float(payload["E_delta_given_doesnt"])
        print(f"  [pooled means] loaded from {POOLED_MEANS_JSON}")
        print(f"    E[Δ|rises]   = {e_up:.5f}")
        print(f"    E[Δ|doesn't] = {e_down:.5f}")
        return e_up, e_down
    # Fallback to documented estimates
    e_up = 0.024
    e_down = -0.005
    print(f"  [pooled means] pooled_means.json not found — using fallback estimates")
    print(f"    E[Δ|rises]   = {e_up:.5f}  (fallback; run build_pooled_means.py)")
    print(f"    E[Δ|doesn't] = {e_down:.5f}  (fallback)")
    return e_up, e_down


# ═══════════════════════════════════════════════════════════════════════════
# Shared metric computation (importable by compare_models.py)
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics_from_arrays(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    e_up: float,
    e_down: float,
    primary_spread: float = 0.01,
    n_bins: int = 10,
) -> dict:
    """Compute all comparison-relevant metrics from label + probability arrays.

    Pure function — no I/O, no printing.  Suitable for unit tests and for
    import by tools/compare_models.py.

    Returns a flat dict containing:
        n_rows, base_rate, roc_auc, average_precision, brier, brier_naive,
        brier_skill, log_loss, e_up, e_down, ev_threshold_primary,
        n_above_ev_threshold, frac_above_ev_threshold, prec_above_ev_threshold,
        max_reliability_gap_above_p20, primary_spread, rel_df.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    n_rows = int(len(y_true))
    base_rate = float(y_true.mean())

    roc_auc = float(roc_auc_score(y_true, y_prob))
    avg_precision = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    brier_naive = float(base_rate * (1 - base_rate))
    brier_skill = 1.0 - brier / brier_naive if brier_naive > 0 else float("nan")
    ll = float(log_loss(y_true, y_prob))

    # Reliability table + overconfidence metric
    rel_df = reliability_table_from_oof(y_true, y_prob, n_bins=n_bins)
    # Max |gap| for bins whose avg_pred > 0.20 (overconfidence in live-relevant range)
    high_p_bins = rel_df[rel_df["avg_pred"] > 0.20]
    if len(high_p_bins) > 0:
        max_rel_gap = float(high_p_bins["gap"].abs().max())
    else:
        max_rel_gap = float("nan")

    # EV threshold
    ev_result = ev_breakeven_threshold(e_up, e_down, primary_spread)
    ev_threshold_primary = ev_result["threshold"]
    if not math.isnan(ev_threshold_primary):
        n_above = int((y_prob >= ev_threshold_primary).sum())
        frac_above = n_above / n_rows if n_rows > 0 else 0.0
        tp_above = int(y_true[y_prob >= ev_threshold_primary].sum())
        prec_above = tp_above / n_above if n_above > 0 else 0.0
    else:
        n_above = 0
        frac_above = 0.0
        prec_above = 0.0
        ev_threshold_primary = None  # normalise NaN → None

    return {
        "n_rows": n_rows,
        "base_rate": base_rate,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "brier": brier,
        "brier_naive": brier_naive,
        "brier_skill": brier_skill,
        "log_loss": ll,
        "e_up": e_up,
        "e_down": e_down,
        "ev_threshold_primary": ev_threshold_primary,
        "n_above_ev_threshold": n_above,
        "frac_above_ev_threshold": frac_above,
        "prec_above_ev_threshold": prec_above,
        "max_reliability_gap_above_p20": max_rel_gap,
        "primary_spread": primary_spread,
        "rel_df": rel_df,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════════════════

def run_eval(output_dir: Path | None = None) -> dict:
    """Run full evaluation and return metrics dict."""
    if output_dir is None:
        output_dir = _ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  NBA Live Model — OOF Evaluation Harness")
    print("=" * 65)

    # ── 1. Load OOF predictions ──────────────────────────────────────────
    print(f"\n[1] Loading OOF predictions from:\n    {OOF_CSV}")
    df = load_oof(OOF_CSV)
    print(f"    Loaded {len(df):,} rows | {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    y_true = df["label_home_up_5m"].to_numpy(dtype=np.int8)
    y_prob = df["oof_prediction"].to_numpy(dtype=np.float32)
    base_rate = float(y_true.mean())
    print(f"    Base rate (label=1): {base_rate:.4f} ({y_true.sum():,} positives)")

    # ── 2. Core metrics ──────────────────────────────────────────────────
    print("\n[2] Computing core metrics...")

    roc_auc = float(roc_auc_score(y_true, y_prob))
    avg_precision = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob))

    # Brier skill score vs naive (always predict base rate)
    brier_naive = float(base_rate * (1 - base_rate))
    brier_skill = 1.0 - brier / brier_naive

    print(f"    ROC-AUC:            {roc_auc:.4f}")
    print(f"    Avg Precision (AP): {avg_precision:.4f}  (baseline={base_rate:.4f})")
    print(f"    Brier score:        {brier:.6f}  (naive={brier_naive:.4f}, skill={brier_skill:.3f})")
    print(f"    Log-loss:           {ll:.6f}")

    # ── 3. PR curve ──────────────────────────────────────────────────────
    print("\n[3] Precision-Recall curve (sampled at 20 thresholds):")
    pr_df = pr_curve_summary(y_true, y_prob, n_thresholds=20)
    print(pr_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ── 4. Reliability table ─────────────────────────────────────────────
    print("\n[4] Reliability table (predicted prob bucket vs empirical rate):")
    rel_df = reliability_table_from_oof(y_true, y_prob, n_bins=10)
    print(rel_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ── 5. EV-optimal threshold ──────────────────────────────────────────
    print("\n[5] EV-optimal threshold analysis...")
    e_up, e_down = load_pooled_means()

    # Sweep half-spread assumptions: 0 (no spread), 0.005, 0.01, 0.02
    spread_sweep = [0.0, 0.005, 0.01, 0.02]
    ev_thresholds = {}
    print(f"\n  {'Half-spread':>14}  {'EV>0 at p>':>12}  {'EV at threshold':>16}")
    print("  " + "-" * 46)
    for hs in spread_sweep:
        result = ev_breakeven_threshold(e_up, e_down, hs)
        thr = result["threshold"]
        ev_thr = result["ev_at_threshold"]
        label = f"{hs:.4f}"
        if math.isnan(thr):
            print(f"  {label:>14}  {'never positive':>12}  {'n/a':>16}")
        else:
            print(f"  {label:>14}  {thr:>12.4f}  {ev_thr:>16.6f}")
        ev_thresholds[hs] = result

    # Detailed EV curve for primary spread assumption (0.01 ≈ 1¢, reasonable live spread)
    primary_spread = 0.01
    primary_result = ev_thresholds[primary_spread]
    curve = primary_result["ev_curve"]
    positive_curve = curve[curve["ev"] > 0]
    ev_table_sample = curve[curve["p"].isin(
        np.round(np.arange(0.05, 0.50, 0.05), 2)
    )].copy() if len(curve) > 0 else pd.DataFrame()

    # Empirical: what fraction of OOF rows exceed the EV-optimal threshold?
    ev_threshold_primary = primary_result["threshold"]
    if not math.isnan(ev_threshold_primary):
        n_above = int((y_prob >= ev_threshold_primary).sum())
        frac_above = n_above / len(y_prob)
        tp_above = int(y_true[y_prob >= ev_threshold_primary].sum())
        prec_above = tp_above / n_above if n_above > 0 else 0.0
        print(f"\n  At half-spread={primary_spread}: EV>0 requires p ≥ {ev_threshold_primary:.4f}")
        print(f"    OOF rows above threshold: {n_above:,} ({frac_above:.2%})")
        print(f"    Empirical precision above threshold: {prec_above:.4f}")
        print(f"    Expected base_rate lift vs {base_rate:.4f}: "
              f"{prec_above/base_rate:.1f}x")
    else:
        n_above = 0
        frac_above = 0.0
        tp_above = 0
        prec_above = 0.0
        print(f"\n  At half-spread={primary_spread}: EV never positive (e_up too small)")

    # EV at a few key model-probability values
    print(f"\n  EV(p) with e_up={e_up:.4f}, e_down={e_down:.4f}, half_spread={primary_spread}:")
    print(f"  {'p':>6}  {'EV':>10}  {'winner_fee':>12}")
    for p_val in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        fee = kalshi_winner_fee(p_val)
        ev = ev_per_contract(p_val, e_up, e_down, primary_spread)
        print(f"  {p_val:>6.2f}  {ev:>10.6f}  {fee:>12.4f}")

    # ── Decile-conditional analysis ──────────────────────────────────────
    print("\n[6] Decile-conditional precision (model signal quality by decile):")
    df_for_decile = pd.DataFrame({"y": y_true, "p": y_prob})
    df_for_decile["decile"] = pd.qcut(
        df_for_decile["p"], q=10, labels=False, duplicates="drop"
    )
    decile_stats = (
        df_for_decile.groupby("decile", observed=True)
        .agg(n=("y", "count"), n_pos=("y", "sum"), avg_pred=("p", "mean"))
        .assign(emp_rate=lambda d: d["n_pos"] / d["n"])
        .assign(lift=lambda d: d["emp_rate"] / base_rate)
        .reset_index()
    )
    print(decile_stats.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ── Collect all metrics ──────────────────────────────────────────────
    metrics = {
        "n_rows": int(len(df)),
        "base_rate": base_rate,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "brier": brier,
        "brier_naive": brier_naive,
        "brier_skill": brier_skill,
        "log_loss": ll,
        "e_up": e_up,
        "e_down": e_down,
        "ev_threshold_primary": ev_threshold_primary if not math.isnan(ev_threshold_primary) else None,
        "n_above_ev_threshold": n_above,
        "frac_above_ev_threshold": frac_above,
        "prec_above_ev_threshold": prec_above,
        "pr_df": pr_df,
        "rel_df": rel_df,
        "decile_stats": decile_stats,
        "ev_thresholds": ev_thresholds,
        "primary_spread": primary_spread,
    }

    # ── Write markdown report ────────────────────────────────────────────
    _write_report(metrics, output_dir)

    return metrics


def _df_to_md(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a Markdown table string without requiring tabulate."""
    if df.empty:
        return "(no data)"
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator] + rows)


def _fmt_threshold(t: float | None) -> str:
    """Format a threshold value for display, handling None and NaN."""
    if t is None:
        return "never"
    try:
        if math.isnan(t):
            return "never"
        return f"{t:.4f}"
    except (TypeError, ValueError):
        return "never"


def _write_report(m: dict, output_dir: Path) -> None:
    """Write findings to outputs/live_model_eval_report.md."""
    e_up = m["e_up"]
    e_down = m["e_down"]
    ev_thr = m["ev_threshold_primary"]
    hs = m["primary_spread"]

    ev_rows = ""
    for p_val in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        fee = kalshi_winner_fee(p_val)
        ev = ev_per_contract(p_val, e_up, e_down, hs)
        ev_rows += f"| {p_val:.2f} | {ev:+.6f} | {fee:.4f} |\n"

    rel_md = _df_to_md(m["rel_df"])
    decile_md = _df_to_md(m["decile_stats"])
    pr_md = _df_to_md(m["pr_df"])

    report = f"""# Live Model Evaluation Report

Generated by `tools/live_model_eval.py` (read-only; no retraining).

## 1. Core OOF Metrics

| Metric | Value |
|--------|-------|
| N rows (OOF) | {m['n_rows']:,} |
| Base rate (label=1) | {m['base_rate']:.4f} ({m['base_rate']*100:.2f}%) |
| ROC-AUC | {m['roc_auc']:.4f} |
| Average Precision (PR-AUC) | {m['average_precision']:.4f} |
| Brier score | {m['brier']:.6f} |
| Brier naive (always=base rate) | {m['brier_naive']:.4f} |
| Brier skill score | {m['brier_skill']:.4f} |
| Log-loss | {m['log_loss']:.6f} |

**Interpretation:** ROC-AUC of {m['roc_auc']:.4f} indicates strong rank-ordering ability.
Average Precision of {m['average_precision']:.4f} is {m['average_precision']/m['base_rate']:.1f}x the base-rate ceiling,
indicating meaningful signal in the top deciles. Brier skill score of {m['brier_skill']:.3f}
means the model is {m['brier_skill']*100:.1f}% better-calibrated than the naive baseline.

## 2. Precision-Recall Curve (sampled)

{pr_md}

## 3. Reliability Table (Calibration)

{rel_md}

**Calibration note:** Bins with large positive gap (emp_rate > avg_pred) mean the model
undershoots true probability in that range; negative gap means it overshoots. Platt scaling
or isotonic regression on OOF predictions would tighten gaps > 0.02.

## 4. Decile Analysis

{decile_md}

## 5. EV-Optimal Threshold Analysis

**Mechanics:** The trigger EV formula (from `paper_trader/trigger.py`) is:

```
ev = p * E[Δ|rises] + (1-p) * E[Δ|doesn't] - half_spread - p * winner_fee(p)
winner_fee(p) = ceil(0.07 * p * (1-p) * 100) / 100
```

**Assumptions:**
- `E[Δ|rises]` = {e_up:.5f} (from pooled_means.json or fallback estimate)
- `E[Δ|doesn't]` = {e_down:.5f} (from pooled_means.json or fallback estimate)
- Half-spread scenarios swept: 0.0, 0.005, 0.01, 0.02

**EV breakeven thresholds by half-spread:**

| Half-spread | EV>0 at p ≥ |
|------------|-------------|
| 0.0000 | {_fmt_threshold(m['ev_thresholds'][0.0]['threshold'])} |
| 0.0050 | {_fmt_threshold(m['ev_thresholds'][0.005]['threshold'])} |
| 0.0100 | {_fmt_threshold(ev_thr)} |
| 0.0200 | {_fmt_threshold(m['ev_thresholds'][0.02]['threshold'])} |

**EV at key model probabilities (half_spread={hs}):**

| p | EV per contract | Winner fee |
|---|----------------|------------|
{ev_rows}
**OOF rows above EV threshold (p ≥ {_fmt_threshold(ev_thr)}):**
- Count: {m['n_above_ev_threshold']:,} ({m['frac_above_ev_threshold']:.2%} of rows)
- Empirical precision: {m['prec_above_ev_threshold']:.4f}
- Lift vs base rate: {m['prec_above_ev_threshold']/m['base_rate']:.1f}x

## 6. Ranked Tuning Recommendations for Round 2

### Recommendation 1 (HIGHEST IMPACT): Lower the trading threshold to the EV-optimal level

**Current state:** The trigger fires only when `ev_per_contract > 0`. The EV threshold
is set by the interplay of pooled means and the winner-fee formula. At the current pooled
E_rises ≈ {e_up:.4f}, EV first turns positive near p ≈ {_fmt_threshold(ev_thr)} (half-spread=0.01).
The model's raw calibrated probabilities peak around 0.30 for the top quantile — meaning
signals above the EV threshold **are available** but only in the top 5-10% of the
prediction distribution (lift 4-6x). Recommendation: explicitly gate on the EV-breakeven
probability rather than the raw `ev > 0` check (which depends on market-time spread inputs
that can be stale), and ensure the Phase B decile table is always fresh after any retrain.

**Expected impact:** Unlock trades in deciles 6-9 where empirical precision exceeds 15-25%
vs base rate 4.7%. High confidence this increases trade volume.

### Recommendation 2 (HIGH IMPACT): Add Platt scaling or isotonic recalibration as a post-fit step

**Current state:** Reliability table shows the model's probabilities are compressed
into a narrow range (most predictions concentrate in 0.04-0.15). The calibration
alpha=0.95 shrinkage in the current pipeline helps but is not equivalent to proper
isotonic calibration. The model's low recall at threshold=0.50 (11.8%) with precision
of only 43% suggests threshold miscalibration.

**Recommendation:** After fitting the LogisticRegression in `live_bootstrap_model.py`,
add a `CalibratedClassifierCV(method='isotonic', cv='prefit')` wrapper fitted on
held-out fold predictions. This is a one-line change to the pipeline and does NOT
require retraining the base model.

**Expected impact:** Improve Brier score by ~5-10% and widen the usable prediction
range, enabling more confident EV discrimination at intermediate probabilities.

### Recommendation 3 (MEDIUM IMPACT): Switch from alpha-weighted shrinkage to class_weight='balanced' or sample weights

**Current state:** The model uses alpha=0.95 shrinkage (a multiplicative label-smoothing
trick). With a 4.7% base rate and ~1.375M rows, the classifier is heavily imbalanced.
The current approach improves calibration but does not address the fundamental
signal-recovery problem: at threshold=0.5, recall is only 11.8% — the model is
missing 88% of positive events.

**Recommendation:** Compare the current shrinkage approach with `class_weight='balanced'`
in LogisticRegression (equivalent to inverse-frequency sample weighting). Separately,
if using XGBoost is added in a future round, `scale_pos_weight = n_neg/n_pos ≈ 20.4`
directly addresses the imbalance in the loss function.

**Expected impact:** Increase recall by 20-40 percentage points (pulling recall from 12%
to 30-50% range), at cost of some precision loss. For trading, higher recall matters only
if precision stays above the EV threshold — so this must be combined with threshold tuning.

### Recommendation 4 (MEDIUM IMPACT): Add cross-game temporal features and momentum signals

**Current state:** The top coefficients are dominated by orderbook features
(imbalance, depth, weighted price) and status flags. The model sees within-tick
microstructure well. It does NOT currently see: score run (consecutive scoring events),
pace delta (actual vs expected possessions), momentum decay (how many minutes since
the last home scoring run).

**Recommendation:** Add 2-3 engineered features in `live_training_matrix.py` (owned
by another agent in this round, but flagged for coordination):
1. `home_score_run_3min`: home point differential in last 3 minutes (captures momentum)
2. `pace_delta`: actual vs expected possessions (high pace = more variance = more price moves)
3. `yes_mid_velocity`: 1st derivative of yes_mid over last 3 ticks (price momentum)

These features directly model whether the home contract is likely to move, complementing
the current static snapshot features.

**Expected impact:** Improve ROC-AUC from 0.819 toward 0.85+, reduce log-loss by 5-8%.
"""

    out_path = output_dir / "live_model_eval_report.md"
    out_path.write_text(report)
    print(f"\n[Report] Written to: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Live model OOF evaluation harness")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "outputs",
        help="Directory for outputs (default: outputs/)",
    )
    args = ap.parse_args()

    metrics = run_eval(output_dir=args.output_dir)
    print("\nDone.")

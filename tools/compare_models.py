"""Model comparison + promotion decision tool.

Compares two or more model/OOF-prediction sources side-by-side against the
deployed baseline model.  Uses shared metric helpers from live_model_eval.py —
EV math is NOT reimplemented here.

Usage
-----
# Compare deployed model vs one or more OOF CSVs
python tools/compare_models.py \\
    --baseline outputs/live_home_up_5m_bootstrap_oof_predictions.csv \\
    --candidates outputs/candidate_isotonic_oof.csv outputs/candidate_momentum_oof.csv \\
    --names deployed isotonic momentum

# Compare using pkl model files (loads OOF from standard path then runs inference —
# or accepts an explicit --oof-col column name)
python tools/compare_models.py \\
    --baseline outputs/live_home_up_5m_bootstrap_oof_predictions.csv \\
    --candidates outputs/live_home_up_5m_bootstrap_oof_predictions.csv \\
    --names baseline same_as_baseline

# Write comparison report to outputs/
python tools/compare_models.py \\
    --baseline outputs/live_home_up_5m_bootstrap_oof_predictions.csv \\
    --write-report

Columns expected in OOF CSVs
-----------------------------
    label_home_up_5m   — integer 0/1 label (required)
    oof_prediction     — calibrated probability (required, unless --oof-col overrides)

If a candidate path is missing, it is skipped with a clear message.

Promotion rule
--------------
A candidate is PROMOTABLE if ALL of the following hold:
    1. Money metric (n_above_ev_threshold) increases OR max_reliability_gap_above_p20
       shrinks by > OVERCONF_IMPROVE_MIN.
    2. Brier score does NOT worsen by more than BRIER_TOLERANCE.
    3. Log-loss does NOT worsen by more than LOG_LOSS_TOLERANCE.
    4. ROC-AUC does NOT worsen by more than ROC_AUC_TOLERANCE.
    5. No obvious calibration regression: max_reliability_gap_above_p20 does NOT
       increase by more than OVERCONF_REGRESSION_MAX (independent of condition 1).

Constants are module-level so tests can override them.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Repo root on sys.path ────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.live_model_eval import (
    compute_metrics_from_arrays,
    ev_breakeven_threshold,
    kalshi_winner_fee,
    load_pooled_means,
)

# ── Promotion thresholds (tunable) ──────────────────────────────────────────
# Candidate is penalised if Brier worsens by more than this
BRIER_TOLERANCE: float = 0.0005
# Candidate is penalised if log-loss worsens by more than this
LOG_LOSS_TOLERANCE: float = 0.005
# Candidate is penalised if ROC-AUC drops by more than this
ROC_AUC_TOLERANCE: float = 0.002
# Overconfidence gap improvement considered meaningful (condition 1, OR branch)
OVERCONF_IMPROVE_MIN: float = 0.01
# Max allowed worsening of reliability gap (condition 5)
OVERCONF_REGRESSION_MAX: float = 0.02


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_oof_csv(path: Path, oof_col: str = "oof_prediction") -> tuple[np.ndarray, np.ndarray]:
    """Load (y_true, y_prob) from an OOF CSV.

    Returns numpy arrays of int8 labels and float32 probabilities.
    Raises FileNotFoundError if path does not exist.
    Raises KeyError if expected columns are missing.
    """
    usecols = ["label_home_up_5m", oof_col]
    dtype = {"label_home_up_5m": "int8", oof_col: "float32"}
    df = pd.read_csv(path, usecols=usecols, dtype=dtype)
    y_true = df["label_home_up_5m"].to_numpy(dtype=np.int8)
    y_prob = df[oof_col].to_numpy(dtype=np.float32)
    return y_true, y_prob


def load_source(
    path: Path,
    oof_col: str = "oof_prediction",
    label: str = "model",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Try to load a source (CSV).  Returns None on any loading failure, printing a message."""
    if not path.exists():
        print(f"  [SKIP] {label}: path not found — {path}")
        return None
    try:
        y_true, y_prob = load_oof_csv(path, oof_col=oof_col)
        print(f"  [OK]   {label}: loaded {len(y_true):,} rows from {path.name}")
        return y_true, y_prob
    except KeyError as exc:
        print(f"  [SKIP] {label}: missing column in CSV — {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"  [SKIP] {label}: failed to load — {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Promotion rule (pure function — fully testable without I/O)
# ═══════════════════════════════════════════════════════════════════════════

def is_promotable(
    baseline: dict,
    candidate: dict,
    brier_tolerance: float = BRIER_TOLERANCE,
    log_loss_tolerance: float = LOG_LOSS_TOLERANCE,
    roc_auc_tolerance: float = ROC_AUC_TOLERANCE,
    overconf_improve_min: float = OVERCONF_IMPROVE_MIN,
    overconf_regression_max: float = OVERCONF_REGRESSION_MAX,
) -> tuple[bool, list[str]]:
    """Decide whether a candidate model should be promoted over the baseline.

    Parameters
    ----------
    baseline, candidate:
        Metric dicts as returned by compute_metrics_from_arrays().
        Required keys:
            n_above_ev_threshold        (int)   money-metric: rows above EV threshold
            max_reliability_gap_above_p20 (float) overconfidence metric
            brier                       (float)
            log_loss                    (float)
            roc_auc                     (float)

    Returns
    -------
    (promotable: bool, reasons: list[str])
        reasons lists one sentence per checked condition — both passes and failures.
    """
    reasons: list[str] = []

    # ── helpers ──────────────────────────────────────────────────────────────
    def _get(d: dict, key: str, default=None):
        v = d.get(key, default)
        if v is None:
            return default
        try:
            if math.isnan(float(v)):
                return default
        except (TypeError, ValueError):
            pass
        return v

    b_money = _get(baseline, "n_above_ev_threshold", 0)
    c_money = _get(candidate, "n_above_ev_threshold", 0)

    b_gap = _get(baseline, "max_reliability_gap_above_p20", float("nan"))
    c_gap = _get(candidate, "max_reliability_gap_above_p20", float("nan"))

    b_brier = _get(baseline, "brier", float("nan"))
    c_brier = _get(candidate, "brier", float("nan"))

    b_ll = _get(baseline, "log_loss", float("nan"))
    c_ll = _get(candidate, "log_loss", float("nan"))

    b_auc = _get(baseline, "roc_auc", float("nan"))
    c_auc = _get(candidate, "roc_auc", float("nan"))

    passes = []  # True/False per condition

    # ── Condition 1: money metric improves OR overconfidence shrinks meaningfully
    money_improved = int(c_money) > int(b_money)
    gap_shrunk = (
        not (math.isnan(b_gap) or math.isnan(c_gap))
        and (b_gap - c_gap) >= overconf_improve_min
    )
    cond1 = money_improved or gap_shrunk
    passes.append(cond1)
    if money_improved:
        reasons.append(
            f"PASS  [C1-money]     EV-positive rows: {b_money:,} → {c_money:,} "
            f"(+{c_money - b_money:,})"
        )
    elif gap_shrunk:
        reasons.append(
            f"PASS  [C1-overconf]  Reliability gap shrunk: {b_gap:.4f} → {c_gap:.4f} "
            f"(improvement {b_gap - c_gap:.4f} ≥ {overconf_improve_min})"
        )
    else:
        delta_money = c_money - b_money
        gap_str = (
            f"gap {b_gap:.4f} → {c_gap:.4f} (Δ{c_gap - b_gap:+.4f})"
            if not (math.isnan(b_gap) or math.isnan(c_gap))
            else "gap unavailable"
        )
        reasons.append(
            f"FAIL  [C1]           No improvement: money {b_money:,} → {c_money:,} "
            f"(Δ{delta_money:+,}); {gap_str}"
        )

    # ── Condition 2: Brier does NOT worsen beyond tolerance
    if math.isnan(b_brier) or math.isnan(c_brier):
        cond2 = True
        reasons.append("PASS  [C2-brier]     Brier comparison not available (skipped).")
    else:
        brier_delta = c_brier - b_brier  # positive = worse
        cond2 = brier_delta <= brier_tolerance
        status = "PASS" if cond2 else "FAIL"
        reasons.append(
            f"{status}  [C2-brier]     Brier: {b_brier:.6f} → {c_brier:.6f} "
            f"(Δ{brier_delta:+.6f}; tolerance ≤ {brier_tolerance})"
        )
    passes.append(cond2)

    # ── Condition 3: Log-loss does NOT worsen beyond tolerance
    if math.isnan(b_ll) or math.isnan(c_ll):
        cond3 = True
        reasons.append("PASS  [C3-logloss]   Log-loss comparison not available (skipped).")
    else:
        ll_delta = c_ll - b_ll
        cond3 = ll_delta <= log_loss_tolerance
        status = "PASS" if cond3 else "FAIL"
        reasons.append(
            f"{status}  [C3-logloss]   Log-loss: {b_ll:.6f} → {c_ll:.6f} "
            f"(Δ{ll_delta:+.6f}; tolerance ≤ {log_loss_tolerance})"
        )
    passes.append(cond3)

    # ── Condition 4: ROC-AUC does NOT drop more than tolerance
    if math.isnan(b_auc) or math.isnan(c_auc):
        cond4 = True
        reasons.append("PASS  [C4-rocauc]    ROC-AUC comparison not available (skipped).")
    else:
        auc_delta = c_auc - b_auc  # positive = better
        cond4 = auc_delta >= -roc_auc_tolerance
        status = "PASS" if cond4 else "FAIL"
        reasons.append(
            f"{status}  [C4-rocauc]    ROC-AUC: {b_auc:.4f} → {c_auc:.4f} "
            f"(Δ{auc_delta:+.4f}; max drop {roc_auc_tolerance})"
        )
    passes.append(cond4)

    # ── Condition 5: Reliability gap does NOT worsen significantly
    if math.isnan(b_gap) or math.isnan(c_gap):
        cond5 = True
        reasons.append("PASS  [C5-calib]     Reliability gap comparison not available (skipped).")
    else:
        gap_worsening = c_gap - b_gap  # positive = worse
        cond5 = gap_worsening <= overconf_regression_max
        status = "PASS" if cond5 else "FAIL"
        reasons.append(
            f"{status}  [C5-calib]     Reliability gap: {b_gap:.4f} → {c_gap:.4f} "
            f"(Δ{gap_worsening:+.4f}; max allowed regression {overconf_regression_max})"
        )
    passes.append(cond5)

    promotable = all(passes)
    return promotable, reasons


# ═══════════════════════════════════════════════════════════════════════════
# Comparison table assembly
# ═══════════════════════════════════════════════════════════════════════════

def build_comparison_table(
    baseline_metrics: dict,
    candidates: list[tuple[str, dict]],
) -> pd.DataFrame:
    """Build a side-by-side comparison DataFrame.

    Parameters
    ----------
    baseline_metrics : dict  — metric dict for the deployed model
    candidates : list of (name, metric_dict) pairs

    Returns
    -------
    DataFrame with rows = metric names, columns = [metric, baseline, cand1, cand2, ...]
    Delta columns (cand - baseline) are appended for each candidate.
    """
    METRIC_ROWS = [
        ("roc_auc",                      "ROC-AUC",                     ".4f",  False),
        ("average_precision",            "PR-AUC",                      ".4f",  False),
        ("brier",                        "Brier Score",                  ".6f",  True),
        ("log_loss",                     "Log-Loss",                     ".6f",  True),
        ("max_reliability_gap_above_p20","Max Reliability Gap (p>0.20)", ".4f",  True),
        ("n_above_ev_threshold",         "EV+ Rows (money metric)",      "d",    False),
        ("frac_above_ev_threshold",      "EV+ Rows (%)",                 ".4%",  False),
        ("prec_above_ev_threshold",      "Precision @ EV threshold",     ".4f",  False),
        ("ev_threshold_primary",         "EV Threshold (p≥)",           ".4f",  True),
        ("n_rows",                       "N Rows",                       "d",    False),
        ("base_rate",                    "Base Rate",                    ".4f",  False),
    ]

    def _fmt(val, fmt: str):
        if val is None:
            return "N/A"
        try:
            if math.isnan(float(val)):
                return "N/A"
        except (TypeError, ValueError):
            pass
        if fmt == "d":
            return f"{int(val):,}"
        if fmt.endswith("%"):
            return f"{float(val):{fmt}}"
        return f"{float(val):{fmt}}"

    def _delta_fmt(baseline_val, cand_val, fmt: str, lower_is_better: bool):
        if baseline_val is None or cand_val is None:
            return "N/A"
        try:
            b = float(baseline_val)
            c = float(cand_val)
            if math.isnan(b) or math.isnan(c):
                return "N/A"
        except (TypeError, ValueError):
            return "N/A"
        delta = c - b
        if fmt == "d":
            delta_str = f"{int(round(delta)):+,}"
        elif fmt.endswith("%"):
            delta_str = f"{delta:+.2%}"
        else:
            delta_str = f"{delta:+{fmt[1:]}}"  # strip leading dot for sign
            delta_str = f"{delta:+{fmt}}"
        # Arrow: up = improvement, down = regression
        if abs(delta) < 1e-10:
            arrow = "  "
        elif (lower_is_better and delta < 0) or (not lower_is_better and delta > 0):
            arrow = " ↑"
        else:
            arrow = " ↓"
        return f"{delta_str}{arrow}"

    rows = []
    col_names = ["Metric", "Baseline"]
    for cand_name, _ in candidates:
        col_names.append(cand_name)
        col_names.append(f"Δ {cand_name}")

    for key, label, fmt, lower_is_better in METRIC_ROWS:
        b_val = baseline_metrics.get(key)
        row = {"Metric": label, "Baseline": _fmt(b_val, fmt)}
        for cand_name, cand_m in candidates:
            c_val = cand_m.get(key)
            row[cand_name] = _fmt(c_val, fmt)
            row[f"Δ {cand_name}"] = _delta_fmt(b_val, c_val, fmt, lower_is_better)
        rows.append(row)

    return pd.DataFrame(rows, columns=col_names)


# ═══════════════════════════════════════════════════════════════════════════
# Report writer
# ═══════════════════════════════════════════════════════════════════════════

def _table_to_str(df: pd.DataFrame, col_width: int = 26) -> str:
    """Format the comparison DataFrame as a plain-text table."""
    lines = []
    header = "  ".join(str(c).ljust(col_width) for c in df.columns)
    separator = "  ".join("-" * col_width for _ in df.columns)
    lines.append(separator)
    lines.append(header)
    lines.append(separator)
    for _, row in df.iterrows():
        lines.append("  ".join(str(v).ljust(col_width) for v in row))
    lines.append(separator)
    return "\n".join(lines)


def write_comparison_report(
    output_dir: Path,
    comparison_df: pd.DataFrame,
    promotability: list[tuple[str, bool, list[str]]],
) -> Path:
    """Write a markdown comparison report to output_dir/compare_models_report.md."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "compare_models_report.md"

    sections = ["# Model Comparison Report\n\nGenerated by `tools/compare_models.py`.\n"]

    # Comparison table
    sections.append("## Side-by-Side Metrics\n")
    # Markdown table
    cols = list(comparison_df.columns)
    md_header = "| " + " | ".join(cols) + " |"
    md_sep = "| " + " | ".join("---" for _ in cols) + " |"
    md_rows = [md_header, md_sep]
    for _, row in comparison_df.iterrows():
        md_rows.append("| " + " | ".join(str(v) for v in row) + " |")
    sections.append("\n".join(md_rows) + "\n")

    # Promotability decisions
    sections.append("## Promotion Decisions\n")
    for cand_name, promotable, reasons in promotability:
        verdict = "**PROMOTABLE**" if promotable else "**NOT PROMOTABLE**"
        sections.append(f"### {cand_name}: {verdict}\n")
        for r in reasons:
            sections.append(f"- {r}")
        sections.append("")

    out_path.write_text("\n".join(sections))
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_comparison(
    baseline_path: Path,
    candidate_paths: list[Path],
    names: list[str] | None = None,
    oof_col: str = "oof_prediction",
    output_dir: Path | None = None,
    write_report: bool = False,
) -> dict:
    """Run full model comparison.

    Parameters
    ----------
    baseline_path    : Path to the baseline OOF CSV.
    candidate_paths  : List of paths to candidate OOF CSVs.
    names            : Optional list of display names (baseline + candidates).
                       If provided, first name = baseline label.
    oof_col          : Column name for predicted probabilities (default: oof_prediction).
    output_dir       : Directory for the written report.
    write_report     : Whether to write outputs/compare_models_report.md.

    Returns
    -------
    dict with keys: baseline_metrics, candidates (list of (name, metrics)),
                    comparison_df, promotability
    """
    if output_dir is None:
        output_dir = _ROOT / "outputs"

    # Resolve names
    if names is None:
        names = []
    baseline_name = names[0] if len(names) > 0 else "baseline"
    cand_names = (
        names[1:] if len(names) > 1
        else [p.stem for p in candidate_paths]
    )
    # Pad cand_names if fewer provided than paths
    while len(cand_names) < len(candidate_paths):
        cand_names.append(candidate_paths[len(cand_names)].stem)

    print("=" * 65)
    print("  NBA Live Model — Model Comparison Harness")
    print("=" * 65)

    # ── Load pooled means (shared across all models for fair EV comparison) ──
    print("\n[1] Loading pooled means for EV threshold calculation...")
    e_up, e_down = load_pooled_means()

    # ── Load baseline ────────────────────────────────────────────────────────
    print(f"\n[2] Loading sources...")
    baseline_data = load_source(baseline_path, oof_col=oof_col, label=baseline_name)
    if baseline_data is None:
        print(f"\nERROR: Baseline could not be loaded from {baseline_path}. Aborting.")
        return {}

    y_true_base, y_prob_base = baseline_data
    print(f"     Computing metrics for baseline ({baseline_name})...")
    baseline_metrics = compute_metrics_from_arrays(y_true_base, y_prob_base, e_up, e_down)

    # ── Load candidates ──────────────────────────────────────────────────────
    loaded_candidates: list[tuple[str, dict]] = []
    for cpath, cname in zip(candidate_paths, cand_names):
        cdata = load_source(cpath, oof_col=oof_col, label=cname)
        if cdata is None:
            continue
        y_true_c, y_prob_c = cdata
        print(f"     Computing metrics for {cname}...")
        cand_metrics = compute_metrics_from_arrays(y_true_c, y_prob_c, e_up, e_down)
        loaded_candidates.append((cname, cand_metrics))

    if not loaded_candidates:
        print("\nWARN: No candidate models could be loaded. Only baseline metrics available.")

    # ── Build comparison table ───────────────────────────────────────────────
    print("\n[3] Building comparison table...")
    comparison_df = build_comparison_table(baseline_metrics, loaded_candidates)

    print("\n" + _table_to_str(comparison_df, col_width=28))

    # ── Promotion decisions ──────────────────────────────────────────────────
    print("\n[4] Promotion checklist...")
    promotability: list[tuple[str, bool, list[str]]] = []
    for cname, cand_m in loaded_candidates:
        promotable, reasons = is_promotable(baseline_metrics, cand_m)
        promotability.append((cname, promotable, reasons))

        verdict = "PROMOTABLE" if promotable else "NOT PROMOTABLE"
        print(f"\n  --- {cname}: {verdict} ---")
        for r in reasons:
            print(f"    {r}")

    # ── Write report ─────────────────────────────────────────────────────────
    if write_report and loaded_candidates:
        report_path = write_comparison_report(output_dir, comparison_df, promotability)
        print(f"\n[Report] Written to: {report_path}")

    return {
        "baseline_metrics": baseline_metrics,
        "baseline_name": baseline_name,
        "candidates": loaded_candidates,
        "comparison_df": comparison_df,
        "promotability": promotability,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Compare NBA live model candidates against the deployed baseline."
    )
    ap.add_argument(
        "--baseline",
        type=Path,
        default=_ROOT / "outputs" / "live_home_up_5m_bootstrap_oof_predictions.csv",
        help="Path to the baseline OOF CSV (default: outputs/live_home_up_5m_bootstrap_oof_predictions.csv)",
    )
    ap.add_argument(
        "--candidates",
        type=Path,
        nargs="+",
        default=[],
        help="Paths to one or more candidate OOF CSVs.",
    )
    ap.add_argument(
        "--names",
        nargs="+",
        default=None,
        help=(
            "Display names: first = baseline label, rest = candidate labels. "
            "E.g. --names deployed isotonic momentum"
        ),
    )
    ap.add_argument(
        "--oof-col",
        default="oof_prediction",
        help="Column name for model predictions in the CSVs (default: oof_prediction).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "outputs",
        help="Directory for the written report (default: outputs/).",
    )
    ap.add_argument(
        "--write-report",
        action="store_true",
        help="Write a markdown report to outputs/compare_models_report.md.",
    )
    args = ap.parse_args()

    result = run_comparison(
        baseline_path=args.baseline,
        candidate_paths=args.candidates,
        names=args.names,
        oof_col=args.oof_col,
        output_dir=args.output_dir,
        write_report=args.write_report,
    )
    print("\nDone.")

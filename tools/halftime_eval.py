"""Honest evaluation of the live halftime-leader predictor: dumb baselines +
a lead-time curve on a temporal hold-out. Research/prediction only."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python tools/halftime_eval.py` to import repo-root modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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


def _safe_metric(fn, y_true, y_prob) -> float:
    try:
        return float(fn(y_true, y_prob))
    except Exception:
        return float("nan")


def main(argv=None) -> int:
    """End-to-end honest hold-out evaluation of the halftime-leader predictor.

    Builds the first-half matrix, restricts to a TEST date window (out of
    sample vs the model's train cutoff), scores it, and compares model vs the
    current-leader / ELO / diffusion baselines, plus a lead-time curve.
    """
    import argparse
    import pickle
    from datetime import datetime, timezone
    from pathlib import Path

    import numpy as np
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    from live_bootstrap_model import build_bootstrap_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/candidates/live_halftime_leader.pkl")
    parser.add_argument("--dates-from", default="2026-04-29")
    parser.add_argument("--dates-to", default="2026-05-08")
    parser.add_argument("--out", default="outputs/halftime_eval_report.md")
    parser.add_argument("--bin-minutes", type=int, default=4)
    args = parser.parse_args(argv)

    with open(args.model_path, "rb") as fh:
        bundle = pickle.load(fh)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    target = bundle["target"]

    # Build the full first-half matrix (all dates), then restrict to TEST window.
    _matrix, labeled_rows, _feat, _meta = build_bootstrap_dataset(
        target=target, first_half_only=True,
    )
    if labeled_rows.empty:
        print("No labeled first-half rows available.")
        return 1

    rows = labeled_rows.copy()
    rows["captured_at"] = pd.to_datetime(rows["captured_at"], utc=True, errors="coerce")
    lo = pd.Timestamp(args.dates_from, tz="UTC")
    hi = pd.Timestamp(args.dates_to, tz="UTC") + pd.Timedelta(days=1)
    rows = rows[(rows["captured_at"] >= lo) & (rows["captured_at"] < hi)].copy()
    rows = rows.dropna(subset=[target, "score_margin_home", "seconds_elapsed"]).copy()
    if rows.empty:
        print("No TEST-window rows after filtering; cannot evaluate.")
        return 1

    y_true = rows[target].astype(int).to_numpy()
    X = rows.reindex(columns=feature_cols)
    model_prob = model.predict_proba(X)[:, 1]

    margin = rows["score_margin_home"].astype(float)
    sec_elapsed = rows["seconds_elapsed"].astype(float)
    sec_left_half = (1440.0 - sec_elapsed).clip(lower=0.0)
    if "pregame_home_win_prob" in rows.columns:
        elo_src = rows["pregame_home_win_prob"].fillna(0.5)
    else:
        elo_src = pd.Series(0.5, index=rows.index)

    cur = margin.apply(current_leader_prob).to_numpy()
    elo = elo_src.apply(elo_prior_prob).to_numpy()
    dif = np.array([diffusion_prob(m, s) for m, s in zip(margin, sec_left_half)])

    group_col = "group_id" if "group_id" in rows.columns else "ticker"
    n_games = int(rows[group_col].nunique())

    def metrics(p):
        return {
            "auc": _safe_metric(roc_auc_score, y_true, p),
            "brier": _safe_metric(brier_score_loss, y_true, p),
            "logloss": _safe_metric(
                lambda yt, pp: log_loss(yt, np.clip(pp, 1e-6, 1 - 1e-6)), y_true, p),
        }

    m = {"model": metrics(model_prob), "current_leader": metrics(cur),
         "elo": metrics(elo), "diffusion": metrics(dif)}

    curve = lead_time_curve(pd.DataFrame({
        "minutes_elapsed": (sec_elapsed / 60.0).to_numpy(),
        "y_true": y_true,
        "model_prob": model_prob,
        "baseline_prob": cur,  # vs current-leader: the key early-game comparison
    }), bin_minutes=args.bin_minutes)

    beats_cur = m["model"]["brier"] < m["current_leader"]["brier"]
    beats_dif = m["model"]["brier"] < m["diffusion"]["brier"]
    # early bins = first third of the half (< 8 game-minutes)
    early = curve[curve["minute_bin"] < 8]
    early_edge = bool(len(early)) and bool((early["model_brier"] < early["baseline_brier"]).all())

    if n_games < 8:
        verdict = "TOO THIN TO CONCLUDE — fewer than 8 hold-out games; treat all numbers as directional only."
    elif beats_cur and beats_dif and early_edge:
        verdict = "PROMISING — model beats current-leader AND diffusion on Brier, and wins the early bins. Confirm on more game-days."
    elif beats_cur and beats_dif:
        verdict = "CAUTION — model beats the baselines overall but not clearly in the early bins (where the value would be). Marginal."
    else:
        verdict = "NO-GO — model does not beat the dumb baselines out of sample; no demonstrable edge."

    def row(name, d):
        return f"| {name} | {d['auc']:.4f} | {d['brier']:.4f} | {d['logloss']:.4f} |"

    lines = [
        "# Halftime-Leader Predictor — Out-of-Sample Evaluation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Model: `{args.model_path}` (target `{target}`)",
        f"TEST window: {args.dates_from} .. {args.dates_to} (out of sample vs train cutoff)",
        f"Rows: {len(rows)}  |  Hold-out games (effective N): **{n_games}**",
        "",
        "## Overall metrics (higher AUC better; lower Brier/log-loss better)",
        "",
        "| Predictor | ROC-AUC | Brier | Log-loss |",
        "|---|---|---|---|",
        row("**model**", m["model"]),
        row("current-leader", m["current_leader"]),
        row("ELO prior", m["elo"]),
        row("diffusion", m["diffusion"]),
        "",
        "## Lead-time curve (model vs current-leader, per game-minute bin)",
        "",
        "| minute_bin | n | model_brier | baseline_brier | model_acc | baseline_acc |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in curve.iterrows():
        lines.append(
            f"| {int(r['minute_bin'])} | {int(r['n'])} | {r['model_brier']:.4f} | "
            f"{r['baseline_brier']:.4f} | {r['model_acc']:.3f} | {r['baseline_acc']:.3f} |")
    lines += [
        "",
        "## Verdict",
        "",
        verdict,
        "",
        f"(effective N is the number of hold-out games, not the {len(rows)} ticks; "
        "ticks within a game are highly correlated.)",
        "",
    ]
    report = "\n".join(lines)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report)
    print(report)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

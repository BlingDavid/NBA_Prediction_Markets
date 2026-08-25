"""
Train a bootstrap in-game model on the first live-label target.

Current default target:
  - `label_home_up_5m`

Why this exists:
  - the project's main model is still a pregame home-win model
  - live snapshot rows are highly autocorrelated within the same contract
  - this trainer evaluates by contract group to avoid leakage
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED
from live_labeler import build_labeled_training_set
from live_training_matrix import build_in_game_training_matrix


DEFAULT_TARGET = "label_home_up_5m"
DEFAULT_MODEL_NAME = "live_home_up_5m_bootstrap"
LABEL_COLUMNS = {
    "label_final_home_win",
    "label_yes_mid_move_5m",
    "label_market_home_implied_move_5m",
    "label_yes_up_5m",
    "label_home_up_5m",
    "label_beats_close_yes",
    "label_beats_close_home",
}


def _as_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _safe_auc(y_true: pd.Series, y_prob: np.ndarray) -> float | None:
    if pd.Series(y_true).nunique(dropna=True) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def _safe_avg_precision(y_true: pd.Series, y_prob: np.ndarray) -> float | None:
    if pd.Series(y_true).nunique(dropna=True) < 2:
        return None
    return float(average_precision_score(y_true, y_prob))


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.25,
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _apply_final_predictions(
    scored_matrix: pd.DataFrame,
    labeled_rows: pd.DataFrame,
    final_model: Pipeline,
    feature_cols: list[str],
    base_rate: float,
    calibration_alpha: float,
) -> pd.DataFrame:
    """
    Populate prediction columns on scored_matrix honestly:
      - unlabeled rows → final-model predictions (shrunk)
      - labeled rows   → their out-of-fold predictions (held out during CV)
    `prediction_basis` flags which rows came from which source so downstream
    eval can't accidentally mix in-sample scores with out-of-sample ones.
    """
    scored = scored_matrix.copy()
    raw = final_model.predict_proba(scored[feature_cols])[:, 1]
    scored["pred_bootstrap_home_up_5m_raw"] = raw
    scored["pred_bootstrap_home_up_5m"] = _apply_shrinkage(raw, base_rate, calibration_alpha)
    scored["prediction_basis"] = "final_model"

    if not labeled_rows.empty and {"oof_prediction_raw", "oof_prediction"}.issubset(labeled_rows.columns):
        scored.loc[labeled_rows.index, "pred_bootstrap_home_up_5m_raw"] = labeled_rows["oof_prediction_raw"]
        scored.loc[labeled_rows.index, "pred_bootstrap_home_up_5m"] = labeled_rows["oof_prediction"]
        scored.loc[labeled_rows.index, "prediction_basis"] = "oof"

    return scored


def _select_feature_columns(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], dict]:
    selected = []
    dropped = {}

    for col in feature_cols:
        if col not in df.columns:
            dropped[col] = "missing_from_matrix"
            continue

        non_null = int(df[col].notna().sum())
        unique_non_null = int(df[col].dropna().nunique())

        if non_null == 0:
            dropped[col] = "all_missing"
            continue
        if unique_non_null <= 1:
            dropped[col] = "constant_or_single_value"
            continue

        selected.append(col)

    return selected, dropped


def _build_group_weights(groups: pd.Series) -> pd.Series:
    counts = groups.value_counts()
    weights = groups.map(lambda g: 1.0 / counts[g]).astype(float)
    return weights * (len(weights) / weights.sum())


def _binary_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = pd.Series(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "count": int(len(y_true)),
        "positive_rate": round(float(y_true.mean()), 6),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 6),
        "logloss": round(float(log_loss(y_true, y_prob, labels=[0, 1])), 6),
        "roc_auc": None if _safe_auc(y_true, y_prob) is None else round(_safe_auc(y_true, y_prob), 6),
        "average_precision": None
        if _safe_avg_precision(y_true, y_prob) is None
        else round(_safe_avg_precision(y_true, y_prob), 6),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def _apply_shrinkage(y_prob: np.ndarray, base_rate: float, alpha: float) -> np.ndarray:
    y_prob = np.asarray(y_prob, dtype=float)
    shrunk = base_rate + alpha * (y_prob - base_rate)
    return np.clip(shrunk, 1e-6, 1 - 1e-6)


def _tune_shrinkage(y_true: pd.Series, y_prob: np.ndarray) -> dict:
    base_rate = float(pd.Series(y_true).mean())
    candidates = []

    for alpha in np.linspace(0.0, 1.0, 21):
        calibrated = _apply_shrinkage(y_prob, base_rate, float(alpha))
        metrics = _binary_metrics(y_true, calibrated)
        candidates.append(
            {
                "alpha": round(float(alpha), 4),
                "metrics": metrics,
            }
        )

    best = min(candidates, key=lambda row: row["metrics"]["logloss"])
    return {
        "base_rate": round(base_rate, 6),
        "alpha": best["alpha"],
        "selection_metric": "logloss",
        "best_metrics": best["metrics"],
        "grid": candidates,
    }


def _lift_table(
    y_true: pd.Series,
    y_prob: np.ndarray,
    quantiles: tuple[float, ...] = (0.70, 0.80, 0.90, 0.95),
) -> list[dict]:
    y_true = pd.Series(y_true).astype(int)
    y_prob = pd.Series(np.asarray(y_prob, dtype=float), index=y_true.index)
    base_rate = float(y_true.mean())
    rows = []

    for quantile in quantiles:
        cutoff = float(y_prob.quantile(quantile))
        mask = y_prob >= cutoff
        bucket_rate = float(y_true[mask].mean()) if int(mask.sum()) else 0.0
        rows.append(
            {
                "quantile": round(float(quantile), 2),
                "rows": int(mask.sum()),
                "score_cutoff": round(cutoff, 6),
                "positive_rate": round(bucket_rate, 6),
                "lift_vs_base": round(bucket_rate / base_rate, 6) if base_rate > 0 else None,
            }
        )

    return rows


def _top_coefficients(model: Pipeline, feature_cols: list[str], top_n: int = 15) -> list[dict]:
    classifier = model.named_steps["classifier"]
    coefs = classifier.coef_[0]

    rows = [
        {
            "feature": feature,
            "coefficient": round(float(coef), 6),
            "abs_coefficient": round(float(abs(coef)), 6),
        }
        for feature, coef in zip(feature_cols, coefs)
    ]
    rows.sort(key=lambda row: row["abs_coefficient"], reverse=True)
    return rows[:top_n]


def filter_matrix_by_cutoff(
    matrix: pd.DataFrame,
    train_cutoff_date: str | None,
) -> pd.DataFrame:
    """Drop rows at or after *train_cutoff_date* (exclusive upper bound).

    Parameters
    ----------
    matrix : pd.DataFrame
        Must already have a timezone-aware ``captured_at`` column (datetime64[ns, UTC]).
    train_cutoff_date : str or None
        ISO date string, e.g. ``"2026-04-29"``.  If None, the matrix is
        returned unchanged — preserving backward-compatible default behaviour.

    Returns
    -------
    pd.DataFrame
        Filtered copy (or the original if cutoff is None).

    Notes
    -----
    Only ``captured_at`` is used for filtering.  Rows where ``captured_at``
    is NaT are retained (they could not be confirmed to be out-of-range).
    """
    if train_cutoff_date is None:
        return matrix
    cutoff = pd.Timestamp(train_cutoff_date, tz="UTC")
    mask = matrix["captured_at"].isna() | (matrix["captured_at"] < cutoff)
    return matrix[mask].copy()


def filter_first_half(matrix: pd.DataFrame, enabled: bool = True) -> pd.DataFrame:
    """Restrict to first-half ticks (period in {1, 2})."""
    if not enabled or "period" not in matrix.columns:
        return matrix
    period = pd.to_numeric(matrix["period"], errors="coerce")
    return matrix[period.isin([1, 2])].copy()


def build_bootstrap_dataset(
    target: str = DEFAULT_TARGET,
    horizon_minutes: int = 5,
    max_lag_minutes: int | None = None,
    only_resolved: bool = False,
    only_home_side: bool = True,
    require_game_state: bool = False,
    live_only: bool = False,
    train_cutoff_date: str | None = None,
    first_half_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    labeled = build_labeled_training_set(
        horizon_minutes=horizon_minutes,
        max_lag_minutes=max_lag_minutes,
        only_resolved=only_resolved,
    )
    matrix, feature_cols = build_in_game_training_matrix(
        labeled,
        only_home_side=only_home_side,
    )

    if matrix.empty:
        return matrix, pd.DataFrame(), [], {}

    matrix = matrix.copy()
    matrix["captured_at"] = _as_timestamp(matrix["captured_at"])
    matrix[target] = pd.to_numeric(matrix[target], errors="coerce")

    if require_game_state and "flag_has_game_state" in matrix.columns:
        matrix = matrix[matrix["flag_has_game_state"] == 1].copy()
    if live_only and "flag_status_live" in matrix.columns:
        matrix = matrix[matrix["flag_status_live"] == 1].copy()

    # Temporal hold-out: restrict BOTH the scored matrix and the labeled rows
    # to captured_at < train_cutoff_date.  This enforces a clean train/test
    # split without touching the deployed model or its outputs.
    matrix = filter_matrix_by_cutoff(matrix, train_cutoff_date)
    matrix = filter_first_half(matrix, enabled=first_half_only)

    labeled_rows = matrix[matrix[target].notna()].copy()
    if labeled_rows.empty:
        return matrix, labeled_rows, [], {}

    selected_features, dropped_features = _select_feature_columns(labeled_rows, feature_cols)
    if not selected_features:
        raise ValueError("No usable feature columns remain after filtering missing/constant columns.")

    labeled_rows["group_id"] = labeled_rows["ticker"].fillna(labeled_rows["game_key"])
    labeled_rows["sample_weight"] = _build_group_weights(labeled_rows["group_id"])

    metadata = {
        "target": target,
        "horizon_minutes": horizon_minutes,
        "max_lag_minutes": max_lag_minutes,
        "only_resolved": only_resolved,
        "only_home_side": only_home_side,
        "require_game_state": require_game_state,
        "live_only": live_only,
        "train_cutoff_date": train_cutoff_date,
        "first_half_only": first_half_only,
        "all_matrix_rows": int(len(matrix)),
        "labeled_rows": int(len(labeled_rows)),
        "groups": int(labeled_rows["group_id"].nunique()),
        "feature_count": int(len(selected_features)),
        "dropped_features": dropped_features,
    }
    return matrix, labeled_rows, selected_features, metadata


def train_bootstrap_model(
    target: str = DEFAULT_TARGET,
    horizon_minutes: int = 5,
    max_lag_minutes: int | None = None,
    only_resolved: bool = False,
    only_home_side: bool = True,
    require_game_state: bool = False,
    live_only: bool = False,
    train_cutoff_date: str | None = None,
    first_half_only: bool = False,
    model_name: str = DEFAULT_MODEL_NAME,
    models_dir: Path | None = None,
    outputs_dir: Path | None = None,
) -> dict:
    """Train and persist a bootstrap in-game model.

    Parameters
    ----------
    models_dir : Path or None
        Directory to write the .pkl artifact. Defaults to MODELS_DIR from config.
        Pass an alternative (e.g. ``models/candidates/``) to avoid overwriting
        the deployed model.
    outputs_dir : Path or None
        Directory to write report, OOF predictions, and scored-rows CSV.
        Defaults to OUTPUTS_DIR from config. ``latest_*`` symlink artifacts are
        only written when outputs_dir is None (i.e. the default path), so
        candidate runs do not pollute the live-trader's outputs.
    """
    matrix, labeled_rows, feature_cols, metadata = build_bootstrap_dataset(
        target=target,
        horizon_minutes=horizon_minutes,
        max_lag_minutes=max_lag_minutes,
        only_resolved=only_resolved,
        only_home_side=only_home_side,
        require_game_state=require_game_state,
        live_only=live_only,
        train_cutoff_date=train_cutoff_date,
        first_half_only=first_half_only,
    )

    if labeled_rows.empty:
        raise ValueError(f"No labeled rows available for target `{target}`.")

    X = labeled_rows[feature_cols]
    y = labeled_rows[target].astype(int)
    groups = labeled_rows["group_id"]
    sample_weight = labeled_rows["sample_weight"]

    positive_groups = int(labeled_rows.groupby("group_id")[target].max().sum())
    n_groups = int(groups.nunique())
    if y.nunique() < 2:
        raise ValueError(f"Target `{target}` has only one class in the labeled sample.")
    if positive_groups < 2 or n_groups < 4:
        raise ValueError(
            "Not enough grouped signal to train safely. Need at least 2 positive groups and 4 groups total."
        )

    n_splits = min(5, n_groups, positive_groups)
    n_splits = max(2, n_splits)
    cv = GroupKFold(n_splits=n_splits)

    oof_prob_raw = pd.Series(index=labeled_rows.index, dtype=float)
    fold_reports = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        model = _build_pipeline()
        model.fit(
            X_train,
            y_train,
            classifier__sample_weight=sample_weight.iloc[train_idx].values,
        )

        fold_prob = model.predict_proba(X_test)[:, 1]
        oof_prob_raw.iloc[test_idx] = fold_prob

        fold_groups = groups.iloc[test_idx]
        fold_reports.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_groups": int(groups.iloc[train_idx].nunique()),
                "test_groups": int(fold_groups.nunique()),
                "test_positive_rate": round(float(y_test.mean()), 6),
                "test_metrics_raw": _binary_metrics(y_test, fold_prob),
            }
        )

    oof_prob_raw = oof_prob_raw.sort_index()
    labeled_rows = labeled_rows.sort_index().copy()
    labeled_rows["oof_prediction_raw"] = oof_prob_raw

    calibration = _tune_shrinkage(y, oof_prob_raw.values)
    calibration_alpha = float(calibration["alpha"])
    base_rate = float(y.mean())
    oof_prob = _apply_shrinkage(oof_prob_raw.values, base_rate, calibration_alpha)
    labeled_rows["oof_prediction"] = oof_prob

    oof_metrics_raw = _binary_metrics(y, oof_prob_raw.values)
    oof_metrics = _binary_metrics(y, oof_prob)
    baseline_prob = np.repeat(base_rate, len(y))
    baseline_metrics = _binary_metrics(y, baseline_prob)
    lift_table = _lift_table(y, oof_prob)

    labeled_rows["oof_fold"] = np.nan
    for fold_report, (_, test_idx) in zip(fold_reports, cv.split(X, y, groups), strict=False):
        labeled_rows.iloc[test_idx, labeled_rows.columns.get_loc("oof_fold")] = fold_report["fold"]

    for fold_report in fold_reports:
        fold_mask = labeled_rows["oof_fold"] == fold_report["fold"]
        fold_report["test_metrics"] = _binary_metrics(
            labeled_rows.loc[fold_mask, target],
            labeled_rows.loc[fold_mask, "oof_prediction"],
        )

    final_model = _build_pipeline()
    final_model.fit(
        X,
        y,
        classifier__sample_weight=sample_weight.values,
    )

    scored_matrix = _apply_final_predictions(
        scored_matrix=matrix,
        labeled_rows=labeled_rows,
        final_model=final_model,
        feature_cols=feature_cols,
        base_rate=base_rate,
        calibration_alpha=calibration_alpha,
    )
    if target in scored_matrix.columns:
        scored_matrix["target_available"] = scored_matrix[target].notna().astype(int)

    trained_at = datetime.now(timezone.utc).isoformat()
    _models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    _outputs_dir = Path(outputs_dir) if outputs_dir is not None else OUTPUTS_DIR
    _models_dir.mkdir(parents=True, exist_ok=True)
    _outputs_dir.mkdir(parents=True, exist_ok=True)
    _using_default_outputs = outputs_dir is None

    model_path = _models_dir / f"{model_name}.pkl"
    report_path = _outputs_dir / f"{model_name}_report.json"
    oof_path = _outputs_dir / f"{model_name}_oof_predictions.csv"
    scored_path = _outputs_dir / f"{model_name}_scored_rows.csv"
    # latest_* artifacts are only written on the default output path so
    # candidate runs don't overwrite the live-trader's reference files.
    latest_report_path = OUTPUTS_DIR / "latest_live_bootstrap_report.json"
    latest_scored_path = OUTPUTS_DIR / "latest_live_bootstrap_scored_rows.csv"

    artifact = {
        "model": final_model,
        "feature_cols": feature_cols,
        "target": target,
        "trained_at": trained_at,
        # sklearn_version is stamped here so load_model() can verify that the
        # runtime sklearn matches the version used at pickle time. Mismatches
        # cause silent predict() failures (see paper_trader/scorer.py).
        "sklearn_version": sklearn.__version__,
        "metadata": metadata,
        "calibration": calibration,
        "oof_metrics_raw": oof_metrics_raw,
        "oof_metrics": oof_metrics,
        "baseline_metrics": baseline_metrics,
        "lift_table": lift_table,
        "fold_reports": fold_reports,
        "top_coefficients": _top_coefficients(final_model, feature_cols),
    }

    with model_path.open("wb") as fh:
        pickle.dump(artifact, fh)

    oof_export = labeled_rows[
        [
            "ticker",
            "game_key",
            "game_date",
            "captured_at",
            "home_team",
            "away_team",
            "group_id",
            target,
            "oof_prediction_raw",
            "oof_prediction",
        ]
    ].sort_values(["game_date", "ticker", "captured_at"])
    oof_export.to_csv(oof_path, index=False)

    scored_sorted = scored_matrix.sort_values(["game_date", "ticker", "captured_at"])
    scored_sorted.to_csv(scored_path, index=False)
    if _using_default_outputs:
        scored_sorted.to_csv(latest_scored_path, index=False)

    report = {
        "model_name": model_name,
        "model_path": str(model_path),
        "report_path": str(report_path),
        "target": target,
        "trained_at": trained_at,
        "metadata": metadata,
        "calibration": calibration,
        "oof_metrics_raw": oof_metrics_raw,
        "oof_metrics": oof_metrics,
        "baseline_metrics": baseline_metrics,
        "lift_table": lift_table,
        "fold_reports": fold_reports,
        "top_coefficients": artifact["top_coefficients"],
        "oof_predictions_path": str(oof_path),
        "scored_rows_path": str(scored_path),
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    if _using_default_outputs:
        latest_report_path.write_text(json.dumps(report, indent=2, default=str))

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Train the first bootstrap in-game model on live-labeled rows.",
    )
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Target label column to train.")
    parser.add_argument("--horizon", type=int, default=5, help="Forward label horizon in minutes.")
    parser.add_argument("--max-lag", type=int, default=None, help="Max allowed lag past horizon in minutes.")
    parser.add_argument(
        "--resolved-only",
        action="store_true",
        help="Train only on rows with resolved final outcomes available.",
    )
    parser.add_argument(
        "--all-sides",
        action="store_true",
        help="Keep both home/away contract rows instead of only home-side rows.",
    )
    parser.add_argument(
        "--require-game-state",
        action="store_true",
        help="Keep only rows that matched an ESPN game-state snapshot.",
    )
    parser.add_argument(
        "--live-only",
        action="store_true",
        help="Keep only rows where ESPN status_state is live (`in`).",
    )
    parser.add_argument(
        "--train-cutoff-date",
        type=str,
        default=None,
        help=(
            "ISO date string (e.g. '2026-04-29').  Only rows with "
            "captured_at BEFORE this date are used for training and scoring. "
            "Use to enforce a temporal hold-out (train on earlier dates, "
            "evaluate on later dates).  Default: None (use all rows)."
        ),
    )
    parser.add_argument("--first-half-only", action="store_true",
                        help="Train only on first-half ticks (period 1-2).")
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Artifact name prefix under models/ and outputs/.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help=(
            "Override the directory where the .pkl artifact is written. "
            "Default: models/ (from config.MODELS_DIR). "
            "Use models/candidates/ for non-destructive candidate runs."
        ),
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default=None,
        help=(
            "Override the directory where report/OOF/scored-rows are written. "
            "Default: outputs/ (from config.OUTPUTS_DIR). "
            "Use outputs/candidates/ for non-destructive candidate runs. "
            "latest_* files are only written when this is the default path."
        ),
    )
    args = parser.parse_args()

    try:
        report = train_bootstrap_model(
            target=args.target,
            horizon_minutes=args.horizon,
            max_lag_minutes=args.max_lag,
            only_resolved=args.resolved_only,
            only_home_side=not args.all_sides,
            require_game_state=args.require_game_state,
            live_only=args.live_only,
            train_cutoff_date=args.train_cutoff_date,
            first_half_only=args.first_half_only,
            model_name=args.model_name,
            models_dir=Path(args.models_dir) if args.models_dir else None,
            outputs_dir=Path(args.outputs_dir) if args.outputs_dir else None,
        )
    except ValueError as exc:
        print(f"  ERROR: {exc}")
        return

    print(f"  Model:                 {report['model_name']}")
    print(f"  Target:                {report['target']}")
    print(f"  Labeled rows:          {report['metadata']['labeled_rows']}")
    print(f"  Contract groups:       {report['metadata']['groups']}")
    print(f"  Features kept:         {report['metadata']['feature_count']}")
    print(f"  Calibration alpha:     {report['calibration']['alpha']}")
    print(f"  OOF ROC AUC:           {report['oof_metrics']['roc_auc']}")
    print(f"  OOF Avg Precision:     {report['oof_metrics']['average_precision']}")
    print(f"  OOF Brier:             {report['oof_metrics']['brier']}")
    print(f"  Baseline Brier:        {report['baseline_metrics']['brier']}")
    print(f"  Top-70% lift:          {report['lift_table'][0]['lift_vs_base']}")
    print(f"  Model saved:           {report['model_path']}")
    print(f"  Report saved:          {report.get('report_path', report['scored_rows_path'])}")
    print(f"  Scored rows saved:     {report['scored_rows_path']}")


if __name__ == "__main__":
    main()

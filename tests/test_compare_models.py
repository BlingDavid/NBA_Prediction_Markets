"""Unit tests for tools/compare_models.py.

Tests cover:
    - is_promotable() with synthetic metric dicts
    - build_comparison_table() with synthetic metric dicts
    - load_oof_csv() with a tiny in-memory CSV (tmpdir)
    - load_source() graceful skip on missing path

No large file I/O is performed.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure repo root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.compare_models import (
    BRIER_TOLERANCE,
    LOG_LOSS_TOLERANCE,
    OVERCONF_IMPROVE_MIN,
    OVERCONF_REGRESSION_MAX,
    ROC_AUC_TOLERANCE,
    build_comparison_table,
    is_promotable,
    load_oof_csv,
    load_source,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures: tiny synthetic metric dicts
# ═══════════════════════════════════════════════════════════════════════════

def _baseline_metrics() -> dict:
    """Plausible baseline metric dict."""
    return {
        "n_rows": 10000,
        "base_rate": 0.05,
        "roc_auc": 0.819,
        "average_precision": 0.12,
        "brier": 0.048,
        "brier_naive": 0.0475,
        "brier_skill": 0.01,
        "log_loss": 0.212,
        "e_up": 0.024,
        "e_down": -0.005,
        "ev_threshold_primary": 0.08,
        "n_above_ev_threshold": 500,
        "frac_above_ev_threshold": 0.05,
        "prec_above_ev_threshold": 0.15,
        "max_reliability_gap_above_p20": 0.06,
        "primary_spread": 0.01,
    }


def _better_candidate() -> dict:
    """Candidate that should be promotable: more EV+ rows, same or better everywhere."""
    return {
        "n_rows": 10000,
        "base_rate": 0.05,
        "roc_auc": 0.825,
        "average_precision": 0.13,
        "brier": 0.047,           # slightly better
        "brier_naive": 0.0475,
        "brier_skill": 0.011,
        "log_loss": 0.210,         # slightly better
        "e_up": 0.024,
        "e_down": -0.005,
        "ev_threshold_primary": 0.07,
        "n_above_ev_threshold": 600,  # more EV+ rows
        "frac_above_ev_threshold": 0.06,
        "prec_above_ev_threshold": 0.16,
        "max_reliability_gap_above_p20": 0.04,  # better calibration
        "primary_spread": 0.01,
    }


def _worse_candidate() -> dict:
    """Candidate that should NOT be promotable: worse on most metrics."""
    return {
        "n_rows": 10000,
        "base_rate": 0.05,
        "roc_auc": 0.810,          # worse by 0.009 > ROC_AUC_TOLERANCE
        "average_precision": 0.11,
        "brier": 0.052,            # worse by 0.004 > BRIER_TOLERANCE
        "brier_naive": 0.0475,
        "brier_skill": 0.005,
        "log_loss": 0.225,          # worse by 0.013 > LOG_LOSS_TOLERANCE
        "e_up": 0.024,
        "e_down": -0.005,
        "ev_threshold_primary": 0.09,
        "n_above_ev_threshold": 400,  # fewer EV+ rows
        "frac_above_ev_threshold": 0.04,
        "prec_above_ev_threshold": 0.13,
        "max_reliability_gap_above_p20": 0.10,  # worse by 0.04 > OVERCONF_REGRESSION_MAX
        "primary_spread": 0.01,
    }


def _only_calib_better() -> dict:
    """Candidate where money metric is flat but overconfidence shrinks enough."""
    base = _baseline_metrics().copy()
    # Same money metric
    base["n_above_ev_threshold"] = 500
    # Overconfidence shrinks by OVERCONF_IMPROVE_MIN + a bit
    base["max_reliability_gap_above_p20"] = 0.06 - OVERCONF_IMPROVE_MIN - 0.005
    # Everything else identical or slightly better
    return base


# ═══════════════════════════════════════════════════════════════════════════
# is_promotable tests
# ═══════════════════════════════════════════════════════════════════════════

class TestIsPromotable:

    def test_better_candidate_is_promotable(self):
        promotable, reasons = is_promotable(_baseline_metrics(), _better_candidate())
        assert promotable is True, f"Expected promotable. Reasons:\n" + "\n".join(reasons)

    def test_worse_candidate_is_not_promotable(self):
        promotable, reasons = is_promotable(_baseline_metrics(), _worse_candidate())
        assert promotable is False, f"Expected NOT promotable. Reasons:\n" + "\n".join(reasons)

    def test_returns_tuple(self):
        result = is_promotable(_baseline_metrics(), _better_candidate())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_second_element_is_list(self):
        _, reasons = is_promotable(_baseline_metrics(), _better_candidate())
        assert isinstance(reasons, list)

    def test_reasons_are_strings(self):
        _, reasons = is_promotable(_baseline_metrics(), _better_candidate())
        assert all(isinstance(r, str) for r in reasons)

    def test_five_conditions_produce_five_reasons(self):
        _, reasons = is_promotable(_baseline_metrics(), _better_candidate())
        # Each condition adds exactly one reason string
        assert len(reasons) == 5

    def test_money_metric_improvement_alone_satisfies_c1(self):
        """Even if calibration worsens slightly, more EV+ rows still passes C1."""
        cand = _baseline_metrics().copy()
        cand["n_above_ev_threshold"] = 600  # money metric up
        cand["max_reliability_gap_above_p20"] = 0.08  # gap slightly worse
        promotable, reasons = is_promotable(_baseline_metrics(), cand)
        # C1 should pass via money branch
        c1_reason = reasons[0]
        assert "PASS" in c1_reason
        assert "money" in c1_reason.lower() or "EV" in c1_reason

    def test_calibration_improvement_alone_satisfies_c1(self):
        """If money stays flat but overconfidence shrinks enough, C1 passes."""
        promotable, reasons = is_promotable(_baseline_metrics(), _only_calib_better())
        c1_reason = reasons[0]
        assert "PASS" in c1_reason

    def test_brier_regression_fails_c2(self):
        cand = _baseline_metrics().copy()
        cand["brier"] += BRIER_TOLERANCE + 0.001  # exceeds tolerance
        cand["n_above_ev_threshold"] = 600  # pass C1
        promotable, reasons = is_promotable(_baseline_metrics(), cand)
        assert promotable is False
        c2_reason = reasons[1]
        assert "FAIL" in c2_reason

    def test_logloss_regression_fails_c3(self):
        cand = _baseline_metrics().copy()
        cand["log_loss"] += LOG_LOSS_TOLERANCE + 0.001
        cand["n_above_ev_threshold"] = 600  # pass C1
        promotable, reasons = is_promotable(_baseline_metrics(), cand)
        assert promotable is False
        c3_reason = reasons[2]
        assert "FAIL" in c3_reason

    def test_roc_auc_drop_fails_c4(self):
        cand = _baseline_metrics().copy()
        cand["roc_auc"] -= ROC_AUC_TOLERANCE + 0.001
        cand["n_above_ev_threshold"] = 600  # pass C1
        promotable, reasons = is_promotable(_baseline_metrics(), cand)
        assert promotable is False
        c4_reason = reasons[3]
        assert "FAIL" in c4_reason

    def test_calibration_regression_fails_c5(self):
        cand = _baseline_metrics().copy()
        cand["max_reliability_gap_above_p20"] = (
            _baseline_metrics()["max_reliability_gap_above_p20"]
            + OVERCONF_REGRESSION_MAX + 0.005
        )
        cand["n_above_ev_threshold"] = 600  # pass C1
        promotable, reasons = is_promotable(_baseline_metrics(), cand)
        assert promotable is False
        c5_reason = reasons[4]
        assert "FAIL" in c5_reason

    def test_nan_metrics_skip_gracefully(self):
        """If a metric is NaN (unavailable), that condition should be skipped (PASS)."""
        cand = _baseline_metrics().copy()
        cand["brier"] = float("nan")
        cand["log_loss"] = float("nan")
        cand["roc_auc"] = float("nan")
        cand["max_reliability_gap_above_p20"] = float("nan")
        cand["n_above_ev_threshold"] = 600  # pass C1
        promotable, reasons = is_promotable(_baseline_metrics(), cand)
        # C2/C3/C4/C5 should be skipped (PASS) when candidate has NaN
        assert "PASS" in reasons[1]  # brier skipped
        assert "PASS" in reasons[2]  # log_loss skipped
        assert "PASS" in reasons[3]  # roc_auc skipped
        assert "PASS" in reasons[4]  # calib skipped
        assert promotable is True

    def test_none_metrics_skip_gracefully(self):
        """None values for a metric should be treated as unavailable."""
        cand = _baseline_metrics().copy()
        cand["brier"] = None
        cand["n_above_ev_threshold"] = 600  # pass C1
        promotable, reasons = is_promotable(_baseline_metrics(), cand)
        assert "PASS" in reasons[1]  # brier comparison skipped

    def test_identical_baseline_and_candidate_not_promotable(self):
        """Identical metrics: money metric doesn't improve and gap doesn't shrink."""
        base = _baseline_metrics()
        promotable, _ = is_promotable(base, base.copy())
        assert promotable is False

    def test_custom_tolerances_accepted(self):
        """Tolerances can be overridden per call."""
        cand = _baseline_metrics().copy()
        cand["brier"] += 0.002  # would fail default BRIER_TOLERANCE=0.0005
        cand["n_above_ev_threshold"] = 600
        # With a loose tolerance, should pass C2
        promotable, reasons = is_promotable(
            _baseline_metrics(), cand, brier_tolerance=0.005
        )
        assert "PASS" in reasons[1]


# ═══════════════════════════════════════════════════════════════════════════
# build_comparison_table tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildComparisonTable:

    def test_returns_dataframe(self):
        df = build_comparison_table(_baseline_metrics(), [("candidate", _better_candidate())])
        assert isinstance(df, pd.DataFrame)

    def test_has_metric_and_baseline_columns(self):
        df = build_comparison_table(_baseline_metrics(), [])
        assert "Metric" in df.columns
        assert "Baseline" in df.columns

    def test_candidate_columns_added(self):
        df = build_comparison_table(
            _baseline_metrics(),
            [("isotonic", _better_candidate()), ("momentum", _worse_candidate())],
        )
        assert "isotonic" in df.columns
        assert "momentum" in df.columns
        assert "Δ isotonic" in df.columns
        assert "Δ momentum" in df.columns

    def test_money_metric_row_present(self):
        df = build_comparison_table(_baseline_metrics(), [("cand", _better_candidate())])
        metrics_col = df["Metric"].tolist()
        assert any("EV+" in m for m in metrics_col), f"EV+ row missing. Got: {metrics_col}"

    def test_roc_auc_row_present(self):
        df = build_comparison_table(_baseline_metrics(), [("cand", _better_candidate())])
        metrics_col = df["Metric"].tolist()
        assert any("ROC" in m for m in metrics_col)

    def test_brier_row_present(self):
        df = build_comparison_table(_baseline_metrics(), [("cand", _better_candidate())])
        assert any("Brier" in m for m in df["Metric"].tolist())

    def test_log_loss_row_present(self):
        df = build_comparison_table(_baseline_metrics(), [("cand", _better_candidate())])
        assert any("Log" in m for m in df["Metric"].tolist())

    def test_reliability_gap_row_present(self):
        df = build_comparison_table(_baseline_metrics(), [("cand", _better_candidate())])
        assert any("Reliability" in m for m in df["Metric"].tolist())

    def test_no_candidates_still_produces_table(self):
        df = build_comparison_table(_baseline_metrics(), [])
        assert len(df) > 0
        assert "Metric" in df.columns
        assert "Baseline" in df.columns

    def test_delta_column_shows_improvement_arrow(self):
        """For ROC-AUC (higher is better), a positive delta should show ↑."""
        df = build_comparison_table(_baseline_metrics(), [("cand", _better_candidate())])
        roc_row = df[df["Metric"].str.contains("ROC")]
        assert len(roc_row) == 1
        delta_val = roc_row["Δ cand"].iloc[0]
        assert "↑" in delta_val, f"Expected ↑ in delta. Got: {delta_val}"

    def test_delta_column_shows_regression_arrow(self):
        """For Brier (lower is better), a positive delta (worsening) should show ↓."""
        df = build_comparison_table(_baseline_metrics(), [("cand", _worse_candidate())])
        brier_row = df[df["Metric"].str.contains("Brier Score")]
        assert len(brier_row) == 1
        delta_val = brier_row["Δ cand"].iloc[0]
        assert "↓" in delta_val, f"Expected ↓ in delta for worsening Brier. Got: {delta_val}"

    def test_missing_metric_shows_na(self):
        """If a metric key is absent from a candidate dict, show N/A."""
        cand = _better_candidate().copy()
        del cand["roc_auc"]
        df = build_comparison_table(_baseline_metrics(), [("cand", cand)])
        roc_row = df[df["Metric"].str.contains("ROC")]
        assert roc_row["cand"].iloc[0] == "N/A"


# ═══════════════════════════════════════════════════════════════════════════
# load_oof_csv tests
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadOofCsv:

    def _write_oof_csv(self, tmp_path: Path, n: int = 100) -> Path:
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "label_home_up_5m": rng.integers(0, 2, n).astype(np.int8),
            "oof_prediction": rng.uniform(0.01, 0.5, n).astype(np.float32),
            "extra_col": "ignored",
        })
        p = tmp_path / "oof_test.csv"
        df.to_csv(p, index=False)
        return p

    def test_loads_correct_shapes(self, tmp_path):
        p = self._write_oof_csv(tmp_path, n=200)
        y_true, y_prob = load_oof_csv(p)
        assert len(y_true) == 200
        assert len(y_prob) == 200

    def test_labels_are_0_or_1(self, tmp_path):
        p = self._write_oof_csv(tmp_path)
        y_true, _ = load_oof_csv(p)
        assert set(np.unique(y_true)).issubset({0, 1})

    def test_probs_in_0_1(self, tmp_path):
        p = self._write_oof_csv(tmp_path)
        _, y_prob = load_oof_csv(p)
        assert (y_prob >= 0.0).all() and (y_prob <= 1.0).all()

    def test_missing_col_raises_key_error(self, tmp_path):
        df = pd.DataFrame({"label_home_up_5m": [0, 1], "wrong_col": [0.1, 0.2]})
        p = tmp_path / "bad.csv"
        df.to_csv(p, index=False)
        with pytest.raises((KeyError, ValueError)):
            load_oof_csv(p)

    def test_custom_oof_col(self, tmp_path):
        df = pd.DataFrame({
            "label_home_up_5m": [0, 1, 0],
            "recalibrated_prob": [0.1, 0.8, 0.2],
        })
        p = tmp_path / "custom.csv"
        df.to_csv(p, index=False)
        y_true, y_prob = load_oof_csv(p, oof_col="recalibrated_prob")
        assert len(y_true) == 3


# ═══════════════════════════════════════════════════════════════════════════
# load_source tests
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadSource:

    def test_missing_path_returns_none(self, capsys):
        result = load_source(Path("/nonexistent/path/to/oof.csv"), label="test_model")
        assert result is None
        captured = capsys.readouterr()
        assert "SKIP" in captured.out

    def test_existing_file_returns_arrays(self, tmp_path):
        df = pd.DataFrame({
            "label_home_up_5m": [0, 1, 0, 1],
            "oof_prediction": [0.1, 0.7, 0.2, 0.8],
        })
        p = tmp_path / "oof.csv"
        df.to_csv(p, index=False)
        result = load_source(p, label="test")
        assert result is not None
        y_true, y_prob = result
        assert len(y_true) == 4

    def test_bad_column_returns_none(self, tmp_path, capsys):
        df = pd.DataFrame({
            "label_home_up_5m": [0, 1],
            "wrong_col": [0.1, 0.2],
        })
        p = tmp_path / "bad.csv"
        df.to_csv(p, index=False)
        result = load_source(p, label="bad_model")
        assert result is None
        captured = capsys.readouterr()
        assert "SKIP" in captured.out

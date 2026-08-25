"""Unit tests for the temporal hold-out filter functions added in the
decisive-edge experiment.

Covers:
  - live_bootstrap_model.filter_matrix_by_cutoff
  - tools.build_pooled_means.filter_scored_rows_by_date
  - tools.cost_replay._available_feature_files  (date-range logic)

All tests use tiny synthetic DataFrames / fixture paths — no CSVs are
read and no models are loaded.  Tests are deterministic and fast.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from live_bootstrap_model import filter_matrix_by_cutoff
from tools.build_pooled_means import filter_scored_rows_by_date
from tools.cost_replay import _available_feature_files, _CORRUPT_FILES


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_matrix(dates: list[str]) -> pd.DataFrame:
    """Build a minimal matrix with timezone-aware captured_at timestamps."""
    return pd.DataFrame(
        {"captured_at": pd.to_datetime(dates, utc=True), "value": range(len(dates))}
    )


def _make_scored_rows(dates: list[str]) -> pd.DataFrame:
    """Build a minimal scored_rows-like frame with string captured_at."""
    labels = ([1, 0] * (len(dates) // 2 + 1))[:len(dates)]
    return pd.DataFrame({"captured_at": dates, "label_home_up_5m": labels})


# ═══════════════════════════════════════════════════════════════════════════
# filter_matrix_by_cutoff (live_bootstrap_model)
# ═══════════════════════════════════════════════════════════════════════════

class TestFilterMatrixByCutoff:
    def test_none_cutoff_returns_unchanged(self):
        df = _make_matrix(["2026-04-28", "2026-04-29", "2026-04-30"])
        out = filter_matrix_by_cutoff(df, None)
        assert len(out) == 3

    def test_cutoff_excludes_on_and_after(self):
        df = _make_matrix(["2026-04-27", "2026-04-28", "2026-04-29", "2026-04-30"])
        out = filter_matrix_by_cutoff(df, "2026-04-29")
        # Only dates strictly before 2026-04-29 are kept
        assert len(out) == 2
        assert list(out["value"]) == [0, 1]

    def test_cutoff_keeps_rows_strictly_before(self):
        df = _make_matrix(["2026-04-28 23:59:59", "2026-04-29 00:00:00"])
        out = filter_matrix_by_cutoff(df, "2026-04-29")
        assert len(out) == 1

    def test_cutoff_empty_result_when_all_after(self):
        df = _make_matrix(["2026-05-01", "2026-05-02"])
        out = filter_matrix_by_cutoff(df, "2026-04-29")
        assert len(out) == 0

    def test_nat_rows_are_retained(self):
        """Rows with NaT captured_at cannot be confirmed out-of-range — keep them."""
        df = pd.DataFrame({
            "captured_at": pd.to_datetime(["2026-04-29", None], utc=True),
            "value": [1, 2],
        })
        out = filter_matrix_by_cutoff(df, "2026-04-29")
        # 2026-04-29 itself is excluded, NaT row is kept
        assert len(out) == 1
        assert out["value"].iloc[0] == 2

    def test_returns_copy_not_view(self):
        df = _make_matrix(["2026-04-28"])
        out = filter_matrix_by_cutoff(df, "2026-04-29")
        out["value"] = 999
        # Original must be unchanged
        assert df["value"].iloc[0] != 999

    def test_all_rows_kept_when_before_earliest_date(self):
        df = _make_matrix(["2026-04-28", "2026-04-29"])
        out = filter_matrix_by_cutoff(df, "2026-12-31")
        assert len(out) == 2


# ═══════════════════════════════════════════════════════════════════════════
# filter_scored_rows_by_date (tools.build_pooled_means)
# ═══════════════════════════════════════════════════════════════════════════

class TestFilterScoredRowsByDate:
    def test_none_dates_before_returns_unchanged(self):
        df = _make_scored_rows(["2026-04-28", "2026-04-29", "2026-04-30"])
        out = filter_scored_rows_by_date(df, None)
        assert len(out) == 3

    def test_cutoff_drops_rows_on_or_after(self):
        df = _make_scored_rows(["2026-04-27", "2026-04-28", "2026-04-29", "2026-04-30"])
        out = filter_scored_rows_by_date(df, "2026-04-29")
        assert len(out) == 2

    def test_no_captured_at_column_returns_unchanged(self):
        df = pd.DataFrame({"label_home_up_5m": [0, 1]})
        out = filter_scored_rows_by_date(df, "2026-04-29")
        assert len(out) == 2

    def test_nat_rows_retained(self):
        df = pd.DataFrame({
            "captured_at": ["2026-04-29", None, "2026-04-28"],
            "label_home_up_5m": [1, 0, 1],
        })
        out = filter_scored_rows_by_date(df, "2026-04-29")
        # 2026-04-29 excluded, NaT retained, 2026-04-28 retained → 2 rows
        assert len(out) == 2

    def test_returns_copy_not_view(self):
        df = _make_scored_rows(["2026-04-28"])
        out = filter_scored_rows_by_date(df, "2026-04-29")
        out.iloc[0, 0] = "mutated"
        assert df.iloc[0, 0] != "mutated"

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame({"captured_at": pd.Series([], dtype="object"), "label_home_up_5m": []})
        out = filter_scored_rows_by_date(df, "2026-04-29")
        assert len(out) == 0


# ═══════════════════════════════════════════════════════════════════════════
# _available_feature_files date-range logic (tools.cost_replay)
# ═══════════════════════════════════════════════════════════════════════════

class TestAvailableFeatureFiles:
    """Create a tiny temp directory with stub CSV filenames and test filtering."""

    @pytest.fixture()
    def tmp_features_dir(self, tmp_path):
        """Populate a temp dir with live_features_<date>.csv stubs."""
        dates = [
            "2026-04-28",
            "2026-04-29",
            "2026-04-30",
            "2026-05-01",
            "2026-05-05",
        ]
        for d in dates:
            (tmp_path / f"live_features_{d}.csv").write_text("col\n1\n")
        # Also write a corrupt-named file to test exclusion
        for name in _CORRUPT_FILES:
            (tmp_path / name).write_text("col\n1\n")
        return tmp_path

    def test_no_filters_returns_all_non_corrupt(self, tmp_features_dir):
        files = _available_feature_files(tmp_features_dir)
        dates = [f.stem.replace("live_features_", "") for f in files]
        assert "2026-04-28" in dates
        assert "2026-05-05" in dates
        for corrupt in _CORRUPT_FILES:
            assert corrupt not in [f.name for f in files]

    def test_dates_from_excludes_earlier(self, tmp_features_dir):
        files = _available_feature_files(tmp_features_dir, dates_from="2026-04-29")
        dates = [f.stem.replace("live_features_", "") for f in files]
        assert "2026-04-28" not in dates
        assert "2026-04-29" in dates
        assert "2026-05-05" in dates

    def test_dates_to_excludes_later(self, tmp_features_dir):
        files = _available_feature_files(tmp_features_dir, dates_to="2026-04-30")
        dates = [f.stem.replace("live_features_", "") for f in files]
        assert "2026-04-28" in dates
        assert "2026-04-29" in dates
        assert "2026-04-30" in dates
        assert "2026-05-01" not in dates

    def test_dates_from_and_dates_to_narrows_range(self, tmp_features_dir):
        files = _available_feature_files(
            tmp_features_dir, dates_from="2026-04-29", dates_to="2026-04-30"
        )
        dates = [f.stem.replace("live_features_", "") for f in files]
        assert dates == ["2026-04-29", "2026-04-30"]

    def test_no_files_in_range_returns_empty(self, tmp_features_dir):
        files = _available_feature_files(
            tmp_features_dir, dates_from="2026-06-01", dates_to="2026-06-30"
        )
        assert files == []

    def test_result_is_sorted(self, tmp_features_dir):
        files = _available_feature_files(tmp_features_dir)
        names = [f.name for f in files]
        assert names == sorted(names)

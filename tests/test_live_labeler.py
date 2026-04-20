"""
Tests for live_labeler.attach_forward_market_labels.

These characterize the existing per-row loop behavior so the upcoming
vectorized (pd.merge_asof) refactor can be verified against the same
contract: same filter semantics, same label columns, same NA handling.
"""

from __future__ import annotations

import pandas as pd
import pytest

from live_labeler import attach_forward_market_labels


def _make_row(ticker: str, t: pd.Timestamp, **fields) -> dict:
    row = {
        "ticker": ticker,
        "captured_at": t,
        "yes_mid": 0.5,
        "yes_ask": 0.52,
        "market_home_implied": 0.5,
        "last_price": 0.51,
        "volume": 100,
    }
    row.update(fields)
    return row


def _base_time() -> pd.Timestamp:
    return pd.Timestamp("2025-04-18T19:00:00Z")


class TestAttachForwardMarketLabels:
    def test_empty_input_returns_empty_copy(self):
        empty = pd.DataFrame(columns=[
            "ticker", "captured_at", "yes_mid", "yes_ask",
            "market_home_implied", "last_price", "volume",
        ])
        result = attach_forward_market_labels(empty, horizon_minutes=5)
        assert result.empty

    def test_single_ticker_within_horizon_populates_labels(self):
        t0 = _base_time()
        df = pd.DataFrame([
            _make_row("A", t0,                         yes_mid=0.50, yes_ask=0.52, market_home_implied=0.50, last_price=0.51, volume=100),
            _make_row("A", t0 + pd.Timedelta(minutes=2), yes_mid=0.55, yes_ask=0.57, market_home_implied=0.55, last_price=0.56, volume=110),
            _make_row("A", t0 + pd.Timedelta(minutes=5), yes_mid=0.60, yes_ask=0.62, market_home_implied=0.60, last_price=0.61, volume=130),
        ])

        out = attach_forward_market_labels(df, horizon_minutes=5)
        row0 = out.sort_values("captured_at").iloc[0]

        # Row 0 (t=0) must match row at t=5 (first snapshot with captured_at >= t+5m).
        assert row0["label_yes_mid_5m"] == pytest.approx(0.60)
        assert row0["label_yes_ask_5m"] == pytest.approx(0.62)
        assert row0["label_market_home_implied_5m"] == pytest.approx(0.60)
        assert row0["label_last_price_5m"] == pytest.approx(0.61)
        assert row0["label_volume_5m"] == pytest.approx(130)
        assert row0["label_forward_delay_min_5m"] == pytest.approx(0.0)

    def test_future_beyond_max_lag_is_na(self):
        t0 = _base_time()
        df = pd.DataFrame([
            _make_row("A", t0),
            # Next snapshot is 100 minutes out — far beyond default max_lag (10m for horizon=5).
            _make_row("A", t0 + pd.Timedelta(minutes=100)),
        ])

        out = attach_forward_market_labels(df, horizon_minutes=5)
        row0 = out.sort_values("captured_at").iloc[0]

        assert pd.isna(row0["label_yes_mid_5m"])
        assert pd.isna(row0["label_forward_delay_min_5m"])
        assert pd.isna(row0["label_future_captured_at_5m"])

    def test_no_future_snapshot_is_na(self):
        df = pd.DataFrame([_make_row("A", _base_time())])
        out = attach_forward_market_labels(df, horizon_minutes=5)
        assert pd.isna(out.iloc[0]["label_yes_mid_5m"])
        assert pd.isna(out.iloc[0]["label_forward_delay_min_5m"])

    def test_last_row_has_no_future(self):
        t0 = _base_time()
        df = pd.DataFrame([
            _make_row("A", t0),
            _make_row("A", t0 + pd.Timedelta(minutes=2)),
        ])

        out = attach_forward_market_labels(df, horizon_minutes=5)
        # Both rows lack a future snapshot at >= captured_at + 5m.
        assert out["label_yes_mid_5m"].isna().all()

    def test_multiple_tickers_isolate(self):
        t0 = _base_time()
        df = pd.DataFrame([
            _make_row("A", t0,                         yes_mid=0.10, yes_ask=0.12, market_home_implied=0.10),
            _make_row("B", t0,                         yes_mid=0.90, yes_ask=0.92, market_home_implied=0.90),
            _make_row("A", t0 + pd.Timedelta(minutes=5), yes_mid=0.20, yes_ask=0.22, market_home_implied=0.20),
            _make_row("B", t0 + pd.Timedelta(minutes=5), yes_mid=0.80, yes_ask=0.82, market_home_implied=0.80),
        ])

        out = attach_forward_market_labels(df, horizon_minutes=5).sort_values(["ticker", "captured_at"])

        # Row 0 of ticker A must pull from ticker A's future, not ticker B's.
        a_row0 = out[(out["ticker"] == "A")].iloc[0]
        b_row0 = out[(out["ticker"] == "B")].iloc[0]

        assert a_row0["label_yes_mid_5m"] == pytest.approx(0.20)
        assert b_row0["label_yes_mid_5m"] == pytest.approx(0.80)

    def test_exact_horizon_boundary_matches(self):
        """A snapshot at exactly t+horizon is a valid match (>= target_time)."""
        t0 = _base_time()
        df = pd.DataFrame([
            _make_row("A", t0,                         yes_mid=0.30),
            _make_row("A", t0 + pd.Timedelta(minutes=5), yes_mid=0.40),
        ])

        out = attach_forward_market_labels(df, horizon_minutes=5)
        row0 = out.sort_values("captured_at").iloc[0]

        assert row0["label_yes_mid_5m"] == pytest.approx(0.40)
        assert row0["label_forward_delay_min_5m"] == pytest.approx(0.0)

    def test_move_columns_equal_future_minus_current(self):
        t0 = _base_time()
        df = pd.DataFrame([
            _make_row("A", t0,                         yes_mid=0.50, yes_ask=0.52, market_home_implied=0.50, last_price=0.51, volume=100),
            _make_row("A", t0 + pd.Timedelta(minutes=5), yes_mid=0.58, yes_ask=0.60, market_home_implied=0.58, last_price=0.59, volume=140),
        ])

        out = attach_forward_market_labels(df, horizon_minutes=5)
        row0 = out.sort_values("captured_at").iloc[0]

        assert row0["label_yes_mid_move_5m"] == pytest.approx(0.08)
        assert row0["label_yes_ask_move_5m"] == pytest.approx(0.08)
        assert row0["label_market_home_implied_move_5m"] == pytest.approx(0.08)
        assert row0["label_last_price_move_5m"] == pytest.approx(0.08)
        assert row0["label_volume_change_5m"] == pytest.approx(40)

    def test_home_up_and_yes_up_flags(self):
        """home_up/yes_up are 1 when move > eps (0.001), 0 otherwise, NA when no future."""
        t0 = _base_time()
        df = pd.DataFrame([
            # Row 0: market moves up for home and yes
            _make_row("A", t0,                         yes_mid=0.50, market_home_implied=0.50),
            # Row 1 @ t=5m: target for row 0 (up)
            _make_row("A", t0 + pd.Timedelta(minutes=5), yes_mid=0.60, market_home_implied=0.60),
            # Row 2 @ t=10m: target for row 1 (down from row 1)
            _make_row("A", t0 + pd.Timedelta(minutes=10), yes_mid=0.55, market_home_implied=0.55),
        ])

        out = attach_forward_market_labels(df, horizon_minutes=5).sort_values("captured_at").reset_index(drop=True)

        assert out.loc[0, "label_home_up_5m"] == 1
        assert out.loc[0, "label_yes_up_5m"] == 1
        assert out.loc[1, "label_home_up_5m"] == 0
        assert out.loc[1, "label_yes_up_5m"] == 0
        # Row 2 has no future within max_lag → NA
        assert pd.isna(out.loc[2, "label_home_up_5m"])
        assert pd.isna(out.loc[2, "label_yes_up_5m"])

    def test_custom_max_lag_truncates(self):
        """Explicit max_lag_minutes=3 rejects a future snapshot 4 min after target."""
        t0 = _base_time()
        df = pd.DataFrame([
            _make_row("A", t0),
            # target_time = t0+5m; this future is at t0+9m → delay=4m > 3m max_lag
            _make_row("A", t0 + pd.Timedelta(minutes=9)),
        ])

        out = attach_forward_market_labels(df, horizon_minutes=5, max_lag_minutes=3)
        row0 = out.sort_values("captured_at").iloc[0]

        assert pd.isna(row0["label_yes_mid_5m"])

"""
Tests for momentum/velocity features:
  - home_score_run_3min
  - yes_mid_velocity
  - pace_delta

Two dimensions of coverage:
  1. Feature correctness on tiny synthetic sequences (unit tests).
  2. Train/live PARITY: the same sorted input row yields identical feature
     values whether computed through the training-matrix path
     (live_training_matrix.compute_momentum_features) or through the
     live-scorer path (paper_trader.scorer.engineer_features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from live_training_matrix import compute_momentum_features, SCORE_RUN_GAME_SECONDS
from paper_trader.scorer import engineer_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ticker_rows(
    n: int = 10,
    ticker: str = "T1",
    start_home: float = 0.0,
    start_away: float = 0.0,
    home_pts_per_tick: float = 2.0,
    away_pts_per_tick: float = 1.5,
    start_seconds: float = 60.0,
    seconds_per_tick: float = 20.0,
    start_mid: float = 0.50,
    mid_delta_per_tick: float = 0.005,
    pregame_total: float = 220.0,
) -> pd.DataFrame:
    """Build a synthetic sorted sequence of feature rows for one ticker."""
    rows = []
    for i in range(n):
        captured_at = pd.Timestamp("2026-04-20T19:00:00Z") + pd.Timedelta(seconds=i * seconds_per_tick)
        seconds_elapsed = start_seconds + i * seconds_per_tick
        home_score = start_home + i * home_pts_per_tick
        away_score = start_away + i * away_pts_per_tick
        yes_mid = start_mid + i * mid_delta_per_tick
        total_points = home_score + away_score
        rows.append({
            "captured_at": captured_at,
            "ticker": ticker,
            "game_key": "G1",
            "home_score": home_score,
            "away_score": away_score,
            "score_margin_home": home_score - away_score,
            "total_points": total_points,
            "yes_bid": yes_mid - 0.01,
            "yes_ask": yes_mid + 0.01,
            "yes_mid": yes_mid,
            "no_bid": 1 - yes_mid - 0.01,
            "no_ask": 1 - yes_mid + 0.01,
            "last_price": yes_mid,
            "market_home_implied": yes_mid,
            "seconds_elapsed": seconds_elapsed,
            "seconds_left_in_period": max(0, 720 - seconds_elapsed),
            "period": 1,
            "volume": 1000.0,
            "open_interest": 500.0,
            "yes_depth_notional_3": 200.0,
            "yes_depth_notional_5": 300.0,
            "no_depth_notional_3": 190.0,
            "no_depth_notional_5": 290.0,
            "yes_weighted_price_3": yes_mid + 0.002,
            "no_weighted_price_3": 1 - yes_mid - 0.002,
            "espn_home_implied": yes_mid,
            "espn_away_implied": 1 - yes_mid,
            "market_consensus_home": yes_mid,
            "pregame_home_win_prob": 0.50,
            "pregame_away_win_prob": 0.50,
            "pregame_spread": 0.0,
            "pregame_total": pregame_total,
            "pregame_edge_home": 0.0,
            "consensus_gap_home": 0.0,
            "status_state": "in",
        })
    df = pd.DataFrame(rows)
    df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
    return df


# ---------------------------------------------------------------------------
# Unit tests: compute_momentum_features correctness
# ---------------------------------------------------------------------------

class TestHomeScoringRun:
    def test_nan_when_only_one_row(self):
        df = _make_ticker_rows(n=1)
        out = compute_momentum_features(df)
        assert np.isnan(out["home_score_run_3min"].iloc[0])

    def test_run_increases_when_home_dominates(self):
        """Home scores 4 pts/tick, away scores 1 pt/tick → net run should be positive."""
        df = _make_ticker_rows(n=15, home_pts_per_tick=4.0, away_pts_per_tick=1.0,
                               start_seconds=120.0, seconds_per_tick=20.0)
        out = compute_momentum_features(df)
        # Rows after index 0 should have a positive run
        later_runs = out["home_score_run_3min"].dropna()
        assert len(later_runs) > 0
        assert (later_runs > 0).all()

    def test_run_negative_when_away_dominates(self):
        """Away scores 4 pts/tick, home scores 1 pt/tick → net run should be negative."""
        df = _make_ticker_rows(n=15, home_pts_per_tick=1.0, away_pts_per_tick=4.0,
                               start_seconds=120.0, seconds_per_tick=20.0)
        out = compute_momentum_features(df)
        later_runs = out["home_score_run_3min"].dropna()
        assert len(later_runs) > 0
        assert (later_runs < 0).all()

    def test_run_zero_when_tied_scoring(self):
        """Equal scoring rate → net run should be 0."""
        df = _make_ticker_rows(n=15, home_pts_per_tick=2.0, away_pts_per_tick=2.0,
                               start_seconds=120.0, seconds_per_tick=20.0)
        out = compute_momentum_features(df)
        later_runs = out["home_score_run_3min"].dropna()
        assert len(later_runs) > 0
        np.testing.assert_allclose(later_runs.values, 0.0, atol=1e-9)

    def test_run_bounded_to_trailing_window(self):
        """With a 3-min window at 20s/tick, run should only reflect last ~9 ticks."""
        # Home scores 4 pts in ticks 0-4 (early burst), then 0 in ticks 5-14
        rows_part1 = _make_ticker_rows(n=5, home_pts_per_tick=4.0, away_pts_per_tick=0.0,
                                       start_seconds=120.0, seconds_per_tick=20.0)
        rows_part2 = _make_ticker_rows(n=10, home_pts_per_tick=0.0, away_pts_per_tick=0.0,
                                       start_seconds=220.0, seconds_per_tick=20.0,
                                       start_home=float(rows_part1["home_score"].iloc[-1]),
                                       start_away=float(rows_part1["away_score"].iloc[-1]))
        df = pd.concat([rows_part1, rows_part2], ignore_index=True)
        df = df.sort_values(["ticker", "captured_at"]).reset_index(drop=True)
        out = compute_momentum_features(df)
        # The last row's run should be 0 (no scoring in the trailing ~3 min window)
        last_run = out["home_score_run_3min"].iloc[-1]
        assert last_run == pytest.approx(0.0, abs=1e-9)

    def test_nan_when_scores_missing(self):
        df = _make_ticker_rows(n=5)
        df["home_score"] = np.nan
        out = compute_momentum_features(df)
        assert out["home_score_run_3min"].isna().all()


class TestYesMidVelocity:
    def test_nan_for_first_three_rows(self):
        """Velocity needs shift(3), so first 3 rows should be NaN."""
        df = _make_ticker_rows(n=5)
        out = compute_momentum_features(df)
        assert out["yes_mid_velocity"].iloc[:3].isna().all()

    def test_velocity_matches_manual_formula(self):
        """velocity[i] = (mid[i] - mid[i-3]) / 3."""
        delta = 0.010
        df = _make_ticker_rows(n=10, start_mid=0.50, mid_delta_per_tick=delta)
        out = compute_momentum_features(df)
        # For row 3: (mid[3] - mid[0]) / 3 = (0.53 - 0.50) / 3 = delta
        assert out["yes_mid_velocity"].iloc[3] == pytest.approx(delta, rel=1e-6)

    def test_velocity_negative_when_mid_falls(self):
        df = _make_ticker_rows(n=10, start_mid=0.70, mid_delta_per_tick=-0.008)
        out = compute_momentum_features(df)
        vels = out["yes_mid_velocity"].dropna()
        assert (vels < 0).all()

    def test_velocity_zero_when_mid_flat(self):
        df = _make_ticker_rows(n=10, mid_delta_per_tick=0.0)
        out = compute_momentum_features(df)
        vels = out["yes_mid_velocity"].dropna()
        np.testing.assert_allclose(vels.values, 0.0, atol=1e-12)

    def test_nan_when_yes_mid_missing(self):
        df = _make_ticker_rows(n=10)
        df["yes_mid"] = np.nan
        out = compute_momentum_features(df)
        assert out["yes_mid_velocity"].isna().all()


class TestPaceDelta:
    def test_nan_when_seconds_elapsed_below_minimum(self):
        """pace_delta should be NaN when seconds_elapsed < PACE_DELTA_MIN_SECONDS."""
        df = _make_ticker_rows(n=5, start_seconds=0.0, seconds_per_tick=10.0)
        out = compute_momentum_features(df)
        # All rows have seconds_elapsed < 60
        assert out["pace_delta"].isna().all()

    def test_pace_delta_zero_when_at_expected_rate(self):
        """If the game is scoring at exactly the pregame total rate, delta = 0.

        For each row: pace_delta = total_points / (seconds/60) * 48 - pregame_total.
        If total_points = pregame_total * (seconds_elapsed / (48*60)), then pace_delta = 0.
        We verify this for a single fixed instant (n=1).
        """
        pregame = 220.0
        # Pick a single instant: 24 min elapsed (half-time), exactly half of points scored
        seconds = 24 * 60  # 1440 s
        total_pts = pregame * (seconds / (48 * 60))  # exactly on pace
        row = {
            "captured_at": pd.Timestamp("2026-04-20T19:00:00Z"),
            "ticker": "T1",
            "home_score": total_pts / 2,
            "away_score": total_pts / 2,
            "score_margin_home": 0.0,
            "total_points": total_pts,
            "yes_mid": 0.50,
            "seconds_elapsed": seconds,
            "pregame_total": pregame,
        }
        df = pd.DataFrame([row])
        df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
        out = compute_momentum_features(df)
        valid = out["pace_delta"].dropna()
        assert len(valid) == 1
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-9)

    def test_pace_delta_positive_when_high_scoring(self):
        """If scoring faster than expected, projected_total > pregame_total → positive delta."""
        pregame = 220.0
        seconds = 24 * 60  # 24 minutes elapsed
        total_pts = 130.0  # exceeds expected 110
        df = _make_ticker_rows(n=3, start_seconds=seconds, seconds_per_tick=20.0,
                               pregame_total=pregame)
        df["total_points"] = total_pts
        df["home_score"] = total_pts / 2
        df["away_score"] = total_pts / 2
        out = compute_momentum_features(df)
        valid = out["pace_delta"].dropna()
        assert (valid > 0).all()

    def test_pace_delta_negative_when_low_scoring(self):
        pregame = 220.0
        seconds = 24 * 60
        total_pts = 90.0  # below expected 110
        df = _make_ticker_rows(n=3, start_seconds=seconds, seconds_per_tick=20.0,
                               pregame_total=pregame)
        df["total_points"] = total_pts
        df["home_score"] = total_pts / 2
        df["away_score"] = total_pts / 2
        out = compute_momentum_features(df)
        valid = out["pace_delta"].dropna()
        assert (valid < 0).all()

    def test_nan_when_pregame_total_missing(self):
        df = _make_ticker_rows(n=5, start_seconds=120.0)
        df["pregame_total"] = np.nan
        out = compute_momentum_features(df)
        assert out["pace_delta"].isna().all()


class TestMultipleTickers:
    def test_momentum_computed_independently_per_ticker(self):
        """Each ticker's momentum window must not bleed into another ticker's."""
        df1 = _make_ticker_rows(n=10, ticker="T1", home_pts_per_tick=4.0,
                                away_pts_per_tick=1.0, start_seconds=120.0)
        df2 = _make_ticker_rows(n=10, ticker="T2", home_pts_per_tick=1.0,
                                away_pts_per_tick=4.0, start_seconds=120.0)
        combined = pd.concat([df1, df2], ignore_index=True)
        out = compute_momentum_features(combined)

        t1_runs = out[out["ticker"] == "T1"]["home_score_run_3min"].dropna()
        t2_runs = out[out["ticker"] == "T2"]["home_score_run_3min"].dropna()
        assert (t1_runs > 0).all(), "T1 should have positive runs"
        assert (t2_runs < 0).all(), "T2 should have negative runs"


# ---------------------------------------------------------------------------
# PARITY TESTS: training path vs live-scorer path must produce identical results
# ---------------------------------------------------------------------------

class TestTrainingLiveParity:
    """Assert that compute_momentum_features (training path) and
    engineer_features (live-scorer path) produce identical momentum feature
    values when given the same sorted input dataframe.
    """

    def _base_row(self) -> dict:
        return dict(
            ticker="KXNBAGAME-26APR28LALHOU-LAL",
            game_key="2026-04-28_LAL_HOU",
            bet_side="home",
            status_state="in",
            period=2,
            seconds_left_in_period=300.0,
            yes_bid=0.55,
            yes_ask=0.57,
            no_bid=0.43,
            no_ask=0.45,
            last_price=0.56,
            market_home_implied=0.56,
            volume=10000.0,
            open_interest=2000.0,
            yes_depth_notional_3=300.0,
            yes_depth_notional_5=500.0,
            no_depth_notional_3=280.0,
            no_depth_notional_5=480.0,
            yes_weighted_price_3=0.555,
            no_weighted_price_3=0.435,
            espn_home_implied=np.nan,
            espn_away_implied=np.nan,
            market_consensus_home=np.nan,
            pregame_home_win_prob=0.50,
            pregame_away_win_prob=0.50,
            pregame_spread=0.0,
            pregame_total=220.0,
            pregame_edge_home=0.06,
            consensus_gap_home=np.nan,
        )

    def _make_sequence(self, n: int = 10) -> pd.DataFrame:
        base = self._base_row()
        rows = []
        for i in range(n):
            row = base.copy()
            row["captured_at"] = (
                pd.Timestamp("2026-04-28T19:00:00Z") + pd.Timedelta(seconds=i * 20)
            )
            row["seconds_elapsed"] = 600.0 + i * 20.0
            row["home_score"] = 40.0 + i * 2
            row["away_score"] = 38.0 + i * 1
            row["score_margin_home"] = row["home_score"] - row["away_score"]
            row["total_points"] = row["home_score"] + row["away_score"]
            row["yes_mid"] = 0.55 + i * 0.003
            rows.append(row)
        df = pd.DataFrame(rows)
        df["captured_at"] = pd.to_datetime(df["captured_at"], utc=True)
        return df

    def test_home_score_run_identical(self):
        """home_score_run_3min must be identical from training path and live path."""
        df = self._make_sequence(n=15)

        # Training path: compute_momentum_features directly
        train_out = compute_momentum_features(df.copy())
        train_runs = train_out["home_score_run_3min"].values

        # Live-scorer path: engineer_features (which calls compute_momentum_features)
        live_out = engineer_features(df.copy())
        live_runs = live_out["home_score_run_3min"].values

        np.testing.assert_array_equal(
            np.isnan(train_runs), np.isnan(live_runs),
            err_msg="NaN mask must match between training and live paths",
        )
        valid = ~np.isnan(train_runs)
        if valid.any():
            np.testing.assert_allclose(
                train_runs[valid], live_runs[valid],
                atol=1e-12,
                err_msg="home_score_run_3min values must be bit-exact",
            )

    def test_yes_mid_velocity_identical(self):
        """yes_mid_velocity must be identical from training path and live path."""
        df = self._make_sequence(n=10)

        train_out = compute_momentum_features(df.copy())
        live_out = engineer_features(df.copy())

        train_vel = train_out["yes_mid_velocity"].values
        live_vel = live_out["yes_mid_velocity"].values

        np.testing.assert_array_equal(
            np.isnan(train_vel), np.isnan(live_vel),
            err_msg="NaN mask must match between training and live paths",
        )
        valid = ~np.isnan(train_vel)
        if valid.any():
            np.testing.assert_allclose(
                train_vel[valid], live_vel[valid],
                atol=1e-12,
                err_msg="yes_mid_velocity values must be bit-exact",
            )

    def test_pace_delta_identical(self):
        """pace_delta must be identical from training path and live path."""
        df = self._make_sequence(n=10)

        train_out = compute_momentum_features(df.copy())
        live_out = engineer_features(df.copy())

        train_pd = train_out["pace_delta"].values
        live_pd = live_out["pace_delta"].values

        np.testing.assert_array_equal(
            np.isnan(train_pd), np.isnan(live_pd),
            err_msg="NaN mask must match between training and live paths",
        )
        valid = ~np.isnan(train_pd)
        if valid.any():
            np.testing.assert_allclose(
                train_pd[valid], live_pd[valid],
                atol=1e-12,
                err_msg="pace_delta values must be bit-exact",
            )

    def test_single_row_gracefully_returns_nan_momentum(self):
        """A single-row df (typical live-scorer tick) should return NaN for all momentum
        features — the model's SimpleImputer handles imputation from training medians."""
        df = self._make_sequence(n=1)
        out = engineer_features(df)
        assert np.isnan(out["home_score_run_3min"].iloc[0])
        assert np.isnan(out["yes_mid_velocity"].iloc[0])
        # pace_delta may or may not be NaN depending on elapsed time (here 600s > 60s min)
        # but it should always be finite or NaN — never raise
        val = out["pace_delta"].iloc[0]
        assert np.isfinite(val) or np.isnan(val)

    def test_parity_with_two_tickers(self):
        """Multi-ticker df: parity holds and tickers are computed independently."""
        df1 = self._make_sequence(n=8)
        df2 = self._make_sequence(n=8)
        df2["ticker"] = "KXNBAGAME-26APR28LALHOU-HOU"
        df2["yes_mid"] = 0.45 - df2.index * 0.002
        combined = pd.concat([df1, df2], ignore_index=True).sort_values(
            ["ticker", "captured_at"]
        )

        train_out = compute_momentum_features(combined.copy())
        live_out = engineer_features(combined.copy())

        for col in ["home_score_run_3min", "yes_mid_velocity", "pace_delta"]:
            tv = train_out[col].values
            lv = live_out[col].values
            np.testing.assert_array_equal(
                np.isnan(tv), np.isnan(lv),
                err_msg=f"NaN mask mismatch for {col}",
            )
            valid = ~np.isnan(tv)
            if valid.any():
                np.testing.assert_allclose(
                    tv[valid], lv[valid], atol=1e-12,
                    err_msg=f"{col} values must be bit-exact across paths",
                )


class TestMomentumFeaturesInTrainingMatrix:
    """Verify that build_in_game_training_matrix includes momentum features in feature_cols."""

    def test_momentum_features_in_feature_cols(self):
        from live_training_matrix import build_in_game_training_matrix

        base = {
            "ticker": "T", "event_ticker": "E", "game_key": "G",
            "game_date": "2026-04-20",
            "home_team": "H", "away_team": "A",
            "game_status": "in", "status_state": "in",
            "period": 2, "seconds_elapsed": 700.0, "seconds_left_in_period": 200.0,
            "home_score": 60.0, "away_score": 55.0,
            "score_margin_home": 5.0, "total_points": 115.0,
            "yes_bid": 0.55, "yes_ask": 0.57, "yes_mid": 0.56,
            "no_bid": 0.43, "no_ask": 0.45, "last_price": 0.56,
            "market_home_implied": 0.56,
            "volume": 100.0, "open_interest": 50.0,
            "yes_depth_notional_3": 1.0, "yes_depth_notional_5": 1.0,
            "no_depth_notional_3": 1.0, "no_depth_notional_5": 1.0,
            "yes_weighted_price_3": 0.56, "no_weighted_price_3": 0.44,
            "espn_home_implied": 0.56, "espn_away_implied": 0.44,
            "oddsapi_home_consensus": 0.56, "oddsapi_away_consensus": 0.44,
            "oddsapi_books": 5.0,
            "market_consensus_home": 0.56,
            "pregame_home_win_prob": 0.50, "pregame_away_win_prob": 0.50,
            "pregame_spread": 0.0, "pregame_total": 220.0,
            "pregame_edge_home": 0.0, "consensus_gap_home": 0.0,
            "label_yes_mid_move_5m": 0.0,
            "label_market_home_implied_move_5m": 0.0,
            "label_yes_up_5m": 0, "label_home_up_5m": 1,
            "label_beats_close_yes": 0, "label_beats_close_home": 1,
            "label_final_home_win": 1,
            "bet_side": "home",
        }
        # Build two rows to allow some momentum computation
        rows = []
        for i in range(4):
            row = base.copy()
            row["captured_at"] = f"2026-04-20T19:{10+i}:00Z"
            row["seconds_elapsed"] = 700.0 + i * 30
            row["home_score"] = 60.0 + i * 2
            row["away_score"] = 55.0 + i * 1
            row["total_points"] = row["home_score"] + row["away_score"]
            row["yes_mid"] = 0.56 + i * 0.005
            rows.append(row)
        labeled = pd.DataFrame(rows)

        _, feature_cols = build_in_game_training_matrix(labeled)
        assert "home_score_run_3min" in feature_cols
        assert "yes_mid_velocity" in feature_cols
        assert "pace_delta" in feature_cols

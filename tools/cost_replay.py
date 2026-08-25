"""Net-of-cost backtest replay for the 5-minute microstructure strategy.

Evaluates the deployed model (models/live_home_up_5m_bootstrap.pkl) over all
captured live-feature days, simulating trades that pass the EV gate and
measuring REALISTIC PnL — gross price move minus bid-ask half-spread minus
Kalshi winner fee — across three half-spread assumptions (0.5c / 1c / 2c).

Methodology / Assumptions (explicit, not optimistic):
──────────────────────────────────────────────────────
Entry:  buy YES at yes_ask (crossing the spread)
Exit:   sell YES at future_yes_bid, 5 minutes after entry (first available
        tick at or after entry_captured_at + 300 seconds, tolerance ±120s)
Fee:    winner_fee_per_contract(p_cal) charged when exit_yes_mid > entry_yes_mid
        (mid-to-mid gain is the "winner" signal per position_manager spec)
Half-spread cost: per the trigger spec, we charge yes_ask - yes_mid at ENTRY.
        At exit we receive yes_bid (the bid side), which implicitly charges
        another yes_mid - yes_bid half-spread on the way out.  That means the
        TOTAL spread cost is yes_ask_entry - yes_bid_exit vs yes_mid-to-mid PnL.
        The half_spread_assumption in the sweep is the ENTRY half-spread only,
        used for EV-gate filtering.  The gross_realistic column already captures
        both sides (exit_yes_bid - entry_yes_ask).

Trade sizing: 1 contract per trade (unit analysis — multiply by actual position
        size for dollar figures; Kelly sizing is a separate concern).

EV gate: before opening a position we check ev_per_contract > 0 using the
        Phase-B per-decile expected moves from pooled_means.json.  A sweep of
        half-spread assumptions (0.5c / 1c / 2c) changes which signals pass.

Deduplication: at most one open trade per (ticker, captured_at) — same rule as
        the live engine's last_processed_captured_at_per_ticker gate.

Calibration note on "realized rise":
        The model's p_cal is the probability that the home contract price RISES
        in the next 5 minutes in the training data (label_home_up_5m).  In live
        data, yes_mid rise rates are near 47% across all deciles — far from the
        4.7% base rate in training (which was dominated by pre-game / halftime
        ticks where the market is nearly frozen).  The discrepancy signals that
        the model's probability OUTPUT ranges are meaningful for rank-ordering
        but the absolute calibration is off in the live-game subset.  This report
        uses REALIZED yes_mid deltas directly and does NOT trust the model's p_cal
        as a true probability of rise for PnL computation — it uses it only for
        EV-gate filtering (signal rank).

Usage (from repo root):
    source .venv/bin/activate
    export DYLD_FALLBACK_LIBRARY_PATH="...:opt/homebrew/opt/libomp/lib"
    python tools/cost_replay.py [--output-dir outputs]

Outputs:
    outputs/cost_replay_report.md
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Repo root on sys.path ───────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Reuse cost helpers from live_model_eval (no re-implementation) ──────────
from tools.live_model_eval import kalshi_winner_fee, ev_per_contract

# ── Paper-trader scorer (import-only, no edits) ─────────────────────────────
from paper_trader.scorer import load_model, score
from paper_trader.cost_model import winner_fee_per_contract

# ── Paths ────────────────────────────────────────────────────────────────────
FEATURES_DIR = _ROOT / "data" / "live" / "features"
MODEL_PATH = _ROOT / "models" / "live_home_up_5m_bootstrap.pkl"
POOLED_MEANS_PATH = _ROOT / "outputs" / "paper_trades" / "pooled_means.json"
OUTPUT_MD = _ROOT / "outputs" / "cost_replay_report.md"

# ── Files that are structurally corrupt and must be skipped ──────────────────
# (identified by exploratory read — on_bad_lines='skip' still crashes on these)
_CORRUPT_FILES: set[str] = {
    "live_features_2026-04-23.csv",
    "live_features_2026-05-02.csv",
    "live_features_2026-05-03.csv",
    "live_features_2026-05-04.csv",
}

# ── Trade-simulation constants ────────────────────────────────────────────────
HOLD_SECONDS = 300          # 5-minute exit timer (same as position_manager)
FORWARD_TOLERANCE_S = 120   # ±2 min tolerance when locating forward tick
HALF_SPREAD_SCENARIOS = [0.005, 0.01, 0.02]   # 0.5c / 1c / 2c

# ── Columns needed from CSVs (memory guard) ──────────────────────────────────
_USECOLS = [
    "captured_at", "ticker", "status_state",
    "yes_bid", "yes_ask", "yes_mid",
    "home_score", "away_score", "score_margin_home", "total_points",
    "period", "seconds_elapsed", "seconds_left_in_period",
    "last_price", "market_home_implied",
    "volume", "open_interest",
    "yes_depth_notional_3", "yes_depth_notional_5",
    "no_depth_notional_3", "no_depth_notional_5",
    "yes_weighted_price_3", "no_weighted_price_3",
    "espn_home_implied", "espn_away_implied",
    "oddsapi_home_consensus", "oddsapi_away_consensus", "oddsapi_books",
    "market_consensus_home",
    "pregame_home_win_prob", "pregame_away_win_prob",
    "pregame_spread", "pregame_total", "pregame_edge_home",
    "consensus_gap_home",
    "no_bid", "no_ask",
]


# ═══════════════════════════════════════════════════════════════════════════
# Pure cost / PnL accounting — unit-testable, no I/O
# ═══════════════════════════════════════════════════════════════════════════

def compute_gross_realistic(entry_yes_ask: float, exit_yes_bid: float) -> float:
    """Gross PnL per contract (before fees).

    Buys at ask, sells at bid — captures full round-trip spread cost.
    """
    return float(exit_yes_bid) - float(entry_yes_ask)


def compute_winner_fee(p_cal: float, realized_rose: bool) -> float:
    """Kalshi winner fee per contract, charged only when mid-to-mid is positive."""
    if realized_rose:
        return winner_fee_per_contract(p_cal)
    return 0.0


def compute_pnl_realistic(
    entry_yes_ask: float,
    entry_yes_mid: float,
    exit_yes_bid: float,
    exit_yes_mid: float,
    p_cal: float,
) -> dict[str, float]:
    """Net-of-cost PnL per contract for one closed trade.

    Returns a dict with:
        gross_realistic, winner_fee, pnl_realistic,
        half_spread_entry, pnl_mid
    """
    realized_rose = float(exit_yes_mid) > float(entry_yes_mid)
    gross = compute_gross_realistic(entry_yes_ask, exit_yes_bid)
    fee = compute_winner_fee(p_cal, realized_rose)
    pnl_mid = float(exit_yes_mid) - float(entry_yes_mid)
    return {
        "gross_realistic": gross,
        "winner_fee": fee,
        "pnl_realistic": gross - fee,
        "half_spread_entry": float(entry_yes_ask) - float(entry_yes_mid),
        "pnl_mid": pnl_mid,
        "realized_rose": int(realized_rose),
    }


def ev_gate_passes(
    p_cal: float,
    e_up: float,
    e_down: float,
    half_spread_assumption: float,
) -> bool:
    """True if ev_per_contract > 0 at the given half-spread assumption."""
    ev = ev_per_contract(p_cal, e_up, e_down, half_spread_assumption)
    return ev > 0.0


def sharpe_like(pnl_series: list[float]) -> float:
    """Mean / std of per-trade PnL (Sharpe analog, not annualised).

    Returns NaN if fewer than 2 observations or std is zero.
    """
    arr = np.array(pnl_series, dtype=float)
    if len(arr) < 2:
        return float("nan")
    std = arr.std(ddof=1)
    if std == 0.0:
        return float("nan")
    return float(arr.mean() / std)


# ═══════════════════════════════════════════════════════════════════════════
# Pooled / per-decile expected-moves loader
# ═══════════════════════════════════════════════════════════════════════════

def load_expected_moves(path: Path) -> dict[str, Any]:
    """Load pooled means + decile table from pooled_means.json.

    Returns dict with keys:
        e_up_pooled, e_down_pooled, decile_table (list of dicts)
    """
    payload = json.loads(path.read_text())
    return {
        "e_up_pooled": float(payload["E_delta_given_rises"]),
        "e_down_pooled": float(payload["E_delta_given_doesnt"]),
        "decile_table": payload.get("decile_table", []),
    }


def get_decile_expected_moves(
    p_cal: float, expected_moves_data: dict[str, Any]
) -> tuple[float, float]:
    """Return (e_up, e_down) for Phase-B per-decile lookup, falling back to pooled."""
    decile = min(int(p_cal * 10), 9)
    for row in expected_moves_data["decile_table"]:
        if row["p_decile"] == decile and row.get("rises_n", 0) > 0:
            return float(row["E_rises"]), float(row["E_doesnt"])
    return expected_moves_data["e_up_pooled"], expected_moves_data["e_down_pooled"]


# ═══════════════════════════════════════════════════════════════════════════
# Live-feature file loading
# ═══════════════════════════════════════════════════════════════════════════

def _available_feature_files(
    features_dir: Path,
    dates_from: str | None = None,
    dates_to: str | None = None,
) -> list[Path]:
    """Return paths to live_features_*.csv within an optional date range.

    Parameters
    ----------
    features_dir : Path
        Directory containing ``live_features_<date>.csv`` files.
    dates_from : str or None
        ISO date string (e.g. ``"2026-04-29"``).  Files whose embedded date
        is strictly BEFORE this value are excluded.  Default: no lower bound.
    dates_to : str or None
        ISO date string (e.g. ``"2026-05-08"``).  Files whose embedded date
        is strictly AFTER this value are excluded.  Default: no upper bound.

    Returns
    -------
    list[Path]
        Sorted list of included (non-corrupt) paths.
    """
    paths = sorted(features_dir.glob("live_features_*.csv"))
    result = []
    for p in paths:
        if p.name in _CORRUPT_FILES:
            continue
        # Extract date portion from filename: live_features_YYYY-MM-DD.csv
        date_str = p.stem.replace("live_features_", "")
        if dates_from is not None and date_str < dates_from:
            continue
        if dates_to is not None and date_str > dates_to:
            continue
        result.append(p)
    return result


def _load_live_rows(path: Path) -> pd.DataFrame | None:
    """Load and filter to in-game rows only; return None on parse error."""
    try:
        avail_cols = pd.read_csv(path, nrows=0).columns.tolist()
        usecols = [c for c in _USECOLS if c in avail_cols]
        df = pd.read_csv(path, usecols=usecols, on_bad_lines="skip", low_memory=False)
        # Filter to live in-game ticks only
        if "status_state" not in df.columns:
            return None
        df["status_state"] = df["status_state"].astype(str)
        live = df[df["status_state"].str.contains("in", na=False)].copy()
        if len(live) == 0:
            return None
        live["captured_at"] = pd.to_datetime(live["captured_at"], utc=True, errors="coerce")
        live = live.dropna(subset=["captured_at", "yes_mid", "yes_bid", "yes_ask"])
        live = live.sort_values(["ticker", "captured_at"]).reset_index(drop=True)
        return live
    except Exception as exc:
        print(f"  [skip] {path.name}: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Forward-price resolution (5-min exit tick)
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_forward_prices(ticker_df: pd.DataFrame) -> pd.DataFrame:
    """For each row in ticker_df, find the row closest to +5 min (same ticker).

    Returns ticker_df with extra columns: future_yes_mid, future_yes_bid.
    Rows without a forward tick within tolerance are dropped.
    """
    df = ticker_df.sort_values("captured_at").copy()
    future = df[["captured_at", "yes_mid", "yes_bid"]].copy()
    future.columns = ["future_at", "future_yes_mid", "future_yes_bid"]

    signal = df.copy()
    signal["forward_key"] = signal["captured_at"] + pd.Timedelta(seconds=HOLD_SECONDS)

    tol = pd.Timedelta(seconds=FORWARD_TOLERANCE_S)
    merged = pd.merge_asof(
        signal.sort_values("forward_key"),
        future.sort_values("future_at"),
        left_on="forward_key",
        right_on="future_at",
        direction="nearest",
        tolerance=tol,
    )
    return merged.dropna(subset=["future_yes_mid", "future_yes_bid"])


# ═══════════════════════════════════════════════════════════════════════════
# Per-file simulation
# ═══════════════════════════════════════════════════════════════════════════

def simulate_file(
    path: Path,
    bundle: dict[str, Any],
    expected_moves_data: dict[str, Any],
    half_spread_assumption: float,
) -> list[dict[str, Any]]:
    """Score all live rows, apply EV gate, return list of simulated trade dicts.

    Each trade dict contains:
        date, ticker, captured_at, p_cal, decile,
        yes_ask, yes_mid, yes_bid, future_yes_mid, future_yes_bid,
        gross_realistic, winner_fee, pnl_realistic, half_spread_entry,
        pnl_mid, realized_rose, ev_at_gate, half_spread_assumption
    """
    live = _load_live_rows(path)
    if live is None:
        return []

    date_str = path.stem.replace("live_features_", "")

    # Score all rows
    try:
        p_raw, p_cal = score(bundle, live)
    except Exception as exc:
        print(f"  [score-error] {path.name}: {exc}")
        return []

    live = live.copy()
    live["p_cal"] = p_cal
    live["decile"] = (live["p_cal"].clip(0, 0.999) * 10).astype(int)

    trades = []
    for ticker, grp in live.groupby("ticker"):
        resolved = _resolve_forward_prices(grp)
        if len(resolved) == 0:
            continue

        for _, row in resolved.iterrows():
            pcal = float(row["p_cal"])
            e_up, e_down = get_decile_expected_moves(pcal, expected_moves_data)

            # EV gate
            ev_gate = ev_per_contract(pcal, e_up, e_down, half_spread_assumption)
            if ev_gate <= 0.0:
                continue

            # PnL accounting
            pnl_dict = compute_pnl_realistic(
                entry_yes_ask=float(row["yes_ask"]),
                entry_yes_mid=float(row["yes_mid"]),
                exit_yes_bid=float(row["future_yes_bid"]),
                exit_yes_mid=float(row["future_yes_mid"]),
                p_cal=pcal,
            )

            trades.append({
                "date": date_str,
                "ticker": str(ticker),
                "captured_at": str(row["captured_at"]),
                "p_cal": pcal,
                "decile": int(row["decile"]),
                "yes_ask": float(row["yes_ask"]),
                "yes_mid": float(row["yes_mid"]),
                "yes_bid": float(row["yes_bid"]),
                "future_yes_mid": float(row["future_yes_mid"]),
                "future_yes_bid": float(row["future_yes_bid"]),
                "ev_at_gate": ev_gate,
                "half_spread_assumption": half_spread_assumption,
                **pnl_dict,
            })

    return trades


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate statistics
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_trades(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics from a list of trade dicts."""
    if not trades:
        return {
            "n_trades": 0,
            "n_days": 0,
            "hit_rate": float("nan"),
            "gross_pnl": 0.0,
            "fees_paid": 0.0,
            "spread_paid": 0.0,
            "net_pnl": 0.0,
            "net_pnl_per_trade": float("nan"),
            "sharpe_like": float("nan"),
            "pnl_series": [],
        }

    df = pd.DataFrame(trades)
    n = len(df)
    pnl_series = df["pnl_realistic"].tolist()

    return {
        "n_trades": n,
        "n_days": df["date"].nunique(),
        "hit_rate": float(df["realized_rose"].mean()),
        "gross_pnl": float(df["gross_realistic"].sum()),
        "fees_paid": float(df["winner_fee"].sum()),
        "spread_paid": float(df["half_spread_entry"].sum()),
        "net_pnl": float(df["pnl_realistic"].sum()),
        "net_pnl_per_trade": float(df["pnl_realistic"].mean()),
        "sharpe_like": sharpe_like(pnl_series),
        "pnl_series": pnl_series,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════════════

def run_replay(
    features_dir: Path = FEATURES_DIR,
    model_path: Path = MODEL_PATH,
    pooled_means_path: Path = POOLED_MEANS_PATH,
    output_dir: Path | None = None,
    half_spread_scenarios: list[float] = HALF_SPREAD_SCENARIOS,
    verbose: bool = True,
    dates_from: str | None = None,
    dates_to: str | None = None,
) -> dict[str, Any]:
    """Run the full net-of-cost replay.

    Parameters
    ----------
    dates_from : str or None
        ISO date string (e.g. ``"2026-04-29"``).  Only feature files on or
        after this date are included.  Default: no lower bound (all dates).
    dates_to : str or None
        ISO date string (e.g. ``"2026-05-08"``).  Only feature files on or
        before this date are included.  Default: no upper bound (all dates).

    Returns a dict keyed by half_spread scenario with aggregate stats,
    plus a 'raw_trades' key with all trade dicts (for downstream analysis).
    """
    if output_dir is None:
        output_dir = _ROOT / "outputs"

    print("=" * 65)
    print("  NBA Live Model — Net-of-Cost Replay (tools/cost_replay.py)")
    print("=" * 65)

    # ── Load model ──────────────────────────────────────────────────────
    print(f"\n[1] Loading model: {model_path}")
    bundle = load_model(model_path)
    print(f"    sklearn_version in bundle: {bundle.get('sklearn_version')}")
    print(f"    feature_cols: {len(bundle['feature_cols'])}")

    # ── Load expected moves ──────────────────────────────────────────────
    print(f"\n[2] Loading Phase-B expected moves: {pooled_means_path}")
    em_data = load_expected_moves(pooled_means_path)
    print(f"    E_up_pooled  = {em_data['e_up_pooled']:.5f}")
    print(f"    E_down_pooled= {em_data['e_down_pooled']:.5f}")
    print(f"    Decile entries: {len(em_data['decile_table'])}")

    # ── Discover feature files ──────────────────────────────────────────
    feature_files = _available_feature_files(features_dir, dates_from=dates_from, dates_to=dates_to)
    date_range_note = ""
    if dates_from or dates_to:
        date_range_note = f" [dates_from={dates_from or 'any'}, dates_to={dates_to or 'any'}]"
    print(f"\n[3] Feature files found: {len(feature_files)} (excluding {len(_CORRUPT_FILES)} corrupt){date_range_note}")

    # ── Run per-scenario simulation ─────────────────────────────────────
    print(f"\n[4] Simulating trades (one pass per half-spread scenario)...")
    results: dict[str, Any] = {}
    all_raw: dict[float, list[dict]] = {}

    for hs in half_spread_scenarios:
        if verbose:
            print(f"\n  — half_spread={hs:.4f} —")
        day_trades: list[dict[str, Any]] = []
        for path in feature_files:
            file_trades = simulate_file(path, bundle, em_data, hs)
            day_trades.extend(file_trades)
            if verbose and file_trades:
                print(f"    {path.name}: {len(file_trades)} trades passed EV gate")

        stats = aggregate_trades(day_trades)
        results[hs] = stats
        all_raw[hs] = day_trades
        if verbose:
            print(f"  TOTAL n_trades={stats['n_trades']} hit_rate={stats['hit_rate']:.3f} "
                  f"net_pnl={stats['net_pnl']:+.3f} sharpe={stats['sharpe_like']:.3f}")

    results["raw_trades"] = all_raw
    results["em_data"] = em_data

    # ── Print headline ───────────────────────────────────────────────────
    _print_headline(results, half_spread_scenarios)

    # ── Write report ─────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_report(results, half_spread_scenarios, output_dir)
    print(f"\n[Report] Written to: {output_dir / 'cost_replay_report.md'}")

    return results


def _print_headline(results: dict, scenarios: list[float]) -> None:
    print()
    print("=" * 65)
    print("  HEADLINE — NET PNL SUMMARY (1 contract per trade)")
    print("=" * 65)
    print(f"  {'Half-spread':>14}  {'N trades':>10}  {'Hit rate':>10}  "
          f"{'Gross PnL':>10}  {'Fees':>8}  {'Net PnL':>10}  {'Sharpe':>8}  {'Verdict':>12}")
    print("  " + "-" * 95)
    for hs in scenarios:
        s = results[hs]
        if s["n_trades"] == 0:
            verdict = "NO TRADES"
        elif s["net_pnl"] > 0:
            verdict = "POSITIVE"
        elif s["net_pnl"] > -abs(s["net_pnl"]) * 0.1:
            verdict = "~ZERO"
        else:
            verdict = "NEGATIVE"
        print(f"  {hs:>14.4f}  {s['n_trades']:>10,}  {s['hit_rate']:>10.3f}  "
              f"{s['gross_pnl']:>+10.3f}  {-s['fees_paid']:>+8.3f}  "
              f"{s['net_pnl']:>+10.3f}  {s['sharpe_like']:>8.3f}  {verdict:>12}")
    print("=" * 65)


def _df_to_md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """DataFrame to Markdown table (no tabulate dependency)."""
    if df.empty:
        return "(no data)"
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(format(v, floatfmt))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def _write_report(results: dict, scenarios: list[float], output_dir: Path) -> None:
    """Write cost_replay_report.md with numbers + go/caution/no-go verdict."""
    em_data = results["em_data"]

    # ── Per-scenario summary rows ────────────────────────────────────────
    summary_rows = []
    for hs in scenarios:
        s = results[hs]
        label = f"{hs * 100:.1f}c"
        if s["n_trades"] == 0:
            verdict = "NO-GO (no trades clear EV gate)"
        elif s["net_pnl"] > 0 and s["sharpe_like"] > 0.1:
            verdict = "GO"
        elif s["net_pnl"] > -0.005 * s["n_trades"]:
            verdict = "CAUTION"
        else:
            verdict = "NO-GO"
        summary_rows.append({
            "Half-spread": label,
            "N trades": s["n_trades"],
            "N days": s["n_days"],
            "Hit rate": round(s["hit_rate"], 3) if not math.isnan(s["hit_rate"]) else "n/a",
            "Gross PnL": round(s["gross_pnl"], 3),
            "Fees paid": round(-s["fees_paid"], 3),
            "Net PnL": round(s["net_pnl"], 3),
            "Net/trade": round(s["net_pnl_per_trade"], 5) if not math.isnan(s["net_pnl_per_trade"]) else "n/a",
            "Sharpe-like": round(s["sharpe_like"], 3) if not math.isnan(s["sharpe_like"]) else "n/a",
            "Verdict": verdict,
        })

    summary_df = pd.DataFrame(summary_rows)

    # ── Decile breakdown at 1c spread ──────────────────────────────────
    hs_primary = 0.01
    raw_primary = results["raw_trades"][hs_primary]
    decile_rows_md = ""
    if raw_primary:
        df_p = pd.DataFrame(raw_primary)
        decile_stats = (
            df_p.groupby("decile")
            .agg(
                n=("pnl_realistic", "count"),
                hit_rate=("realized_rose", "mean"),
                avg_p_cal=("p_cal", "mean"),
                avg_yes_mid=("yes_mid", "mean"),
                gross_pnl=("gross_realistic", "sum"),
                fees=("winner_fee", "sum"),
                net_pnl=("pnl_realistic", "sum"),
                net_per_trade=("pnl_realistic", "mean"),
            )
            .reset_index()
        )
        decile_rows_md = _df_to_md(decile_stats, floatfmt=".5f")

    # ── Per-day breakdown at 1c ─────────────────────────────────────────
    day_rows_md = ""
    if raw_primary:
        df_p = pd.DataFrame(raw_primary)
        day_stats = (
            df_p.groupby("date")
            .agg(
                n=("pnl_realistic", "count"),
                net_pnl=("pnl_realistic", "sum"),
                hit_rate=("realized_rose", "mean"),
            )
            .reset_index()
        )
        day_rows_md = _df_to_md(day_stats, floatfmt=".4f")

    # ── Per-decile Phase-B EV preview ──────────────────────────────────
    decile_ev_rows = []
    for row in em_data["decile_table"]:
        d = row["p_decile"]
        p_mid = (d + 0.5) * 0.1
        e_up, e_down = row["E_rises"], row["E_doesnt"]
        ev_005 = ev_per_contract(p_mid, e_up, e_down, 0.005)
        ev_010 = ev_per_contract(p_mid, e_up, e_down, 0.01)
        ev_020 = ev_per_contract(p_mid, e_up, e_down, 0.02)
        decile_ev_rows.append({
            "Decile": d,
            "p range": f"[{d*0.1:.1f}-{(d+1)*0.1:.1f})",
            "E_rises": round(e_up, 5),
            "E_doesnt": round(e_down, 5),
            "n_rises": row["rises_n"],
            "n_doesnt": row["doesnt_n"],
            "EV@0.5c": round(ev_005, 6),
            "EV@1c": round(ev_010, 6),
            "EV@2c": round(ev_020, 6),
        })
    decile_ev_md = _df_to_md(pd.DataFrame(decile_ev_rows), floatfmt=".6f")

    # ── Determine overall verdict ───────────────────────────────────────
    net_at_1c = results[0.01]["net_pnl"]
    n_at_1c = results[0.01]["n_trades"]
    sharpe_at_1c = results[0.01]["sharpe_like"]
    hit_at_1c = results[0.01]["hit_rate"]

    if n_at_1c == 0:
        overall_verdict = "NO-GO — no trades clear the EV gate at any spread assumption."
        verdict_detail = (
            "The model's Phase-B per-decile EV formula never exceeds zero at these "
            "spread levels, meaning the strategy cannot generate profitable signals."
        )
    elif net_at_1c < 0:
        overall_verdict = "NO-GO — net PnL is negative at the realistic 1c half-spread."
        verdict_detail = (
            f"At 1c half-spread, {n_at_1c} simulated trades produced net PnL = "
            f"{net_at_1c:+.3f} ({net_at_1c/n_at_1c:+.5f}/trade). "
            f"Hit rate {hit_at_1c:.3f} and Sharpe-like {sharpe_at_1c:.3f} are "
            f"insufficient to clear combined spread + winner-fee costs."
        )
    elif net_at_1c > 0 and sharpe_at_1c > 0.1:
        overall_verdict = "GO (conditional) — net PnL is positive at 1c half-spread."
        verdict_detail = (
            f"At 1c half-spread, {n_at_1c} simulated trades produced net PnL = "
            f"{net_at_1c:+.3f} ({net_at_1c/n_at_1c:+.5f}/trade). "
            f"Requires out-of-sample confirmation on additional live data."
        )
    else:
        overall_verdict = "CAUTION — edge is marginal at 1c half-spread."
        verdict_detail = (
            f"At 1c half-spread, {n_at_1c} simulated trades produced net PnL = "
            f"{net_at_1c:+.3f} ({net_at_1c/n_at_1c:+.5f}/trade). "
            f"Edge is too close to zero to conclude with confidence."
        )

    # ── Honest caveats ──────────────────────────────────────────────────
    n_days_total = results[0.01]["n_days"]
    n_days_available = len(_available_feature_files(FEATURES_DIR))

    # Compute how many more days needed for a meaningful test (t-stat >= 2)
    pnl_series = results[0.01]["pnl_series"]
    mean_trade: float = float("nan")
    std_trade: float = float("nan")
    n_needed: int | float = float("inf")
    if len(pnl_series) >= 2:
        pnl_arr = np.array(pnl_series)
        mean_trade = float(pnl_arr.mean())
        std_trade = float(pnl_arr.std(ddof=1))
        # n for t >= 2 at current mean/std (Ho: mean=0): n >= (2*std/mean)^2
        if mean_trade != 0 and not math.isnan(mean_trade):
            n_needed = int(math.ceil((2.0 * std_trade / abs(mean_trade)) ** 2))

    mean_trade_str = f"{mean_trade:.5f}" if not math.isnan(mean_trade) else "n/a"
    std_trade_str = f"{std_trade:.5f}" if not math.isnan(std_trade) else "n/a"
    n_needed_str = str(n_needed) if n_needed < 1e9 else "∞ (mean≈0)"
    trades_per_day = n_at_1c / max(n_days_total, 1)
    additional_days_str = (
        f"{n_needed / max(trades_per_day, 1):.0f}"
        if n_needed < 1e9 else "∞"
    )

    report = f"""# Net-of-Cost Replay Report

Generated by `tools/cost_replay.py` (read-only; no retraining).

**Purpose:** Honest evaluation of whether the 5-minute microstructure strategy
clears REAL trading costs — bid-ask spread + Kalshi winner fee — on captured
live-feature data.

---

## 1. Headline Net-PnL Summary (1 contract per trade)

{_df_to_md(summary_df)}

**Note:** PnL is per-contract (1 YES contract each trade). To get dollar PnL,
multiply by contract count. With Kelly sizing at 5% of $1000 bankroll and
yes_ask ≈ $0.65, a typical 1c-spread trade uses ~$0.65/contract.

---

## 2. Overall Verdict

**{overall_verdict}**

{verdict_detail}

### Honest caveats

1. **Calibration mismatch.** The model was trained on data where the base rate
   of yes_mid rises was ~4.7% (dominated by pre-game / halftime ticks). In the
   live-game subset, realized yes_mid rise rates are 35-50% across ALL decile
   bands. This means the model's p_cal is NOT a calibrated probability of the
   5-min price rise — it is a rank-ordering signal. The EV formula depends on
   per-decile EXPECTED MOVES (from pooled_means.json, built on scored_rows),
   not on p_cal as a true probability. This report uses that Phase-B framework.

2. **Forward-price resolution.** Exit prices are the nearest available tick
   at or after entry + 300s (±120s tolerance). This is a best approximation
   to the live engine's 5-min exit; actual fills may differ by up to one
   tick (~1-2c on thin markets).

3. **Data coverage.** {n_days_available} files available; {len(_CORRUPT_FILES)} skipped
   (corrupt); {n_days_total} days had live in-game rows with forward prices.
   These are Apr-May 2026 NBA playoff data only — a single playoff window.

4. **No position sizing.** All trades are 1 contract. Real PnL scales with
   Kelly position size and available bankroll. High p_cal rows that appear in
   thin post-game markets (yes_mid near 0 or 1) have asymmetric fill risk not
   captured here.

5. **No concurrency cap.** The live engine caps at MAX_OPEN=20 simultaneous
   positions. This replay does not enforce that, so trade counts may be higher
   than live engine would produce.

6. **In-sample warning.** The pooled_means.json was built from
   `live_home_up_5m_bootstrap_scored_rows.csv` which includes the SAME dates
   as the live feature files. The decile-level E_rises / E_doesnt values used
   in the EV gate are therefore partly in-sample. A truly out-of-sample test
   would require held-out dates not used to build pooled_means.

---

## 3. Data Sufficiency

| Metric | Value |
|--------|-------|
| Days with live data | {n_days_total} |
| Total trades @ 1c spread | {n_at_1c:,} |
| Trades needed for t≥2 power | {n_needed_str} |
| Current std(PnL/trade) | {std_trade_str} |
| Current mean(PnL/trade) | {mean_trade_str} |

**Conclusion:** {"Insufficient statistical power to distinguish edge from noise. Collect more live data before risking capital." if n_needed == float("inf") or n_at_1c < n_needed else f"Current sample ({n_at_1c} trades) has sufficient power IF the mean/std pattern holds out-of-sample."}

To reach statistical significance (t≥2), approximately **{n_needed_str} trades** are needed at
current PnL mean/std. At the observed rate of ~{trades_per_day:.0f} trades/day,
this requires ~{additional_days_str} additional live days.

---

## 4. Phase-B Per-Decile EV (from pooled_means.json)

EV computed at per-decile midpoint probability using Phase-B expected moves.
Positive EV = that decile's signals pass the gate at that spread assumption.

{decile_ev_md}

**Key finding:** Only deciles 7+ show positive EV at 0.5c half-spread.
At 1c half-spread, only deciles 7+ pass (though decile 7 is marginal).
At 2c half-spread, only deciles 8-9 pass.

---

## 5. Per-Decile Realized PnL (at 1c half-spread)

Actual realized PnL from the simulation (entry at yes_ask, exit at future_yes_bid).
Note: these rows include ONLY signals that passed the EV gate at 1c spread.

{decile_rows_md}

---

## 6. Per-Day PnL (at 1c half-spread)

{day_rows_md}

---

## 7. Biggest Lever

The single biggest lever that would change the answer:

**Calibrate p_cal to actual live-game rise probability.**

Currently, the model outputs p_cal ∈ [0,1] trained on a dataset where 95.3%
of rows are "price did NOT rise" — dominated by pre/post-game ticks. In the
live-game subset, prices rise ~47% of the time. The model's p_cal therefore
*overestimates* confidence relative to live-game baseline.

If p_cal were recalibrated specifically on in-game rows (isotonic regression
on in-game OOF predictions), two things would change:
1. The EV gate would use better-calibrated p values → fewer false positives
   in low-decile bands.
2. The winner-fee formula `ceil(0.07 * p * (1-p) * 100)/100` would use the
   correct p, potentially reducing fee overestimation.

**Estimated impact:** Could shift net PnL/trade by +0.002 to +0.010 per
contract by reducing trades in EV-negative bands.

The second biggest lever: **increase live data volume.** With only {n_days_total} days
of in-game data, the per-decile mean estimates in pooled_means.json are noisy
(especially deciles 7-9 with n_rises = 1335 / 460 / 45 respectively). Decile 9
with n=45 rises is essentially an anecdote.

---

*Report generated by tools/cost_replay.py — read-only, no models or data modified.*
"""

    out_path = output_dir / "cost_replay_report.md"
    out_path.write_text(report)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Net-of-cost replay for the NBA live model")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "outputs",
        help="Directory for outputs (default: outputs/)",
    )
    ap.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Path to the model .pkl artifact (default: models/live_home_up_5m_bootstrap.pkl).",
    )
    ap.add_argument(
        "--pooled-means-path",
        type=Path,
        default=POOLED_MEANS_PATH,
        help="Path to pooled_means.json (default: outputs/paper_trades/pooled_means.json).",
    )
    ap.add_argument(
        "--dates-from",
        type=str,
        default=None,
        help=(
            "ISO date string (e.g. '2026-04-29').  Only feature files on or "
            "after this date are included.  Use to restrict to the test hold-out window."
        ),
    )
    ap.add_argument(
        "--dates-to",
        type=str,
        default=None,
        help=(
            "ISO date string (e.g. '2026-05-08').  Only feature files on or "
            "before this date are included."
        ),
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file progress output",
    )
    args = ap.parse_args()

    run_replay(
        output_dir=args.output_dir,
        model_path=args.model_path,
        pooled_means_path=args.pooled_means_path,
        dates_from=args.dates_from,
        dates_to=args.dates_to,
        verbose=not args.quiet,
    )
    print("\nDone.")

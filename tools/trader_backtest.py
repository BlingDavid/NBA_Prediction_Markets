"""Replay one historical live_features_<date>.csv through the same pipeline.

Produces outputs in --output-dir mirroring the live structure. The replay
groups feed rows by captured_at into "ticks", processes each tick exactly as
the live engine would, then advances simulated time to the next captured_at.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from run_paper_trader import build_engine_context, process_tick


def _write_chunk(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path,
                        default=Path("models/live_home_up_5m_bootstrap.pkl"))
    parser.add_argument("--pooled-path", type=Path,
                        default=Path("outputs/paper_trades/pooled_means.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap rows for speed (debugging).")
    args = parser.parse_args(argv)

    full = pd.read_csv(args.features_path)
    if args.max_rows:
        full = full.head(args.max_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = args.output_dir / "_replay_chunk.csv"

    # Process tick-by-tick; each chunk is the rows at one captured_at.
    captured_groups = sorted(full["captured_at"].astype(str).unique())
    start = time.monotonic()
    for capture in captured_groups:
        chunk = full[full["captured_at"].astype(str) == capture]
        _write_chunk(chunk_path, chunk)
        ctx = build_engine_context(
            model_path=args.model_path,
            pooled_path=args.pooled_path,
            feed_path=chunk_path,
            output_dir=args.output_dir,
            bankroll=1000.0,
            mode_sizing="A",
            live_or_paper="paper",
        )
        process_tick(ctx)
    elapsed = time.monotonic() - start
    print(f"backtest done: ticks={len(captured_groups)} rows={len(full)} elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

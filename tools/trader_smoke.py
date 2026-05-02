"""10-minute smoke run: connects to the real feed, evaluates gates and
scoring, but rejects every trigger so no positions are opened."""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_paper_trader import build_engine_context, process_tick
from paper_trader import trigger as _trigger


def _force_reject(*_args, **_kwargs):
    return False, 0.0, "smoke_off"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=int, default=10)
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--features-path", type=Path,
                        default=Path("data/live/features/latest_features.csv"))
    parser.add_argument("--model-path", type=Path,
                        default=Path("models/live_home_up_5m_bootstrap.pkl"))
    parser.add_argument("--pooled-path", type=Path,
                        default=Path("outputs/paper_trades/pooled_means.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/paper_trades_smoke"))
    args = parser.parse_args(argv)

    _trigger.evaluate = _force_reject

    ctx = build_engine_context(
        model_path=args.model_path, pooled_path=args.pooled_path,
        feed_path=args.features_path, output_dir=args.output_dir,
        bankroll=1000.0, mode_sizing="A", live_or_paper="paper",
        interval_seconds=args.interval,
    )

    deadline = datetime.now(timezone.utc).timestamp() + args.minutes * 60
    ticks = 0
    errors = 0
    while datetime.now(timezone.utc).timestamp() < deadline:
        try:
            process_tick(ctx)
            ticks += 1
        except Exception:
            errors += 1
        time.sleep(args.interval)
    print(f"smoke complete: ticks={ticks} errors={errors} dir={args.output_dir}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

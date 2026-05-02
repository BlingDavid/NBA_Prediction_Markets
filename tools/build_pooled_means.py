"""One-shot: compute Phase A pooled conditional means from the bootstrap
scored_rows artifact.

Output: outputs/paper_trades/pooled_means.json with structure:
    {
      "source": "outputs/live_home_up_5m_bootstrap_scored_rows.csv",
      "rows_total": <int>,
      "rows_home_side_labeled": <int>,
      "delta_col": "label_yes_mid_move_5m",
      "label_col": "label_home_up_5m",
      "E_delta_given_rises": <float>,
      "E_delta_given_doesnt": <float>,
      "var_delta_pooled": <float>,
      "rises_count": <int>,
      "doesnt_count": <int>,
      "built_at": "<iso>"
    }
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SCORED_ROWS = Path("outputs/live_home_up_5m_bootstrap_scored_rows.csv")
OUT = Path("outputs/paper_trades/pooled_means.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-rows", type=Path, default=SCORED_ROWS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    cols = ["bet_side", "label_home_up_5m", "label_yes_mid_move_5m"]
    df = pd.read_csv(args.scored_rows, usecols=cols)
    df = df[df["bet_side"].astype(str).str.lower() == "home"]
    df = df.dropna(subset=["label_home_up_5m", "label_yes_mid_move_5m"])

    rises = df[df["label_home_up_5m"] == 1]["label_yes_mid_move_5m"]
    doesnt = df[df["label_home_up_5m"] == 0]["label_yes_mid_move_5m"]
    pooled_var = float(df["label_yes_mid_move_5m"].var(ddof=0))

    payload = {
        "source": str(args.scored_rows),
        "rows_total": int(len(df)),
        "rows_home_side_labeled": int(len(df)),
        "delta_col": "label_yes_mid_move_5m",
        "label_col": "label_home_up_5m",
        "E_delta_given_rises": float(rises.mean()),
        "E_delta_given_doesnt": float(doesnt.mean()),
        "var_delta_pooled": pooled_var,
        "rises_count": int(len(rises)),
        "doesnt_count": int(len(doesnt)),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}: rises={payload['rises_count']} "
          f"E[rises]={payload['E_delta_given_rises']:.6f} "
          f"E[doesnt]={payload['E_delta_given_doesnt']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

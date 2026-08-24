"""One-shot: compute Phase A pooled conditional means + the per-decile table
from the bootstrap scored_rows artifact. The decile_table lets the trader run
Phase B (per-decile expected moves) from boot, instead of waiting for 200
closed live trades — without it, pooled means dominate every signal and
upper-decile edge is invisible to the trigger.

Output: outputs/paper_trades/pooled_means.json with structure:
    {
      "source": "outputs/live_home_up_5m_bootstrap_scored_rows.csv",
      "rows_total": <int>,
      "rows_home_side_labeled": <int>,
      "delta_col": "label_yes_mid_move_5m",
      "label_col": "label_home_up_5m",
      "pred_col": "pred_bootstrap_home_up_5m",
      "E_delta_given_rises": <float>,
      "E_delta_given_doesnt": <float>,
      "var_delta_pooled": <float>,
      "rises_count": <int>,
      "doesnt_count": <int>,
      "decile_table": [{p_decile, E_rises, E_doesnt, var, rises_n, doesnt_n}, ... 10 rows],
      "built_at": "<iso>"
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from paper_trader.move_size_estimator import per_decile_means


SCORED_ROWS = Path("outputs/live_home_up_5m_bootstrap_scored_rows.csv")
OUT = Path("outputs/paper_trades/pooled_means.json")
PRED_COL = "pred_bootstrap_home_up_5m"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-rows", type=Path, default=SCORED_ROWS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    cols = ["bet_side", "label_home_up_5m", "label_yes_mid_move_5m", PRED_COL]
    df_all = pd.read_csv(args.scored_rows, usecols=cols)
    rows_total = int(len(df_all))
    df = df_all[df_all["bet_side"].astype(str).str.lower() == "home"]
    df = df.dropna(subset=["label_home_up_5m", "label_yes_mid_move_5m", PRED_COL])

    rises = df[df["label_home_up_5m"] == 1]["label_yes_mid_move_5m"]
    doesnt = df[df["label_home_up_5m"] == 0]["label_yes_mid_move_5m"]
    pooled_var = float(df["label_yes_mid_move_5m"].var(ddof=0))

    decile_input = df.rename(columns={PRED_COL: "p_calibrated"})[
        ["p_calibrated", "label_home_up_5m", "label_yes_mid_move_5m"]
    ]
    decile_table = per_decile_means(decile_input, delta_col="label_yes_mid_move_5m")

    payload = {
        "source": str(args.scored_rows),
        "rows_total": rows_total,
        "rows_home_side_labeled": int(len(df)),
        "delta_col": "label_yes_mid_move_5m",
        "label_col": "label_home_up_5m",
        "pred_col": PRED_COL,
        "E_delta_given_rises": float(rises.mean()),
        "E_delta_given_doesnt": float(doesnt.mean()),
        "var_delta_pooled": pooled_var,
        "rises_count": int(len(rises)),
        "doesnt_count": int(len(doesnt)),
        "decile_table": decile_table.to_dict(orient="records"),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}: rises={payload['rises_count']} "
          f"E[rises]={payload['E_delta_given_rises']:.6f} "
          f"E[doesnt]={payload['E_delta_given_doesnt']:.6f} "
          f"deciles={len(payload['decile_table'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

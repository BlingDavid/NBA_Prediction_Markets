"""Local dashboard for the NBA pre-game model.

A single page: a "Run today's picks" button (triggers run_pipeline.py in the
background) plus the current bet card as a sorted, color-coded table.
Read-only view of outputs/bet_card.csv + trigger the pre-game run — no trading.

Run with:  ./run.sh web    (or:  python webapp/app.py)
"""
from __future__ import annotations

import csv
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, redirect, render_template, url_for

REPO = Path(__file__).resolve().parent.parent
BET_CARD = REPO / "outputs" / "bet_card.csv"

app = Flask(__name__)

# Simple in-process run state (single-user local app).
RUN_STATE: dict = {"running": False, "started": None, "finished": None,
                   "returncode": None, "log_tail": ""}
_lock = threading.Lock()


def _run_pipeline() -> None:
    """Run the pre-game pipeline as a subprocess (inherits the env set by run.sh)."""
    proc = subprocess.run(
        [sys.executable, "run_pipeline.py", "--predict", "--min-edge", "0.05",
         "--bankroll", "2000"],
        cwd=str(REPO), capture_output=True, text=True,
    )
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-10:])
    with _lock:
        RUN_STATE.update(running=False,
                         finished=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         returncode=proc.returncode, log_tail=tail)


def _pct(v) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _money(v) -> str:
    try:
        return f"${float(v):,.0f}"
    except (TypeError, ValueError):
        return "—"


def load_bet_card():
    """Return (display_rows_sorted_by_edge_desc, last_updated_str)."""
    if not BET_CARD.exists():
        return [], None
    with open(BET_CARD, newline="") as fh:
        raw = list(csv.DictReader(fh))

    def edge_val(r) -> float:
        try:
            return float(r.get("edge") or 0)
        except ValueError:
            return 0.0

    raw.sort(key=edge_val, reverse=True)
    rows = []
    for r in raw:
        e = edge_val(r)
        rows.append({
            "game": f'{r.get("away_team", "")} @ {r.get("home_team", "")}',
            "pick": r.get("selection") or f'{r.get("side", "")} {r.get("market", "")}'.strip(),
            "edge_pct": f"{e * 100:+.1f}%",
            "model_pct": _pct(r.get("model_prob")),
            "market_pct": _pct(r.get("implied_prob")),
            "size": _money(r.get("bet_size")),
            "cls": "big" if e >= 0.06 else ("mid" if e >= 0.03 else "thin"),
        })
    updated = datetime.fromtimestamp(BET_CARD.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return rows, updated


@app.route("/")
def index():
    rows, updated = load_bet_card()
    with _lock:
        state = dict(RUN_STATE)
    return render_template("index.html", rows=rows, updated=updated, state=state)


@app.route("/run", methods=["POST"])
def run():
    with _lock:
        if not RUN_STATE["running"]:
            RUN_STATE.update(running=True,
                             started=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                             finished=None, returncode=None, log_tail="")
            threading.Thread(target=_run_pipeline, daemon=True).start()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)

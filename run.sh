#!/usr/bin/env bash
#
# run.sh — one-shot launcher for the NBA model.
# Handles the environment setup (venv + libomp) once, then runs a chosen command.
#
# Usage:  ./run.sh <command> [extra args...]
#         ./run.sh help
#
# Any unknown command is passed straight through with the environment set,
# e.g.  ./run.sh python tools/halftime_eval.py --help
#
set -uo pipefail

# --- Resolve repo dir from this script's location (works from anywhere) ---
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# --- Environment: venv + the libomp path xgboost needs ---
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "!! .venv not found in $REPO_DIR — create it or fix the path." >&2
  exit 1
fi
export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:-}:/opt/homebrew/opt/libomp/lib"

CMD="${1:-help}"; shift || true

case "$CMD" in
  # ---- Pre-game model ----
  pregame)   python run_pipeline.py --predict --min-edge "${MIN_EDGE:-0.05}" --bankroll "${BANKROLL:-2000}" "$@" ;;
  scan)      python run_pipeline.py --scan "$@" ;;
  backtest)  python run_pipeline.py --backtest "$@" ;;
  ev)        python ev_analyzer.py "$@" ;;               # e.g. ./run.sh ev KXNBAGAME-26MAY05LALOKC-LAL

  # ---- Live in-game paper trader ----
  trader)    python run_paper_trader.py "${@:---once}" ;; # default: single tick
  status)    python tools/trader_status.py "$@" ;;

  # ---- Training ----
  train-live)     python live_bootstrap_model.py "$@" ;;
  pooled-means)   python tools/build_pooled_means.py "$@" ;;
  train-halftime) python live_bootstrap_model.py --target label_halftime_home_lead --first-half-only \
                    --train-cutoff-date "${CUTOFF:-2026-04-29}" --model-name live_halftime_leader \
                    --models-dir models/candidates --outputs-dir outputs/candidates "$@" ;;

  # ---- Evaluation tools ----
  eval)          python tools/live_model_eval.py "$@" ;;
  cost)          python tools/cost_replay.py "$@" ;;
  eval-halftime) python tools/halftime_eval.py \
                    --model-path "${MODEL:-models/candidates/live_halftime_leader.pkl}" \
                    --dates-from "${FROM:-2026-04-29}" --dates-to "${TO:-2026-05-08}" \
                    --out "${OUT:-outputs/halftime_eval_report.md}" "$@" ;;

  # ---- Tests / interactive ----
  test)   python -m pytest "${@:-tests/}" -q ;;         # no args = full suite; else the given paths
  py)     exec python "$@" ;;                            # REPL (or run a .py) with env set
  web)    ( sleep 1.5 && open "http://localhost:8000" >/dev/null 2>&1 ) &
          exec python webapp/app.py "$@" ;;              # dashboard at http://localhost:8000

  help|-h|--help)
    echo "run.sh — one-shot launcher for the NBA model (handles venv + libomp, then runs a command)."
    echo ""
    echo "Usage:  ./run.sh <command> [extra args...]"
    echo "        ./run.sh python tools/halftime_eval.py --help   # unknown cmds pass through w/ env set"
    echo ""
    echo "Commands:"
    echo "  pregame [args]     predict + find edges   (env: MIN_EDGE, BANKROLL)"
    echo "  scan | backtest    market scan / walk-forward backtest"
    echo "  ev <TICKER>        single-game EV analysis"
    echo "  trader [args]      paper trader (default --once)"
    echo "  status             read-only trader status"
    echo "  train-live [args]  retrain the deployed live model"
    echo "  pooled-means       refresh Phase B seed after retraining"
    echo "  train-halftime     train the halftime-leader candidate (env: CUTOFF)"
    echo "  eval | cost        live-model eval / net-of-cost replay"
    echo "  eval-halftime      halftime hold-out eval (env: MODEL, FROM, TO, OUT)"
    echo "  web                launch the dashboard at http://localhost:8000"
    echo "  test [args]        run the pytest suite"
    echo "  py [file]          python REPL (or run a script) with env set"
    echo "  <anything else>    run it verbatim with the env set"
    ;;

  *) exec "$CMD" "$@" ;;                                 # passthrough with env set
esac

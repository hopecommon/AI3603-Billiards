#!/usr/bin/env bash

# Quick Test
# N_GAMES_ALL=1 QUIET=1 LOG_FLUSH_SECONDS=1 bash scripts/refresh_paper_data.sh

# Formal Run
# QUIET=1 bash scripts/refresh_paper_data.sh
# tail -f paper/results/run.log
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

# By default, respect per-match `n_games` specified in `experiments/suites/paper_suite.json`
# (final matchups use 120; ablations use fewer games to keep runtime reasonable).
# If you want to override ALL matches, set N_GAMES_ALL=...
N_GAMES_ALL="${N_GAMES_ALL:-}"
SEED="${SEED:-42}"
QUIET="${QUIET:-0}"
LOG="${LOG:-paper/results/run.log}"
mkdir -p "$(dirname "${LOG}")"
LOG_FLUSH_SECONDS="${LOG_FLUSH_SECONDS:-10}"

if [ -n "${N_GAMES_ALL}" ]; then
  echo "[1/4] Running experiment suite (override n_games=${N_GAMES_ALL}, seed=${SEED})..."
  if [ "${QUIET}" = "1" ]; then
    python experiments/run_suite.py --suite experiments/suites/paper_suite.json --out paper/results --n-games "${N_GAMES_ALL}" --seed "${SEED}" --fixed-seed --quiet --progress --log-file "${LOG}" --log-flush-seconds "${LOG_FLUSH_SECONDS}"
  else
    python experiments/run_suite.py --suite experiments/suites/paper_suite.json --out paper/results --n-games "${N_GAMES_ALL}" --seed "${SEED}" --fixed-seed --no-quiet --progress --log-file "${LOG}" --log-flush-seconds "${LOG_FLUSH_SECONDS}"
  fi
else
  echo "[1/4] Running experiment suite (use per-match n_games, seed=${SEED})..."
  if [ "${QUIET}" = "1" ]; then
    python experiments/run_suite.py --suite experiments/suites/paper_suite.json --out paper/results --seed "${SEED}" --fixed-seed --quiet --progress --log-file "${LOG}" --log-flush-seconds "${LOG_FLUSH_SECONDS}"
  else
    python experiments/run_suite.py --suite experiments/suites/paper_suite.json --out paper/results --seed "${SEED}" --fixed-seed --no-quiet --progress --log-file "${LOG}" --log-flush-seconds "${LOG_FLUSH_SECONDS}"
  fi
fi

echo "[2/4] Generating LaTeX tables/macros from results..."
python paper/update_results.py --results paper/results/summary.json

echo "[3/4] Regenerating figures..."
python paper/generate_figures.py --results paper/results/summary.json --outdir paper

echo "[4/4] Done. Compile the paper with:"
echo "  cd paper && ./compile.sh"
echo ""
echo "Artifacts:"
echo "  - paper/results/summary.json"
echo "  - paper/results/generated_results.tex"
echo "  - paper/results/generated_table_winrate.tex"
echo "  - paper/results/generated_table_ablation.tex"
echo "  - paper/figure*.pdf"
echo "  - ${LOG}"

# Experiments

This folder provides a reproducible evaluation harness to generate:

- `paper/results/summary.json` (aggregated suite results)
- `paper/results/<match_id>/{summary.json,games.jsonl}` (per-match structured logs)
- the win-rate/ablation tables and figures used by `paper/main.tex`

## Quick Start (End-to-End)

From repository root:

```bash
bash scripts/refresh_paper_data.sh
cd paper && ./compile.sh
```

Progress bars:
- If `tqdm` is installed, `run_suite.py` shows per-match progress and `run_match.py` shows per-game progress.
- If `tqdm` is not installed, scripts still run normally without progress bars.
Tip: for long runs, set `QUIET=1` when using `scripts/refresh_paper_data.sh` to keep the terminal clean while preserving progress bars.
Logs are always appended to `paper/results/run.log` by default; override with `LOG=...`.

## Run a Single Match

```bash
python experiments/run_match.py \
  --agent-a BasicAgentPro \
  --agent-b NewAgent \
  --agent-b-config experiments/configs/final.json \
  --n-games 120 \
  --seed 42 \
  --fixed-seed \
  --quiet \
  --out paper/results/final_vs_pro
```

Outputs:
- `paper/results/final_vs_pro/summary.json`: aggregate metrics (win rate, avg time, over-180s rate)
- `paper/results/final_vs_pro/games.jsonl`: one JSON record per game

## Run the Full Suite (Paper)

```bash
python experiments/run_suite.py \
  --suite experiments/suites/paper_suite.json \
  --out paper/results \
  --seed 42 \
  --fixed-seed \
  --quiet
```

The suite file defines per-match `n_games` (final matchups use more games; ablations use fewer to save time).
If you want to override **all** matches to the same game count, use `--n-games` (or `N_GAMES_ALL` with `scripts/refresh_paper_data.sh`):

```bash
python experiments/run_suite.py --suite experiments/suites/paper_suite.json --out paper/results --n-games 120 --seed 42 --fixed-seed --quiet
N_GAMES_ALL=120 bash scripts/refresh_paper_data.sh
```

## Ablations & Configs

`experiments/configs/*.json` are **agent overrides** passed to `NewAgent(config=...)`.

Typical patterns:
- `final.json`: full system
- `no_cma.json`: disables CMA-ES refinement
- `no_pruning.json`: expands candidate generation (more targets/pockets/candidates)
- `no_strategy.json`: disables strategic bonus and safety planning
- `with_catastrophic_penalty.json`: enables an additional soft catastrophic foul penalty (ablation; default is disabled)
- `ghost_only.json`: near-geometry-only baseline

## Notes on the 180s Budget

The suite reports `over_budget_rate`, which is the fraction of games whose wall-clock time exceeded 180 seconds.
This is a **reference average budget** indicator (not a strict per-game disqualification rule).

# Repository Guidelines

This repository contains an 8-ball pool (billiards) AI project built around the provided `PoolEnv` simulator and pluggable agents in `agents/`.

## Project Structure & Module Organization

- `poolenv.py`: game environment and rules (course-provided; treat as read-only for final evaluation).
- `agents/`: agent implementations.
  - `agents/basic_agent.py`, `agents/basic_agent_pro.py`: baseline opponents.
  - `agents/new_agent.py`: your main submission agent (exported as `NewAgent`).
- `evaluate.py`: simple local tournament script (prints verbose logs).
- `experiments/`: reproducible experiment suite used by the paper (structured JSON outputs).
- `paper/`: IEEE-style report (`paper/main.tex`) and figure generator (`paper/generate_figures.py`).

## Build, Test, and Development Commands

- Run a quick match (verbose): `python evaluate.py`
- Run the paper experiment suite (structured outputs):  
  `python experiments/run_suite.py --suite experiments/suites/paper_suite.json --out paper/results --n-games 120 --seed 42 --fixed-seed --quiet`
- Refresh paper tables + figures from results: `bash scripts/refresh_paper_data.sh`
- Compile the paper (interactive): `cd paper && ./compile.sh`  
  Non-interactive: `cd paper && bash compile_ci.sh`

## Coding Style & Naming Conventions

- Python: 4-space indentation, keep functions small and single-purpose.
- Prefer explicit names (`max_target_balls`, `timeout_base`) over abbreviations.
- Keep `NewAgent()` backward-compatible; optional configs should not change defaults unexpectedly.

## Testing Guidelines

No unit-test suite is provided. Validate changes by running:
- `python experiments/run_match.py ... --n-games 1` for smoke tests.
- Full 120-game runs for final numbers used in `paper/`.

## Commit & Pull Request Guidelines

- Commit messages are short and imperative (often Chinese/English mixed), e.g. “优化…”, “update”.
- PRs should include: what changed, how to reproduce results (commands), and updated `paper/results/*` artifacts if paper numbers changed.

## Configuration & Reproducibility Tips

- Use `--fixed-seed` for reproducible tournaments; record `paper/results/summary.json` alongside generated figures and tables.
- `agents/new_agent.py` uses timeouts; results can vary across hardware. Prefer reporting the structured suite outputs for consistency.


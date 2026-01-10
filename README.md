# AI3603-Billiards

AI3603 course project: a pool (8-ball) agent built on the provided `PoolEnv` simulator, with reproducible experiments and an IEEE-format report.

## Repository Layout

| Path | What it is | Notes |
|---|---|---|
| `poolenv.py` | Environment + rules | Course-provided; treat as read-only for final evaluation |
| `agents/` | Agents | `agents/new_agent.py` exports `NewAgent` (our final agent) |
| `evaluate.py` | Local quick evaluation | Useful for manual debugging |
| `experiments/` | Reproducible tournament runner | Produces structured `summary.json` outputs |
| `scripts/refresh_paper_data.sh` | Paper data pipeline | Runs suite → generates tables → regenerates figures |
| `paper/` | IEEE report | `paper/main.tex`, `paper/generate_figures.py` |

## Environment Setup (Recommended: Python 3.10 + uv)

We recommend Python **3.10** (the paper results were produced on macOS arm64 with Python 3.10.19).

1) Install `uv` (see https://github.com/astral-sh/uv)

2) Sync dependencies from `pyproject.toml` / `uv.lock`:

```bash
uv sync --python 3.10
```

**⚠️ Important:** The `--python 3.10` flag is **required** when using `uv sync`. Without it, `uv` may default to Python 3.12+ which will fail due to missing `panda3d` wheels for those versions. If you see dependency resolution errors mentioning `panda3d` or `pooltool-billiards`, ensure you're using Python 3.10 explicitly.

3) Sanity check key deps:

```bash
uv run python -c "import pooltool, cma, numpy; print('ok')"
```

### Alternative (pip)

If you do not use `uv`, make sure at least `cma` and `pooltool-billiards` are installed:

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
python -c "import pooltool, cma; print('ok')"
```

When using `pip`, ensure you're using Python 3.10 (not 3.12+) to avoid dependency conflicts with `panda3d`.

## Quick Run (Smoke Test)

Run a small match with verbose logs:

```bash
python experiments/run_match.py --agent-a BasicAgent --agent-b NewAgent \
  --agent-b-config experiments/configs/final.json \
  --n-games 1 --seed 42 --fixed-seed \
  --progress --out /tmp/match_smoke
```

## Reproducing Paper Results

One command to regenerate the paper data pipeline:

```bash
bash scripts/refresh_paper_data.sh -f
```

Artifacts:
- `paper/results/summary.json` (authoritative numbers)
- `paper/results/generated_results.tex` (LaTeX macros)
- `paper/results/generated_table_*.tex` (LaTeX tables)
- `paper/figure*.pdf` (figures used in the report)

## Building the Report (PDF)

CI-style build:

```bash
bash paper/compile_ci.sh
```

Interactive build (opens `.aux`/`bbl` as needed):

```bash
cd paper && ./compile.sh
```

## Notes

- Runtime metrics depend on hardware; always report the structured suite outputs (`paper/results/summary.json`) rather than ad-hoc logs.
- The 180s budget used in plots/tables is a reference budget (slight overruns may be tolerated by the evaluation harness).

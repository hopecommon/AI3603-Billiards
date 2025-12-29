# Paper & Experiments Reproducibility Guide

This folder contains the IEEE-style report (`paper/main.tex`) and all scripts needed to regenerate the **numbers and figures** used in the paper from scratch.

## Prerequisites

- Python environment that can import `pooltool` and run the simulator.
- Python packages typically needed for the paper pipeline:
  - `numpy`
  - `matplotlib` (for figure generation)
  - `bayesian-optimization` (used by baseline agents)
- LaTeX toolchain for IEEE papers:
  - `pdflatex`, `bibtex`

Notes:
- `agents/new_agent.py` optionally uses `cma` (CMA-ES). If `cma` is not installed, the agent will automatically skip CMA-ES and still run.
- The course “180s” setting is treated as a **reference average budget**, not a strict per-game cutoff. We report an **Over-180s rate** as a diagnostic metric.

## One-Command Workflow (Recommended)

From the repository root:

```bash
bash scripts/refresh_paper_data.sh
cd paper && ./compile.sh
```

If you prefer fewer logs while keeping progress bars:

```bash
QUIET=1 bash scripts/refresh_paper_data.sh
```

Logs are always appended to `paper/results/run.log` by default. Override with:

```bash
LOG=paper/results/my_run.log bash scripts/refresh_paper_data.sh
```

To reduce perceived buffering when `tail -f` watching logs, you can force more frequent flushes:

```bash
LOG_FLUSH_SECONDS=2 bash scripts/refresh_paper_data.sh
```

This does:
1. Runs the full experiment suite (matchups + ablations) and writes structured results to `paper/results/`.
2. Generates LaTeX tables/macros in `paper/results/generated_*.tex`.
3. Regenerates figures `paper/figure*.pdf`.
4. Compiles `paper/main.pdf`.

## What Gets Generated (Artifacts)

After running the suite, key outputs are:

- Aggregated suite summary: `paper/results/summary.json`
- Per-match outputs: `paper/results/<match_id>/{summary.json,games.jsonl}`
- Paper macros (numbers): `paper/results/generated_results.tex`
- Paper tables: `paper/results/generated_table_winrate.tex`, `paper/results/generated_table_ablation.tex`
- Figures: `paper/figure1_pipeline.pdf` … `paper/figure5_case_study.pdf`

## Running Experiments Manually

### 1) Run a single matchup

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

### 2) Run the full “paper suite”

```bash
python experiments/run_suite.py \
  --suite experiments/suites/paper_suite.json \
  --out paper/results \
  --n-games 120 \
  --seed 42 \
  --fixed-seed \
  --quiet
```

You can override the seed via environment variables when using `scripts/refresh_paper_data.sh`:

```bash
SEED=7 bash scripts/refresh_paper_data.sh
```

If you want to override **all** matches to the same game count (not recommended for the default suite, because ablations are intentionally smaller), use:

```bash
N_GAMES_ALL=120 bash scripts/refresh_paper_data.sh
```

## Updating Paper Numbers and Figures

### Update tables/macros from results

```bash
python paper/update_results.py --results paper/results/summary.json
```

### Regenerate figures (reads results first; falls back to logs if missing)

```bash
python paper/generate_figures.py --results paper/results/summary.json --outdir paper
```

## Compiling the PDF

Interactive compilation:

```bash
cd paper && ./compile.sh
```

Non-interactive (CI-style):

```bash
cd paper && bash compile_ci.sh
```

If citations or links do not resolve, ensure you run BibTeX (both scripts do this) and recompile twice.

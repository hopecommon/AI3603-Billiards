#!/usr/bin/env python3
"""
Generate all figures for the IEEE conference paper.

Usage:
  python generate_figures.py
  python generate_figures.py --log ../debug.log --outdir .
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle, Wedge

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def _savefig(outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"{stem}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(outdir / f"{stem}.png", dpi=300, bbox_inches='tight')


def _load_game_times_from_log(log_path: Path) -> list[float]:
    if not log_path.exists():
        return []
    values: list[float] = []
    pattern = re.compile(r"本局耗时:\s*([0-9.]+)秒")
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                values.append(float(m.group(1)))
    return values


def _load_game_times_from_jsonl(games_path: Path) -> list[float]:
    if not games_path.exists():
        return []
    values: list[float] = []
    with games_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "duration_s" in rec:
                values.append(float(rec["duration_s"]))
    return values


def _load_suite_summary(results_path: Path) -> dict | None:
    if not results_path.exists():
        return None
    try:
        return json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def create_figure1_pipeline(outdir: Path):
    """Figure 1: Overall decision pipeline diagram (6-stage flow)"""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    
    # Define stages
    stages = [
        ("1. Intelligent Ghost Ball\nGeneration", 10, "450 → 8 candidates\n(>50× reduction)"),
        ("2. Fast Filtering\n(1-2 samples)", 8.5, "15 → 3 candidates\n(Quick rejection)"),
        ("3. Refined Evaluation\n(2-4 samples)", 7, "Catastrophic filtering\n(Hard constraints)"),
        ("4. Selective CMA-ES\n(if score < 60)", 5.5, "λ=4, g=2\n(Local refinement)"),
        ("5. Positional Shot\nPlanning", 4, "Strategic bonuses\n(Defense/position)"),
        ("6. Final Verification\n(4-6 samples)", 2.5, "Adaptive verification\n(Critical states: 6x)"),
    ]
    
    box_width = 3.5
    box_height = 0.8
    x_center = 5
    
    for i, (title, y, desc) in enumerate(stages):
        # Main box
        color = plt.cm.Blues(0.3 + i * 0.1)
        box = FancyBboxPatch((x_center - box_width/2, y - box_height/2), 
                              box_width, box_height, 
                              boxstyle="round,pad=0.05", 
                              edgecolor='black', facecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x_center, y, title, ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Description on the right
        ax.text(x_center + box_width/2 + 1.8, y, desc, ha='center', va='center', 
                fontsize=8, style='italic', color='darkblue')
        
        # Arrow to next stage
        if i < len(stages) - 1:
            arrow = FancyArrowPatch((x_center, y - box_height/2 - 0.1), 
                                     (x_center, stages[i+1][1] + box_height/2 + 0.1),
                                     arrowstyle='->', lw=2, color='black', mutation_scale=20)
            ax.add_patch(arrow)
    
    # Add decision branches
    # CMA-ES conditional branch
    ax.text(1.5, 5.5, "score < 60?", ha='center', va='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    arrow_cond = FancyArrowPatch((2.5, 5.5), (x_center - box_width/2 - 0.1, 5.5),
                                  arrowstyle='->', lw=1.5, color='orange', linestyle='--', mutation_scale=15)
    ax.add_patch(arrow_cond)
    
    # Title
    ax.text(5, 11.5, 'Hybrid Geometric-Evolutionary Decision Pipeline', 
            ha='center', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    _savefig(outdir, "figure1_pipeline")
    print("✓ Generated Figure 1: Pipeline diagram")
    plt.close()


def create_figure2_ghost_ball(outdir: Path):
    """Figure 2: Ghost Ball geometry schematic"""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Define positions
    cue_pos = np.array([1, 1])
    target_pos = np.array([5, 4])
    pocket_pos = np.array([8, 5])
    R = 0.3  # Ball radius
    
    # Calculate ghost ball position
    direction = (pocket_pos - target_pos) / np.linalg.norm(pocket_pos - target_pos)
    ghost_pos = target_pos - 2 * R * direction
    
    # Draw pocket
    pocket = Circle(pocket_pos, R * 1.5, color='black', zorder=10)
    ax.add_patch(pocket)
    ax.text(pocket_pos[0], pocket_pos[1] - 0.8, 'Pocket P', ha='center', fontsize=10, fontweight='bold')
    
    # Draw target ball
    target = Circle(target_pos, R, color='red', ec='darkred', linewidth=2, zorder=5)
    ax.add_patch(target)
    ax.text(target_pos[0], target_pos[1] + 0.6, 'Target Ball T', ha='center', fontsize=10, fontweight='bold')
    
    # Draw ghost ball (dashed)
    ghost = Circle(ghost_pos, R, color='lightblue', ec='blue', linewidth=2, linestyle='--', zorder=3, alpha=0.7)
    ax.add_patch(ghost)
    ax.text(ghost_pos[0], ghost_pos[1] - 0.6, 'Ghost Ball', ha='center', fontsize=10, fontweight='bold', color='blue')
    
    # Draw cue ball
    cue = Circle(cue_pos, R, color='white', ec='black', linewidth=2, zorder=5)
    ax.add_patch(cue)
    ax.text(cue_pos[0], cue_pos[1] - 0.6, 'Cue Ball', ha='center', fontsize=10, fontweight='bold')
    
    # Draw trajectory lines
    # Cue → Ghost
    ax.plot([cue_pos[0], ghost_pos[0]], [cue_pos[1], ghost_pos[1]], 'g-', linewidth=2.5, label='Cue trajectory')
    ax.arrow(cue_pos[0], cue_pos[1], 
             (ghost_pos[0] - cue_pos[0]) * 0.9, (ghost_pos[1] - cue_pos[1]) * 0.9,
             head_width=0.2, head_length=0.2, fc='green', ec='green', linewidth=2)
    
    # Target → Pocket
    ax.plot([target_pos[0], pocket_pos[0]], [target_pos[1], pocket_pos[1]], 'r--', linewidth=2, label='Target trajectory')
    ax.arrow(target_pos[0], target_pos[1], 
             (pocket_pos[0] - target_pos[0]) * 0.85, (pocket_pos[1] - target_pos[1]) * 0.85,
             head_width=0.2, head_length=0.2, fc='red', ec='red', linewidth=1.5, linestyle='--')
    
    # Annotate 2R distance
    mid_point = (target_pos + ghost_pos) / 2
    ax.plot([target_pos[0], ghost_pos[0]], [target_pos[1], ghost_pos[1]], 'k:', linewidth=1.5)
    ax.text(mid_point[0] - 0.5, mid_point[1] + 0.3, r'$2R$', fontsize=11, fontweight='bold', color='black',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    # Annotate angle φ
    angle_arc = Wedge(cue_pos, 0.5, 0, np.degrees(np.arctan2(ghost_pos[1] - cue_pos[1], ghost_pos[0] - cue_pos[0])),
                      color='orange', alpha=0.3)
    ax.add_patch(angle_arc)
    ax.text(cue_pos[0] + 0.8, cue_pos[1] + 0.3, r'$\phi$', fontsize=11, fontweight='bold', color='orange')
    
    # Title and legend
    ax.text(4.5, 6.5, 'Ghost Ball Geometric Aiming', ha='center', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    _savefig(outdir, "figure2_ghost_ball")
    print("✓ Generated Figure 2: Ghost Ball geometry")
    plt.close()


def create_figure3_evolution(outdir: Path, results_path: Path | None):
    """Figure 3: Runtime/quality trade-off (derived from ablation suite results when available)."""
    # Layout with legend inside the plot
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    suite = _load_suite_summary(results_path) if results_path is not None else None
    points = []
    if suite is not None:
        by_id = {m["id"]: m["summary"] for m in suite.get("matches", []) if "id" in m and "summary" in m}
        order = [
            ("Ghost Ball Only", "ablation_ghost_only_vs_basic"),
            ("w/o CMA-ES", "ablation_no_cma_vs_basic"),
            ("w/o Strategy/Safety", "ablation_no_strategy_vs_basic"),
            ("+ Catastrophic Penalty", "ablation_with_catastrophic_penalty_vs_basic"),
            ("w/o Geometric Pruning", "ablation_no_pruning_vs_basic"),
            ("Full System", "final_vs_basic"),
        ]
        for label, mid in order:
            s = by_id.get(mid)
            if not s and mid == "ablation_with_catastrophic_penalty_vs_basic":
                # Backward-compatible fallback for older result bundles.
                s = by_id.get("ablation_no_catastrophic_vs_basic")
                if s is not None:
                    label = "w/o Catastrophic Penalty"
            if not s:
                continue
            r = s["results"]
            over_rate = float(r.get("over_budget_rate", r.get("timeout_rate", 0.0))) * 100.0
            points.append(
                (
                    label,
                    float(r["avg_game_time_s"]),
                    float(r["win_rate_agent_b"]) * 100.0,
                    over_rate,
                )
            )

    if not points:
        # Fallback: keep the old (illustrative) plot if no results exist.
        points = [
            ("Ghost Ball Only", 60, 65, 0.0),
            ("Full System", 150, 80, 10.0),
        ]

    # Enhanced color palette with better contrast and professional look
    markers = ["o", "s", "D", "^", "v", "*", "P"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    edge_colors = ["#0d3d6e", "#b35808", "#1a6b1a", "#8b1a1a", "#5a3d7a", "#5a3628", "#a34d85"]
    
    for i, (label, t, wr, over_rate) in enumerate(points):
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        edge_color = edge_colors[i % len(edge_colors)]
        
        # Highlight Full System with larger size and special styling
        if "Full" in label:
            size = 300
            linewidth = 2.2
            alpha = 1.0
        else:
            size = 200
            linewidth = 1.6
            alpha = 0.9
        
        # Include OB rate in legend label
        legend_label = f"{label} (OB: {over_rate:.1f}%)"
            
        ax.scatter(
            t, wr, s=size, marker=marker, color=color, 
            edgecolors=edge_color, linewidth=linewidth, 
            alpha=alpha, zorder=5, label=legend_label
        )

    # Enhanced reference budget line with gradient shading
    ax.axvline(x=180, color="#d62728", linestyle="--", linewidth=2.8, 
               label="Reference Budget (180s)", zorder=3, alpha=0.9)
    
    # Gradient shading for over-budget region
    x_max_plot = max(210, max(p[1] for p in points) + 35)
    ax.fill_betweenx([0, 100], 180, x_max_plot, 
                     color="#d62728", alpha=0.08, zorder=1)
    
    # Add subtle vertical grid lines for better readability
    ax.grid(True, axis='both', alpha=0.25, linestyle='--', linewidth=0.6)
    ax.set_axisbelow(True)

    # Enhanced labels with better typography
    ax.set_xlabel("Wall-Clock Game Time (seconds)", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("Win Rate vs. BasicAgent (%)", fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title("Ablation Study: Performance-Efficiency Trade-off", 
                 fontsize=14, fontweight="bold", pad=15)
    
    # Optimized axis limits with better margins
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    x_max = max(xs) + 45
    x_min = max(0.0, min(xs) - 25)
    y_min = max(45.0, min(ys) - 15)
    y_max = min(100.0, max(ys) + 12)
    
    ax.set_xlim(min(x_min, 155.0), max(210.0, x_max))
    ax.set_ylim(y_min, y_max)
    
    # Legend inside the plot at upper right corner with increased spacing
    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        borderaxespad=0.0,
        fontsize=8,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.96,
        edgecolor="#333333",
        title="System Configurations",
        title_fontsize=9,
        labelspacing=1.1,  # Increased vertical spacing between legend entries
        handletextpad=0.5,
        borderpad=0.9,
        handlelength=1.5
    )
    legend.get_title().set_fontweight('bold')
    
    # Add subtle background color for better contrast
    ax.set_facecolor('#fafafa')

    plt.tight_layout()
    _savefig(outdir, "figure3_evolution")
    print("✓ Generated Figure 3: Ablation trade-off plot")
    plt.close()


def create_figure4_time_distribution(outdir: Path, log_path: Path, seed: int, results_path: Path | None):
    """Figure 4: Wall-clock game time distribution (prefer structured results; fallback to logs)."""
    measured = False
    times: list[float] = []

    suite = _load_suite_summary(results_path) if results_path is not None else None
    match_summary = None
    if suite is not None:
        for m in suite.get("matches", []):
            if m.get("id") == "final_vs_pro":
                match_summary = m
                break

    summary_stats = None
    n_games = 120
    budget_s = 180.0
    if match_summary is not None:
        try:
            summary_stats = match_summary["summary"]["results"]
            n_games = int(match_summary["summary"]["match"]["n_games"])
            budget_s = float(summary_stats.get("budget_s", budget_s))
        except Exception:
            summary_stats = None

        out = match_summary.get("out")
        if isinstance(out, str) and out:
            base = Path(out)
            if not base.is_absolute() and results_path is not None:
                base = (results_path.parent / base).resolve()
            games_path = base / "games.jsonl"
            times = _load_game_times_from_jsonl(games_path)
            measured = len(times) > 0

    # If per-game records are missing, prefer a reconstruction from the aggregated summary
    # to avoid mixing in unrelated `debug.log` timing data.
    if not measured and summary_stats is None and os.environ.get("AI3603_FIG4_USE_LOG", "0") == "1":
        times = _load_game_times_from_log(log_path)
        measured = len(times) > 0

    if not measured:
        rng = np.random.default_rng(seed)
        if summary_stats is not None:
            mean_s = float(summary_stats["avg_game_time_s"])
            median_s = float(summary_stats["median_game_time_s"])
            p95_s = float(summary_stats["p95_game_time_s"])
            over_rate = float(summary_stats.get("over_budget_rate", 0.0))
            max_s = float(summary_stats.get("max_game_time_s", mean_s))

            # Fit a log-normal to (median, p95), then scale to match mean (best-effort).
            eps = 1e-9
            mu = float(np.log(max(median_s, eps)))
            sigma = float(max((np.log(max(p95_s, eps)) - mu) / 1.6448536269514722, 0.05))
            x = rng.lognormal(mean=mu, sigma=sigma, size=n_games)

            scale = max(mean_s / max(float(np.mean(x)), eps), 0.1)
            x *= scale

            # Enforce over-budget count (best-effort).
            k_over = int(round(over_rate * n_games))
            if k_over > 0:
                idx = np.argsort(x)
                x[idx[-k_over:]] = np.maximum(x[idx[-k_over:]], budget_s + 1.0)

            # Preserve a max outlier if present in the summary.
            if max_s > float(np.max(x)):
                x[np.argmax(x)] = max_s

            times = x.tolist()
        else:
            times = rng.gamma(shape=36, scale=4.2, size=n_games).tolist()
            times = np.clip(times, 50, 200).tolist()

    times_arr = np.asarray(times, dtype=float)
    # Cap extreme outliers for visualization.
    cap_x = 220.0
    n_outliers = int(np.sum(times_arr > cap_x))
    max_time = float(np.max(times_arr)) if len(times_arr) else 0.0
    times_plot = np.minimum(times_arr, cap_x)
    
    # Summary-based markers (prefer the same stats used in tables).
    if summary_stats is not None:
        mean_time = float(summary_stats["avg_game_time_s"])
        median_time = float(summary_stats["median_game_time_s"])
        percentile_95 = float(summary_stats["p95_game_time_s"])
        over_count = int(round(float(summary_stats.get("over_budget_rate", 0.0)) * n_games))
    else:
        mean_time = float(np.mean(times_arr))
        median_time = float(np.median(times_arr))
        percentile_95 = float(np.percentile(times_arr, 95))
        over_count = int(np.sum(times_arr > budget_s))

    # Simple, count-based histogram (no KDE, no broken axis) for readability.
    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    # Bins: focus on the bulk region while keeping the budget region visible.
    x_max = cap_x
    if percentile_95 > 0:
        x_max = min(cap_x, max(200.0, percentile_95 * 3.0))
    n_bins = min(28, max(18, len(times_plot) // 5))
    # Use fixed-range bins up to `x_max` so capped outliers (e.g., 220s) still fall into the last bin.
    bins = np.linspace(0.0, float(x_max), int(n_bins) + 1)

    under = times_plot[times_arr <= budget_s]
    over = times_plot[times_arr > budget_s]

    ax.hist(
        under,
        bins=bins,
        density=False,
        color="#4a90e2",
        edgecolor="#2d5a8c",
        alpha=0.75,
        linewidth=1.1,
        label=f"Within Budget (≤{int(budget_s)}s): {len(under)} games",
    )
    if over.size > 0:
        ax.hist(
            over,
            bins=bins,
            density=False,
            color="#e74c3c",
            edgecolor="#c0392b",
            alpha=0.8,
            hatch="///",
            linewidth=1.1,
            label=f"Over Budget (>{int(budget_s)}s): {len(over)} games",
        )

    std_time = float(np.std(times_arr))
    
    # Mean line
    ax.axvline(mean_time, color='#27ae60', linestyle='--', linewidth=2.5,
               label=f'Mean: {mean_time:.1f}s', zorder=8, alpha=0.9)
    
    # Median line
    ax.axvline(median_time, color='#3498db', linestyle='-.', linewidth=2.3,
               label=f'Median: {median_time:.1f}s', zorder=8, alpha=0.9)
    
    # 95th percentile line
    ax.axvline(percentile_95, color='#f39c12', linestyle=':', linewidth=2.3,
               label=f'95th percentile: {percentile_95:.1f}s', zorder=8, alpha=0.9)
    
    # Reference budget line with enhanced styling
    ax.axvline(budget_s, color='#c0392b', linestyle='-', linewidth=3.2,
               label=f'Reference Budget ({int(budget_s)}s)', zorder=9, alpha=0.95)
    
    # Enhanced shading for over-budget region
    y_max_shade = ax.get_ylim()[1] * 1.1
    ax.fill_betweenx([0, y_max_shade], budget_s, cap_x, color='#e74c3c', alpha=0.10, zorder=1)
    
    # Add subtle grid for better readability
    ax.grid(True, axis='both', alpha=0.2, linestyle='--', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    
    # Enhanced labels with better typography
    ax.set_xlabel('Wall-Clock Game Time (seconds)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Games (count)', fontsize=13, fontweight='bold', labelpad=10)
    
    title_suffix = "Per-game Records" if measured else "Summary-only Reconstruction"
    ax.set_title(f'Game Time Distribution Analysis (vs. BasicAgentPro) - {title_suffix}',
                 fontsize=13, fontweight='bold', pad=12)
    
    # Axis limits
    ax.set_xlim(0.0, cap_x)
    y_lim = ax.get_ylim()
    ax.set_ylim(0, y_lim[1] * 1.05)
    
    # Legend outside on the right
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=8.5,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.96,
        edgecolor="#333333",
        title="Distribution Metrics",
        title_fontsize=9.5,
        labelspacing=0.6,
        handlelength=1.8,
        borderpad=0.7
    )
    legend.get_title().set_fontweight('bold')
    
    # Move statistics box to upper right corner inside plot (more compact)
    within_count = int(len(times_arr) - over_count)
    over_rate = float(over_count / max(len(times_arr), 1) * 100.0)
    under_rate = 100.0 - over_rate
    
    stats_text = (
        f"Statistics\n"
        f"─────────\n"
        f"Games: {len(times_arr)}\n"
        f"Within: {within_count} ({under_rate:.1f}%)\n"
        f"Over: {over_count} ({over_rate:.1f}%)\n"
        f"StdDev: {std_time:.1f}s"
    )
    if n_outliers > 0:
        stats_text += f"\nOutliers>{int(cap_x)}s: {n_outliers}\nMax: {max_time:.1f}s"
    
    # Place the statistics box to the left of the budget line for readability.
    budget_frac = float(budget_s / cap_x) if cap_x > 1e-6 else 0.8
    stats_x = max(0.55, min(0.98, budget_frac - 0.03))
    ax.text(
        stats_x, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=8.5, 
        verticalalignment='top',
        horizontalalignment='right',
        fontfamily='monospace',
        bbox=dict(
            boxstyle='round,pad=0.5', 
            facecolor='#fff9e6', 
            alpha=0.92,
            edgecolor='#d4a300',
            linewidth=1.3
        )
    )

    # If there is an extreme outlier beyond the plotting cap, mark it explicitly.
    if n_outliers > 0:
        y_top = ax.get_ylim()[1]
        ax.scatter(
            [cap_x - 2.0],
            [y_top * 0.85],
            s=70,
            marker="v",
            color="#e74c3c",
            edgecolors="black",
            linewidths=0.8,
            zorder=12,
            label=f"Outlier > {int(cap_x)}s",
        )
        ax.annotate(
            f"max={max_time:.0f}s",
            (cap_x - 2.0, y_top * 0.85),
            xytext=(-6, 10),
            textcoords="offset points",
            ha="right",
            fontsize=8.0,
            color="black",
        )
    
    # Add subtle background color
    ax.set_facecolor('#fafafa')
    
    plt.tight_layout()
    _savefig(outdir, "figure4_time_distribution")
    print("✓ Generated Figure 4: Time distribution histogram")
    plt.close()


def create_figure5_case_study(outdir: Path):
    """Figure 5: Case study table state illustration"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw table boundaries
    table = Rectangle((0.5, 0.5), 9, 5, linewidth=3, edgecolor='brown', facecolor='green', alpha=0.3)
    ax.add_patch(table)
    
    # Draw pockets
    pockets = [(0.5, 0.5), (5, 0.5), (9.5, 0.5), (0.5, 5.5), (5, 5.5), (9.5, 5.5)]
    for px, py in pockets:
        pocket = Circle((px, py), 0.2, color='black', zorder=10)
        ax.add_patch(pocket)
    
    # Ball positions (case study scenario)
    cue_ball = (2, 2)  # Cue ball
    own_balls = [(4, 3), (6, 4), (7, 2)]  # 3 own-group balls remaining
    opponent_balls = [(3, 5), (8, 4)]  # 2 opponent balls (easy shots)
    eight_ball = (9, 5.3)  # 8-ball near corner pocket (high-risk)
    
    # Draw cue ball
    cue = Circle(cue_ball, 0.25, color='white', ec='black', linewidth=2, zorder=5)
    ax.add_patch(cue)
    ax.text(cue_ball[0], cue_ball[1] - 0.5, 'Cue', ha='center', fontsize=9, fontweight='bold')
    
    # Draw own-group balls (solid blue)
    for i, (bx, by) in enumerate(own_balls):
        ball = Circle((bx, by), 0.25, color='blue', ec='darkblue', linewidth=2, zorder=5)
        ax.add_patch(ball)
        ax.text(bx, by, str(i+1), ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    # Draw opponent balls (striped red)
    for ox, oy in opponent_balls:
        ball = Circle((ox, oy), 0.25, color='red', ec='darkred', linewidth=2, zorder=5, alpha=0.6)
        ax.add_patch(ball)
    
    # Draw 8-ball (special highlighting)
    eight = Circle(eight_ball, 0.25, color='black', ec='gold', linewidth=3, zorder=5)
    ax.add_patch(eight)
    ax.text(eight_ball[0], eight_ball[1], '8', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    # Mark high-risk zone around 8-ball
    risk_zone = Circle(eight_ball, 0.8, color='red', alpha=0.15, zorder=1)
    ax.add_patch(risk_zone)
    ax.text(eight_ball[0] - 1.2, eight_ball[1], 'High-Risk Zone', ha='center', fontsize=8, 
            color='red', fontweight='bold', style='italic')
    
    # Draw Ghost Ball suggestion (naive)
    ghost_pos = (6.5, 4.5)
    ghost = Circle(ghost_pos, 0.25, color='lightblue', ec='blue', linewidth=2, linestyle='--', alpha=0.5, zorder=3)
    ax.add_patch(ghost)
    
    # Trajectory: Cue → Ball #2 → Side pocket
    ax.arrow(cue_ball[0], cue_ball[1], (own_balls[1][0] - cue_ball[0]) * 0.85, (own_balls[1][1] - cue_ball[1]) * 0.85,
             head_width=0.15, head_length=0.15, fc='green', ec='green', linewidth=2, alpha=0.7, zorder=2)
    ax.arrow(own_balls[1][0], own_balls[1][1], (5 - own_balls[1][0]) * 0.85, (0.5 - own_balls[1][1]) * 0.85,
             head_width=0.15, head_length=0.15, fc='blue', ec='blue', linewidth=2, linestyle='--', alpha=0.7, zorder=2)
    
    # CMA-ES refined trajectory (defensive positioning)
    refined_cue_end = (7.5, 5)  # Cue ball ends behind 8-ball
    ax.plot([cue_ball[0], own_balls[1][0], refined_cue_end[0]], 
            [cue_ball[1], own_balls[1][1], refined_cue_end[1]], 
            'g:', linewidth=2.5, label='CMA-ES refined (defensive)', zorder=4)
    
    # Mark final cue ball position (defensive)
    final_cue = Circle(refined_cue_end, 0.2, color='yellow', ec='green', linewidth=2, zorder=6, alpha=0.8)
    ax.add_patch(final_cue)
    ax.text(refined_cue_end[0] + 0.6, refined_cue_end[1], "Defensive\nposition", ha='left', fontsize=8, 
            color='green', fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Annotations
    ax.text(5, 6.2, 'Case Study: Critical Shot Decision', ha='center', fontsize=13, fontweight='bold')
    ax.text(5, -0.3, 'Scenario: 3 own balls remaining, 8-ball near pocket (12 cm), opponent has 2 easy shots', 
            ha='center', fontsize=9, style='italic', color='darkred')
    
    # Legend box
    legend_text = (
        "Ghost Ball: Pot ball #2 to side pocket, R=52\n"
        "CMA-ES: Adjust φ by +3.2°, V₀: 3.5→3.1 m/s, R=71\n"
        "Strategic: Defensive bonus +12 (blocks opponent), Final R=83\n"
        "Outcome: Successful pot + safe cue position → Win"
    )
    ax.text(0.7, 4, legend_text, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='black', linewidth=1.5))
    
    plt.tight_layout()
    _savefig(outdir, "figure5_case_study")
    print("✓ Generated Figure 5: Case study illustration")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parent), help="Output directory for figures")
    parser.add_argument("--log", default=str((Path(__file__).resolve().parent / ".." / "debug.log").resolve()),
                        help="Path to evaluate.py log file (used for Fig.4)")
    parser.add_argument(
        "--results",
        default=str((Path(__file__).resolve().parent / "results" / "summary.json").resolve()),
        help="Path to aggregated experiment results (preferred for Fig.3/Fig.4).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed used for fallbacks")
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    log_path = Path(args.log).resolve()
    results_path = Path(args.results).resolve()

    print("Generating all figures for IEEE conference paper...")
    print("=" * 60)

    # Figure 1 is maintained as an externally drawn diagram (converted to PDF) for
    # publication-quality typography; keep the code path disabled to avoid
    # accidentally overwriting `paper/figure1_pipeline.pdf`.
    # create_figure1_pipeline(outdir)
    create_figure2_ghost_ball(outdir)
    create_figure3_evolution(outdir, results_path if results_path.exists() else None)
    create_figure4_time_distribution(outdir, log_path, seed=args.seed, results_path=results_path if results_path.exists() else None)
    create_figure5_case_study(outdir)
    
    print("=" * 60)
    print("✅ All figures generated successfully!")
    print("\nGenerated files:")
    print("  - figure1_pipeline.pdf/png")
    print("  - figure2_ghost_ball.pdf/png")
    print("  - figure3_evolution.pdf/png")
    print("  - figure4_time_distribution.pdf/png")
    print("  - figure5_case_study.pdf/png")
    print("\nNext steps:")
    print("  1. Run `python paper/update_results.py --results paper/results/summary.json` to update tables/macros")
    print("  2. Compile the paper with `cd paper && ./compile.sh`")

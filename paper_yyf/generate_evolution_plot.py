import matplotlib.pyplot as plt
import os

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def generate_evolution_plot():
    # Data from paper_yyf/results/generated_table_ablation.tex
    # (Label, Avg Time (s), Win Rate (%))
    data = [
        ("Ghost Ball Only", 15.6, 81.7),
        ("Full System", 19.4, 93.3),
        ("w/o CMA-ES", 24.6, 85.0),
        ("w/o Geometric Pruning", 52.9, 89.2),
        ("w/o Strategy/Safety", 29.4, 76.7),
        ("+ Catastrophic Penalty", 19.8, 84.2),
    ]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Color and marker mapping
    markers = {
        "Full System": "*",
        "Ghost Ball Only": "o",
        "w/o CMA-ES": "s",
        "w/o Geometric Pruning": "D",
        "w/o Strategy/Safety": "^",
        "+ Catastrophic Penalty": "v"
    }
    
    colors = {
        "Full System": "#d62728", # Red
        "Ghost Ball Only": "#1f77b4", # Blue
        "w/o CMA-ES": "#ff7f0e", # Orange
        "w/o Geometric Pruning": "#2ca02c", # Green
        "w/o Strategy/Safety": "#9467bd", # Purple
        "+ Catastrophic Penalty": "#8c564b" # Brown
    }

    for label, time, wr in data:
        marker = markers.get(label, "o")
        color = colors.get(label, "#333333")
        size = 250 if label == "Full System" else 120
        edgecolor = "black" if label == "Full System" else color
        linewidth = 1.5 if label == "Full System" else 1.0
        
        ax.scatter(time, wr, s=size, marker=marker, color=color, 
                   edgecolors=edgecolor, linewidths=linewidth, label=label, zorder=5)

    # Reference budget line
    ax.axvline(x=180, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    ax.text(175, 75, 'Reference Budget (180s)', rotation=90, verticalalignment='center', color='gray', fontsize=9)

    # Grid and styling
    ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
    ax.set_xlabel('Average Game Time (seconds)', fontweight='bold')
    ax.set_ylabel('Win Rate vs. BasicAgent (%)', fontweight='bold')
    ax.set_title('Ablation Study: Performance-Efficiency Trade-off', fontweight='bold', pad=15)
    
    # Axis limits
    ax.set_xlim(10, 60)
    ax.set_ylim(70, 100)

    # Legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, borderpad=1)

    # Shading for the Pareto-ish frontier (optional but looks good)
    # The "Full System" is clearly at the top-left of the cluster
    
    plt.tight_layout()
    
    # Save files
    output_path = "figure3_evolution"
    plt.savefig(f"{output_path}.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"{output_path}.png", bbox_inches='tight', dpi=300)
    print(f"Successfully generated {output_path}.pdf and .png")

if __name__ == "__main__":
    generate_evolution_plot()

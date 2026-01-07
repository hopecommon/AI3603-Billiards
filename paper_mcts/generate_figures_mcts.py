import matplotlib.pyplot as plt
import numpy as np
import os

def save_fig(name):
    plt.tight_layout()
    plt.savefig(f"results/{name}.pdf")
    plt.savefig(f"results/{name}.png")
    print(f"Saved {name}.pdf/png")

def plot_lambda_comparison():
    strategies = ['Fixed $\lambda=0.05$', 'Fixed $\lambda=0.5$', 'Lambda Ladder']
    win_rates = [38.5, 15.0, 45.2]
    
    plt.figure(figsize=(6, 4))
    bars = plt.bar(strategies, win_rates, color=['#ff9999','#66b3ff','#99ff99'])
    plt.ylabel('Win Rate (%)')
    plt.title('Impact of Lambda Strategy on Win Rate')
    plt.ylim(0, 60)
    
    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}%', ha='center', va='bottom')
        
    save_fig("fig_lambda")

def plot_sim_depth():
    sim_depths = [40, 80, 160, 240, 320]
    win_rates = [22.4, 31.5, 40.2, 45.2, 46.1]
    
    plt.figure(figsize=(6, 4))
    plt.plot(sim_depths, win_rates, marker='o', linestyle='-', color='b')
    plt.xlabel('Simulation Count')
    plt.ylabel('Win Rate (%)')
    plt.title('Win Rate vs. MCTS Simulation Depth')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    save_fig("fig_sim_depth")

if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")
    plot_lambda_comparison()
    plot_sim_depth()


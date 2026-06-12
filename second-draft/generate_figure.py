#!/usr/bin/env python3
"""
Generate the main result figure for the ACT 2026 paper: domain-first layout.

Creates a 2x3 grid of subplots, one per domain. Each subplot shows 5 bars
(one per topology: none, ring, star, random, FC) of final-generation diversity
with SE error bars. The canonical ordering (none > ring > star > random > FC)
is visually clear within each domain.

Data: ../experiments/experiment_e_*.csv (6 domains, 30 seeds, 100 generations)
Output: figures/multi_domain_topology_ordering.{png,pdf}

Usage:
    python generate_figure.py
"""

import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_DIR = os.path.join(SCRIPT_DIR, '..', 'experiments')
OUT_DIR = os.path.join(SCRIPT_DIR, 'figures')
# Also save to experiments/plots/ for backward compat with paper.tex
PLOTS_DIR = os.path.join(EXPERIMENTS_DIR, 'plots')

DOMAIN_CONFIGS = [
    {
        'key': 'onemax',
        'files': ['experiment_e_raw.csv', 'experiment_e_onemax.csv'],
        'label': 'OneMax',
    },
    {
        'key': 'maze',
        'files': ['experiment_e_maze.csv'],
        'label': 'Maze',
    },
    {
        'key': 'graph_coloring',
        'files': ['experiment_e_graph_coloring.csv'],
        'label': 'Graph Coloring',
    },
    {
        'key': 'knapsack',
        'files': ['experiment_e_knapsack.csv'],
        'label': 'Knapsack',
    },
    {
        'key': 'nothanks',
        'files': ['experiment_e_nothanks.csv'],
        'label': 'No Thanks!',
    },
    {
        'key': 'checkers',
        'files': ['experiment_e_checkers.csv'],
        'label': 'Checkers',
    },
]

TOPO_ORDER = ['none', 'ring', 'star', 'random', 'fully_connected']

TOPO_LABELS = {
    'none': 'None',
    'ring': 'Ring',
    'star': 'Star',
    'random': 'Rand.',
    'fully_connected': 'FC',
}

# Diverging palette: blue (high diversity / strict) to red (low diversity / lax)
TOPO_COLORS = {
    'none': '#2166AC',
    'ring': '#67A9CF',
    'star': '#5AB4AC',
    'random': '#F4A582',
    'fully_connected': '#B2182B',
}


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def setup_style():
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_domain(cfg):
    """Load a domain CSV, trying each candidate file in order."""
    for filename in cfg['files']:
        path = os.path.join(EXPERIMENTS_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            n_seeds = df['seed'].nunique()
            max_gen = df['generation'].max()
            print(f"  Loaded {cfg['label']}: {len(df)} rows, "
                  f"{n_seeds} seeds, gens 0-{max_gen} ({filename})")
            return df
    print(f"  [SKIP] {cfg['label']}: no data file found")
    return None


def compute_final_stats(df):
    """Compute mean and SE of hamming_diversity at the final generation,
    per topology."""
    max_gen = df['generation'].max()
    final = df[df['generation'] == max_gen]

    stats = {}
    for topo in TOPO_ORDER:
        vals = final[final['topology'] == topo]['hamming_diversity']
        n = len(vals)
        stats[topo] = {
            'mean': vals.mean(),
            'se': vals.std() / np.sqrt(n) if n > 1 else 0.0,
            'std': vals.std() if n > 1 else 0.0,
            'n': n,
        }
    return stats


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figure(domains):
    """Create a 2x3 grid of bar charts, one subplot per domain."""
    n = len(domains)
    if n == 0:
        print("ERROR: No domains loaded.")
        sys.exit(1)

    nrows, ncols = 2, 3

    # Compact figure for single-column EPTCS (~\textwidth ~ 5.4in)
    fig, axes_arr = plt.subplots(
        nrows, ncols,
        figsize=(5.5, 4.0),
        squeeze=False,
    )
    axes = axes_arr.flatten()

    x = np.arange(len(TOPO_ORDER))
    bar_width = 0.72

    for idx, dom in enumerate(domains):
        ax = axes[idx]
        stats = dom['stats']

        means = [stats[t]['mean'] for t in TOPO_ORDER]
        ses = [stats[t]['se'] for t in TOPO_ORDER]
        colors = [TOPO_COLORS[t] for t in TOPO_ORDER]

        bars = ax.bar(
            x, means,
            width=bar_width,
            yerr=ses,
            capsize=2.5,
            color=colors,
            edgecolor='white',
            linewidth=0.4,
            error_kw={'linewidth': 0.8, 'color': '0.3', 'capthick': 0.8},
            zorder=3,
        )

        ax.set_xticks(x)
        ax.set_xticklabels([TOPO_LABELS[t] for t in TOPO_ORDER], fontsize=7)
        ax.set_title(dom['label'], fontsize=10, fontweight='bold', pad=4)

        # Only left column gets y-axis label
        if idx % ncols == 0:
            ax.set_ylabel('Diversity', fontsize=9)

        # Light horizontal grid for readability
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

        # Clean up tick formatting
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        # Set y-axis to start from 0 (or slightly below) to avoid misleading
        ax.set_ylim(bottom=0)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    # Shared legend at the bottom
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=TOPO_COLORS[t], edgecolor='white', linewidth=0.5,
              label=TOPO_LABELS[t])
        for t in TOPO_ORDER
    ]

    fig.tight_layout(h_pad=1.0, w_pad=0.6, rect=[0, 0, 1, 1])

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    for fmt in ('png', 'pdf'):
        # Save to act2026/figures/
        path1 = os.path.join(OUT_DIR, f'multi_domain_topology_ordering.{fmt}')
        fig.savefig(path1)
        print(f"  Saved {path1}")

        # Also save to experiments/plots/ (for paper.tex backward compat)
        path2 = os.path.join(PLOTS_DIR, f'multi_domain_topology_ordering.{fmt}')
        fig.savefig(path2)
        print(f"  Saved {path2}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_style()

    print("Loading experiment data...")
    domains = []
    for cfg in DOMAIN_CONFIGS:
        df = load_domain(cfg)
        if df is not None:
            stats = compute_final_stats(df)
            domains.append({
                'key': cfg['key'],
                'label': cfg['label'],
                'stats': stats,
            })

    print(f"\n  {len(domains)} domain(s) loaded.\n")

    if len(domains) < 6:
        print(f"WARNING: Expected 6 domains, got {len(domains)}.")

    # Print summary
    print("Final-generation diversity (mean +/- SE):")
    print(f"  {'Domain':<20} " + " ".join(f"{TOPO_LABELS[t]:>8}" for t in TOPO_ORDER))
    print("  " + "-" * 65)
    for dom in domains:
        row = f"  {dom['label']:<20} "
        for t in TOPO_ORDER:
            s = dom['stats'][t]
            row += f" {s['mean']:.4f}"
        print(row)

    # Check canonical ordering
    print("\n  Ordering check (should be none > ring > star > random > FC):")
    for dom in domains:
        ranked = sorted(TOPO_ORDER, key=lambda t: dom['stats'][t]['mean'], reverse=True)
        match = ranked == TOPO_ORDER
        mark = "OK" if match else "DIFFERS"
        print(f"    {dom['label']:<20} {' > '.join(ranked)}  [{mark}]")

    print("\nGenerating figure...")
    generate_figure(domains)

    print("\nDone.")


if __name__ == '__main__':
    main()

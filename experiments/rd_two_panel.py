#!/usr/bin/env python3
"""
Two-panel R(d) figure for the ACT 2026 paper.

Left panel:  Global R (Kuramoto order parameter) bar chart by topology.
Right panel: R(d) line plots by topology, where d = graph distance between
             island pairs.

R = 1 - population_divergence (Kuramoto proxy).
R(d) = mean coherence for island pairs at graph distance d.

Topological distances for 5 islands (0-4):
    Ring (cycle 0-1-2-3-4-0):
        d=1: (0,1),(1,2),(2,3),(3,4),(4,0)  [5 pairs]
        d=2: (0,2),(0,3),(1,3),(1,4),(2,4)  [5 pairs]
    Star (hub=0):
        d=1: (0,1),(0,2),(0,3),(0,4)        [4 pairs, hub-spoke]
        d=2: (1,2),(1,3),(1,4),(2,3),(2,4),(3,4)  [6 pairs, spoke-spoke]
    Fully connected:
        d=1: all 10 pairs
    Random:
        d=1: all 10 pairs (approximate — edges vary by seed/gen)
    None:
        d=inf: all 10 pairs (no edges)

Usage:
    python3 rd_two_panel.py                           # maze (default)
    python3 rd_two_panel.py --domain onemax
    python3 rd_two_panel.py --domain maze
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Configuration — matches plot_fingerprints.py color scheme
# ---------------------------------------------------------------------------

COLORS = {
    "none": "#2166AC",
    "ring": "#67A9CF",
    "star": "#5AB4AC",
    "random": "#F4A582",
    "fully_connected": "#B2182B",
}

TOPOLOGY_ORDER = ["none", "ring", "star", "random", "fully_connected"]

TOPOLOGY_LABELS = {
    "none": "None\n(isolated)",
    "ring": "Ring",
    "star": "Star",
    "random": "Random",
    "fully_connected": "Fully\nconnected",
}

TOPOLOGY_LABELS_SHORT = {
    "none": "None (isolated)",
    "ring": "Ring",
    "star": "Star",
    "random": "Random",
    "fully_connected": "Fully connected",
}

N_ISLANDS = 5

# All pairwise divergence column names (upper triangle, i < j)
DIV_COLS = [f'div_{i}_{j}' for i in range(N_ISLANDS)
            for j in range(i + 1, N_ISLANDS)]

# --- Pair-distance classifications ---

# Ring: cycle 0-1-2-3-4-0
RING_PAIRS_BY_DIST = defaultdict(list)
for _i in range(N_ISLANDS):
    for _j in range(_i + 1, N_ISLANDS):
        _d = min(abs(_i - _j), N_ISLANDS - abs(_i - _j))
        RING_PAIRS_BY_DIST[_d].append(f'div_{_i}_{_j}')

# Star: hub = island 0
STAR_PAIRS_BY_DIST = defaultdict(list)
for _i in range(N_ISLANDS):
    for _j in range(_i + 1, N_ISLANDS):
        if _i == 0 or _j == 0:
            STAR_PAIRS_BY_DIST[1].append(f'div_{_i}_{_j}')
        else:
            STAR_PAIRS_BY_DIST[2].append(f'div_{_i}_{_j}')

# Fully connected: all pairs at d=1
FC_PAIRS_BY_DIST = {1: list(DIV_COLS)}

# Random: approximate all pairs as d=1
RANDOM_PAIRS_BY_DIST = {1: list(DIV_COLS)}

# None: all pairs at d=inf
NONE_PAIRS_BY_DIST = {float('inf'): list(DIV_COLS)}

DOMAIN_FILES = {
    "onemax": "experiment_e_per_island.csv",
    "maze": "experiment_e_maze.csv",
}


def get_pair_distances(topology):
    """Return dict mapping distance -> list of div column names."""
    if topology == 'ring':
        return dict(RING_PAIRS_BY_DIST)
    elif topology == 'star':
        return dict(STAR_PAIRS_BY_DIST)
    elif topology == 'fully_connected':
        return dict(FC_PAIRS_BY_DIST)
    elif topology == 'random':
        return dict(RANDOM_PAIRS_BY_DIST)
    elif topology == 'none':
        return dict(NONE_PAIRS_BY_DIST)
    else:
        raise ValueError(f"Unknown topology: {topology}")


# ---------------------------------------------------------------------------
# Style setup — matches plot_fingerprints.py
# ---------------------------------------------------------------------------

def setup_style():
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(filepath):
    """Load per-island CSV into structured dict.

    Returns: dict[topology][seed] = list of row dicts, sorted by generation.
    """
    data = defaultdict(lambda: defaultdict(list))
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            topo = row['topology']
            seed = int(row['seed'])
            data[topo][seed].append(row)

    # Sort each seed's rows by generation
    for topo in data:
        for seed in data[topo]:
            data[topo][seed].sort(key=lambda r: int(r['generation']))

    return data


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute_global_r(data, ss_window=20):
    """Compute global R = 1 - population_divergence per topology.

    Uses steady-state average (last ss_window generations), consistent with
    kuramoto_analysis.py.

    Returns: dict with keys = topology, values = dict with:
        'mean': float, 'std': float, 'per_seed_means': array
    """
    results = {}

    for topo in TOPOLOGY_ORDER:
        if topo not in data:
            continue

        seeds = sorted(data[topo].keys())
        per_seed_means = []

        for seed in seeds:
            rows = data[topo][seed]
            max_gen = max(int(r['generation']) for r in rows)
            start_gen = max_gen - ss_window + 1

            ss_rows = [r for r in rows if int(r['generation']) >= start_gen]
            r_values = [1.0 - float(r['population_divergence']) for r in ss_rows]
            per_seed_means.append(np.mean(r_values))

        per_seed_means = np.array(per_seed_means)
        results[topo] = {
            'mean': float(np.mean(per_seed_means)),
            'std': float(np.std(per_seed_means, ddof=1)),
            'per_seed_means': per_seed_means,
        }

    return results


def compute_rd(data, ss_window=20):
    """Compute R(d) per topology — coherence by graph distance.

    Uses steady-state average (last ss_window generations).

    Returns: dict[topology] = list of dicts, each with:
        'distance': int or float('inf'),
        'mean': float,
        'std': float (across seeds),
        'sem': float,
        'per_seed_means': array of shape (n_seeds,)
    """
    results = {}

    for topo in TOPOLOGY_ORDER:
        if topo not in data:
            continue

        pair_dists = get_pair_distances(topo)
        seeds = sorted(data[topo].keys())

        topo_results = []

        for dist in sorted(pair_dists.keys()):
            cols = pair_dists[dist]
            per_seed_means = []

            for seed in seeds:
                rows = data[topo][seed]
                max_gen = max(int(r['generation']) for r in rows)
                start_gen = max_gen - ss_window + 1
                ss_rows = [r for r in rows if int(r['generation']) >= start_gen]

                # Mean coherence over pairs and steady-state window
                coherences = []
                for row in ss_rows:
                    for col in cols:
                        coherences.append(1.0 - float(row[col]))
                per_seed_means.append(np.mean(coherences))

            per_seed_means = np.array(per_seed_means)
            n = len(per_seed_means)
            topo_results.append({
                'distance': dist,
                'mean': float(np.mean(per_seed_means)),
                'std': float(np.std(per_seed_means, ddof=1)),
                'sem': float(np.std(per_seed_means, ddof=1) / np.sqrt(n)),
                'per_seed_means': per_seed_means,
            })

        results[topo] = topo_results

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_two_panel(global_r, rd, domain_name, output_path):
    """Create the two-panel figure.

    Left:  Bar chart of global R by topology.
    Right: R(d) line plots, one line per topology.
    """
    setup_style()

    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(10, 4))

    # ── Left panel: Global R bar chart ──────────────────────────────────

    x = np.arange(len(TOPOLOGY_ORDER))
    means = [global_r[t]['mean'] for t in TOPOLOGY_ORDER]
    stds = [global_r[t]['std'] for t in TOPOLOGY_ORDER]
    bar_colors = [COLORS[t] for t in TOPOLOGY_ORDER]

    bars = ax_bar.bar(
        x, means, yerr=stds,
        capsize=4,
        color=bar_colors,
        edgecolor='white',
        linewidth=0.8,
        width=0.65,
        error_kw={'linewidth': 1.2, 'color': '0.3'},
    )

    # Value labels on bars
    for bar_obj, mean_val, std_val in zip(bars, means, stds):
        ax_bar.text(
            bar_obj.get_x() + bar_obj.get_width() / 2,
            bar_obj.get_height() + std_val + 0.008,
            f'{mean_val:.3f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold', color='0.3',
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([TOPOLOGY_LABELS[t] for t in TOPOLOGY_ORDER], fontsize=9)
    ax_bar.set_ylabel('Global $R$ (Kuramoto order parameter)')
    ax_bar.set_title(f'(a) Global synchronization $R$', fontsize=12, fontweight='bold')

    # Y-axis: start from a sensible minimum to show differentiation
    all_vals = [m - s for m, s in zip(means, stds)]
    y_floor = max(0, min(all_vals) - 0.03)
    y_ceil = max(means) + max(stds) + 0.03
    ax_bar.set_ylim(y_floor, min(y_ceil, 1.0))
    ax_bar.yaxis.set_major_locator(ticker.MultipleLocator(0.02))

    # Strict -> Lax annotation
    ax_bar.annotate(
        '',
        xy=(4.3, -0.04), xytext=(-0.3, -0.04),
        xycoords=('data', 'axes fraction'),
        textcoords=('data', 'axes fraction'),
        arrowprops=dict(arrowstyle='->', color='0.5', linewidth=1.0),
        annotation_clip=False,
    )
    ax_bar.text(
        2, -0.10,
        'strict                                                       lax',
        transform=ax_bar.get_xaxis_transform(),
        ha='center', va='top', fontsize=8, color='0.5', style='italic',
    )

    # ── Right panel: R(d) line plots ────────────────────────────────────

    # Marker styles for each topology
    markers = {
        'none': 's',
        'ring': 'o',
        'star': 'D',
        'random': '^',
        'fully_connected': 'v',
    }

    # For the x-axis, we need to map all possible distances.
    # Ring: d=1,2. Star: d=1,2. FC: d=1. Random: d=1. None: d=inf.
    # We'll plot inf as a separate x position (e.g., 3.5) with a break.

    # Collect all finite distances
    finite_dists = set()
    for topo in TOPOLOGY_ORDER:
        if topo not in rd:
            continue
        for entry in rd[topo]:
            if entry['distance'] != float('inf'):
                finite_dists.add(entry['distance'])

    finite_dists = sorted(finite_dists)
    # x positions for finite distances
    x_map = {d: d for d in finite_dists}
    # Position for infinity
    x_inf = max(finite_dists) + 1.2 if finite_dists else 1

    # Plot each topology
    for topo in TOPOLOGY_ORDER:
        if topo not in rd:
            continue

        x_vals = []
        y_vals = []
        y_errs = []

        for entry in rd[topo]:
            d = entry['distance']
            if d == float('inf'):
                x_vals.append(x_inf)
            else:
                x_vals.append(x_map[d])
            y_vals.append(entry['mean'])
            y_errs.append(entry['std'])

        ax_line.errorbar(
            x_vals, y_vals, yerr=y_errs,
            color=COLORS[topo],
            marker=markers[topo],
            markersize=7,
            linewidth=2.0,
            capsize=4,
            capthick=1.2,
            label=TOPOLOGY_LABELS_SHORT[topo],
            linestyle='-' if len(x_vals) > 1 else 'none',
            zorder=3 if topo in ('ring', 'star') else 2,
        )

    # X-axis formatting
    all_x_ticks = list(finite_dists) + [x_inf]
    all_x_labels = [str(d) for d in finite_dists] + ['$\\infty$']
    ax_line.set_xticks(all_x_ticks)
    ax_line.set_xticklabels(all_x_labels, fontsize=10)
    ax_line.set_xlabel('Graph distance $d$')
    ax_line.set_ylabel('Pairwise coherence $R(d)$')
    ax_line.set_title(f'(b) Coherence by topological distance', fontsize=12, fontweight='bold')

    # Y-axis: same range as left panel for visual comparability
    ax_line.set_ylim(y_floor, min(y_ceil, 1.0))
    ax_line.yaxis.set_major_locator(ticker.MultipleLocator(0.02))

    # Legend
    ax_line.legend(
        frameon=True, framealpha=0.9, edgecolor='0.8',
        loc='lower left', fontsize=8.5,
    )

    # Light grid on right panel
    ax_line.grid(axis='y', alpha=0.2, linewidth=0.5)

    # Add a visual break between d=2 and d=inf
    if finite_dists and x_inf is not None:
        break_x = max(finite_dists) + 0.6
        ax_line.axvline(break_x, color='0.7', linewidth=0.8, linestyle=':', alpha=0.5)

    # Suptitle
    domain_title = domain_name.capitalize() if domain_name != 'onemax' else 'OneMax'
    fig.suptitle(
        f'Synchronization structure by topology — {domain_title} domain',
        fontsize=13, fontweight='bold', y=1.02,
    )

    plt.tight_layout()

    # Save
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Figure saved: {output_path}")
    print(f"Figure saved: {pdf_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Two-panel R(d) figure: global R + coherence by distance')
    parser.add_argument('--domain', choices=['onemax', 'maze'], default='maze',
                        help='Domain to plot (default: maze)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = DOMAIN_FILES[args.domain]
    csv_path = os.path.join(script_dir, csv_file)

    if not os.path.exists(csv_path):
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.domain} data: {csv_path}")
    data = load_data(csv_path)

    n_combos = sum(len(seeds) for seeds in data.values())
    print(f"Loaded {n_combos} topology-seed combos")
    for topo in TOPOLOGY_ORDER:
        if topo in data:
            n_seeds = len(data[topo])
            n_gens = len(data[topo][next(iter(data[topo]))])
            print(f"  {topo}: {n_seeds} seeds x {n_gens} generations")

    # Compute
    print("\nComputing global R (steady-state, last 20 gens)...")
    global_r = compute_global_r(data)
    for topo in TOPOLOGY_ORDER:
        if topo in global_r:
            r = global_r[topo]
            print(f"  {topo:>18}: R = {r['mean']:.4f} +/- {r['std']:.4f}")

    print("\nComputing R(d) by topological distance...")
    rd = compute_rd(data)
    for topo in TOPOLOGY_ORDER:
        if topo in rd:
            for entry in rd[topo]:
                d = entry['distance']
                d_str = 'inf' if d == float('inf') else str(d)
                print(f"  {topo:>18} d={d_str}: R(d) = {entry['mean']:.4f} "
                      f"+/- {entry['std']:.4f}")

    # Plot
    plot_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    output_path = os.path.join(plot_dir, f'rd_two_panel_{args.domain}.png')

    print(f"\nGenerating two-panel figure...")
    plot_two_panel(global_r, rd, args.domain, output_path)

    print("\nDone.")


if __name__ == '__main__':
    main()

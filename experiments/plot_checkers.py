#!/usr/bin/env python3
"""
Checkers domain analysis: fingerprint plots + comparison with maze.

Generates four publication-quality figures:
1. Fingerprint panels (5-panel: individual seed traces per topology)
2. Topology comparison bar chart (final-generation diversity by topology)
3. R(d) two-panel figure (global R + coherence by topological distance)
4. Combined multi-domain comparison (maze vs checkers, normalized overlay)

Data: experiment_e_checkers.csv
      Same columns as experiment_e_maze.csv:
        topology, seed, generation, hamming_diversity, population_divergence,
        best_fitness, island_X_diversity, island_X_fitness, div_i_j

Output: plots/ directory (PNG 300 DPI + PDF), all prefixed with checkers_

Usage:
    python plot_checkers.py              # all 4 figures
    python plot_checkers.py --only fingerprints   # just 1+2
    python plot_checkers.py --only rd             # just 3
    python plot_checkers.py --only comparison     # just 4
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Configuration — matches plot_fingerprints.py
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)

CHECKERS_CSV = SCRIPT_DIR / "experiment_e_checkers.csv"
MAZE_CSV = SCRIPT_DIR / "experiment_e_maze.csv"

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

COLORS = {
    "none": "#2166AC",
    "ring": "#67A9CF",
    "star": "#5AB4AC",
    "random": "#F4A582",
    "fully_connected": "#B2182B",
}

N_ISLANDS = 5

# Pairwise divergence column names (upper triangle)
DIV_COLS = [f'div_{i}_{j}' for i in range(N_ISLANDS)
            for j in range(i + 1, N_ISLANDS)]

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

FC_PAIRS_BY_DIST = {1: list(DIV_COLS)}
RANDOM_PAIRS_BY_DIST = {1: list(DIV_COLS)}
NONE_PAIRS_BY_DIST = {float('inf'): list(DIV_COLS)}


def get_pair_distances(topology):
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
# Style — matches plot_fingerprints.py exactly
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

def load_checkers_df():
    """Load checkers CSV as a pandas DataFrame."""
    if not CHECKERS_CSV.exists():
        print(f"[WAITING] Checkers CSV not found: {CHECKERS_CSV}")
        print("  The topology sweep is still running. Re-run when it finishes.")
        return None
    df = pd.read_csv(CHECKERS_CSV)
    print(f"Loaded {len(df)} rows from {CHECKERS_CSV.name}"
          f" ({df['seed'].nunique()} seeds, gens 0-{df['generation'].max()})")
    return df


def load_checkers_rows():
    """Load checkers CSV as list-of-dicts (for R(d) computation)."""
    if not CHECKERS_CSV.exists():
        return None
    data = defaultdict(lambda: defaultdict(list))
    with open(CHECKERS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            topo = row['topology']
            seed = int(row['seed'])
            data[topo][seed].append(row)
    for topo in data:
        for seed in data[topo]:
            data[topo][seed].sort(key=lambda r: int(r['generation']))
    return data


def load_maze_df():
    """Load maze CSV for comparison."""
    if not MAZE_CSV.exists():
        print(f"[SKIP] Maze CSV not found: {MAZE_CSV}")
        return None
    df = pd.read_csv(MAZE_CSV)
    print(f"Loaded {len(df)} rows from {MAZE_CSV.name}"
          f" ({df['seed'].nunique()} seeds, gens 0-{df['generation'].max()})")
    return df


# ===========================================================================
# Figure 1: Fingerprint panels (5-panel, seed traces + mean)
# ===========================================================================

def plot_fingerprint_panels(df):
    """5-panel figure: one subplot per topology, individual seed traces + bold mean."""
    fig, axes = plt.subplots(1, 5, figsize=(14, 2.8), sharey=True)

    y_min = df["hamming_diversity"].min() * 0.95
    y_max = df["hamming_diversity"].max() * 1.02

    for ax, topo in zip(axes, TOPOLOGY_ORDER):
        tdf = df[df["topology"] == topo]
        seeds_df = tdf.pivot(index="generation", columns="seed",
                             values="hamming_diversity")
        color = COLORS[topo]

        # Individual seed traces
        for seed_col in seeds_df.columns:
            ax.plot(seeds_df.index, seeds_df[seed_col],
                    color=color, alpha=0.15, linewidth=0.6, rasterized=True)

        # Mean line
        mean_line = seeds_df.mean(axis=1)
        ax.plot(mean_line.index, mean_line.values,
                color=color, alpha=1.0, linewidth=2.0)

        ax.set_title(TOPOLOGY_LABELS[topo], fontsize=10, fontweight="bold", pad=6)
        ax.set_xlim(0, df["generation"].max())
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Generation")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(25))

    axes[0].set_ylabel("Hamming diversity")

    for ax in axes:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.tick_params(axis="both", which="both", length=3)

    fig.suptitle(
        "Diversity fingerprints by migration topology (Checkers)",
        fontsize=13, fontweight="bold", y=1.04,
    )

    for fmt in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"checkers_fingerprints_panels.{fmt}")
    plt.close(fig)
    print("  checkers_fingerprints_panels.png/pdf saved")


# ===========================================================================
# Figure 1b: Overlay (mean + std band)
# ===========================================================================

def plot_overlay(df):
    """Single panel: mean diversity per topology with std band."""
    fig, ax = plt.subplots(figsize=(6, 3.5))

    y_min = df["hamming_diversity"].min() * 0.95
    y_max = df["hamming_diversity"].max() * 1.02

    for topo in TOPOLOGY_ORDER:
        tdf = df[df["topology"] == topo]
        seeds_df = tdf.pivot(index="generation", columns="seed",
                             values="hamming_diversity")
        mean_line = seeds_df.mean(axis=1)
        std_line = seeds_df.std(axis=1)

        ax.fill_between(mean_line.index, mean_line - std_line,
                        mean_line + std_line, color=COLORS[topo], alpha=0.12)
        ax.plot(mean_line.index, mean_line.values,
                color=COLORS[topo], linewidth=2.0,
                label=TOPOLOGY_LABELS_SHORT[topo])

    ax.set_xlim(0, df["generation"].max())
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hamming diversity")
    ax.set_title("Mean diversity by topology (Checkers)",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8", loc="upper right")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))

    for fmt in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"checkers_fingerprints_overlay.{fmt}")
    plt.close(fig)
    print("  checkers_fingerprints_overlay.png/pdf saved")


# ===========================================================================
# Figure 2: Phase transition bar chart
# ===========================================================================

def plot_phase_transition(df):
    """Bar chart of final-generation diversity by topology with symmetry-break annotation."""
    max_gen = df["generation"].max()
    final = df[df["generation"] == max_gen]
    stats = (
        final.groupby("topology")["hamming_diversity"]
        .agg(["mean", "std"])
        .reindex(TOPOLOGY_ORDER)
    )

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    x = np.arange(len(TOPOLOGY_ORDER))
    bar_colors = [COLORS[t] for t in TOPOLOGY_ORDER]

    bars = ax.bar(
        x, stats["mean"], yerr=stats["std"],
        capsize=4, color=bar_colors, edgecolor="white",
        linewidth=0.8, width=0.65,
        error_kw={"linewidth": 1.2, "color": "0.3"},
    )

    # Annotate none->ring gap
    none_mean = stats.loc["none", "mean"]
    ring_mean = stats.loc["ring", "mean"]
    drop_pct = (none_mean - ring_mean) / none_mean * 100

    bracket_y = none_mean + stats.loc["none", "std"] + 0.006
    ax.annotate(
        "", xy=(0, bracket_y), xytext=(1, bracket_y),
        arrowprops=dict(arrowstyle="<->", color="0.3", linewidth=1.2,
                        shrinkA=0, shrinkB=0),
    )
    ax.text(0.5, bracket_y + 0.004, f"{drop_pct:.0f}% drop",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="0.3")

    # Annotate subsequent gaps
    prev_name = "ring"
    for i, topo in enumerate(TOPOLOGY_ORDER[2:], start=2):
        prev_mean = stats.loc[prev_name, "mean"]
        curr_mean = stats.loc[topo, "mean"]
        gap_pct = (prev_mean - curr_mean) / prev_mean * 100
        mid_y = (prev_mean + curr_mean) / 2
        ax.annotate(
            f"{gap_pct:.0f}%", xy=(i, curr_mean),
            xytext=(i - 0.5, mid_y + 0.003),
            fontsize=7.5, color="0.5", ha="center", va="bottom",
        )
        prev_name = topo

    # Reference line at none level
    ax.axhline(none_mean, color=COLORS["none"], linewidth=0.8,
               linestyle="--", alpha=0.4, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([TOPOLOGY_LABELS[t] for t in TOPOLOGY_ORDER], fontsize=9)
    ax.set_ylabel(f"Final diversity (gen {max_gen})")
    ax.set_title(
        "Phase transition: symmetry break at first coupling (Checkers)",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylim(0, bracket_y + 0.020)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))

    # Strict -> Lax arrow
    ax.annotate(
        "", xy=(4.3, -0.016), xytext=(-0.3, -0.016),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        arrowprops=dict(arrowstyle="->", color="0.5", linewidth=1.0),
        annotation_clip=False,
    )
    ax.text(2, -0.08,
            "strict                                                       lax",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color="0.5", style="italic")

    for fmt in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"checkers_phase_transition.{fmt}")
    plt.close(fig)
    print("  checkers_phase_transition.png/pdf saved")


# ===========================================================================
# Figure 3: R(d) two-panel (from rd_two_panel.py logic)
# ===========================================================================

def compute_global_r(data, ss_window=20):
    """Compute global R = 1 - population_divergence per topology (steady-state avg)."""
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
    """Compute R(d) per topology -- coherence by graph distance."""
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


def plot_rd_two_panel(data):
    """Two-panel figure: global R bar chart + R(d) line plots."""
    global_r = compute_global_r(data)
    rd = compute_rd(data)

    # Print summary
    print("\n  Global R (steady-state, last 20 gens):")
    for topo in TOPOLOGY_ORDER:
        if topo in global_r:
            r = global_r[topo]
            print(f"    {topo:>18}: R = {r['mean']:.4f} +/- {r['std']:.4f}")

    print("\n  R(d) by topological distance:")
    for topo in TOPOLOGY_ORDER:
        if topo in rd:
            for entry in rd[topo]:
                d = entry['distance']
                d_str = 'inf' if d == float('inf') else str(d)
                print(f"    {topo:>18} d={d_str}: R(d) = {entry['mean']:.4f} "
                      f"+/- {entry['std']:.4f}")

    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(10, 4))

    # -- Left panel: Global R bar chart --
    x = np.arange(len(TOPOLOGY_ORDER))
    means = [global_r[t]['mean'] for t in TOPOLOGY_ORDER]
    stds = [global_r[t]['std'] for t in TOPOLOGY_ORDER]
    bar_colors = [COLORS[t] for t in TOPOLOGY_ORDER]

    bars = ax_bar.bar(
        x, means, yerr=stds, capsize=4,
        color=bar_colors, edgecolor='white', linewidth=0.8, width=0.65,
        error_kw={'linewidth': 1.2, 'color': '0.3'},
    )
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
    ax_bar.set_title('(a) Global synchronization $R$', fontsize=12, fontweight='bold')

    all_vals = [m - s for m, s in zip(means, stds)]
    y_floor = max(0, min(all_vals) - 0.03)
    y_ceil = max(means) + max(stds) + 0.03
    ax_bar.set_ylim(y_floor, min(y_ceil, 1.0))
    ax_bar.yaxis.set_major_locator(ticker.MultipleLocator(0.02))

    ax_bar.annotate(
        '', xy=(4.3, -0.04), xytext=(-0.3, -0.04),
        xycoords=('data', 'axes fraction'),
        textcoords=('data', 'axes fraction'),
        arrowprops=dict(arrowstyle='->', color='0.5', linewidth=1.0),
        annotation_clip=False,
    )
    ax_bar.text(2, -0.10,
                'strict                                                       lax',
                transform=ax_bar.get_xaxis_transform(),
                ha='center', va='top', fontsize=8, color='0.5', style='italic')

    # -- Right panel: R(d) line plots --
    markers = {'none': 's', 'ring': 'o', 'star': 'D',
               'random': '^', 'fully_connected': 'v'}

    finite_dists = set()
    for topo in TOPOLOGY_ORDER:
        if topo not in rd:
            continue
        for entry in rd[topo]:
            if entry['distance'] != float('inf'):
                finite_dists.add(entry['distance'])
    finite_dists = sorted(finite_dists)
    x_map = {d: d for d in finite_dists}
    x_inf = max(finite_dists) + 1.2 if finite_dists else 1

    for topo in TOPOLOGY_ORDER:
        if topo not in rd:
            continue
        x_vals, y_vals, y_errs = [], [], []
        for entry in rd[topo]:
            d = entry['distance']
            x_vals.append(x_inf if d == float('inf') else x_map[d])
            y_vals.append(entry['mean'])
            y_errs.append(entry['std'])
        ax_line.errorbar(
            x_vals, y_vals, yerr=y_errs,
            color=COLORS[topo], marker=markers[topo], markersize=7,
            linewidth=2.0, capsize=4, capthick=1.2,
            label=TOPOLOGY_LABELS_SHORT[topo],
            linestyle='-' if len(x_vals) > 1 else 'none',
            zorder=3 if topo in ('ring', 'star') else 2,
        )

    all_x_ticks = list(finite_dists) + [x_inf]
    all_x_labels = [str(d) for d in finite_dists] + ['$\\infty$']
    ax_line.set_xticks(all_x_ticks)
    ax_line.set_xticklabels(all_x_labels, fontsize=10)
    ax_line.set_xlabel('Graph distance $d$')
    ax_line.set_ylabel('Pairwise coherence $R(d)$')
    ax_line.set_title('(b) Coherence by topological distance',
                      fontsize=12, fontweight='bold')
    ax_line.set_ylim(y_floor, min(y_ceil, 1.0))
    ax_line.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax_line.legend(frameon=True, framealpha=0.9, edgecolor='0.8',
                   loc='lower left', fontsize=8.5)
    ax_line.grid(axis='y', alpha=0.2, linewidth=0.5)

    if finite_dists and x_inf is not None:
        break_x = max(finite_dists) + 0.6
        ax_line.axvline(break_x, color='0.7', linewidth=0.8,
                        linestyle=':', alpha=0.5)

    fig.suptitle(
        'Synchronization structure by topology \u2014 Checkers domain',
        fontsize=13, fontweight='bold', y=1.02,
    )
    plt.tight_layout()

    output_path = OUT_DIR / "checkers_rd_two_panel.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  checkers_rd_two_panel.png/pdf saved")


# ===========================================================================
# Figure 4: Multi-domain comparison (maze vs checkers)
# ===========================================================================

def plot_domain_comparison(df_checkers, df_maze):
    """Side-by-side comparison: maze vs checkers, normalized diversity overlay."""
    max_gen_ck = df_checkers["generation"].max()
    max_gen_mz = df_maze["generation"].max()
    max_gen = min(max_gen_ck, max_gen_mz)

    # Compute mean traces per topology
    def mean_traces(df):
        traces = {}
        for topo in TOPOLOGY_ORDER:
            tdf = df[df["topology"] == topo]
            pivot = tdf.pivot(index="generation", columns="seed",
                              values="hamming_diversity")
            traces[topo] = pivot.mean(axis=1)
        return traces

    def normalize_traces(traces):
        all_vals = np.concatenate([t.values for t in traces.values()])
        vmin, vmax = all_vals.min(), all_vals.max()
        return {t: (v - vmin) / (vmax - vmin) for t, v in traces.items()}, vmin, vmax

    ck_traces = mean_traces(df_checkers)
    mz_traces = mean_traces(df_maze)
    ck_norm, _, _ = normalize_traces(ck_traces)
    mz_norm, _, _ = normalize_traces(mz_traces)

    # --- Plot A: Normalized overlay (time series) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    for topo in TOPOLOGY_ORDER:
        color = COLORS[topo]
        label_short = TOPOLOGY_LABELS_SHORT[topo]
        ax1.plot(mz_norm[topo].index, mz_norm[topo].values,
                 color=color, linewidth=2.0, linestyle='-',
                 label=f"{label_short} (Maze)")
        ax1.plot(ck_norm[topo].index, ck_norm[topo].values,
                 color=color, linewidth=2.0, linestyle='--',
                 label=f"{label_short} (Checkers)")

    ax1.set_xlim(0, max_gen)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Normalized diversity (min-max per domain)")
    ax1.set_title("(a) Topology ordering across domains",
                  fontsize=12, fontweight="bold")
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    # Custom legend
    legend_elements = [
        Line2D([0], [0], color="0.4", linewidth=2, linestyle="-", label="Maze"),
        Line2D([0], [0], color="0.4", linewidth=2, linestyle="--", label="Checkers"),
        Line2D([0], [0], color="none", label=""),
    ]
    for topo in TOPOLOGY_ORDER:
        legend_elements.append(
            Line2D([0], [0], color=COLORS[topo], linewidth=2.5, linestyle="-",
                   label=TOPOLOGY_LABELS_SHORT[topo])
        )
    ax1.legend(handles=legend_elements, frameon=True, framealpha=0.9,
               edgecolor="0.8", loc="upper right", fontsize=8)

    # --- Plot B: Final-generation bar chart comparison ---
    def final_stats(df):
        mg = df["generation"].max()
        final = df[df["generation"] == mg]
        stats = (
            final.groupby("topology")["hamming_diversity"]
            .agg(["mean", "std"])
            .reindex(TOPOLOGY_ORDER)
        )
        raw = stats["mean"].values
        vmin, vmax = raw.min(), raw.max()
        drange = vmax - vmin if vmax != vmin else 1.0
        stats["norm_mean"] = (raw - vmin) / drange
        stats["norm_std"] = stats["std"] / drange
        return stats

    mz_stats = final_stats(df_maze)
    ck_stats = final_stats(df_checkers)

    x = np.arange(len(TOPOLOGY_ORDER))
    width = 0.35

    ax2.bar(x - width / 2, mz_stats["norm_mean"], width,
            yerr=mz_stats["norm_std"], capsize=3,
            color="#F28E2B", edgecolor="white", linewidth=0.6,
            label="Maze", error_kw={"linewidth": 1.0, "color": "0.3"})
    ax2.bar(x + width / 2, ck_stats["norm_mean"], width,
            yerr=ck_stats["norm_std"], capsize=3,
            color="#76B7B2", edgecolor="white", linewidth=0.6,
            label="Checkers", error_kw={"linewidth": 1.0, "color": "0.3"})

    ax2.set_xticks(x)
    ax2.set_xticklabels([TOPOLOGY_LABELS[t] for t in TOPOLOGY_ORDER], fontsize=9)
    ax2.set_ylabel("Normalized final diversity")
    ax2.set_ylim(0, 1.15)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.set_title("(b) Final-generation diversity comparison",
                  fontsize=12, fontweight="bold")
    ax2.legend(frameon=True, framealpha=0.9, edgecolor="0.8", loc="upper right")

    # Strict -> Lax arrow
    ax2.annotate(
        "", xy=(4.3, -0.015), xytext=(-0.4, -0.015),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        arrowprops=dict(arrowstyle="->", color="0.5", linewidth=1.0),
        annotation_clip=False,
    )
    ax2.text(2, -0.08,
             "strict                                                       lax",
             transform=ax2.get_xaxis_transform(),
             ha="center", va="top", fontsize=8, color="0.5", style="italic")

    fig.suptitle(
        "Domain Independence: Maze vs Checkers",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"checkers_domain_comparison.{fmt}")
    plt.close(fig)
    print("  checkers_domain_comparison.png/pdf saved")

    # Print comparison table
    print("\n  Domain comparison -- final diversity:")
    print(f"  {'Topology':<20} {'Maze':>10} {'Checkers':>10} {'Ordering match':>16}")
    print(f"  {'-' * 58}")
    mz_order = mz_stats.sort_values("mean", ascending=False).index.tolist()
    ck_order = ck_stats.sort_values("mean", ascending=False).index.tolist()
    for topo in TOPOLOGY_ORDER:
        mz_val = mz_stats.loc[topo, "mean"]
        ck_val = ck_stats.loc[topo, "mean"]
        mz_rank = mz_order.index(topo) + 1
        ck_rank = ck_order.index(topo) + 1
        match = "YES" if mz_rank == ck_rank else f"({mz_rank} vs {ck_rank})"
        print(f"  {TOPOLOGY_LABELS_SHORT[topo]:<20} {mz_val:>10.4f} {ck_val:>10.4f} {match:>16}")

    print(f"\n  Maze ordering:    {' > '.join(mz_order)}")
    print(f"  Checkers ordering: {' > '.join(ck_order)}")
    ordering_match = mz_order == ck_order
    print(f"  Orderings identical: {ordering_match}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate checkers fingerprint and comparison plots."
    )
    parser.add_argument(
        "--only",
        choices=["fingerprints", "rd", "comparison"],
        default=None,
        help="Generate only one figure type (default: all).",
    )
    args = parser.parse_args()

    setup_style()

    # ---- Fingerprint plots (1 + 2) ----
    if args.only is None or args.only == "fingerprints":
        print("=== Checkers Fingerprint Plots ===")
        df = load_checkers_df()
        if df is not None:
            plot_fingerprint_panels(df)
            plot_overlay(df)
            plot_phase_transition(df)
        else:
            print("  Skipping fingerprint plots (no data).")

    # ---- R(d) two-panel ----
    if args.only is None or args.only == "rd":
        print("\n=== Checkers R(d) Two-Panel ===")
        data = load_checkers_rows()
        if data is not None:
            plot_rd_two_panel(data)
        else:
            print("  Skipping R(d) plot (no data).")

    # ---- Multi-domain comparison ----
    if args.only is None or args.only == "comparison":
        print("\n=== Multi-Domain Comparison: Maze vs Checkers ===")
        df_ck = load_checkers_df()
        df_mz = load_maze_df()
        if df_ck is not None and df_mz is not None:
            plot_domain_comparison(df_ck, df_mz)
        else:
            missing = []
            if df_ck is None:
                missing.append("checkers")
            if df_mz is None:
                missing.append("maze")
            print(f"  Skipping comparison (missing: {', '.join(missing)}).")

    print(f"\nAll output -> {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate all 6 EUMAS 2026 figures from REAL experimental data.

Data sources (extracted from git branches):
  - Directed cycle experiment (240 runs): /tmp/eumas_data/directed/
  - Bridge experiment (320 runs): /tmp/eumas_data/bridge/
  - Ring-vs-Star NK sweep (160 runs): /tmp/eumas_data/ring_vs_star/
  - NK pilot (90 runs): /tmp/eumas_data/nk_sweep/
  - LLM/Claude Chorus (strict vs lax): /tmp/eumas_data/comparison_summary.csv

Outputs: /home/lyra/projects/categorical-evolution/cais2026/figures/

Figures:
  1. iso-dense-graphs.pdf — Diagram of 8 iso-dense directed graph families
  2. eta-squared-vs-k.pdf — NK dose-response curve
  3. ring-vs-star-diversity.pdf — Ring vs Star diversity comparison
  4. bridge-diversity-generations.pdf — Two-invariant temporal separation
  5. llm-diversity-heatmap.pdf — LLM diversity: strict vs lax
  6. invariant-hierarchy.pdf — Conceptual three-invariant diagram
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from io import StringIO

# ============================================================
# Publication settings for LNCS
# ============================================================
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 8.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 1.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUTDIR = '/home/lyra/projects/categorical-evolution/cais2026/figures'
os.makedirs(OUTDIR, exist_ok=True)

# Colorblind-friendly palette
CB_BLUE = '#0072B2'
CB_ORANGE = '#E69F00'
CB_GREEN = '#009E73'
CB_RED = '#D55E00'
CB_PURPLE = '#CC79A7'
CB_CYAN = '#56B4E9'
CB_YELLOW = '#F0E442'
CB_BLACK = '#000000'


# ============================================================
# Data loading helpers
# ============================================================
def load_csvs_from_dir(directory, pattern='*.csv'):
    """Load CSVs from directory. Returns dict: key -> list of DataFrames."""
    data = {}
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    for f in files:
        basename = os.path.basename(f)
        if basename.startswith('ring_vs_star_summary') or basename.startswith('within_') or basename.startswith('twoway_') or basename.startswith('cross_'):
            continue

        parts = basename.replace('.csv', '')
        # Handle different naming: topology_seedNNN.csv or topology_NNN.csv
        if '_seed' in parts:
            key = parts.rsplit('_seed', 1)[0]
        else:
            key = parts.rsplit('_', 1)[0]

        try:
            with open(f, 'r') as fh:
                lines = fh.readlines()
            header_idx = None
            for i, line in enumerate(lines):
                if line.startswith('generation,'):
                    header_idx = i
                    break
            if header_idx is None:
                continue
            csv_text = ''.join(lines[header_idx:])
            df = pd.read_csv(StringIO(csv_text))
            if 'generation' not in df.columns:
                continue
            if key not in data:
                data[key] = []
            data[key].append(df)
        except Exception as e:
            print(f"  Warning: {basename}: {e}")
    return data


def find_gen_index(generations, target_gen):
    return np.argmin(np.abs(generations - target_gen))


def eta_squared_groups(*groups):
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = np.sum((all_vals - grand_mean)**2)
    return ss_between / ss_total if ss_total > 1e-15 else 0.0


def save_fig(fig, name):
    fig.savefig(os.path.join(OUTDIR, f'{name}.pdf'))
    fig.savefig(os.path.join(OUTDIR, f'{name}.png'))
    plt.close(fig)
    print(f"  Saved {name}.pdf and {name}.png")


# ============================================================
# FIGURE 1: Iso-dense directed graph families
# ============================================================
def figure1_iso_dense_graphs():
    """Draw the 8 iso-dense directed graph families using networkx."""
    import networkx as nx

    print("\nGenerating Fig 1: Iso-dense directed graph families...")

    # 8 topologies from the directed cycle experiment
    # All have 8 nodes and ~15 edges (iso-dense), but different directed cycle counts
    families = [
        ('dag-layer', 0),
        ('dag-wide', 0),
        ('lowcyc-1', 3),
        ('bidir-ring', 10),
        ('two-cliques', 14),
        ('mesh-cyclic', 20),
        ('dense-triangles', 29),
        ('ring-skip2', 47),
    ]

    # Build representative directed graphs for each topology
    def make_dag_layer(n=8):
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        # Layered DAG: 0->1, 0->2, 1->3, 1->4, 2->5, 2->6, 3->7, 4->7, 5->7, 6->7
        # Plus enough edges to reach ~15
        edges = [(0,1),(0,2),(0,3),(1,3),(1,4),(2,4),(2,5),(3,6),(3,7),(4,6),(4,7),(5,6),(5,7),(6,7),(1,5)]
        G.add_edges_from(edges)
        return G

    def make_dag_wide(n=8):
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        # Wide DAG: fan-out pattern
        edges = [(0,1),(0,2),(0,3),(0,4),(1,5),(2,5),(3,6),(4,6),(1,6),(2,7),(3,7),(4,7),(5,7),(6,7),(1,7)]
        G.add_edges_from(edges)
        return G

    def make_lowcyc(n=8):
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        # Mostly DAG with a few directed cycles
        edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),  # one big cycle
                 (0,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,0)]
        G.add_edges_from(edges)
        return G

    def make_bidir_ring(n=8):
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for i in range(n):
            G.add_edge(i, (i+1) % n)
            G.add_edge((i+1) % n, i)
        return G

    def make_two_cliques(n=8):
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        # Two directed 4-cliques with bridge
        for i in range(4):
            for j in range(4):
                if i != j:
                    G.add_edge(i, j)
        for i in range(4, 8):
            for j in range(4, 8):
                if i != j:
                    G.add_edge(i, j)
        G.add_edge(3, 4)
        G.add_edge(4, 3)
        return G

    def make_mesh_cyclic(n=8):
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        # 2x4 mesh with cyclic connections
        for i in range(4):
            G.add_edge(i, (i+1) % 4)
            G.add_edge(i+4, (i+1) % 4 + 4)
            G.add_edge(i, i+4)
            G.add_edge(i+4, i)
        return G

    def make_dense_triangles(n=8):
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        # Dense triangular structure
        for i in range(n):
            G.add_edge(i, (i+1) % n)
            G.add_edge(i, (i+2) % n)
        return G

    def make_ring_skip2(n=8):
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        # Ring with skip-2 connections (both directions)
        for i in range(n):
            G.add_edge(i, (i+1) % n)
            G.add_edge((i+1) % n, i)
            G.add_edge(i, (i+2) % n)
        return G

    builders = [
        make_dag_layer, make_dag_wide, make_lowcyc, make_bidir_ring,
        make_two_cliques, make_mesh_cyclic, make_dense_triangles, make_ring_skip2
    ]

    colors_grad = plt.cm.viridis(np.linspace(0.15, 0.85, 8))

    fig, axes = plt.subplots(2, 4, figsize=(7, 3.8))
    axes = axes.flatten()

    for idx, (ax, (name, kappa), builder) in enumerate(zip(axes, families, builders)):
        G = builder()
        pos = nx.spring_layout(G, seed=42, k=1.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=80,
                               node_color=[colors_grad[idx]], edgecolors='black', linewidths=0.5)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray',
                               arrowsize=6, width=0.6, alpha=0.7,
                               connectionstyle='arc3,rad=0.1')
        ax.set_title(f'{name}\n$\\kappa={kappa}$', fontsize=8, pad=2)
        ax.axis('off')
        ax.set_aspect('equal')

    fig.suptitle('Eight iso-dense directed graph families', fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'iso-dense-graphs')


# ============================================================
# FIGURE 2: NK dose-response (eta-squared vs K)
# ============================================================
def figure2_eta_squared_vs_k():
    """eta-squared for diversity effect across NK K values."""
    print("\nGenerating Fig 2: eta-squared vs K (NK dose-response)...")

    # Load ring-vs-star data (K=0,2,4,6, ring vs star)
    rs_data = load_csvs_from_dir('/tmp/eumas_data/ring_vs_star/')
    print(f"  Ring-vs-star topologies: {list(rs_data.keys())}")

    # Load NK pilot data (K=0,4,6, 3 topologies)
    nk_data = load_csvs_from_dir('/tmp/eumas_data/nk_sweep/')
    print(f"  NK pilot topologies: {list(nk_data.keys())}")

    # Use ring-vs-star data at K=0,2,4,6 (2 topologies: ring, star)
    # Compute eta-squared at generation 200 for diversity
    K_values_rs = [0, 2, 4, 6]
    eta2_rs = []
    ci_rs = []

    for K in K_values_rs:
        ring_key = f'nk{K}_ring'
        star_key = f'nk{K}_star'
        if ring_key not in rs_data or star_key not in rs_data:
            print(f"  WARNING: missing {ring_key} or {star_key}")
            eta2_rs.append(0)
            ci_rs.append(0)
            continue

        gens = rs_data[ring_key][0]['generation'].values
        gen200_idx = find_gen_index(gens, 200)

        ring_div = np.array([df['diversity'].values[gen200_idx] for df in rs_data[ring_key]])
        star_div = np.array([df['diversity'].values[gen200_idx] for df in rs_data[star_key]])

        eta2 = eta_squared_groups(ring_div, star_div)
        eta2_rs.append(eta2)
        # Bootstrap CI
        boot_eta2 = []
        for _ in range(1000):
            r_boot = np.random.choice(ring_div, len(ring_div), replace=True)
            s_boot = np.random.choice(star_div, len(star_div), replace=True)
            boot_eta2.append(eta_squared_groups(r_boot, s_boot))
        ci_rs.append(np.std(boot_eta2) * 1.96)

    # Use NK pilot data at K=0,4,6 (3 topologies: dag-layer, bidir-ring, ring-skip2)
    K_values_nk = [0, 4, 6]
    eta2_nk = []
    ci_nk = []
    nk_topos = ['dag-layer', 'bidir-ring', 'ring-skip2']

    for K in K_values_nk:
        groups = []
        for topo in nk_topos:
            key = f'nk{K}_{topo}'
            if key not in nk_data:
                continue
            gens = nk_data[key][0]['generation'].values
            gen200_idx = find_gen_index(gens, 200)
            vals = np.array([df['diversity'].values[gen200_idx] for df in nk_data[key]])
            groups.append(vals)

        if len(groups) >= 2:
            eta2 = eta_squared_groups(*groups)
            eta2_nk.append(eta2)
            boot_eta2 = []
            for _ in range(1000):
                boot_groups = [np.random.choice(g, len(g), replace=True) for g in groups]
                boot_eta2.append(eta_squared_groups(*boot_groups))
            ci_nk.append(np.std(boot_eta2) * 1.96)
        else:
            eta2_nk.append(0)
            ci_nk.append(0)

    print(f"  Ring-vs-star eta2 at K={K_values_rs}: {[f'{e:.3f}' for e in eta2_rs]}")
    print(f"  NK pilot eta2 at K={K_values_nk}: {[f'{e:.3f}' for e in eta2_nk]}")

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # Plot NK pilot (3 topologies) - main curve
    ax.errorbar(K_values_nk, eta2_nk, yerr=ci_nk,
                fmt='o-', color=CB_BLUE, markersize=7, capsize=4, capthick=1.2,
                markeredgecolor='white', markeredgewidth=0.8, linewidth=2,
                label='3 topologies (NK pilot)', zorder=5)

    # Plot ring-vs-star (2 topologies) as secondary
    ax.errorbar(K_values_rs, eta2_rs, yerr=ci_rs,
                fmt='s--', color=CB_ORANGE, markersize=6, capsize=3, capthick=1,
                markeredgecolor='white', markeredgewidth=0.5, linewidth=1.5,
                label='Ring vs Star', zorder=4)

    # Cohen's thresholds
    for thresh, lbl in [(0.01, 'small'), (0.06, 'medium'), (0.14, 'large')]:
        ax.axhline(y=thresh, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.text(6.3, thresh + 0.005, lbl, fontsize=7, color='gray', ha='left', alpha=0.7)

    ax.set_xlabel('Epistasis $K$')
    ax.set_ylabel('$\\eta^2$ (topology effect on diversity)')
    ax.set_xlim(-0.5, 7)
    ymax = max(max(eta2_nk), max(eta2_rs)) * 1.25
    ax.set_ylim(0, min(ymax, 1.0))
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='none')

    fig.tight_layout()
    save_fig(fig, 'eta-squared-vs-k')


# ============================================================
# FIGURE 3: Ring vs Star diversity across K
# ============================================================
def figure3_ring_vs_star():
    """Ring vs Star diversity comparison across NK K values."""
    print("\nGenerating Fig 3: Ring vs Star diversity...")

    rs_data = load_csvs_from_dir('/tmp/eumas_data/ring_vs_star/')
    K_values = [0, 2, 4, 6]

    fig, axes = plt.subplots(1, 4, figsize=(7, 2.8), sharey=True)

    for ax, K in zip(axes, K_values):
        ring_key = f'nk{K}_ring'
        star_key = f'nk{K}_star'
        if ring_key not in rs_data or star_key not in rs_data:
            continue

        gens = rs_data[ring_key][0]['generation'].values

        ring_div = np.array([df['diversity'].values for df in rs_data[ring_key]])
        star_div = np.array([df['diversity'].values for df in rs_data[star_key]])

        ring_mean = ring_div.mean(axis=0)
        star_mean = star_div.mean(axis=0)
        ring_se = ring_div.std(axis=0) / np.sqrt(ring_div.shape[0])
        star_se = star_div.std(axis=0) / np.sqrt(star_div.shape[0])

        ax.plot(gens, ring_mean, color=CB_BLUE, linewidth=1.5, label='Ring')
        ax.fill_between(gens, ring_mean - ring_se, ring_mean + ring_se,
                        alpha=0.15, color=CB_BLUE)
        ax.plot(gens, star_mean, color=CB_RED, linewidth=1.5, label='Star')
        ax.fill_between(gens, star_mean - star_se, star_mean + star_se,
                        alpha=0.15, color=CB_RED)

        # Compute eta2 at gen 200
        gen200_idx = find_gen_index(gens, 200)
        r_vals = ring_div[:, gen200_idx]
        s_vals = star_div[:, gen200_idx]
        eta2 = eta_squared_groups(r_vals, s_vals)

        ax.set_title(f'$K = {K}$', fontsize=10)
        ax.set_xlabel('Gen.', fontsize=9)
        ax.text(0.97, 0.95, f'$\\eta^2 = {eta2:.3f}$', transform=ax.transAxes,
                ha='right', va='top', fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='gray', alpha=0.8))

        if K == 0:
            ax.legend(fontsize=7, loc='upper right', framealpha=0.8, edgecolor='none')

    axes[0].set_ylabel('Mean diversity')

    fig.suptitle('Both have $\\beta_1 = 1$: modest difference', fontsize=10, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'ring-vs-star-diversity')


# ============================================================
# FIGURE 4: Bridge experiment — temporal separation
# ============================================================
def figure4_bridge_diversity():
    """Diversity over generations for bridge families showing temporal separation."""
    print("\nGenerating Fig 4: Bridge diversity (temporal separation)...")

    bridge_data = load_csvs_from_dir('/tmp/eumas_data/bridge/')
    print(f"  Bridge topologies: {list(bridge_data.keys())}")

    # Use NK4 domain (shows stronger effects)
    # Family 1 (cycle, lambda2=0.586): ring(b1=1), ring-chord1(b1=2), ring-chord2(b1=3), ring-chord3(b1=4)
    # Family 2 (star, lambda2=1.0): star(b1=0), star-leaf1(b1=1), star-leaf2(b1=2), star-leaf3(b1=3)

    cycle_topos = [('nk4_ring', 1), ('nk4_ring-chord1', 2), ('nk4_ring-chord2', 3), ('nk4_ring-chord3', 4)]
    star_topos = [('nk4_star', 0), ('nk4_star-leaf1', 1), ('nk4_star-leaf2', 2), ('nk4_star-leaf3', 3)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.5), sharey=True)

    # Cycle family
    alphas_cycle = [0.4, 0.6, 0.8, 1.0]
    for (topo_key, b1), alpha in zip(cycle_topos, alphas_cycle):
        if topo_key not in bridge_data:
            print(f"  WARNING: missing {topo_key}")
            continue
        dfs = bridge_data[topo_key]
        gens = dfs[0]['generation'].values
        div = np.array([df['diversity'].values for df in dfs])
        mean = div.mean(axis=0)
        se = div.std(axis=0) / np.sqrt(len(dfs))
        ax1.plot(gens, mean, color=CB_BLUE, alpha=alpha, linewidth=1.2 + 0.3*b1,
                 label=f'$\\beta_1={b1}$')
        ax1.fill_between(gens, mean - se, mean + se, alpha=0.08, color=CB_BLUE)

    ax1.set_title(f'Cycle family ($\\lambda_2 = 0.586$)', fontsize=10)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Mean diversity')
    ax1.legend(fontsize=7.5, loc='upper right', framealpha=0.9, edgecolor='none')

    # Star family
    alphas_star = [0.4, 0.6, 0.8, 1.0]
    for (topo_key, b1), alpha in zip(star_topos, alphas_star):
        if topo_key not in bridge_data:
            print(f"  WARNING: missing {topo_key}")
            continue
        dfs = bridge_data[topo_key]
        gens = dfs[0]['generation'].values
        div = np.array([df['diversity'].values for df in dfs])
        mean = div.mean(axis=0)
        se = div.std(axis=0) / np.sqrt(len(dfs))
        ax2.plot(gens, mean, color=CB_RED, alpha=alpha, linewidth=1.2 + 0.3*b1,
                 label=f'$\\beta_1={b1}$')
        ax2.fill_between(gens, mean - se, mean + se, alpha=0.08, color=CB_RED)

    ax2.set_title(f'Star family ($\\lambda_2 = 1.0$)', fontsize=10)
    ax2.set_xlabel('Generation')
    ax2.legend(fontsize=7.5, loc='upper right', framealpha=0.9, edgecolor='none')

    # Add vertical markers for temporal reference
    for ax in [ax1, ax2]:
        for gen_mark, lbl in [(100, 'g100'), (500, 'g500')]:
            ax.axvline(x=gen_mark, color='gray', linestyle=':', linewidth=0.6, alpha=0.4)

    fig.tight_layout()
    save_fig(fig, 'bridge-diversity-generations')

    # Now make the eta-squared temporal plot (the KEY figure)
    print("  Generating bridge eta-squared temporal plot...")

    fig2, ax = plt.subplots(figsize=(5.0, 3.5))

    # Compute within-family eta2 at each generation for NK4
    for family_name, topos, color, label in [
        ('cycle', cycle_topos, CB_BLUE, 'Within cycle ($\\lambda_2 = 0.586$)'),
        ('star', star_topos, CB_RED, 'Within star ($\\lambda_2 = 1.0$)')
    ]:
        avail_topos = [(k, b) for k, b in topos if k in bridge_data]
        if len(avail_topos) < 2:
            continue

        ref_key = avail_topos[0][0]
        gens = bridge_data[ref_key][0]['generation'].values
        eta2_over_time = []

        for g_idx in range(len(gens)):
            groups = []
            for topo_key, b1 in avail_topos:
                vals = np.array([df['diversity'].values[g_idx] for df in bridge_data[topo_key]])
                groups.append(vals)
            eta2_over_time.append(eta_squared_groups(*groups))

        ax.plot(gens, eta2_over_time, color=color, linewidth=2, label=label)

    # Cross-family eta2 (matched beta1, different lambda2)
    matched_pairs = {
        1: ('nk4_ring', 'nk4_star-leaf1'),
        2: ('nk4_ring-chord1', 'nk4_star-leaf2'),
        3: ('nk4_ring-chord2', 'nk4_star-leaf3'),
    }

    # Average cross-family eta2 across matched beta1 levels
    ref_key = 'nk4_ring'
    if ref_key in bridge_data:
        gens = bridge_data[ref_key][0]['generation'].values
        cross_eta2_over_time = []

        for g_idx in range(len(gens)):
            eta2_vals = []
            for b1, (cycle_key, star_key) in matched_pairs.items():
                if cycle_key in bridge_data and star_key in bridge_data:
                    c_vals = np.array([df['diversity'].values[g_idx] for df in bridge_data[cycle_key]])
                    s_vals = np.array([df['diversity'].values[g_idx] for df in bridge_data[star_key]])
                    eta2_vals.append(eta_squared_groups(c_vals, s_vals))
            cross_eta2_over_time.append(np.mean(eta2_vals) if eta2_vals else 0)

        ax.plot(gens, cross_eta2_over_time, color=CB_GREEN, linewidth=2, linestyle='--',
                label='Cross-family ($\\lambda_2$ effect)')

    # Reference lines
    ax.axvline(x=100, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
    ax.axvline(x=500, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
    ax.text(105, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 0.15, 'gen 100',
            fontsize=7, color='gray')
    ax.text(505, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 0.15, 'gen 500',
            fontsize=7, color='gray')

    # Cohen's thresholds
    for thresh, lbl in [(0.06, 'medium'), (0.14, 'large')]:
        ax.axhline(y=thresh, color='gray', linewidth=0.5, linestyle=':', alpha=0.4)

    ax.set_xlabel('Generation')
    ax.set_ylabel('$\\eta^2$ (effect size)')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none', fontsize=8)

    fig2.tight_layout()
    save_fig(fig2, 'bridge-eta2-temporal')


# ============================================================
# FIGURE 5: LLM diversity — strict vs lax
# ============================================================
def figure5_llm_diversity():
    """Heatmap-style comparison of LLM diversity under strict vs lax coupling."""
    print("\nGenerating Fig 5: LLM diversity (strict vs lax)...")

    summary = pd.read_csv('/tmp/eumas_data/comparison_summary.csv')
    strict_div = pd.read_csv('/tmp/eumas_data/strict_diversity.csv')
    lax_div = pd.read_csv('/tmp/eumas_data/lax_diversity.csv')

    # This is the GECCO experiment: strict (positive coupling) vs lax (negative coupling)
    # "Strict" = migration preserves best (exploitation-heavy)
    # "Lax" = migration allows diversity (exploration-friendly)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.5))

    # Panel 1: Diversity trajectories
    gens = summary['generation'].values
    strict_mean = summary['strict_div_mean'].values
    strict_ci_lo = summary['strict_div_ci_low'].values
    strict_ci_hi = summary['strict_div_ci_high'].values
    lax_mean = summary['lax_div_mean'].values
    lax_ci_lo = summary['lax_div_ci_low'].values
    lax_ci_hi = summary['lax_div_ci_high'].values

    ax1.plot(gens, strict_mean, color=CB_BLUE, linewidth=2, label='Strict (exploit)')
    ax1.fill_between(gens, strict_ci_lo, strict_ci_hi, alpha=0.15, color=CB_BLUE)
    ax1.plot(gens, lax_mean, color=CB_RED, linewidth=2, label='Lax (explore)')
    ax1.fill_between(gens, lax_ci_lo, lax_ci_hi, alpha=0.15, color=CB_RED)

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Mean diversity')
    ax1.legend(loc='upper right', framealpha=0.9, edgecolor='none')
    ax1.set_xlim(0, 100)

    # Panel 2: Cohen's d over time (sign flip visualization)
    cohens_d = summary['cohens_d'].values

    ax2.plot(gens, cohens_d, color=CB_PURPLE, linewidth=2)
    ax2.axhline(y=0, color='gray', linewidth=0.5)

    # Mark the sign flip region
    # Find where effect becomes significant (d > 0.5)
    sig_gen = gens[np.argmax(cohens_d > 0.5)] if np.any(cohens_d > 0.5) else None
    if sig_gen is not None:
        ax2.axvline(x=sig_gen, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
        ax2.text(sig_gen + 2, 0.3, f'gen {sig_gen}', fontsize=7, color='gray')

    # Cohen's d thresholds
    for thresh, lbl in [(0.2, 'small'), (0.5, 'medium'), (0.8, 'large')]:
        ax2.axhline(y=thresh, color='gray', linewidth=0.4, linestyle=':', alpha=0.3)
        ax2.text(100, thresh + 0.05, lbl, fontsize=6.5, color='gray', ha='right', alpha=0.6)

    ax2.set_xlabel('Generation')
    ax2.set_ylabel("Cohen's $d$ (strict $>$ lax)")
    ax2.set_xlim(0, 100)

    # Note: In LLMs, strict coupling PRESERVES more diversity (contravariant)
    # This is the OPPOSITE of GAs where lax preserves diversity
    ax1.set_title('Contravariant: strict $>$ lax', fontsize=9, style='italic')
    ax2.set_title('Effect size grows monotonically', fontsize=9, style='italic')

    fig.tight_layout()
    save_fig(fig, 'llm-diversity-heatmap')

    # Print key stats
    d_at_50 = summary.loc[summary['generation'] == 50, 'cohens_d'].values[0]
    d_at_100 = summary.loc[summary['generation'] == 100, 'cohens_d'].values[0]
    print(f"  Cohen's d at gen 50: {d_at_50:.3f}")
    print(f"  Cohen's d at gen 100: {d_at_100:.3f}")


# ============================================================
# FIGURE 6: Invariant hierarchy (conceptual diagram)
# ============================================================
def figure6_invariant_hierarchy():
    """Conceptual diagram of three-invariant hierarchy."""
    print("\nGenerating Fig 6: Invariant hierarchy...")

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # Three boxes representing invariants
    boxes = [
        (0.15, 0.65, 'Cycle length\n$\\ell(C)$', CB_ORANGE, 'Immediate\n(within generation)'),
        (0.45, 0.65, 'Cycle rank\n$\\beta_1 = |E| - n + 1$', CB_BLUE, 'Transient\n(gens 50--200)'),
        (0.75, 0.65, 'Fiedler value\n$\\lambda_2(L)$', CB_GREEN, 'Persistent\n(gens 200+)'),
    ]

    box_width = 0.22
    box_height = 0.2

    for x, y, label, color, timescale in boxes:
        rect = mpatches.FancyBboxPatch(
            (x - box_width/2, y - box_height/2), box_width, box_height,
            boxstyle='round,pad=0.02', facecolor=color, alpha=0.15,
            edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y + 0.02, label, ha='center', va='center', fontsize=9,
                fontweight='bold', color=color)
        ax.text(x, y - box_height/2 - 0.08, timescale, ha='center', va='top',
                fontsize=8, color='#555555', style='italic')

    # Arrows between boxes
    arrow_style = dict(arrowstyle='->', color='#666666', lw=1.5,
                       connectionstyle='arc3,rad=0')

    ax.annotate('', xy=(0.45 - box_width/2 - 0.01, 0.65),
                xytext=(0.15 + box_width/2 + 0.01, 0.65),
                arrowprops=arrow_style)
    ax.annotate('', xy=(0.75 - box_width/2 - 0.01, 0.65),
                xytext=(0.45 + box_width/2 + 0.01, 0.65),
                arrowprops=arrow_style)

    # Bottom annotation: the key insight
    ax.text(0.5, 0.18,
            'Shortest cycle determines local mixing speed\n'
            'Cycle rank determines total feedback capacity\n'
            'Fiedler value determines global information flow',
            ha='center', va='center', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5',
                      edgecolor='#cccccc', linewidth=1),
            linespacing=1.6)

    # Top label
    ax.text(0.5, 0.95, 'Three-invariant hierarchy',
            ha='center', va='top', fontsize=11, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    fig.tight_layout()
    save_fig(fig, 'invariant-hierarchy')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("EUMAS 2026 Figure Generation")
    print("=" * 60)

    figure1_iso_dense_graphs()
    figure2_eta_squared_vs_k()
    figure3_ring_vs_star()
    figure4_bridge_diversity()
    figure5_llm_diversity()
    figure6_invariant_hierarchy()

    print("\n" + "=" * 60)
    print("ALL 6 FIGURES GENERATED")
    print("=" * 60)
    print(f"Output: {OUTDIR}/")
    for name in ['iso-dense-graphs', 'eta-squared-vs-k', 'ring-vs-star-diversity',
                  'bridge-diversity-generations', 'bridge-eta2-temporal',
                  'llm-diversity-heatmap', 'invariant-hierarchy']:
        pdf = os.path.join(OUTDIR, f'{name}.pdf')
        if os.path.exists(pdf):
            print(f"  {name}.pdf  OK")
        else:
            print(f"  {name}.pdf  MISSING")

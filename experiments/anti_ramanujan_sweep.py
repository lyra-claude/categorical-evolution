#!/usr/bin/env python3
"""
Anti-Ramanujan sweep: lambda2 (algebraic connectivity) vs diversity.

Computes the second-smallest eigenvalue of the graph Laplacian (lambda2)
for each migration topology used in our experiments, plus the GP(5,1)
prism graph (k=3 regular, candidate anti-Ramanujan for k=3).

Key insight: anti-Ramanujan graphs MINIMIZE lambda2 for a given degree k.
Lower lambda2 = slower synchronization = more diversity preservation = lax.
Our laxator magnitude should be proportional to 1/lambda2.

For k=2, the ring is the unique anti-Ramanujan graph on n vertices.
For k=3, GP(n,1) (prism/generalized Petersen) is the natural candidate.

Usage:
    python anti_ramanujan_sweep.py
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, 'plots')
N = 5  # Number of islands (matching all experiments)

DOMAIN_CSVS = {
    'OneMax': os.path.join(SCRIPT_DIR, 'experiment_e_raw.csv'),
    'Maze': os.path.join(SCRIPT_DIR, 'experiment_e_maze.csv'),
    'Graph Coloring': os.path.join(SCRIPT_DIR, 'experiment_e_graph_coloring.csv'),
    'Knapsack': os.path.join(SCRIPT_DIR, 'experiment_e_knapsack.csv'),
}

TOPO_ORDER = ['none', 'ring', 'star', 'random', 'fully_connected']


# ---------------------------------------------------------------------------
# Graph construction for each topology
# ---------------------------------------------------------------------------

def build_none_graph(n):
    """No edges. Isolated islands."""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    return G


def build_ring_graph(n):
    """Ring: each island i connects to (i-1) mod n and (i+1) mod n.
    This is the cycle graph C_n. k=2 regular."""
    return nx.cycle_graph(n)


def build_star_graph(n):
    """Star: island 0 (hub) connects to all others.
    Hub has degree n-1, spokes have degree 1. NOT regular."""
    return nx.star_graph(n - 1)  # nx.star_graph(k) creates k+1 nodes


def build_fully_connected_graph(n):
    """Complete graph K_n. k=(n-1) regular."""
    return nx.complete_graph(n)


def build_random_graph(n, num_edges=None):
    """Random topology: n random edges (same count as ring = n edges).
    Since the random topology re-randomizes each migration event,
    we compute lambda2 for the EXPECTED graph: Erdos-Renyi with
    edge probability p = n_edges / C(n,2).

    For n=5, ring has 5 edges, C(5,2) = 10, so p = 0.5.
    We compute the expected Laplacian analytically:
      L_expected = n*p * I - p * J + p * I = n*p*I - p*(J - I)
    Actually, for Erdos-Renyi G(n,p), the expected adjacency matrix is
      E[A] = p*(J - I), where J is all-ones matrix.
    The expected Laplacian is:
      E[L] = diag(E[degree]) - E[A] = (n-1)*p*I - p*(J - I) = n*p*I - p*J
    Eigenvalues of E[L]: np is eigenvalue with multiplicity n-1, and 0 once.
    So E[lambda2] = n*p.

    But let's also compute it empirically: sample 1000 random graphs and average.
    """
    if num_edges is None:
        num_edges = n  # same edge count as ring

    # Sample approach: generate many random graphs with exactly n edges
    lambda2_samples = []
    rng = np.random.default_rng(42)
    for _ in range(1000):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        edges_added = set()
        attempts = 0
        while len(edges_added) < num_edges and attempts < 1000:
            i = rng.integers(0, n)
            j = rng.integers(0, n - 1)
            if j >= i:
                j += 1
            edge = (min(i, j), max(i, j))
            if edge not in edges_added:
                edges_added.add(edge)
                G.add_edge(*edge)
            attempts += 1
        l2 = compute_lambda2(G)
        if l2 is not None:
            lambda2_samples.append(l2)

    return np.mean(lambda2_samples), np.std(lambda2_samples)


def build_prism_graph(n):
    """GP(n,1): Generalized Petersen graph / Prism graph.
    Two concentric n-cycles connected by n "rungs".
    Total: 2n nodes, 3n edges. k=3 regular.

    But wait -- our experiments use n=5 ISLANDS. GP(5,1) has 10 nodes.
    For a fair comparison, we need a k=3 regular graph on 5 nodes.

    A k=3 regular graph on 5 nodes requires 5*3/2 = 7.5 edges.
    This is impossible -- k=3 regular on n=5 doesn't exist (n*k must be even).

    So GP(n,1) with n=5 has 10 nodes. We can't use it directly with 5 islands.

    Options:
    1. Report GP(5,1) on 10 nodes for theoretical comparison.
    2. Find the minimum-lambda2 graph on 5 nodes with various k values.
    3. Compute lambda2 for all possible graphs on 5 nodes.

    Let's do option 2: for each achievable k, find the graph on 5 nodes
    that minimizes lambda2 (the anti-Ramanujan graph).
    """
    # Build the actual GP(5,1) prism graph (10 nodes)
    G = nx.Graph()
    # Outer ring: 0-1-2-3-4-0
    for i in range(n):
        G.add_edge(i, (i + 1) % n)
    # Inner ring: 5-6-7-8-9-5
    for i in range(n):
        G.add_edge(n + i, n + (i + 1) % n)
    # Rungs: 0-5, 1-6, 2-7, 3-8, 4-9
    for i in range(n):
        G.add_edge(i, n + i)
    return G


# ---------------------------------------------------------------------------
# Lambda2 computation
# ---------------------------------------------------------------------------

def compute_lambda2(G):
    """Compute algebraic connectivity (second-smallest eigenvalue of Laplacian).
    Returns None if graph has fewer than 2 nodes."""
    n = G.number_of_nodes()
    if n < 2:
        return None

    L = nx.laplacian_matrix(G).toarray().astype(float)
    eigenvalues = np.sort(np.linalg.eigvalsh(L))

    # lambda2 is the second eigenvalue (first non-zero)
    # For disconnected graphs, lambda2 = 0
    return float(eigenvalues[1])


def compute_lambda2_normalized(G):
    """Compute normalized algebraic connectivity: lambda2 / k
    where k is the average degree. This normalizes for edge count."""
    l2 = compute_lambda2(G)
    if l2 is None:
        return None
    n = G.number_of_nodes()
    avg_degree = 2 * G.number_of_edges() / n if n > 0 else 0
    if avg_degree == 0:
        return 0.0
    return l2 / avg_degree


# ---------------------------------------------------------------------------
# Exhaustive search: minimum-lambda2 graphs on 5 nodes
# ---------------------------------------------------------------------------

def enumerate_k_regular_5(k):
    """Find all k-regular graphs on 5 nodes.
    Only k=0, 2, 4 are possible (n*k must be even for n=5)."""
    if (5 * k) % 2 != 0:
        return []

    from itertools import combinations
    possible_edges = list(combinations(range(5), 2))  # 10 edges total
    num_edges = (5 * k) // 2

    results = []
    for edge_set in combinations(possible_edges, num_edges):
        degrees = [0] * 5
        for i, j in edge_set:
            degrees[i] += 1
            degrees[j] += 1
        if all(d == k for d in degrees):
            G = nx.Graph()
            G.add_nodes_from(range(5))
            G.add_edges_from(edge_set)
            results.append(G)
    return results


def find_min_lambda2_5():
    """For each achievable degree k on 5 nodes, find the graph
    that minimizes lambda2 (the anti-Ramanujan graph)."""
    results = {}
    for k in range(5):
        if (5 * k) % 2 != 0:
            continue
        graphs = enumerate_k_regular_5(k)
        if not graphs:
            continue

        best_l2 = float('inf')
        best_G = None
        all_l2 = []
        for G in graphs:
            l2 = compute_lambda2(G)
            if l2 is not None:
                all_l2.append(l2)
                if l2 < best_l2:
                    best_l2 = l2
                    best_G = G

        results[k] = {
            'min_lambda2': best_l2,
            'max_lambda2': max(all_l2) if all_l2 else None,
            'all_lambda2': sorted(set(round(x, 6) for x in all_l2)),
            'best_graph': best_G,
            'num_graphs': len(graphs),
            'best_edges': list(best_G.edges()) if best_G else [],
        }
    return results


# ---------------------------------------------------------------------------
# Diversity data from experiment CSVs
# ---------------------------------------------------------------------------

def load_final_diversity(csv_path, gen=99):
    """Load final-generation mean Hamming diversity per topology from CSV."""
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    max_gen = df['generation'].max()
    final = df[df['generation'] == max_gen]
    result = {}
    for topo in TOPO_ORDER:
        vals = final[final['topology'] == topo]['hamming_diversity']
        if len(vals) > 0:
            result[topo] = {
                'mean': vals.mean(),
                'se': vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0.0,
                'n': len(vals),
            }
    return result


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("=" * 80)
    print("  ANTI-RAMANUJAN SWEEP")
    print("  lambda2 (algebraic connectivity) vs diversity preservation")
    print(f"  n = {N} islands")
    print("=" * 80)

    # ---- Step 1: Compute lambda2 for each experimental topology ----
    print("\n--- Step 1: Lambda2 for experimental topologies (n=5) ---\n")

    topo_results = {}

    # None (isolated)
    G_none = build_none_graph(N)
    l2_none = compute_lambda2(G_none)
    topo_results['none'] = {
        'lambda2': l2_none,
        'k': 0,
        'edges': G_none.number_of_edges(),
        'regular': True,
        'description': 'Isolated (no edges)',
    }

    # Ring (cycle C5)
    G_ring = build_ring_graph(N)
    l2_ring = compute_lambda2(G_ring)
    topo_results['ring'] = {
        'lambda2': l2_ring,
        'k': 2,
        'edges': G_ring.number_of_edges(),
        'regular': True,
        'description': f'Cycle C_{N} (k=2 regular)',
    }

    # Star
    G_star = build_star_graph(N)
    l2_star = compute_lambda2(G_star)
    degrees_star = [d for _, d in G_star.degree()]
    topo_results['star'] = {
        'lambda2': l2_star,
        'k': f'{min(degrees_star)}-{max(degrees_star)}',
        'edges': G_star.number_of_edges(),
        'regular': False,
        'description': f'Star S_{N} (hub k={max(degrees_star)}, spokes k={min(degrees_star)})',
    }

    # Fully connected
    G_fc = build_fully_connected_graph(N)
    l2_fc = compute_lambda2(G_fc)
    topo_results['fully_connected'] = {
        'lambda2': l2_fc,
        'k': N - 1,
        'edges': G_fc.number_of_edges(),
        'regular': True,
        'description': f'Complete K_{N} (k={N-1} regular)',
    }

    # Random (expected lambda2)
    print("  Computing expected lambda2 for random topology (1000 samples)...")
    rand_mean, rand_std = build_random_graph(N)
    topo_results['random'] = {
        'lambda2': rand_mean,
        'lambda2_std': rand_std,
        'k': f'~{2*N / (N*(N-1)/2):.2f} avg',  # expected avg degree
        'edges': N,
        'regular': False,
        'description': f'Random ({N} edges per event, resampled)',
    }

    # Print table
    print(f"\n  {'Topology':<22s} {'k':>8s} {'Edges':>6s} {'lambda2':>10s} {'Regular':>8s}")
    print("  " + "-" * 60)
    for topo in ['none', 'ring', 'star', 'random', 'fully_connected']:
        r = topo_results[topo]
        l2_str = f"{r['lambda2']:.6f}" if isinstance(r['lambda2'], float) else str(r['lambda2'])
        if 'lambda2_std' in r:
            l2_str += f" +/- {r['lambda2_std']:.4f}"
        print(f"  {topo:<22s} {str(r['k']):>8s} {r['edges']:>6d} {l2_str:>20s} {'Yes' if r['regular'] else 'No':>8s}")

    # ---- Step 2: GP(5,1) prism graph ----
    print("\n--- Step 2: GP(5,1) Prism graph (k=3 regular, 10 nodes) ---\n")
    G_prism = build_prism_graph(N)
    l2_prism = compute_lambda2(G_prism)
    print(f"  GP({N},1) prism graph: 10 nodes, {G_prism.number_of_edges()} edges, k=3 regular")
    print(f"  lambda2 = {l2_prism:.6f}")
    print(f"  Note: This is on 10 nodes, not 5. For comparison only.")

    # ---- Step 3: Exhaustive search on 5 nodes ----
    print("\n--- Step 3: All k-regular graphs on 5 nodes ---\n")
    reg_results = find_min_lambda2_5()
    print(f"  {'k':>3s} {'#graphs':>8s} {'min lambda2':>12s} {'max lambda2':>12s} {'all lambda2 values'}")
    print("  " + "-" * 70)
    for k in sorted(reg_results.keys()):
        r = reg_results[k]
        all_str = ', '.join(f'{v:.4f}' for v in r['all_lambda2'])
        print(f"  {k:>3d} {r['num_graphs']:>8d} {r['min_lambda2']:>12.6f} {r['max_lambda2']:>12.6f}   [{all_str}]")
        if r['best_graph']:
            print(f"      Anti-Ramanujan edges: {r['best_edges']}")

    # ---- Step 4: Load diversity data from experiments ----
    print("\n--- Step 4: Diversity data from experiments ---\n")
    all_diversity = {}
    for domain, path in DOMAIN_CSVS.items():
        div_data = load_final_diversity(path)
        if div_data:
            all_diversity[domain] = div_data
            print(f"  Loaded {domain}")

    # ---- Step 5: Combined table ----
    print("\n" + "=" * 80)
    print("  COMBINED TABLE: lambda2 vs Final-Generation Diversity")
    print("=" * 80)

    # Header
    header = f"  {'Topology':<18s} {'lambda2':>10s}"
    for domain in all_diversity:
        header += f" {domain:>14s}"
    print(header)
    print("  " + "-" * (28 + 15 * len(all_diversity)))

    for topo in TOPO_ORDER:
        r = topo_results[topo]
        l2 = r['lambda2']
        l2_str = f"{l2:.4f}" if isinstance(l2, float) and l2 > 0 else "0.0000"
        if 'lambda2_std' in r:
            l2_str = f"{l2:.4f}*"  # mark as approximate
        row = f"  {topo:<18s} {l2_str:>10s}"
        for domain in all_diversity:
            if topo in all_diversity[domain]:
                d = all_diversity[domain][topo]
                row += f"   {d['mean']:.4f}+/-{d['se']:.4f}"
            else:
                row += f" {'--':>14s}"
        print(row)

    # ---- Step 6: Correlation analysis ----
    print("\n--- Step 6: Correlation: lambda2 vs diversity ---\n")

    # For correlation, use the 4 coupled topologies (exclude 'none' which is trivially disconnected)
    coupled_topos = ['ring', 'star', 'random', 'fully_connected']

    for domain in all_diversity:
        lambda2_vals = []
        diversity_vals = []
        labels = []

        for topo in coupled_topos:
            if topo in all_diversity[domain] and topo in topo_results:
                l2 = topo_results[topo]['lambda2']
                div = all_diversity[domain][topo]['mean']
                lambda2_vals.append(l2)
                diversity_vals.append(div)
                labels.append(topo)

        if len(lambda2_vals) >= 3:
            from scipy import stats as sp_stats
            # Pearson correlation
            r_pearson, p_pearson = sp_stats.pearsonr(lambda2_vals, diversity_vals)
            # Spearman rank correlation
            rho_spearman, p_spearman = sp_stats.spearmanr(lambda2_vals, diversity_vals)

            print(f"  {domain}:")
            print(f"    Pearson  r = {r_pearson:+.4f}, p = {p_pearson:.4f}")
            print(f"    Spearman rho = {rho_spearman:+.4f}, p = {p_spearman:.4f}")

            # Direction check
            direction = "NEGATIVE" if r_pearson < 0 else "POSITIVE"
            print(f"    Direction: {direction} (higher lambda2 -> {'lower' if r_pearson < 0 else 'higher'} diversity)")
        else:
            print(f"  {domain}: insufficient data for correlation")

    # Also with 'none' included
    print("\n  Including 'none' (isolated) topology:")
    for domain in all_diversity:
        lambda2_vals = []
        diversity_vals = []
        for topo in TOPO_ORDER:
            if topo in all_diversity[domain] and topo in topo_results:
                l2 = topo_results[topo]['lambda2']
                div = all_diversity[domain][topo]['mean']
                lambda2_vals.append(l2)
                diversity_vals.append(div)

        if len(lambda2_vals) >= 3:
            from scipy import stats as sp_stats
            r_pearson, p_pearson = sp_stats.pearsonr(lambda2_vals, diversity_vals)
            rho_spearman, p_spearman = sp_stats.spearmanr(lambda2_vals, diversity_vals)
            print(f"    {domain}: Pearson r = {r_pearson:+.4f} (p={p_pearson:.4f}), "
                  f"Spearman rho = {rho_spearman:+.4f} (p={p_spearman:.4f})")

    # ---- Step 7: Key theoretical results ----
    print("\n" + "=" * 80)
    print("  KEY RESULTS")
    print("=" * 80)

    # Ring is anti-Ramanujan for k=2
    print(f"\n  1. Ring (C_5) lambda2 = {l2_ring:.6f}")
    if 2 in reg_results:
        print(f"     All k=2 regular on 5 nodes: {reg_results[2]['all_lambda2']}")
        if abs(l2_ring - reg_results[2]['min_lambda2']) < 1e-6:
            print(f"     CONFIRMED: Ring has MINIMUM lambda2 among all k=2 regular graphs on 5 nodes.")
            print(f"     Ring IS the anti-Ramanujan graph for k=2, n=5.")
        else:
            print(f"     Ring does NOT have minimum lambda2. Min is {reg_results[2]['min_lambda2']:.6f}")

    # Fully connected lambda2
    print(f"\n  2. Fully connected (K_5) lambda2 = {l2_fc:.6f}")
    print(f"     K_n always has lambda2 = n = {N} (Ramanujan-optimal for k=n-1).")

    # Ordering
    ordered = sorted(
        [(topo, topo_results[topo]['lambda2'])
         for topo in coupled_topos],
        key=lambda x: x[1]
    )
    order_str = ' < '.join(f'{t}({l2:.4f})' for t, l2 in ordered)
    print(f"\n  3. Lambda2 ordering (coupled topologies):")
    print(f"     {order_str}")

    # Compare with diversity ordering
    print(f"\n  4. Diversity ordering (coupled topologies, OneMax):")
    if 'OneMax' in all_diversity:
        div_ordered = sorted(
            [(topo, all_diversity['OneMax'][topo]['mean'])
             for topo in coupled_topos if topo in all_diversity['OneMax']],
            key=lambda x: x[1], reverse=True
        )
        div_str = ' > '.join(f'{t}({d:.4f})' for t, d in div_ordered)
        print(f"     {div_str}")

        # Check if orderings are inversely related
        l2_ranks = [t for t, _ in ordered]
        div_ranks = [t for t, _ in div_ordered]
        print(f"\n     Lambda2 order (low->high): {' < '.join(l2_ranks)}")
        print(f"     Diversity order (high->low): {' > '.join(div_ranks)}")
        if l2_ranks == div_ranks:
            print(f"     PERFECT INVERSE CORRELATION: lowest lambda2 = highest diversity")
        else:
            print(f"     Orderings differ (see correlation analysis above)")

    # 1/lambda2 as laxator measure
    print(f"\n  5. Laxator magnitude ~ 1/lambda2:")
    for topo, l2 in ordered:
        if l2 > 0:
            print(f"     {topo:<22s} 1/lambda2 = {1.0/l2:.4f}")
        else:
            print(f"     {topo:<22s} 1/lambda2 = inf (disconnected)")

    # ---- Step 8: Plot ----
    print("\n--- Step 8: Generating plot ---\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: lambda2 vs diversity scatter
    ax = axes[0]
    markers = {'OneMax': 'o', 'Maze': 's', 'Graph Coloring': 'D', 'Knapsack': 'v'}
    colors = {'OneMax': '#4E79A7', 'Maze': '#F28E2B', 'Graph Coloring': '#E15759', 'Knapsack': '#B07AA1'}

    for domain in all_diversity:
        lambda2_vals = []
        diversity_vals = []
        diversity_se = []
        topo_labels = []

        for topo in TOPO_ORDER:
            if topo in all_diversity[domain] and topo in topo_results:
                l2 = topo_results[topo]['lambda2']
                d = all_diversity[domain][topo]
                lambda2_vals.append(l2)
                diversity_vals.append(d['mean'])
                diversity_se.append(d['se'])
                topo_labels.append(topo)

        ax.errorbar(
            lambda2_vals, diversity_vals,
            yerr=diversity_se,
            fmt=markers.get(domain, 'o'),
            color=colors.get(domain, 'gray'),
            markersize=8, capsize=3, linewidth=1,
            label=domain, alpha=0.8,
        )

        # Label each point with topology name (only for OneMax to avoid clutter)
        if domain == 'OneMax':
            for l2, div, topo in zip(lambda2_vals, diversity_vals, topo_labels):
                ax.annotate(
                    topo, (l2, div),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=7, color='0.4',
                )

    ax.set_xlabel('$\\lambda_2$ (algebraic connectivity)', fontsize=11)
    ax.set_ylabel('Final-generation Hamming diversity', fontsize=11)
    ax.set_title('(a) $\\lambda_2$ vs Diversity\nacross 4 domains', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, frameon=True, framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: lambda2 bar chart for topologies
    ax = axes[1]
    topo_display = ['none', 'ring', 'star', 'random', 'fully_connected']
    l2_vals = [topo_results[t]['lambda2'] for t in topo_display]
    bar_colors = ['#2166AC', '#67A9CF', '#5AB4AC', '#F4A582', '#B2182B']

    bars = ax.bar(range(len(topo_display)), l2_vals, color=bar_colors,
                  edgecolor='white', linewidth=1.2, width=0.7)

    # Add error bar for random
    rand_idx = topo_display.index('random')
    if 'lambda2_std' in topo_results['random']:
        ax.errorbar(
            rand_idx, topo_results['random']['lambda2'],
            yerr=topo_results['random']['lambda2_std'],
            fmt='none', capsize=5, color='black', linewidth=1.5,
        )

    # Add value labels on bars
    for i, (bar, l2) in enumerate(zip(bars, l2_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{l2:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add degree annotation
    for i, topo in enumerate(topo_display):
        r = topo_results[topo]
        ax.text(i, -0.15, f'k={r["k"]}', ha='center', va='top',
                fontsize=8, color='0.5', transform=ax.get_xaxis_transform())

    ax.set_xticks(range(len(topo_display)))
    ax.set_xticklabels(['None\n(isolated)', 'Ring', 'Star', 'Random*', 'Fully\nConnected'],
                        fontsize=9)
    ax.set_ylabel('$\\lambda_2$ (algebraic connectivity)', fontsize=11)
    ax.set_title('(b) $\\lambda_2$ by Topology\n(* random = mean over 1000 samples)',
                 fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add arrow: lax -> strict
    ax.annotate(
        '', xy=(len(topo_display) - 0.3, -0.08), xytext=(-0.3, -0.08),
        xycoords=('data', 'axes fraction'),
        textcoords=('data', 'axes fraction'),
        arrowprops=dict(arrowstyle='->', color='0.5', linewidth=1.0),
        annotation_clip=False,
    )
    ax.text((len(topo_display) - 1) / 2, -0.15,
            'anti-Ramanujan (lax)                                             Ramanujan (strict)',
            transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=7, color='0.5', style='italic')

    fig.suptitle(
        'Anti-Ramanujan Sweep: Algebraic Connectivity ($\\lambda_2$) vs Diversity Preservation\n'
        f'n = {N} islands, 30 seeds, generation 99',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()

    for fmt in ('png', 'pdf'):
        path = os.path.join(PLOT_DIR, f'anti_ramanujan_sweep.{fmt}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved {path}")
    plt.close(fig)

    # ---- Step 9: Theoretical prediction for anti-Ramanujan GP(5,1) ----
    print("\n" + "=" * 80)
    print("  PREDICTION: GP(5,1) as k=3 anti-Ramanujan")
    print("=" * 80)
    print(f"\n  GP(5,1) on 10 nodes: lambda2 = {l2_prism:.6f}")
    print(f"  For comparison, K_5 on 5 nodes: lambda2 = {l2_fc:.6f}")
    print(f"  Ring (C_5) on 5 nodes: lambda2 = {l2_ring:.6f}")
    print(f"\n  Note: k=3 regular graphs on 5 nodes are IMPOSSIBLE (5*3 is odd).")
    print(f"  The smallest k=3 regular graph is on 4 nodes (K_4) with lambda2 = 4.0.")
    print(f"  For meaningful k=3 comparison, need even number of nodes (n=6, 8, 10...).")

    # Compute for n=6 (smallest even n for k=3)
    print(f"\n  --- k=3 regular on n=6 nodes ---")
    # Prism graph on 6 nodes: two triangles connected by 3 rungs
    G_prism6 = build_prism_graph(3)  # GP(3,1) = prism on 6 nodes
    l2_prism6 = compute_lambda2(G_prism6)
    print(f"  GP(3,1) prism on 6 nodes: lambda2 = {l2_prism6:.6f}")
    print(f"  K_{3,3} complete bipartite on 6 nodes: lambda2 = {compute_lambda2(nx.complete_bipartite_graph(3,3)):.6f}")

    # The Petersen graph (k=3 regular on 10 nodes)
    G_petersen = nx.petersen_graph()
    l2_petersen = compute_lambda2(G_petersen)
    print(f"\n  Petersen graph (10 nodes, k=3): lambda2 = {l2_petersen:.6f}")
    print(f"  GP(5,1) prism (10 nodes, k=3): lambda2 = {l2_prism:.6f}")
    if l2_prism < l2_petersen:
        print(f"  GP(5,1) has LOWER lambda2 than Petersen -> MORE LAX (anti-Ramanujan direction)")
    else:
        print(f"  Petersen has LOWER lambda2 than GP(5,1) -> Petersen is more anti-Ramanujan")

    # Check all k=2 on n=5 explicitly
    print(f"\n  --- Confirming ring = anti-Ramanujan for k=2, n=5 ---")
    print(f"  C_5 (ring) lambda2 = {l2_ring:.6f}")
    print(f"  Theoretical: for cycle C_n, lambda2 = 2(1 - cos(2pi/n))")
    l2_theory = 2 * (1 - np.cos(2 * np.pi / N))
    print(f"  Computed:    lambda2 = {l2_theory:.6f}")
    print(f"  For k=2 regular on n nodes, C_n is the UNIQUE graph (up to isomorphism).")
    print(f"  Therefore ring is trivially the anti-Ramanujan graph for k=2.")

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()

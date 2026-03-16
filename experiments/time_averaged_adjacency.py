#!/usr/bin/env python3
"""
Time-averaged adjacency analysis for random topology migration.

Tests Claudius's hypothesis: the time-averaged adjacency matrix (sum of all
migration adjacency matrices over a run) has lambda2 HIGHER than the mean of
snapshot lambda2s. This would explain why random topology yields less diversity
than ring despite having lower snapshot lambda2.

Key insight: random topology re-randomizes every migration event. A single
snapshot has ~5 edges on 5 nodes, but over 20 migration events, the cumulative
graph approaches a complete graph (with non-uniform weights). The effective
coupling is therefore much stronger than any single snapshot suggests.

Usage:
    python time_averaged_adjacency.py
"""

import numpy as np
import networkx as nx
from collections import defaultdict


def random_adjacency_snapshot(rng: np.random.Generator, n: int) -> np.ndarray:
    """Generate one random migration adjacency matrix.

    Matches the random_migrate function in onemax_stats.py:
    - Generate n random edges (same count as ring topology)
    - Each edge connects two distinct islands (no self-loops)
    - Edges can repeat (with replacement on edges)

    Returns a symmetric weighted adjacency matrix where entry (i,j) is
    the number of edges between islands i and j in this snapshot.
    """
    adj = np.zeros((n, n), dtype=float)
    for _ in range(n):
        i = rng.integers(0, n)
        j = rng.integers(0, n - 1)
        if j >= i:
            j += 1  # avoid self-loop, matching onemax_stats.py
        adj[i, j] += 1
        adj[j, i] += 1  # symmetric (bidirectional exchange)
    return adj


def compute_lambda2(adj: np.ndarray) -> float:
    """Compute the second-smallest eigenvalue of the Laplacian (algebraic connectivity).

    The Laplacian L = D - A where D is the degree matrix.
    lambda2 (Fiedler value / algebraic connectivity) measures graph connectivity.
    """
    G = nx.from_numpy_array(adj)
    laplacian = nx.laplacian_matrix(G).toarray().astype(float)
    eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))
    # lambda1 should be ~0 (connected graph); lambda2 is algebraic connectivity
    return eigenvalues[1]


def fixed_topology_adjacency(name: str, n: int) -> np.ndarray:
    """Return the adjacency matrix for a fixed topology on n nodes."""
    adj = np.zeros((n, n), dtype=float)

    if name == "ring":
        for i in range(n):
            j = (i + 1) % n
            adj[i, j] = 1
            adj[j, i] = 1

    elif name == "star":
        hub = 0
        for spoke in range(1, n):
            adj[hub, spoke] = 1
            adj[spoke, hub] = 1

    elif name == "fully_connected":
        for i in range(n):
            for j in range(i + 1, n):
                adj[i, j] = 1
                adj[j, i] = 1

    elif name == "none":
        pass  # all zeros

    else:
        raise ValueError(f"Unknown topology: {name}")

    return adj


def run_analysis(n_islands: int = 5, n_migration_events: int = 20,
                 n_runs: int = 100, seed: int = 42):
    """Run the time-averaged adjacency analysis.

    Args:
        n_islands: Number of islands (5 in experiment E)
        n_migration_events: Number of migration events per run (100 gens / 5 freq = 20)
        n_runs: Number of independent runs to average over
        seed: Base random seed
    """
    print("=" * 70)
    print("TIME-AVERAGED ADJACENCY ANALYSIS FOR RANDOM TOPOLOGY")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Islands: {n_islands}")
    print(f"  Migration events per run: {n_migration_events}")
    print(f"  Independent runs: {n_runs}")
    print(f"  Random edges per event: {n_islands} (matching ring edge count)")
    print()

    rng = np.random.default_rng(seed)

    # Storage for results
    all_snapshot_lambda2s = []       # All individual snapshot lambda2 values
    all_time_avg_lambda2s = []       # Time-averaged lambda2 per run
    all_time_avg_matrices = []       # For overall analysis

    for run in range(n_runs):
        run_rng = np.random.default_rng(rng.integers(0, 2**32))

        snapshot_lambda2s_this_run = []
        cumulative_adj = np.zeros((n_islands, n_islands), dtype=float)

        for event in range(n_migration_events):
            # Generate one random adjacency snapshot
            snapshot = random_adjacency_snapshot(run_rng, n_islands)

            # Record snapshot lambda2
            snapshot_lambda2s_this_run.append(compute_lambda2(snapshot))

            # Accumulate into time-averaged matrix
            cumulative_adj += snapshot

        # Compute lambda2 of the time-averaged (cumulative) matrix
        time_avg_lambda2 = compute_lambda2(cumulative_adj)

        all_snapshot_lambda2s.extend(snapshot_lambda2s_this_run)
        all_time_avg_lambda2s.append(time_avg_lambda2)
        all_time_avg_matrices.append(cumulative_adj)

    # --- Results ---
    snapshot_mean = np.mean(all_snapshot_lambda2s)
    snapshot_std = np.std(all_snapshot_lambda2s)
    snapshot_median = np.median(all_snapshot_lambda2s)

    time_avg_mean = np.mean(all_time_avg_lambda2s)
    time_avg_std = np.std(all_time_avg_lambda2s)
    time_avg_median = np.median(all_time_avg_lambda2s)

    print("-" * 70)
    print("SNAPSHOT LAMBDA2 (individual migration events)")
    print("-" * 70)
    print(f"  Mean:   {snapshot_mean:.4f}")
    print(f"  Std:    {snapshot_std:.4f}")
    print(f"  Median: {snapshot_median:.4f}")
    print(f"  Min:    {np.min(all_snapshot_lambda2s):.4f}")
    print(f"  Max:    {np.max(all_snapshot_lambda2s):.4f}")
    print(f"  N:      {len(all_snapshot_lambda2s)} snapshots ({n_runs} runs x {n_migration_events} events)")
    print()

    print("-" * 70)
    print("TIME-AVERAGED LAMBDA2 (cumulative adjacency over full run)")
    print("-" * 70)
    print(f"  Mean:   {time_avg_mean:.4f}")
    print(f"  Std:    {time_avg_std:.4f}")
    print(f"  Median: {time_avg_median:.4f}")
    print(f"  Min:    {np.min(all_time_avg_lambda2s):.4f}")
    print(f"  Max:    {np.max(all_time_avg_lambda2s):.4f}")
    print(f"  N:      {len(all_time_avg_lambda2s)} runs")
    print()

    ratio = time_avg_mean / snapshot_mean
    print("-" * 70)
    print("COMPARISON")
    print("-" * 70)
    print(f"  Time-averaged / Snapshot ratio: {ratio:.2f}x")
    print(f"  Time-averaged - Snapshot diff:  {time_avg_mean - snapshot_mean:.4f}")
    print()

    # --- Fixed topology lambda2 values ---
    print("-" * 70)
    print("FIXED TOPOLOGY LAMBDA2 VALUES (for comparison)")
    print("-" * 70)

    fixed_topologies = ["none", "ring", "star", "fully_connected"]
    fixed_lambda2s = {}
    for topo in fixed_topologies:
        adj = fixed_topology_adjacency(topo, n_islands)
        lam2 = compute_lambda2(adj)
        fixed_lambda2s[topo] = lam2
        print(f"  {topo:20s}: lambda2 = {lam2:.4f}")

    print()

    # Scale fixed topologies to match migration volume
    # Ring has n edges, applied 20 times = total weight 20 per edge
    # To compare fairly, multiply fixed adjacency by n_migration_events
    print("-" * 70)
    print("FIXED TOPOLOGY LAMBDA2 (scaled by migration events for fair comparison)")
    print("-" * 70)
    scaled_fixed_lambda2s = {}
    for topo in fixed_topologies:
        adj = fixed_topology_adjacency(topo, n_islands)
        scaled_adj = adj * n_migration_events
        lam2 = compute_lambda2(scaled_adj)
        scaled_fixed_lambda2s[topo] = lam2
        print(f"  {topo:20s}: lambda2 = {lam2:.4f} (= {fixed_lambda2s[topo]:.4f} x {n_migration_events})")

    print()

    # --- Find closest fixed topology ---
    print("-" * 70)
    print("CLOSEST FIXED TOPOLOGY MATCH")
    print("-" * 70)

    # Compare unscaled (since time-averaged is already cumulative)
    # The time-averaged matrix has total weight = n_islands * n_migration_events = 5 * 20 = 100 edges total
    # Ring scaled would have n_islands * n_migration_events = 100 edges total too
    # So the scaled comparison is fair

    closest_topo = None
    closest_dist = float('inf')
    for topo, lam2 in scaled_fixed_lambda2s.items():
        dist = abs(lam2 - time_avg_mean)
        if dist < closest_dist:
            closest_dist = dist
            closest_topo = topo
        print(f"  |{topo} - time_avg| = |{lam2:.4f} - {time_avg_mean:.4f}| = {dist:.4f}")

    print(f"\n  Closest match: {closest_topo} (distance = {closest_dist:.4f})")
    print()

    # --- Normalized comparison (lambda2 per unit edge weight) ---
    print("-" * 70)
    print("NORMALIZED ANALYSIS (lambda2 / total edge weight)")
    print("-" * 70)

    # For random time-averaged: total weight = n_islands * n_migration_events (each event adds n edges)
    # But some edges may overlap, so actual sum of weights varies
    avg_total_weight = np.mean([m.sum() / 2 for m in all_time_avg_matrices])  # /2 because symmetric
    print(f"  Random time-avg total edge weight: {avg_total_weight:.1f} (expected: {n_islands * n_migration_events})")
    print(f"  Random time-avg lambda2 / total weight: {time_avg_mean / avg_total_weight:.6f}")

    for topo in fixed_topologies:
        adj = fixed_topology_adjacency(topo, n_islands)
        total_weight = adj.sum() / 2
        if total_weight > 0:
            print(f"  {topo:20s} lambda2 / total weight: {fixed_lambda2s[topo] / total_weight:.6f} "
                  f"(edges: {total_weight:.0f})")

    print()

    # --- Edge coverage analysis ---
    print("-" * 70)
    print("EDGE COVERAGE ANALYSIS")
    print("-" * 70)

    max_edges = n_islands * (n_islands - 1) // 2
    edges_covered = []
    for m in all_time_avg_matrices:
        nonzero = np.count_nonzero(np.triu(m, k=1))
        edges_covered.append(nonzero)

    mean_coverage = np.mean(edges_covered)
    print(f"  Maximum possible edges (K_{n_islands}): {max_edges}")
    print(f"  Mean edges covered after {n_migration_events} events: {mean_coverage:.1f} / {max_edges}")
    print(f"  Coverage fraction: {mean_coverage / max_edges:.3f}")
    print()

    # --- Hypothesis test ---
    print("=" * 70)
    print("HYPOTHESIS TEST: Claudius's Prediction")
    print("=" * 70)
    print()
    print(f"  H0: time-averaged lambda2 = mean snapshot lambda2")
    print(f"  H1: time-averaged lambda2 > mean snapshot lambda2 (Claudius)")
    print()
    print(f"  Snapshot mean lambda2:      {snapshot_mean:.4f}")
    print(f"  Time-averaged mean lambda2: {time_avg_mean:.4f}")
    print(f"  Ratio:                      {ratio:.2f}x")
    print()

    if time_avg_mean > snapshot_mean:
        print("  RESULT: Claudius's hypothesis is SUPPORTED.")
        print(f"  Time-averaged lambda2 is {ratio:.1f}x larger than snapshot lambda2.")
    else:
        print("  RESULT: Claudius's hypothesis is NOT supported.")
        print(f"  Time-averaged lambda2 is NOT larger than snapshot lambda2.")

    print()

    # Where does time-averaged random fall in the fixed topology ordering?
    print("  Effective coupling ordering:")
    all_topos = list(scaled_fixed_lambda2s.items()) + [("random_time_avg", time_avg_mean)]
    all_topos.sort(key=lambda x: x[1])
    for topo, lam2 in all_topos:
        marker = " <-- RANDOM TIME-AVERAGED" if topo == "random_time_avg" else ""
        print(f"    {topo:20s}: {lam2:.4f}{marker}")

    print()

    # Diversity ordering from experiments (for reference)
    print("  For reference, observed DIVERSITY ordering (most to least diverse):")
    print("    none > ring > star > random > fully_connected")
    print()
    print("  Expected COUPLING ordering (weakest to strongest):")
    print("    none < ring < star < ??? < fully_connected")
    print()
    print(f"  Random's TIME-AVERAGED lambda2 ({time_avg_mean:.4f}) sits between "
          f"which fixed topologies?")

    # Find where random sits
    for i in range(len(all_topos) - 1):
        if all_topos[i][0] == "random_time_avg" or all_topos[i+1][0] == "random_time_avg":
            if all_topos[i][0] == "random_time_avg":
                print(f"    Between nothing and {all_topos[i+1][0]} (lambda2={all_topos[i+1][1]:.4f})")
            else:
                print(f"    Between {all_topos[i][0]} (lambda2={all_topos[i][1]:.4f}) "
                      f"and {all_topos[i+1][0] if i+1 < len(all_topos) else 'top'}")

    print()

    return {
        'snapshot_mean': snapshot_mean,
        'snapshot_std': snapshot_std,
        'time_avg_mean': time_avg_mean,
        'time_avg_std': time_avg_std,
        'ratio': ratio,
        'fixed_lambda2s': fixed_lambda2s,
        'scaled_fixed_lambda2s': scaled_fixed_lambda2s,
        'closest_topo': closest_topo,
        'edge_coverage': mean_coverage / max_edges,
    }


if __name__ == "__main__":
    results = run_analysis()

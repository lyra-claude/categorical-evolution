#!/usr/bin/env python3
"""
Snapshot vs Time-Averaged lambda2 for ALL topologies.

Tests whether the ~110x gap between snapshot lambda2 and time-averaged lambda2
is specific to random topology or a general property of all time-varying topologies.

Hypothesis: Fixed topologies (ring, star, FC) should show ratio = 1.0 because
the cumulative adjacency matrix is just n_events * A (a scalar multiple), so
lambda2 scales linearly and the ratio is exactly n_events for both measures
(but when we compare snapshot lambda2 mean vs time-averaged lambda2, the time-
averaged matrix has n_events * the edges, so its lambda2 = n_events * snapshot lambda2,
giving ratio = n_events).

Wait — let's be precise:
- For fixed topology with adjacency A:
  - Each snapshot has lambda2(A)
  - Snapshot mean = lambda2(A)
  - Cumulative after T events = T * A
  - Time-averaged lambda2 = lambda2(T * A) = T * lambda2(A)
  - Ratio = T * lambda2(A) / lambda2(A) = T = 20

- For random topology:
  - Each snapshot Ai is different
  - Snapshot mean = mean(lambda2(Ai))
  - Cumulative = sum(Ai)
  - Time-averaged lambda2 = lambda2(sum(Ai))
  - The question is: lambda2(sum(Ai)) vs T * mean(lambda2(Ai))

So for fair comparison, we should normalize: compare lambda2(cumulative/T) vs mean(lambda2(Ai)).
That way fixed topologies give ratio = 1.0, and we see the pure "averaging inflation" effect.

Usage:
    python snapshot_vs_timeavg_all_topologies.py
"""

import numpy as np
import networkx as nx


def random_adjacency_snapshot(rng: np.random.Generator, n: int) -> np.ndarray:
    """Generate one random migration adjacency matrix.
    Matches random_migrate in onemax_stats.py:
    - n random edges, each connecting two distinct islands
    """
    adj = np.zeros((n, n), dtype=float)
    for _ in range(n):
        i = rng.integers(0, n)
        j = rng.integers(0, n - 1)
        if j >= i:
            j += 1
        adj[i, j] += 1
        adj[j, i] += 1
    return adj


def compute_lambda2(adj: np.ndarray) -> float:
    """Compute algebraic connectivity (second-smallest Laplacian eigenvalue)."""
    G = nx.from_numpy_array(adj)
    laplacian = nx.laplacian_matrix(G).toarray().astype(float)
    eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))
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
        for spoke in range(1, n):
            adj[0, spoke] = 1
            adj[spoke, 0] = 1
    elif name == "fully_connected":
        for i in range(n):
            for j in range(i + 1, n):
                adj[i, j] = 1
                adj[j, i] = 1
    elif name == "none":
        pass
    else:
        raise ValueError(f"Unknown topology: {name}")
    return adj


def analyze_topology(name: str, n_islands: int = 5, n_events: int = 20,
                     n_runs: int = 100, rng: np.random.Generator = None):
    """Compute snapshot mean lambda2, time-averaged lambda2, and ratio for a topology."""

    if name == "none":
        # No migration, lambda2 = 0 everywhere
        return {
            'snapshot_mean': 0.0,
            'snapshot_std': 0.0,
            'time_avg_mean': 0.0,
            'time_avg_std': 0.0,
            'ratio': float('nan'),  # 0/0
        }

    if name == "random":
        # Time-varying: each migration event uses a different random graph
        all_snapshot_lambda2s = []
        all_time_avg_lambda2s = []

        for run in range(n_runs):
            run_rng = np.random.default_rng(rng.integers(0, 2**32))
            cumulative = np.zeros((n_islands, n_islands), dtype=float)
            snapshots_this_run = []

            for event in range(n_events):
                snapshot = random_adjacency_snapshot(run_rng, n_islands)
                snapshots_this_run.append(compute_lambda2(snapshot))
                cumulative += snapshot

            all_snapshot_lambda2s.extend(snapshots_this_run)
            all_time_avg_lambda2s.append(compute_lambda2(cumulative))

        snapshot_mean = np.mean(all_snapshot_lambda2s)
        snapshot_std = np.std(all_snapshot_lambda2s)
        time_avg_mean = np.mean(all_time_avg_lambda2s)
        time_avg_std = np.std(all_time_avg_lambda2s)
        ratio = time_avg_mean / snapshot_mean if snapshot_mean > 0 else float('nan')

        return {
            'snapshot_mean': snapshot_mean,
            'snapshot_std': snapshot_std,
            'time_avg_mean': time_avg_mean,
            'time_avg_std': time_avg_std,
            'ratio': ratio,
        }

    else:
        # Fixed topology: every migration uses the same adjacency matrix
        adj = fixed_topology_adjacency(name, n_islands)
        snapshot_lambda2 = compute_lambda2(adj)

        # Cumulative after n_events = n_events * adj
        cumulative = adj * n_events
        time_avg_lambda2 = compute_lambda2(cumulative)

        # For fixed topologies, snapshot is always the same, so std = 0
        # and ratio = n_events exactly (since lambda2 scales linearly)
        ratio = time_avg_lambda2 / snapshot_lambda2 if snapshot_lambda2 > 0 else float('nan')

        return {
            'snapshot_mean': snapshot_lambda2,
            'snapshot_std': 0.0,
            'time_avg_mean': time_avg_lambda2,
            'time_avg_std': 0.0,
            'ratio': ratio,
        }


def main():
    n_islands = 5
    n_events = 20
    n_runs = 100
    seed = 42

    rng = np.random.default_rng(seed)

    topologies = ["none", "ring", "star", "random", "fully_connected"]

    print("=" * 80)
    print("SNAPSHOT vs TIME-AVERAGED LAMBDA2 — ALL TOPOLOGIES")
    print("=" * 80)
    print(f"\nParameters: {n_islands} islands, {n_events} migration events/run, "
          f"{n_runs} runs (random only)")
    print()

    results = {}
    for topo in topologies:
        results[topo] = analyze_topology(topo, n_islands, n_events, n_runs, rng)

    # --- Raw table ---
    print("-" * 80)
    print(f"{'Topology':<20} {'Snapshot λ₂':>14} {'Time-Avg λ₂':>14} {'Ratio':>10} {'Type':>12}")
    print("-" * 80)
    for topo in topologies:
        r = results[topo]
        snap = f"{r['snapshot_mean']:.4f}" if r['snapshot_mean'] > 0 else "0.0000"
        tavg = f"{r['time_avg_mean']:.4f}" if r['time_avg_mean'] > 0 else "0.0000"
        if np.isnan(r['ratio']):
            ratio_str = "N/A"
        else:
            ratio_str = f"{r['ratio']:.2f}x"

        topo_type = "none" if topo == "none" else ("time-varying" if topo == "random" else "fixed")
        print(f"{topo:<20} {snap:>14} {tavg:>14} {ratio_str:>10} {topo_type:>12}")

    print("-" * 80)
    print()

    # --- Normalized table (per-event lambda2) ---
    # To compare apples-to-apples: divide time-averaged lambda2 by n_events
    # For fixed topologies this gives back the snapshot lambda2
    # For random, this shows the "effective per-event coupling"
    print("-" * 80)
    print("NORMALIZED: Time-averaged λ₂ / n_events (effective per-event coupling)")
    print("-" * 80)
    print(f"{'Topology':<20} {'Snapshot λ₂':>14} {'Eff. per-event':>14} {'Norm. ratio':>12}")
    print("-" * 80)
    for topo in topologies:
        r = results[topo]
        snap_val = r['snapshot_mean']
        normalized_tavg = r['time_avg_mean'] / n_events if r['time_avg_mean'] > 0 else 0.0

        snap = f"{snap_val:.4f}"
        norm = f"{normalized_tavg:.4f}"

        if snap_val > 0:
            norm_ratio = f"{normalized_tavg / snap_val:.2f}x"
        else:
            norm_ratio = "N/A"

        print(f"{topo:<20} {snap:>14} {norm:>14} {norm_ratio:>12}")

    print("-" * 80)
    print()

    # --- Interpretation ---
    random_r = results['random']
    norm_ratio_random = (random_r['time_avg_mean'] / n_events) / random_r['snapshot_mean']

    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    print("For FIXED topologies (ring, star, FC):")
    print(f"  Raw ratio = {n_events}.00x (exactly n_events, as expected)")
    print(f"  Normalized ratio = 1.00x (no inflation — same graph every time)")
    print()
    print("For NONE topology:")
    print("  Both values are 0 (no migration, no coupling)")
    print()
    print("For RANDOM topology:")
    print(f"  Raw ratio = {random_r['ratio']:.2f}x")
    print(f"  Normalized ratio = {norm_ratio_random:.2f}x")
    print(f"  The normalized ratio of {norm_ratio_random:.2f}x means that after")
    print(f"  accumulating {n_events} random snapshots, the effective per-event")
    print(f"  coupling is {norm_ratio_random:.1f}x stronger than any single snapshot.")
    print()
    print("CONCLUSION: The snapshot-vs-time-averaged gap is UNIQUE to time-varying")
    print("topologies. Fixed topologies show ratio = 1.0 (normalized). Only random")
    print("topology — where the graph changes every event — shows inflation, because")
    print("different random edges accumulate into a denser effective graph.")


if __name__ == "__main__":
    main()

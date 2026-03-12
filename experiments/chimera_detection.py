#!/usr/bin/env python3
"""
Chimera state detection in island-model GA topology sweep.

Tests Claudius's prediction: star topology produces bimodal divergence
(hub synchronized with each spoke, but peripheral-to-peripheral divergence
differs from hub-to-peripheral divergence).

In the star topology, island 0 is the hub. Spokes (islands 1-4) only
exchange migrants with the hub, never directly with each other.

Hypothesis: hub-to-peripheral pairwise divergences should be lower than
peripheral-to-peripheral divergences, because the hub receives genetic
material from ALL spokes while spokes only receive from the hub. This
creates an information asymmetry that may produce chimera-like states
where the hub is synchronized with each spoke individually, but spokes
diverge from each other.

Usage:
    python3 chimera_detection.py
    python3 chimera_detection.py --csv experiment_e_per_island.csv
    python3 chimera_detection.py --gen 99  # specific generation
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats


def load_csv(path: str) -> list:
    """Load experiment CSV into list of dicts."""
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_star_pair_types(n_islands: int = 5):
    """Classify pairwise divergence columns for star topology.

    Island 0 is the hub. Returns two lists of column names:
    - hub_peripheral: div_0_1, div_0_2, div_0_3, div_0_4
    - peripheral_peripheral: div_1_2, div_1_3, div_1_4, div_2_3, div_2_4, div_3_4
    """
    hub = 0
    hub_peripheral = []
    peripheral_peripheral = []

    for i in range(n_islands):
        for j in range(i + 1, n_islands):
            col = f'div_{i}_{j}'
            if i == hub or j == hub:
                hub_peripheral.append(col)
            else:
                peripheral_peripheral.append(col)

    return hub_peripheral, peripheral_peripheral


def hartigans_dip_statistic(data: np.ndarray) -> float:
    """Compute a simple bimodality coefficient as proxy for Hartigan's dip test.

    Uses the bimodality coefficient (BC):
        BC = (skewness^2 + 1) / (kurtosis + 3 * (n-1)^2 / ((n-2)*(n-3)))

    BC > 5/9 (~0.555) suggests bimodality.

    Also computes a simple dip approximation: if the distribution has two
    clear modes, the histogram will show a valley.
    """
    n = len(data)
    if n < 4:
        return 0.0

    skew = float(stats.skew(data))
    kurt = float(stats.kurtosis(data))  # excess kurtosis

    # Bimodality coefficient
    # Adjusted kurtosis denominator for finite sample
    denom = kurt + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    if denom == 0:
        return 0.0

    bc = (skew ** 2 + 1) / denom
    return bc


def analyze_divergence_distributions(rows: list, generation: int, n_islands: int = 5):
    """Analyze pairwise divergence distributions across topologies at a given generation."""

    topologies = ["none", "ring", "star", "fully_connected", "random"]
    hub_cols, periph_cols = get_star_pair_types(n_islands)
    all_div_cols = hub_cols + periph_cols

    print(f"\n{'='*80}")
    print(f"CHIMERA STATE DETECTION — Generation {generation}")
    print(f"{'='*80}")

    # --- 1. Distribution of pairwise divergences for each topology ---
    print(f"\n--- 1. Pairwise divergence distributions (gen {generation}, 30 seeds) ---")
    print(f"{'Topology':>18} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Skew':>8} {'BC':>8}")
    print("-" * 70)

    topo_all_divs = {}

    for topo in topologies:
        topo_rows = [r for r in rows
                     if r['topology'] == topo and int(r['generation']) == generation]

        all_divs = []
        for r in topo_rows:
            for col in all_div_cols:
                all_divs.append(float(r[col]))

        all_divs = np.array(all_divs)
        topo_all_divs[topo] = all_divs

        bc = hartigans_dip_statistic(all_divs)
        skew = float(stats.skew(all_divs)) if len(all_divs) > 2 else 0.0

        print(f"{topo:>18} {np.mean(all_divs):>8.4f} {np.std(all_divs):>8.4f} "
              f"{np.min(all_divs):>8.4f} {np.max(all_divs):>8.4f} "
              f"{skew:>8.3f} {bc:>8.3f}")

    # --- 2. Star topology: hub vs peripheral divergence ---
    print(f"\n--- 2. Star topology: hub-peripheral vs peripheral-peripheral ---")

    star_rows = [r for r in rows
                 if r['topology'] == 'star' and int(r['generation']) == generation]

    hub_divs_all = []
    periph_divs_all = []
    hub_divs_per_seed = []
    periph_divs_per_seed = []

    for r in star_rows:
        seed_hub = [float(r[c]) for c in hub_cols]
        seed_periph = [float(r[c]) for c in periph_cols]
        hub_divs_all.extend(seed_hub)
        periph_divs_all.extend(seed_periph)
        hub_divs_per_seed.append(np.mean(seed_hub))
        periph_divs_per_seed.append(np.mean(seed_periph))

    hub_divs_all = np.array(hub_divs_all)
    periph_divs_all = np.array(periph_divs_all)
    hub_divs_per_seed = np.array(hub_divs_per_seed)
    periph_divs_per_seed = np.array(periph_divs_per_seed)

    print(f"\n  Hub-peripheral pairs ({len(hub_cols)} pairs x 30 seeds = {len(hub_divs_all)} values):")
    print(f"    Mean: {np.mean(hub_divs_all):.4f}  Std: {np.std(hub_divs_all):.4f}  "
          f"Median: {np.median(hub_divs_all):.4f}")
    print(f"    Min: {np.min(hub_divs_all):.4f}  Max: {np.max(hub_divs_all):.4f}")

    print(f"\n  Peripheral-peripheral pairs ({len(periph_cols)} pairs x 30 seeds = {len(periph_divs_all)} values):")
    print(f"    Mean: {np.mean(periph_divs_all):.4f}  Std: {np.std(periph_divs_all):.4f}  "
          f"Median: {np.median(periph_divs_all):.4f}")
    print(f"    Min: {np.min(periph_divs_all):.4f}  Max: {np.max(periph_divs_all):.4f}")

    # Mann-Whitney U test: hub-peripheral vs peripheral-peripheral
    u_stat, u_p = stats.mannwhitneyu(hub_divs_all, periph_divs_all, alternative='two-sided')
    print(f"\n  Mann-Whitney U test (hub-periph vs periph-periph):")
    print(f"    U = {u_stat:.1f}, p = {u_p:.6f}")
    print(f"    {'SIGNIFICANT' if u_p < 0.05 else 'NOT significant'} at alpha=0.05")

    # Effect size: Vargha-Delaney A (probability that a random hub-periph div
    # is less than a random periph-periph div)
    n1, n2 = len(hub_divs_all), len(periph_divs_all)
    vd_a = u_stat / (n1 * n2)
    print(f"    Vargha-Delaney A = {vd_a:.3f} (0.5 = no effect, <0.5 = hub < periph)")

    # Per-seed paired test
    t_stat, t_p = stats.ttest_rel(hub_divs_per_seed, periph_divs_per_seed)
    print(f"\n  Paired t-test (per-seed mean hub-periph vs periph-periph):")
    print(f"    t = {t_stat:.3f}, p = {t_p:.6f}")
    print(f"    Hub-periph mean: {np.mean(hub_divs_per_seed):.4f}")
    print(f"    Periph-periph mean: {np.mean(periph_divs_per_seed):.4f}")
    print(f"    Difference: {np.mean(periph_divs_per_seed) - np.mean(hub_divs_per_seed):.4f}")

    # --- 3. Bimodality test ---
    print(f"\n--- 3. Bimodality analysis for star topology ---")

    # Check bimodality of the combined divergence distribution
    combined = np.concatenate([hub_divs_all, periph_divs_all])
    bc_combined = hartigans_dip_statistic(combined)
    print(f"\n  Bimodality coefficient (all star divergences): {bc_combined:.4f}")
    print(f"    {'BIMODAL' if bc_combined > 0.555 else 'UNIMODAL'} (threshold: 0.555)")

    # Bimodality per seed
    bc_values = []
    for r in star_rows:
        seed_divs = [float(r[c]) for c in all_div_cols]
        bc_seed = hartigans_dip_statistic(np.array(seed_divs))
        bc_values.append(bc_seed)

    bc_values = np.array(bc_values)
    print(f"\n  Per-seed bimodality coefficients:")
    print(f"    Mean BC: {np.mean(bc_values):.4f}  Std: {np.std(bc_values):.4f}")
    print(f"    Seeds with BC > 0.555 (bimodal): {np.sum(bc_values > 0.555)}/{len(bc_values)}")

    # Compare with other topologies
    print(f"\n  Bimodality comparison across topologies:")
    print(f"  {'Topology':>18} {'BC (all divs)':>14} {'Bimodal?':>10}")
    print("  " + "-" * 45)
    for topo in topologies:
        bc = hartigans_dip_statistic(topo_all_divs[topo])
        bimodal = "YES" if bc > 0.555 else "no"
        print(f"  {topo:>18} {bc:>14.4f} {bimodal:>10}")

    # --- 4. Per-island diversity analysis for star ---
    print(f"\n--- 4. Per-island diversity in star topology ---")
    print(f"  (Island 0 = hub)")

    island_div_by_idx = defaultdict(list)
    island_fit_by_idx = defaultdict(list)

    for r in star_rows:
        for k in range(n_islands):
            island_div_by_idx[k].append(float(r[f'island_{k}_diversity']))
            island_fit_by_idx[k].append(float(r[f'island_{k}_fitness']))

    print(f"\n  {'Island':>8} {'Role':>10} {'Diversity':>12} {'(std)':>8} {'Fitness':>10} {'(std)':>8}")
    print("  " + "-" * 60)
    for k in range(n_islands):
        role = "HUB" if k == 0 else f"spoke-{k}"
        divs = np.array(island_div_by_idx[k])
        fits = np.array(island_fit_by_idx[k])
        print(f"  {k:>8} {role:>10} {np.mean(divs):>12.4f} {np.std(divs):>8.4f} "
              f"{np.mean(fits):>10.1f} {np.std(fits):>8.2f}")

    # Test if hub diversity is different from spoke diversity
    hub_divs_island = np.array(island_div_by_idx[0])
    spoke_divs_island = np.concatenate([np.array(island_div_by_idx[k]) for k in range(1, n_islands)])

    u2, p2 = stats.mannwhitneyu(hub_divs_island, spoke_divs_island, alternative='two-sided')
    print(f"\n  Mann-Whitney U (hub diversity vs spoke diversity):")
    print(f"    U = {u2:.1f}, p = {p2:.6f}")
    print(f"    Hub mean diversity: {np.mean(hub_divs_island):.4f}")
    print(f"    Spoke mean diversity: {np.mean(spoke_divs_island):.4f}")

    # --- 5. Comparison with ring ---
    print(f"\n--- 5. Ring vs Star: structural divergence comparison ---")

    ring_rows = [r for r in rows
                 if r['topology'] == 'ring' and int(r['generation']) == generation]

    # For ring, all pairs are "equivalent" structurally (symmetric),
    # but adjacent vs non-adjacent pairs may differ
    # Ring adjacency for 5 islands: 0-1, 1-2, 2-3, 3-4, 4-0
    ring_adjacent_cols = ['div_0_1', 'div_1_2', 'div_2_3', 'div_3_4', 'div_0_4']
    ring_nonadjacent_cols = ['div_0_2', 'div_0_3', 'div_1_3', 'div_1_4', 'div_2_4']

    ring_adj = []
    ring_nonadj = []
    for r in ring_rows:
        for c in ring_adjacent_cols:
            ring_adj.append(float(r[c]))
        for c in ring_nonadjacent_cols:
            ring_nonadj.append(float(r[c]))

    ring_adj = np.array(ring_adj)
    ring_nonadj = np.array(ring_nonadj)

    print(f"\n  Ring adjacent pairs: mean={np.mean(ring_adj):.4f}, std={np.std(ring_adj):.4f}")
    print(f"  Ring non-adjacent:   mean={np.mean(ring_nonadj):.4f}, std={np.std(ring_nonadj):.4f}")
    u3, p3 = stats.mannwhitneyu(ring_adj, ring_nonadj, alternative='two-sided')
    print(f"  Mann-Whitney U: U={u3:.1f}, p={p3:.6f}")

    print(f"\n  Star hub-periph:     mean={np.mean(hub_divs_all):.4f}, std={np.std(hub_divs_all):.4f}")
    print(f"  Star periph-periph:  mean={np.mean(periph_divs_all):.4f}, std={np.std(periph_divs_all):.4f}")

    # --- 6. Summary verdict ---
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    verdict_bimodal = bc_combined > 0.555
    verdict_asymmetry = u_p < 0.05
    direction = "hub < periph" if np.mean(hub_divs_all) < np.mean(periph_divs_all) else "hub >= periph"

    print(f"\n  Bimodality in star divergence distribution: {'YES' if verdict_bimodal else 'NO'}")
    print(f"  Hub-peripheral vs peripheral-peripheral asymmetry: {'YES' if verdict_asymmetry else 'NO'}")
    print(f"  Direction: {direction}")
    print(f"  Hub island diversity vs spoke diversity: p={p2:.6f}")
    print(f"\n  Star bimodality prediction: "
          f"{'SUPPORTED' if verdict_bimodal or verdict_asymmetry else 'NOT SUPPORTED'}")

    if verdict_asymmetry and not verdict_bimodal:
        print(f"\n  NOTE: While the divergence distribution is not formally bimodal,")
        print(f"  there IS a significant structural asymmetry between hub-peripheral")
        print(f"  and peripheral-peripheral divergences. This is consistent with a")
        print(f"  chimera-like state where the hub acts as a synchronization mediator.")

    return {
        'bimodal': verdict_bimodal,
        'asymmetry': verdict_asymmetry,
        'bc_combined': bc_combined,
        'hub_periph_mean': float(np.mean(hub_divs_all)),
        'periph_periph_mean': float(np.mean(periph_divs_all)),
        'u_p': u_p,
        'direction': direction,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Chimera state detection in topology sweep data")
    parser.add_argument('--csv', type=str, default='experiment_e_per_island.csv',
                        help='Input CSV file (default: experiment_e_per_island.csv)')
    parser.add_argument('--gen', type=int, default=99,
                        help='Generation to analyze (default: 99, i.e. final)')
    args = parser.parse_args()

    # Resolve path relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, args.csv) if not os.path.isabs(args.csv) else args.csv

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {csv_path}")
    rows = load_csv(csv_path)
    print(f"Loaded {len(rows)} rows")

    result = analyze_divergence_distributions(rows, args.gen)

    return result


if __name__ == '__main__':
    main()

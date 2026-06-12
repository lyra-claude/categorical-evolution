#!/usr/bin/env python3
"""
Compute Kendall's W concordance coefficient for the 4 connected topologies
(ring, star, random, fully_connected) across 6 domains.

Excludes "none" (no migration baseline) to test whether the diversity
ordering is preserved among actual migration strategies.

Output:
  - Per-domain mean diversity and rankings
  - Kendall's W, chi-square, p-value
  - Whether ring > star > random > FC holds universally
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- #
#  Configuration                                                               #
# --------------------------------------------------------------------------- #

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_DIR = os.path.join(SCRIPT_DIR, '..', 'experiments')

CONNECTED_TOPOS = ['ring', 'star', 'random', 'fully_connected']

DOMAIN_CONFIGS = {
    'OneMax': {
        'file': 'experiment_e_raw.csv',
        'max_gen': 99,
    },
    'Maze': {
        'file': 'experiment_e_maze.csv',
        'max_gen': 99,
    },
    'graph_coloring': {
        'file': 'experiment_e_graph_coloring.csv',
        'max_gen': 99,
    },
    'knapsack': {
        'file': 'experiment_e_knapsack.csv',
        'max_gen': 99,
    },
    'No Thanks!': {
        'file': 'experiment_e_nothanks.csv',
        'max_gen': 99,
    },
    'checkers': {
        'file': 'experiment_e_checkers.csv',
        'max_gen': 99,
    },
}


def load_domain(config):
    """Load a domain CSV and return the DataFrame."""
    path = os.path.join(EXPERIMENT_DIR, config['file'])
    df = pd.read_csv(path)
    return df


def get_final_gen_diversity(df, max_gen):
    """Get mean diversity at final generation for each connected topology."""
    final = df[df['generation'] == max_gen]
    result = {}
    for topo in CONNECTED_TOPOS:
        vals = final[final['topology'] == topo]['hamming_diversity']
        if len(vals) == 0:
            print(f"  WARNING: No data for topology '{topo}' at generation {max_gen}")
            result[topo] = np.nan
        else:
            result[topo] = vals.mean()
    return result


def rank_topologies(diversity_dict):
    """Rank topologies by diversity (highest = rank 1)."""
    sorted_topos = sorted(diversity_dict.items(), key=lambda x: x[1], reverse=True)
    ranks = {}
    for rank_idx, (topo, _) in enumerate(sorted_topos, 1):
        ranks[topo] = rank_idx
    return ranks


def kendalls_w(rankings_matrix):
    """
    Compute Kendall's W (coefficient of concordance).

    rankings_matrix: k x n array where k = number of judges (domains)
                     and n = number of items (topologies)

    W = 12 * S / (k^2 * (n^3 - n))
    where S = sum of squared deviations of column sums from the mean column sum

    Also returns chi-square = k * (n - 1) * W and p-value.
    """
    k, n = rankings_matrix.shape  # k judges, n items

    # Column sums (sum of ranks for each topology across all domains)
    col_sums = rankings_matrix.sum(axis=0)
    mean_col_sum = col_sums.mean()

    # S = sum of squared deviations
    S = np.sum((col_sums - mean_col_sum) ** 2)

    # Kendall's W
    W = 12 * S / (k**2 * (n**3 - n))

    # Chi-square statistic
    chi2 = k * (n - 1) * W

    # Degrees of freedom
    df = n - 1

    # p-value (chi-square distribution)
    p_value = 1 - stats.chi2.cdf(chi2, df)

    return W, chi2, df, p_value, col_sums


def main():
    print("=" * 70)
    print("Kendall's W for 4 Connected Topologies (excluding 'none')")
    print("=" * 70)
    print()

    # Collect data
    all_diversities = {}
    all_rankings = {}

    for domain_name, config in DOMAIN_CONFIGS.items():
        df = load_domain(config)
        max_gen = config['max_gen']

        # Check actual max generation in data
        actual_max = df['generation'].max()
        if actual_max < max_gen:
            print(f"  NOTE: {domain_name} max generation in data is {actual_max}, using that instead of {max_gen}")
            max_gen = actual_max

        diversity = get_final_gen_diversity(df, max_gen)
        ranks = rank_topologies(diversity)

        all_diversities[domain_name] = diversity
        all_rankings[domain_name] = ranks

    # Print per-domain results
    print("Per-domain mean diversity at final generation:")
    print("-" * 70)
    header = f"{'Domain':<20}" + "".join(f"{t:<18}" for t in CONNECTED_TOPOS)
    print(header)
    print("-" * 70)

    for domain_name in DOMAIN_CONFIGS:
        div = all_diversities[domain_name]
        row = f"{domain_name:<20}"
        for topo in CONNECTED_TOPOS:
            row += f"{div[topo]:<18.6f}"
        print(row)

    print()
    print("Per-domain rankings (1 = highest diversity):")
    print("-" * 70)
    header = f"{'Domain':<20}" + "".join(f"{t:<18}" for t in CONNECTED_TOPOS)
    print(header)
    print("-" * 70)

    for domain_name in DOMAIN_CONFIGS:
        ranks = all_rankings[domain_name]
        row = f"{domain_name:<20}"
        for topo in CONNECTED_TOPOS:
            row += f"{ranks[topo]:<18d}"
        print(row)

    # Build rankings matrix: k domains x n topologies
    domain_names = list(DOMAIN_CONFIGS.keys())
    rankings_matrix = np.array([
        [all_rankings[d][t] for t in CONNECTED_TOPOS]
        for d in domain_names
    ])

    print()
    print("Rankings matrix (rows=domains, cols=topologies):")
    print(f"  Columns: {CONNECTED_TOPOS}")
    print(rankings_matrix)

    # Compute Kendall's W
    W, chi2, df, p_value, col_sums = kendalls_w(rankings_matrix)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Number of judges (domains):  k = {len(domain_names)}")
    print(f"  Number of items (topologies): n = {len(CONNECTED_TOPOS)}")
    print(f"  Column sums of ranks:        {col_sums}")
    print(f"    (ring, star, random, FC)")
    print()
    print(f"  Kendall's W = {W:.6f}")
    print(f"  Chi-square  = {chi2:.4f}")
    print(f"  df          = {df}")
    print(f"  p-value     = {p_value:.6f}")

    if p_value < 0.05:
        print(f"  => Significant at alpha=0.05 (p={p_value:.6f})")
    else:
        print(f"  => NOT significant at alpha=0.05 (p={p_value:.6f})")

    if p_value < 0.01:
        print(f"  => Significant at alpha=0.01 (p={p_value:.6f})")

    if p_value < 0.001:
        print(f"  => Significant at alpha=0.001 (p={p_value:.6f})")

    # Check if ordering ring > star > random > FC is preserved
    print()
    print("=" * 70)
    print("ORDERING CHECK: ring > star > random > FC")
    print("=" * 70)

    expected_order = ['ring', 'star', 'random', 'fully_connected']
    all_match = True

    for domain_name in domain_names:
        div = all_diversities[domain_name]
        ordered = [div[t] for t in expected_order]
        is_decreasing = all(ordered[i] > ordered[i+1] for i in range(len(ordered)-1))
        status = "YES" if is_decreasing else "NO"
        if not is_decreasing:
            all_match = False

        actual_order = sorted(CONNECTED_TOPOS, key=lambda t: div[t], reverse=True)
        print(f"  {domain_name:<20}: {status}  (actual order: {' > '.join(actual_order)})")

    print()
    if all_match:
        print("  => The ordering ring > star > random > FC is PERFECTLY preserved across all 6 domains.")
    else:
        print("  => The ordering ring > star > random > FC is NOT perfectly preserved in all domains.")

    # Also try scipy's Friedman test as a cross-check
    print()
    print("=" * 70)
    print("CROSS-CHECK: Friedman test on raw diversity values")
    print("=" * 70)

    # Collect per-seed final-gen diversity for each topology in each domain
    # Use mean across seeds per domain as the observation
    groups = []
    for topo in CONNECTED_TOPOS:
        topo_means = []
        for domain_name, config in DOMAIN_CONFIGS.items():
            topo_means.append(all_diversities[domain_name][topo])
        groups.append(topo_means)

    # Friedman test expects observations in columns
    try:
        friedman_stat, friedman_p = stats.friedmanchisquare(*groups)
        print(f"  Friedman chi-square = {friedman_stat:.4f}")
        print(f"  Friedman p-value    = {friedman_p:.6f}")
    except Exception as e:
        print(f"  Friedman test failed: {e}")


if __name__ == '__main__':
    main()

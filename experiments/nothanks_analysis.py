#!/usr/bin/env python3
"""
No Thanks! domain analysis — full pipeline.

No Thanks! is a co-evolutionary card game domain where fitness is relative
(tournament-based), unique among our domains. This makes it the strongest test
of domain independence: if the canonical topology ordering holds even when
fitness isn't defined by a fixed landscape, topology truly determines dynamics.

Analyses:
  1. Mean diversity and divergence by topology (with std, SE)
  2. Topology ordering: does it match canonical none > ring > star > random > FC?
  3. Kruskal-Wallis tests for topology effect
  4. Pairwise Mann-Whitney with Bonferroni correction
  5. Domain independence: Spearman rank correlation with other domains
  6. Kuramoto order parameter analysis (r = 1 - divergence)
  7. Per-island analysis (unique: we have island-level data)
  8. Effect sizes (Cohen's d for adjacent topology pairs)
  9. Coupling onset detection

Data: experiment_e_nothanks.csv
  15,000 rows: 5 topologies x 30 seeds x 100 generations
  26 columns including per-island diversity/fitness and pairwise divergences
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import uniform_filter1d
from collections import defaultdict
from itertools import combinations

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'experiment_e_nothanks.csv')

TOPO_ORDER = ['none', 'ring', 'star', 'random', 'fully_connected']
COUPLED_TOPOS = ['ring', 'star', 'random', 'fully_connected']
TOPO_LABELS = {
    'none': 'None (isolated)', 'ring': 'Ring', 'star': 'Star',
    'random': 'Random', 'fully_connected': 'Fully connected',
}

# Other domain data files for cross-domain comparison
OTHER_DOMAINS = {
    'OneMax': os.path.join(SCRIPT_DIR, 'experiment_e_raw.csv'),
    'Maze': os.path.join(SCRIPT_DIR, 'experiment_e_maze.csv'),
    'Graph Coloring': os.path.join(SCRIPT_DIR, 'experiment_e_graph_coloring.csv'),
    'Knapsack': os.path.join(SCRIPT_DIR, 'experiment_e_knapsack.csv'),
}

MIGRATION_START = 5


def load_data():
    """Load No Thanks! CSV."""
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded No Thanks! data: {len(df)} rows")
    print(f"  {df['topology'].nunique()} topologies, {df['seed'].nunique()} seeds per topo, "
          f"gens {df['generation'].min()}-{df['generation'].max()}")
    return df


def load_other_domain(name, path):
    """Load another domain's CSV for comparison."""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ============================================================================
# 1. Basic statistics: mean diversity & divergence by topology
# ============================================================================

def analyze_basic_stats(df):
    """Compute mean diversity and divergence at final generation."""
    max_gen = df['generation'].max()
    final = df[df['generation'] == max_gen]

    print("\n" + "=" * 90)
    print("  1. BASIC STATISTICS: Final-Generation Metrics by Topology")
    print("=" * 90)

    results = {}
    print(f"\n  {'Topology':<20} {'Diversity':>12} {'(std)':>10} {'(SE)':>10} "
          f"{'Divergence':>12} {'(std)':>10} {'(SE)':>10} {'N':>5}")
    print("  " + "-" * 95)

    for topo in TOPO_ORDER:
        sub = final[final['topology'] == topo]
        div_vals = sub['hamming_diversity']
        pop_div_vals = sub['population_divergence']
        n = len(sub)

        results[topo] = {
            'div_mean': div_vals.mean(),
            'div_std': div_vals.std(ddof=1),
            'div_se': div_vals.std(ddof=1) / np.sqrt(n),
            'pop_div_mean': pop_div_vals.mean(),
            'pop_div_std': pop_div_vals.std(ddof=1),
            'pop_div_se': pop_div_vals.std(ddof=1) / np.sqrt(n),
            'n': n,
            'div_values': div_vals.values,
            'pop_div_values': pop_div_vals.values,
        }

        r = results[topo]
        print(f"  {TOPO_LABELS[topo]:<20} {r['div_mean']:>12.6f} {r['div_std']:>10.6f} "
              f"{r['div_se']:>10.6f} {r['pop_div_mean']:>12.6f} {r['pop_div_std']:>10.6f} "
              f"{r['pop_div_se']:>10.6f} {r['n']:>5}")

    return results


# ============================================================================
# 2. Topology ordering
# ============================================================================

def analyze_ordering(basic_stats):
    """Check if ordering matches canonical: none > ring > star > random > FC."""
    print("\n" + "=" * 90)
    print("  2. TOPOLOGY ORDERING")
    print("=" * 90)

    # Sort by diversity (descending)
    div_order = sorted(TOPO_ORDER, key=lambda t: basic_stats[t]['div_mean'], reverse=True)
    canonical = ['none', 'ring', 'star', 'random', 'fully_connected']

    print(f"\n  Diversity ordering (highest -> lowest):")
    print(f"    {' > '.join(TOPO_LABELS[t] for t in div_order)}")
    print(f"\n  Canonical ordering:")
    print(f"    {' > '.join(TOPO_LABELS[t] for t in canonical)}")
    print(f"\n  Match: {div_order == canonical}")

    # Also show divergence ordering (should be inverse)
    pop_div_order = sorted(TOPO_ORDER, key=lambda t: basic_stats[t]['pop_div_mean'])
    print(f"\n  Divergence ordering (lowest -> highest = most coupled -> least):")
    print(f"    {' < '.join(TOPO_LABELS[t] for t in pop_div_order)}")

    # Phase transition: none -> ring drop
    none_div = basic_stats['none']['div_mean']
    ring_div = basic_stats['ring']['div_mean']
    drop_pct = (none_div - ring_div) / none_div * 100
    print(f"\n  Phase transition (none -> ring): {drop_pct:.1f}% diversity drop")
    print(f"    none: {none_div:.6f}")
    print(f"    ring: {ring_div:.6f}")

    # Subsequent drops
    for i in range(len(TOPO_ORDER) - 1):
        t1, t2 = TOPO_ORDER[i], TOPO_ORDER[i + 1]
        d1, d2 = basic_stats[t1]['div_mean'], basic_stats[t2]['div_mean']
        drop = (d1 - d2) / d1 * 100
        print(f"    {t1} -> {t2}: {drop:.1f}% drop")

    return div_order


# ============================================================================
# 3. Kruskal-Wallis tests
# ============================================================================

def analyze_kruskal_wallis(basic_stats):
    """Kruskal-Wallis tests for topology effect on diversity and divergence."""
    print("\n" + "=" * 90)
    print("  3. KRUSKAL-WALLIS TESTS")
    print("=" * 90)

    # Diversity
    div_groups = [basic_stats[t]['div_values'] for t in TOPO_ORDER]
    h_div, p_div = stats.kruskal(*div_groups)
    print(f"\n  Diversity:   H = {h_div:.4f}, p = {p_div:.2e}")
    print(f"    {'***' if p_div < 0.001 else '**' if p_div < 0.01 else '*' if p_div < 0.05 else 'n.s.'}")

    # Divergence
    pop_div_groups = [basic_stats[t]['pop_div_values'] for t in TOPO_ORDER]
    h_pop, p_pop = stats.kruskal(*pop_div_groups)
    print(f"  Divergence:  H = {h_pop:.4f}, p = {p_pop:.2e}")
    print(f"    {'***' if p_pop < 0.001 else '**' if p_pop < 0.01 else '*' if p_pop < 0.05 else 'n.s.'}")

    return {'div_H': h_div, 'div_p': p_div, 'pop_div_H': h_pop, 'pop_div_p': p_pop}


# ============================================================================
# 4. Pairwise Mann-Whitney with Bonferroni correction
# ============================================================================

def analyze_pairwise(basic_stats):
    """Pairwise Mann-Whitney U tests with Bonferroni correction."""
    print("\n" + "=" * 90)
    print("  4. PAIRWISE COMPARISONS (Mann-Whitney U, Bonferroni-corrected)")
    print("=" * 90)

    pairs = list(combinations(TOPO_ORDER, 2))
    n_comparisons = len(pairs)
    alpha = 0.05

    print(f"\n  {n_comparisons} pairwise comparisons, Bonferroni threshold = {alpha/n_comparisons:.4f}")

    # Diversity
    print(f"\n  --- Diversity ---")
    print(f"  {'Pair':<40} {'U':>10} {'p (raw)':>12} {'p (Bonf)':>12} {'Cohen d':>10} {'Sig':>6}")
    print("  " + "-" * 95)

    div_results = []
    for t1, t2 in pairs:
        v1 = basic_stats[t1]['div_values']
        v2 = basic_stats[t2]['div_values']
        u_stat, p_raw = stats.mannwhitneyu(v1, v2, alternative='two-sided')
        p_bonf = min(p_raw * n_comparisons, 1.0)

        # Cohen's d
        pooled_std = np.sqrt((np.var(v1, ddof=1) + np.var(v2, ddof=1)) / 2)
        d = (np.mean(v1) - np.mean(v2)) / pooled_std if pooled_std > 0 else 0

        sig = '***' if p_bonf < 0.001 else '**' if p_bonf < 0.01 else '*' if p_bonf < 0.05 else 'n.s.'
        pair_label = f"{TOPO_LABELS[t1]} vs {TOPO_LABELS[t2]}"
        print(f"  {pair_label:<40} {u_stat:>10.1f} {p_raw:>12.2e} {p_bonf:>12.4f} {d:>10.4f} {sig:>6}")

        div_results.append({
            't1': t1, 't2': t2, 'U': u_stat, 'p_raw': p_raw,
            'p_bonf': p_bonf, 'cohen_d': d, 'sig': sig,
        })

    # Divergence
    print(f"\n  --- Divergence ---")
    print(f"  {'Pair':<40} {'U':>10} {'p (raw)':>12} {'p (Bonf)':>12} {'Cohen d':>10} {'Sig':>6}")
    print("  " + "-" * 95)

    pop_div_results = []
    for t1, t2 in pairs:
        v1 = basic_stats[t1]['pop_div_values']
        v2 = basic_stats[t2]['pop_div_values']
        u_stat, p_raw = stats.mannwhitneyu(v1, v2, alternative='two-sided')
        p_bonf = min(p_raw * n_comparisons, 1.0)

        pooled_std = np.sqrt((np.var(v1, ddof=1) + np.var(v2, ddof=1)) / 2)
        d = (np.mean(v1) - np.mean(v2)) / pooled_std if pooled_std > 0 else 0

        sig = '***' if p_bonf < 0.001 else '**' if p_bonf < 0.01 else '*' if p_bonf < 0.05 else 'n.s.'
        pair_label = f"{TOPO_LABELS[t1]} vs {TOPO_LABELS[t2]}"
        print(f"  {pair_label:<40} {u_stat:>10.1f} {p_raw:>12.2e} {p_bonf:>12.4f} {d:>10.4f} {sig:>6}")

        pop_div_results.append({
            't1': t1, 't2': t2, 'U': u_stat, 'p_raw': p_raw,
            'p_bonf': p_bonf, 'cohen_d': d, 'sig': sig,
        })

    # Adjacent pair effect sizes
    print(f"\n  --- Adjacent Pair Effect Sizes (Diversity) ---")
    adjacent = [('none', 'ring'), ('ring', 'star'), ('star', 'random'), ('random', 'fully_connected')]
    for t1, t2 in adjacent:
        v1 = basic_stats[t1]['div_values']
        v2 = basic_stats[t2]['div_values']
        pooled_std = np.sqrt((np.var(v1, ddof=1) + np.var(v2, ddof=1)) / 2)
        d = (np.mean(v1) - np.mean(v2)) / pooled_std if pooled_std > 0 else 0
        size = 'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small' if abs(d) > 0.2 else 'negligible'
        print(f"    {TOPO_LABELS[t1]} -> {TOPO_LABELS[t2]}: d = {d:.4f} ({size})")

    return {'diversity': div_results, 'divergence': pop_div_results}


# ============================================================================
# 5. Domain independence analysis
# ============================================================================

def analyze_domain_independence(basic_stats):
    """Compare No Thanks! ordering with other domains via Spearman correlation."""
    print("\n" + "=" * 90)
    print("  5. DOMAIN INDEPENDENCE ANALYSIS")
    print("=" * 90)

    # Get No Thanks! ranking
    nt_ranking = sorted(TOPO_ORDER, key=lambda t: basic_stats[t]['div_mean'], reverse=True)
    nt_ranks = {t: i for i, t in enumerate(nt_ranking)}

    print(f"\n  No Thanks! ordering: {' > '.join(nt_ranking)}")

    domain_rankings = {'No Thanks!': nt_ranks}
    domain_orderings = {'No Thanks!': nt_ranking}

    for name, path in OTHER_DOMAINS.items():
        df = load_other_domain(name, path)
        if df is None:
            print(f"  [SKIP] {name}: file not found")
            continue

        max_gen = df['generation'].max()
        final = df[df['generation'] == max_gen]

        means = {}
        for topo in TOPO_ORDER:
            vals = final[final['topology'] == topo]['hamming_diversity']
            means[topo] = vals.mean()

        ranking = sorted(TOPO_ORDER, key=lambda t: means[t], reverse=True)
        ranks = {t: i for i, t in enumerate(ranking)}
        domain_rankings[name] = ranks
        domain_orderings[name] = ranking
        print(f"  {name} ordering: {' > '.join(ranking)}")

    # Pairwise Spearman correlations (between domains)
    print(f"\n  --- Pairwise Spearman Rank Correlations ---")
    all_domains = list(domain_rankings.keys())

    if len(all_domains) < 2:
        print("  Not enough domains for comparison.")
        return None

    rho_matrix = {}
    for d1, d2 in combinations(all_domains, 2):
        ranks1 = [domain_rankings[d1][t] for t in TOPO_ORDER]
        ranks2 = [domain_rankings[d2][t] for t in TOPO_ORDER]
        rho, p = stats.spearmanr(ranks1, ranks2)
        rho_matrix[(d1, d2)] = (rho, p)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        print(f"    {d1} vs {d2}: rho = {rho:.4f}, p = {p:.4f} {sig}")

    # No Thanks! vs all other domains specifically
    print(f"\n  --- No Thanks! vs Each Other Domain ---")
    nt_ranks_list = [nt_ranks[t] for t in TOPO_ORDER]
    for name in OTHER_DOMAINS:
        if name not in domain_rankings:
            continue
        other_ranks = [domain_rankings[name][t] for t in TOPO_ORDER]
        rho, p = stats.spearmanr(nt_ranks_list, other_ranks)
        print(f"    No Thanks! vs {name}: rho = {rho:.4f}, p = {p:.4f}")

    # Overall: do ALL domains have the same ordering?
    print(f"\n  --- Overall Ordering Agreement ---")
    canonical = ['none', 'ring', 'star', 'random', 'fully_connected']
    for name, ordering in domain_orderings.items():
        match = ordering == canonical
        print(f"    {name:<20}: {' > '.join(ordering)} {'[CANONICAL]' if match else ''}")

    n_canonical = sum(1 for o in domain_orderings.values() if o == canonical)
    print(f"\n    {n_canonical}/{len(domain_orderings)} domains show canonical ordering")

    # Friedman test (all domains, all topologies)
    if len(domain_rankings) >= 3:
        # Matrix: domains x topologies, values = ranks
        rank_matrix = []
        domain_names = []
        for name, ranks in domain_rankings.items():
            rank_matrix.append([ranks[t] for t in TOPO_ORDER])
            domain_names.append(name)

        rank_matrix = np.array(rank_matrix)
        # Friedman tests if topology ranking is consistent across domains
        # We need: topologies as "treatments", domains as "blocks"
        # Transpose: columns = topologies, rows = domains
        try:
            friedman_stat, friedman_p = stats.friedmanchisquare(
                *[rank_matrix[:, i] for i in range(len(TOPO_ORDER))]
            )
            print(f"\n  Friedman test (topology ranking consistent across domains):")
            print(f"    chi2 = {friedman_stat:.4f}, p = {friedman_p:.4f}")
        except Exception as e:
            print(f"\n  Friedman test failed: {e}")

    return domain_rankings


# ============================================================================
# 6. Kuramoto order parameter analysis
# ============================================================================

def analyze_kuramoto(df):
    """Kuramoto order parameter r = 1 - population_divergence."""
    print("\n" + "=" * 90)
    print("  6. KURAMOTO ORDER PARAMETER (r = 1 - population_divergence)")
    print("=" * 90)

    max_gen = df['generation'].max()
    final = df[df['generation'] == max_gen]

    print(f"\n  {'Topology':<20} {'r (final)':>12} {'std':>10} {'Interpretation':>20}")
    print("  " + "-" * 65)

    results = {}
    for topo in TOPO_ORDER:
        sub = final[final['topology'] == topo]
        r_vals = 1.0 - sub['population_divergence'].values
        r_mean = np.mean(r_vals)
        r_std = np.std(r_vals, ddof=1)

        if r_mean > 0.95:
            interp = "synchronized"
        elif r_mean > 0.90:
            interp = "near-sync"
        elif r_mean > 0.85:
            interp = "partial sync"
        elif r_mean > 0.80:
            interp = "low coherence"
        else:
            interp = "incoherent"

        results[topo] = {'r_mean': r_mean, 'r_std': r_std, 'r_values': r_vals, 'interp': interp}
        print(f"  {TOPO_LABELS[topo]:<20} {r_mean:>12.6f} {r_std:>10.6f} {interp:>20}")

    return results


# ============================================================================
# 7. Per-island analysis (No Thanks! has island-level data)
# ============================================================================

def analyze_per_island(df):
    """Analyze per-island diversity and fitness patterns."""
    print("\n" + "=" * 90)
    print("  7. PER-ISLAND ANALYSIS")
    print("=" * 90)

    max_gen = df['generation'].max()
    final = df[df['generation'] == max_gen]

    island_div_cols = [f'island_{i}_diversity' for i in range(5)]
    island_fit_cols = [f'island_{i}_fitness' for i in range(5)]
    div_pair_cols = ['div_0_1', 'div_0_2', 'div_0_3', 'div_0_4',
                     'div_1_2', 'div_1_3', 'div_1_4', 'div_2_3', 'div_2_4', 'div_3_4']

    for topo in TOPO_ORDER:
        sub = final[final['topology'] == topo]
        print(f"\n  {TOPO_LABELS[topo]}:")

        # Island diversity spread
        island_divs = sub[island_div_cols].values  # (n_seeds, 5)
        mean_per_island = island_divs.mean(axis=0)
        std_across_islands = island_divs.std(axis=1, ddof=1)  # spread within each seed

        print(f"    Per-island diversity: {', '.join(f'{m:.4f}' for m in mean_per_island)}")
        print(f"    Cross-island std (mean over seeds): {std_across_islands.mean():.6f}")

        # Star topology: island 0 is often the hub
        if topo == 'star':
            hub_div = island_divs[:, 0]  # island 0
            peripheral_divs = island_divs[:, 1:].mean(axis=1)  # islands 1-4
            hub_mean = hub_div.mean()
            periph_mean = peripheral_divs.mean()
            print(f"    Hub (island 0) diversity: {hub_mean:.6f}")
            print(f"    Peripheral (1-4) diversity: {periph_mean:.6f}")
            print(f"    Hub-peripheral gap: {hub_mean - periph_mean:.6f}")

            # Statistical test
            t_stat, p_val = stats.ttest_ind(hub_div, peripheral_divs)
            print(f"    Hub vs peripheral t-test: t={t_stat:.4f}, p={p_val:.4f}")

        # Pairwise divergence matrix
        pair_divs = sub[div_pair_cols].values  # (n_seeds, 10)
        mean_pairs = pair_divs.mean(axis=0)

        if topo == 'star':
            # Hub-connected pairs: div_0_1, div_0_2, div_0_3, div_0_4
            hub_pairs = sub[['div_0_1', 'div_0_2', 'div_0_3', 'div_0_4']].values
            # Peripheral pairs: div_1_2, div_1_3, div_1_4, div_2_3, div_2_4, div_3_4
            periph_pairs = sub[['div_1_2', 'div_1_3', 'div_1_4', 'div_2_3', 'div_2_4', 'div_3_4']].values
            print(f"    Hub-peripheral divergence: {hub_pairs.mean():.6f}")
            print(f"    Peripheral-peripheral divergence: {periph_pairs.mean():.6f}")

    return True


# ============================================================================
# 8. Coupling onset detection
# ============================================================================

def analyze_coupling_onset(df):
    """Detect coupling onset from baseline-corrected r curves."""
    print("\n" + "=" * 90)
    print("  8. COUPLING ONSET DETECTION")
    print("=" * 90)

    # Build r trajectories
    data = {}
    for topo in TOPO_ORDER:
        topo_df = df[df['topology'] == topo]
        seeds = sorted(topo_df['seed'].unique())
        data[topo] = {}
        for seed in seeds:
            seed_df = topo_df[topo_df['seed'] == seed].sort_values('generation')
            data[topo][seed] = {
                'generations': seed_df['generation'].values,
                'r': (1.0 - seed_df['population_divergence'].values),
                'diversity': seed_df['hamming_diversity'].values,
            }

    # Baseline: none topology mean r
    none_seeds = sorted(data['none'].keys())
    none_r_matrix = np.array([data['none'][s]['r'] for s in none_seeds])
    baseline_r = np.mean(none_r_matrix, axis=0)

    print(f"\n  {'Topology':<20} {'Onset gen':>12} {'SE':>10} {'Sig frac':>10}")
    print("  " + "-" * 55)

    results = {}
    for topo in COUPLED_TOPOS:
        seeds = sorted(data[topo].keys())
        onset_gens = []

        for seed in seeds:
            sd = data[topo][seed]
            delta_r = sd['r'] - baseline_r
            gens = sd['generations']

            # Smooth
            if len(delta_r) >= 10:
                delta_smooth = uniform_filter1d(delta_r, size=10)
            else:
                delta_smooth = delta_r

            # Find onset
            mask = gens >= MIGRATION_START
            if not np.any(mask):
                continue
            post_gens = gens[mask]
            post_delta = delta_smooth[mask]

            above = np.where(np.abs(post_delta) > 0.05)[0]
            if len(above) > 0:
                onset_gens.append(int(post_gens[above[0]]))

        onset_arr = np.array(onset_gens, dtype=float)
        valid = onset_arr[~np.isnan(onset_arr)]
        sig_frac = len(valid) / len(seeds) if seeds else 0

        mean_onset = np.mean(valid) if len(valid) > 0 else np.nan
        se_onset = np.std(valid, ddof=1) / np.sqrt(len(valid)) if len(valid) > 1 else 0

        results[topo] = {
            'onset_mean': mean_onset, 'onset_se': se_onset,
            'sig_frac': sig_frac, 'onset_gens': valid,
        }

        print(f"  {TOPO_LABELS[topo]:<20} {mean_onset:>12.1f} {se_onset:>10.2f} {sig_frac:>10.1%}")

    # Onset ordering
    onset_items = [(results[t]['onset_mean'], t) for t in COUPLED_TOPOS
                   if not np.isnan(results[t]['onset_mean'])]
    onset_items.sort()
    if onset_items:
        print(f"\n  Onset ordering (earliest -> latest):")
        print(f"    {' < '.join(f'{TOPO_LABELS[t]}({g:.0f})' for g, t in onset_items)}")

    return results


# ============================================================================
# 9. Fitness analysis
# ============================================================================

def analyze_fitness(df):
    """Analyze best fitness by topology."""
    print("\n" + "=" * 90)
    print("  9. FITNESS ANALYSIS")
    print("=" * 90)

    max_gen = df['generation'].max()
    final = df[df['generation'] == max_gen]

    print(f"\n  {'Topology':<20} {'Best Fitness':>14} {'(std)':>10} {'(SE)':>10}")
    print("  " + "-" * 60)

    fitness_results = {}
    for topo in TOPO_ORDER:
        sub = final[final['topology'] == topo]
        fit_vals = sub['best_fitness']
        fit_mean = fit_vals.mean()
        fit_std = fit_vals.std(ddof=1)
        fit_se = fit_std / np.sqrt(len(sub))
        fitness_results[topo] = {'mean': fit_mean, 'std': fit_std, 'se': fit_se, 'values': fit_vals.values}
        print(f"  {TOPO_LABELS[topo]:<20} {fit_mean:>14.6f} {fit_std:>10.6f} {fit_se:>10.6f}")

    # Ordering
    fit_order = sorted(TOPO_ORDER, key=lambda t: fitness_results[t]['mean'], reverse=True)
    print(f"\n  Fitness ordering (highest -> lowest): {' > '.join(fit_order)}")

    # Kruskal-Wallis on fitness
    fit_groups = [fitness_results[t]['values'] for t in TOPO_ORDER]
    h_fit, p_fit = stats.kruskal(*fit_groups)
    print(f"  Kruskal-Wallis: H = {h_fit:.4f}, p = {p_fit:.2e}")

    return fitness_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 90)
    print("  NO THANKS! DOMAIN — FULL ANALYSIS PIPELINE")
    print("  Co-evolutionary card game | Tournament-based relative fitness")
    print("  5 topologies x 30 seeds x 100 generations = 15,000 rows")
    print("=" * 90)

    df = load_data()

    # Run all analyses
    basic_stats = analyze_basic_stats(df)
    div_order = analyze_ordering(basic_stats)
    kw_results = analyze_kruskal_wallis(basic_stats)
    pairwise = analyze_pairwise(basic_stats)
    domain_rankings = analyze_domain_independence(basic_stats)
    kuramoto = analyze_kuramoto(df)
    per_island = analyze_per_island(df)
    onset = analyze_coupling_onset(df)
    fitness = analyze_fitness(df)

    # ---- SUMMARY ----
    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)

    canonical = ['none', 'ring', 'star', 'random', 'fully_connected']
    ordering_match = div_order == canonical
    print(f"\n  Canonical ordering match: {ordering_match}")
    print(f"  Observed ordering: {' > '.join(div_order)}")
    print(f"  Kruskal-Wallis (diversity): H={kw_results['div_H']:.1f}, p={kw_results['div_p']:.2e}")
    print(f"  Kruskal-Wallis (divergence): H={kw_results['pop_div_H']:.1f}, p={kw_results['pop_div_p']:.2e}")

    # Count significant pairwise comparisons
    n_sig = sum(1 for r in pairwise['diversity'] if r['sig'] != 'n.s.')
    n_total = len(pairwise['diversity'])
    print(f"  Significant pairwise comparisons: {n_sig}/{n_total} (diversity)")

    # Phase transition
    none_div = basic_stats['none']['div_mean']
    ring_div = basic_stats['ring']['div_mean']
    drop = (none_div - ring_div) / none_div * 100
    print(f"  Phase transition (none->ring): {drop:.1f}% drop")

    print(f"\n  KEY FINDING: No Thanks! is a CO-EVOLUTIONARY domain with")
    print(f"  tournament-based relative fitness. The canonical topology ordering")
    if ordering_match:
        print(f"  HOLDS, providing the strongest evidence yet for domain independence.")
        print(f"  When even the fitness landscape itself changes with the population,")
        print(f"  topology STILL determines the diversity dynamics.")
    else:
        print(f"  does NOT match exactly. Observed: {' > '.join(div_order)}")
        print(f"  This may be due to the co-evolutionary fitness landscape.")

    print("\n  DONE.")


if __name__ == '__main__':
    main()

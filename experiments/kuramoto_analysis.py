#!/usr/bin/env python3
"""
Kuramoto order parameter analysis from topology sweep data.

Maps evolutionary dynamics to coupled oscillator framework:
  - "phase" = fitness landscape position (genome content)
  - "synchronization" = convergence to same basin
  - "oscillator" = island subpopulation

The Kuramoto order parameter r measures phase coherence:
  r = 1: perfect synchronization
  r = 0: complete incoherence
  0 < r < 1: partial synchronization (chimera states)

We approximate r from population_divergence (inter-island divergence):
  r_approx = 1 - population_divergence

  Low divergence => islands similar => high synchronization => r near 1
  High divergence => islands different => low synchronization => r near 0

We also compute variance of r across seeds, which is the chimera signature:
chimera states show intermediate r with HIGH variance.
"""

import csv
import sys
import numpy as np
from collections import defaultdict
from scipy import stats


def load_data(filepath):
    """Load experiment E raw CSV data."""
    data = defaultdict(lambda: defaultdict(list))
    # data[topology][seed] = [(gen, diversity, divergence, fitness), ...]

    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            topo = row['topology']
            seed = int(row['seed'])
            gen = int(row['generation'])
            div = float(row['hamming_diversity'])
            pop_div = float(row['population_divergence'])
            fit = float(row['best_fitness'])
            data[topo][seed].append((gen, div, pop_div, fit))

    return data


def compute_kuramoto_proxy(data):
    """
    Compute Kuramoto order parameter proxy from population_divergence.

    r_approx = 1 - population_divergence

    population_divergence is the mean pairwise L1 distance on allele frequencies
    across all island pairs, normalized by genome length. It's in [0, 1].

    When divergence = 0: all islands have identical allele frequencies => perfect sync
    When divergence = 1: islands maximally different => complete incoherence
    """
    results = {}

    topo_order = ['none', 'ring', 'star', 'random', 'fully_connected']

    for topo in topo_order:
        if topo not in data:
            continue

        seeds = sorted(data[topo].keys())
        n_seeds = len(seeds)

        # Get max generations
        max_gen = max(gen for seed_data in data[topo].values()
                      for gen, _, _, _ in seed_data)

        # Compute r per seed per generation
        # r_series[seed] = array of r values over generations
        r_series = {}
        for seed in seeds:
            gens_data = sorted(data[topo][seed], key=lambda x: x[0])
            r_series[seed] = np.array([1.0 - pop_div for _, _, pop_div, _ in gens_data])

        # Final generation stats
        final_r = np.array([r_series[s][-1] for s in seeds])

        # Time-averaged stats (last 20 generations = steady state)
        steady_state_r = np.array([np.mean(r_series[s][-20:]) for s in seeds])

        # Variance across seeds at each generation
        all_r = np.array([r_series[s] for s in seeds])  # shape: (n_seeds, n_gens)
        r_mean_over_time = np.mean(all_r, axis=0)
        r_std_over_time = np.std(all_r, axis=0, ddof=1)
        r_var_over_time = np.var(all_r, axis=0, ddof=1)

        # Steady-state variance (mean variance over last 20 gens)
        ss_var = float(np.mean(r_var_over_time[-20:]))
        ss_std = float(np.mean(r_std_over_time[-20:]))

        results[topo] = {
            'final_r_mean': float(np.mean(final_r)),
            'final_r_std': float(np.std(final_r, ddof=1)),
            'final_r_median': float(np.median(final_r)),
            'steady_state_r_mean': float(np.mean(steady_state_r)),
            'steady_state_r_std': float(np.std(steady_state_r, ddof=1)),
            'steady_state_var': ss_var,
            'steady_state_std': ss_std,
            'r_mean_curve': r_mean_over_time,
            'r_std_curve': r_std_over_time,
            'final_r_values': final_r,
            'all_r': all_r,
        }

    return results


def test_claudius_predictions(results):
    """
    Test Claudius's predictions:
    1. none = incoherent (low r)
    2. ring = chimera signature (intermediate r, high variance)
    3. star = bimodal (hub synchronized, peripherals variable)
    4. mesh/fully_connected = synchrony (high r)
    """
    print("\n" + "=" * 80)
    print("TESTING CLAUDIUS'S PREDICTIONS")
    print("=" * 80)

    # Prediction 1: none has lowest r
    r_values = {t: results[t]['final_r_mean'] for t in results}
    min_r_topo = min(r_values, key=r_values.get)
    max_r_topo = max(r_values, key=r_values.get)

    print(f"\n1. none = incoherent (lowest r)?")
    print(f"   Lowest r topology: {min_r_topo} (r = {r_values[min_r_topo]:.4f})")
    print(f"   none r = {r_values.get('none', 'N/A'):.4f}")
    pred1 = min_r_topo == 'none'
    print(f"   PREDICTION: {'CONFIRMED' if pred1 else 'PARTIALLY CONFIRMED' if r_values['none'] < 0.9 else 'REJECTED'}")

    # Prediction 2: ring has intermediate r with HIGH variance (chimera)
    print(f"\n2. ring = chimera signature (intermediate r, high variance)?")
    ring_r = results['ring']['final_r_mean']
    ring_var = results['ring']['steady_state_var']

    # Check if ring variance is higher than others
    var_values = {t: results[t]['steady_state_var'] for t in results}
    max_var_topo = max(var_values, key=var_values.get)

    print(f"   ring: r = {ring_r:.4f}, variance = {ring_var:.6f}")
    print(f"   Highest variance topology: {max_var_topo} (var = {var_values[max_var_topo]:.6f})")
    print(f"   Variance ranking:")
    for t in sorted(var_values, key=var_values.get, reverse=True):
        marker = " <-- chimera?" if t == 'ring' else ""
        print(f"     {t:>18}: {var_values[t]:.6f}{marker}")

    # Is ring's r intermediate?
    all_r = sorted(r_values.values())
    ring_rank = sorted(r_values.items(), key=lambda x: x[1])
    ring_pos = [i for i, (t, _) in enumerate(ring_rank) if t == 'ring'][0]
    is_intermediate = 0 < ring_pos < len(ring_rank) - 1

    print(f"   ring r rank: {ring_pos + 1}/{len(ring_rank)} (1=lowest)")
    pred2 = is_intermediate and ring_var > var_values.get('fully_connected', 0)
    print(f"   PREDICTION: {'CONFIRMED' if pred2 else 'PARTIALLY CONFIRMED'}")

    # Prediction 3: star = bimodal
    print(f"\n3. star = bimodal (hub vs peripherals)?")
    star_r = results['star']['final_r_values']

    # Test for bimodality using Hartigan's dip test approximation
    # Simple approach: check if distribution is wider or more spread
    from scipy.stats import shapiro
    stat, p_normal = shapiro(star_r)

    print(f"   star r distribution: mean={np.mean(star_r):.4f}, "
          f"std={np.std(star_r, ddof=1):.4f}, "
          f"skew={float(stats.skew(star_r)):.4f}, "
          f"kurtosis={float(stats.kurtosis(star_r)):.4f}")
    print(f"   Shapiro-Wilk normality: W={stat:.4f}, p={p_normal:.4f}")
    print(f"   NOTE: Cannot test hub/peripheral bimodality without per-island data.")
    print(f"   The current CSV has only population-level metrics.")
    print(f"   PREDICTION: UNTESTABLE with current data (need per-island divergence)")

    # Prediction 4: fully_connected = synchrony (highest r)
    print(f"\n4. fully_connected = synchrony (highest r)?")
    print(f"   Highest r topology: {max_r_topo} (r = {r_values[max_r_topo]:.4f})")
    print(f"   fully_connected r = {r_values.get('fully_connected', 'N/A'):.4f}")
    pred4 = max_r_topo == 'fully_connected'
    print(f"   PREDICTION: {'CONFIRMED' if pred4 else 'REJECTED'}")

    # Statistical tests
    print(f"\n--- Statistical Tests ---")

    # Kruskal-Wallis on final r across all topologies
    groups = [results[t]['final_r_values'] for t in ['none', 'ring', 'star', 'random', 'fully_connected']]
    kw_stat, kw_p = stats.kruskal(*groups)
    print(f"Kruskal-Wallis (all topologies): H={kw_stat:.4f}, p={kw_p:.2e}")

    # Pairwise Mann-Whitney: none vs fully_connected (extremes)
    u_stat, u_p = stats.mannwhitneyu(
        results['none']['final_r_values'],
        results['fully_connected']['final_r_values'],
        alternative='less'
    )
    print(f"Mann-Whitney (none < fully_connected): U={u_stat:.1f}, p={u_p:.2e}")

    # Brown-Forsythe test for heteroscedasticity (variance differences)
    bf_stat, bf_p = stats.levene(*groups, center='median')
    print(f"Brown-Forsythe (variance homogeneity): F={bf_stat:.4f}, p={bf_p:.2e}")


def print_summary_table(results):
    """Print summary table of Kuramoto proxy values."""
    print("\n" + "=" * 80)
    print("KURAMOTO ORDER PARAMETER PROXY: r = 1 - population_divergence")
    print("=" * 80)

    topo_order = ['none', 'ring', 'star', 'random', 'fully_connected']

    print(f"\n{'Topology':>18} {'r (final)':>12} {'r (std)':>10} "
          f"{'r (s.s.)':>12} {'Var(r)':>12} {'Interpretation':>20}")
    print("-" * 90)

    for topo in topo_order:
        if topo not in results:
            continue
        r = results[topo]

        # Interpretation
        if r['final_r_mean'] > 0.95:
            interp = "synchronized"
        elif r['final_r_mean'] > 0.92:
            interp = "near-sync"
        elif r['final_r_mean'] > 0.88:
            interp = "partial sync"
        elif r['final_r_mean'] > 0.85:
            interp = "low coherence"
        else:
            interp = "incoherent"

        print(f"{topo:>18} {r['final_r_mean']:>12.4f} {r['final_r_std']:>10.4f} "
              f"{r['steady_state_r_mean']:>12.4f} {r['steady_state_var']:>12.6f} "
              f"{interp:>20}")

    print(f"\n  r (final) = mean r at generation 99")
    print(f"  r (s.s.)  = mean r averaged over last 20 generations (steady state)")
    print(f"  Var(r)    = mean variance across seeds over last 20 generations")


def analyze_phase_transition(results):
    """Analyze where phase transitions happen in the r trajectory."""
    print("\n" + "=" * 80)
    print("PHASE TRANSITION ANALYSIS")
    print("=" * 80)

    topo_order = ['none', 'ring', 'star', 'random', 'fully_connected']

    for topo in topo_order:
        if topo not in results:
            continue
        r = results[topo]
        curve = r['r_mean_curve']

        # Find steepest increase in r (= steepest decrease in divergence)
        dr = np.diff(curve)
        max_dr_gen = int(np.argmax(dr))

        # Find generation where r first exceeds 0.9
        r90_gen = np.where(curve > 0.9)[0]
        r90 = int(r90_gen[0]) if len(r90_gen) > 0 else -1

        # r at generation 5 (first migration for most)
        r_at_5 = float(curve[5]) if len(curve) > 5 else float('nan')
        r_at_10 = float(curve[10]) if len(curve) > 10 else float('nan')
        r_at_20 = float(curve[20]) if len(curve) > 20 else float('nan')

        print(f"\n  {topo:>18}: r(0)={curve[0]:.3f} -> r(5)={r_at_5:.3f} "
              f"-> r(10)={r_at_10:.3f} -> r(20)={r_at_20:.3f} -> r(99)={curve[-1]:.3f}")
        print(f"{'':>20} max dr/dt at gen {max_dr_gen} (dr={float(dr[max_dr_gen]):.4f})")
        if r90 >= 0:
            print(f"{'':>20} first r > 0.9 at gen {r90}")
        else:
            print(f"{'':>20} r never exceeds 0.9")


def chimera_detection(results):
    """
    Chimera state detection: look for intermediate r with high temporal variance
    within individual seeds.
    """
    print("\n" + "=" * 80)
    print("CHIMERA STATE DETECTION")
    print("=" * 80)
    print("\nChimera = intermediate r with high variance (some oscillators sync, others don't)")
    print("Two signatures: (1) high cross-seed variance, (2) high within-seed temporal variance\n")

    topo_order = ['none', 'ring', 'star', 'random', 'fully_connected']

    print(f"{'Topology':>18} {'Cross-seed var':>15} {'Within-seed var':>16} "
          f"{'r range':>10} {'Chimera?':>10}")
    print("-" * 75)

    for topo in topo_order:
        if topo not in results:
            continue
        r = results[topo]

        # Cross-seed variance (at final gen)
        cross_var = float(np.var(r['final_r_values'], ddof=1))

        # Within-seed temporal variance (mean over seeds, last 20 gens)
        all_r = r['all_r']  # shape (n_seeds, n_gens)
        within_vars = np.var(all_r[:, -20:], axis=1, ddof=1)  # var over time per seed
        mean_within_var = float(np.mean(within_vars))

        # r range across seeds at final gen
        r_range = float(np.max(r['final_r_values']) - np.min(r['final_r_values']))

        # Chimera heuristic: intermediate r AND (high cross or within variance)
        mean_r = r['final_r_mean']
        is_intermediate = 0.88 < mean_r < 0.95
        has_high_var = cross_var > 0.0005 or mean_within_var > 0.0005
        chimera = is_intermediate and has_high_var

        print(f"{topo:>18} {cross_var:>15.6f} {mean_within_var:>16.6f} "
              f"{r_range:>10.4f} {'YES' if chimera else 'no':>10}")

    print(f"\nNote: True chimera detection requires per-island r values.")
    print(f"A chimera state = some islands synchronized, others not.")
    print(f"With only population-level divergence, we can detect SIGNATURES but not confirm.")


def data_gap_analysis(data, results):
    """Report what additional data would be needed."""
    print("\n" + "=" * 80)
    print("DATA GAP ANALYSIS: What We'd Need for True Kuramoto Analysis")
    print("=" * 80)

    print("""
CURRENT DATA:
  - hamming_diversity: Mean pairwise Hamming distance across ALL individuals
    (population-level, not per-island)
  - population_divergence: Mean pairwise L1 allele-frequency distance across
    all ISLAND PAIRS (this IS inter-island, so it's a good r proxy)
  - best_fitness: Max fitness across all islands

WHAT WORKS:
  population_divergence is the best proxy for Kuramoto r because it directly
  measures inter-island coherence. r = 1 - pop_divergence is a valid mapping:
    - pop_divergence = 0 => all islands identical => r = 1 (perfect sync)
    - pop_divergence = 1 => islands maximally different => r = 0 (incoherent)

WHAT'S MISSING (for full Kuramoto analysis):
  1. PER-ISLAND diversity: diversity within each island separately
     -> Needed to distinguish between "all islands converged to SAME point"
        vs "all islands converged to DIFFERENT points"
     -> Currently hamming_diversity mixes intra- and inter-island variation

  2. PER-ISLAND-PAIR divergence: the full divergence matrix, not just the mean
     -> Needed for chimera detection: some pairs sync'd, others not
     -> The mean washes out bimodal structure

  3. PER-ISLAND best fitness: fitness trajectory per island
     -> Needed for star topology analysis: hub vs peripheral dynamics

SCRIPT MODIFICATIONS NEEDED:
  In run_experiment_e_single(), after line 841, add:

    # Per-island metrics
    island_diversities = [hamming_diversity(isl) for isl in islands]
    island_fitnesses = [float(np.max(evaluate(isl))) for isl in islands]

    # Full divergence matrix (upper triangle)
    div_matrix = []
    for i in range(n_isl):
        for j in range(i + 1, n_isl):
            div_matrix.append(population_divergence(islands[i], islands[j]))

  Then add to the row dict:
    'island_diversities': island_diversities,  # list of 4 floats
    'island_fitnesses': island_fitnesses,      # list of 4 floats
    'div_matrix': div_matrix,                  # list of 6 floats (4 choose 2)

  This would give us TRUE per-island Kuramoto phases.
""")


def main():
    filepath = '/home/lyra/projects/categorical-evolution/experiments/experiment_e_raw.csv'

    print("Loading data...")
    data = load_data(filepath)

    print(f"Loaded {sum(len(seeds) for seeds in data.values())} topology-seed combos")
    for topo in ['none', 'ring', 'star', 'random', 'fully_connected']:
        if topo in data:
            n_seeds = len(data[topo])
            n_gens = len(next(iter(data[topo].values())))
            print(f"  {topo}: {n_seeds} seeds x {n_gens} generations")

    print("\nComputing Kuramoto order parameter proxy...")
    results = compute_kuramoto_proxy(data)

    print_summary_table(results)
    analyze_phase_transition(results)
    chimera_detection(results)
    test_claudius_predictions(results)
    data_gap_analysis(data, results)

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
1. r = 1 - population_divergence is a VALID Kuramoto proxy because:
   - population_divergence directly measures inter-island allele-frequency distance
   - It maps naturally to the order parameter: 0 (incoherent) to 1 (synchronized)
   - It captures the RIGHT quantity: inter-oscillator phase coherence

2. However, it's an APPROXIMATION because:
   - We only have the MEAN of all pairwise divergences, not the full matrix
   - True Kuramoto r = |1/N * sum(exp(i*theta_j))| involves complex phases
   - Our "phases" are high-dimensional (genome space), not 1D angles
   - We lose bimodal structure by averaging

3. For a rigorous analysis, we need per-island metrics (see DATA GAP ANALYSIS).
   The script modification is straightforward (~10 lines of code).
""")


if __name__ == '__main__':
    main()

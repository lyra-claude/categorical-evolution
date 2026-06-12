#!/usr/bin/env python3
"""
Multi-seed statistical experiments for GECCO 2026 paper:
"Composition Determines Behavior: Diversity Fingerprints and the
Strict/Lax Dichotomy in Genetic Algorithms"

Compares two composition patterns of the SAME GA operators on OneMax:

  Experiment C — STRICT composition (standard pipeline):
      selection → crossover → mutation
      Applied sequentially each generation. This is the textbook GA.

  Experiment D — LAX composition (reordered pipeline):
      mutation → selection → crossover
      Mutation applied BEFORE selection. Individuals are perturbed first,
      then the fittest mutants are selected, then crossover recombines them.
      This changes the diversity dynamics because selection acts on
      already-mutated individuals rather than originals.

The key claim: the same operators, composed differently, produce
measurably different diversity trajectories (diversity fingerprints).

Usage:
    python onemax_stats.py                    # Run both experiments, 30 seeds
    python onemax_stats.py --seeds 10         # Quick test with 10 seeds
    python onemax_stats.py --no-plot          # Skip matplotlib plot
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# GA Configuration
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    population_size: int = 100
    genome_length: int = 100
    tournament_size: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.01  # 1/L where L=100
    max_generations: int = 100


# ---------------------------------------------------------------------------
# Core GA operators
# ---------------------------------------------------------------------------

def random_population(rng: np.random.Generator, pop_size: int,
                      genome_length: int) -> np.ndarray:
    """Generate random binary population. Shape: (pop_size, genome_length)."""
    return rng.integers(0, 2, size=(pop_size, genome_length), dtype=np.int8)


def evaluate(pop: np.ndarray) -> np.ndarray:
    """OneMax fitness: count 1-bits per individual."""
    return pop.sum(axis=1).astype(float)


def tournament_select(rng: np.random.Generator, pop: np.ndarray,
                      fitnesses: np.ndarray, tournament_size: int) -> np.ndarray:
    """Tournament selection with replacement.

    For each slot in the new population, pick `tournament_size` random
    individuals and keep the one with highest fitness.
    """
    n = len(pop)
    selected_indices = np.empty(n, dtype=int)
    for i in range(n):
        contestants = rng.integers(0, n, size=tournament_size)
        best_idx = contestants[np.argmax(fitnesses[contestants])]
        selected_indices[i] = best_idx
    return pop[selected_indices].copy()


def one_point_crossover(rng: np.random.Generator, pop: np.ndarray,
                        crossover_rate: float) -> np.ndarray:
    """One-point crossover on sequential pairs.

    Pairs individuals (0,1), (2,3), ... and with probability `crossover_rate`
    picks a random crossover point and swaps tails.
    """
    result = pop.copy()
    n = len(pop)
    genome_len = pop.shape[1]

    i = 0
    while i + 1 < n:
        if rng.random() < crossover_rate:
            point = rng.integers(1, genome_len)  # [1, genome_len)
            result[i, point:], result[i + 1, point:] = (
                pop[i + 1, point:].copy(), pop[i, point:].copy()
            )
        i += 2
    return result


def point_mutate(rng: np.random.Generator, pop: np.ndarray,
                 mutation_rate: float) -> np.ndarray:
    """Per-bit flip mutation. Each bit flips independently with probability
    `mutation_rate`."""
    result = pop.copy()
    mask = rng.random(size=pop.shape) < mutation_rate
    result[mask] = 1 - result[mask]
    return result


# ---------------------------------------------------------------------------
# Composition patterns
# ---------------------------------------------------------------------------

def strict_pipeline(rng: np.random.Generator, pop: np.ndarray,
                    config: GAConfig) -> np.ndarray:
    """STRICT composition: selection → crossover → mutation.

    This is the standard/textbook GA pipeline. Selection acts on the
    current population, producing a mating pool. Crossover recombines
    the mating pool. Mutation perturbs the offspring. The key property:
    selection sees the UN-mutated population from the previous generation.
    """
    fitnesses = evaluate(pop)
    selected = tournament_select(rng, pop, fitnesses, config.tournament_size)
    crossed = one_point_crossover(rng, selected, config.crossover_rate)
    mutated = point_mutate(rng, crossed, config.mutation_rate)
    return mutated


def lax_pipeline(rng: np.random.Generator, pop: np.ndarray,
                 config: GAConfig) -> np.ndarray:
    """LAX composition: mutation → selection → crossover.

    Mutation is applied FIRST, creating a cloud of variants around each
    individual. Selection then picks the fittest mutants. Crossover
    recombines the selected mutants. The key property: selection sees
    ALREADY-mutated individuals, which changes the selection gradient.

    With strict composition, a fit individual survives selection unchanged
    and only gets mutated afterward. With lax composition, even fit
    individuals are mutated before selection can preserve them, so
    selection operates on a noisier fitness signal.
    """
    mutated = point_mutate(rng, pop, config.mutation_rate)
    fitnesses = evaluate(mutated)
    selected = tournament_select(rng, mutated, fitnesses, config.tournament_size)
    crossed = one_point_crossover(rng, selected, config.crossover_rate)
    return crossed


# ---------------------------------------------------------------------------
# Diversity metric
# ---------------------------------------------------------------------------

def hamming_diversity(pop: np.ndarray) -> float:
    """Mean pairwise Hamming distance, normalized to [0, 1].

    Uses the efficient formula: for each bit position, count k ones among
    n individuals. The number of disagreeing pairs is k*(n-k). Sum over
    positions, divide by total pairs and genome length.

    Result is 0 when all individuals are identical, approaches 0.5 for
    a uniformly random population.
    """
    n = len(pop)
    if n < 2:
        return 0.0
    genome_len = pop.shape[1]
    ones_per_position = pop.sum(axis=0).astype(float)
    disagreeing = ones_per_position * (n - ones_per_position)
    total_pairs = n * (n - 1) / 2
    return float(np.sum(disagreeing) / total_pairs / genome_len)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(rng: np.random.Generator, config: GAConfig,
               pipeline_fn, label: str) -> dict:
    """Run a single GA with the given pipeline for `max_generations` gens.

    Returns a dict with per-generation diversity, fitness stats, and metadata.
    """
    pop = random_population(rng, config.population_size, config.genome_length)

    diversity_trace = []
    best_fitness_trace = []
    mean_fitness_trace = []

    # Record generation 0
    fitnesses = evaluate(pop)
    diversity_trace.append(hamming_diversity(pop))
    best_fitness_trace.append(float(np.max(fitnesses)))
    mean_fitness_trace.append(float(np.mean(fitnesses)))

    for gen in range(1, config.max_generations + 1):
        pop = pipeline_fn(rng, pop, config)
        fitnesses = evaluate(pop)
        diversity_trace.append(hamming_diversity(pop))
        best_fitness_trace.append(float(np.max(fitnesses)))
        mean_fitness_trace.append(float(np.mean(fitnesses)))

    return {
        'label': label,
        'diversity': diversity_trace,       # length = max_generations + 1
        'best_fitness': best_fitness_trace,
        'mean_fitness': mean_fitness_trace,
    }


# ---------------------------------------------------------------------------
# Multi-seed experiment
# ---------------------------------------------------------------------------

def run_multi_seed(config: GAConfig, pipeline_fn, label: str,
                   num_seeds: int = 30) -> dict:
    """Run `num_seeds` independent runs and collect diversity traces.

    Returns a dict with:
      - diversity_matrix: shape (num_seeds, max_generations+1)
      - best_fitness_matrix: same shape
      - mean_fitness_matrix: same shape
      - label: str
    """
    n_gens = config.max_generations + 1
    diversity_matrix = np.zeros((num_seeds, n_gens))
    best_fitness_matrix = np.zeros((num_seeds, n_gens))
    mean_fitness_matrix = np.zeros((num_seeds, n_gens))

    for seed in range(num_seeds):
        rng = np.random.default_rng(seed)
        result = run_single(rng, config, pipeline_fn, label)
        diversity_matrix[seed] = result['diversity']
        best_fitness_matrix[seed] = result['best_fitness']
        mean_fitness_matrix[seed] = result['mean_fitness']

        if (seed + 1) % 10 == 0 or seed + 1 == num_seeds:
            print(f"  {label}: {seed + 1}/{num_seeds} seeds complete",
                  flush=True)

    return {
        'label': label,
        'diversity_matrix': diversity_matrix,
        'best_fitness_matrix': best_fitness_matrix,
        'mean_fitness_matrix': mean_fitness_matrix,
        'num_seeds': num_seeds,
        'config': config,
    }


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def compute_summary(results: dict, generation: int) -> dict:
    """Compute summary statistics at a given generation."""
    div_at_gen = results['diversity_matrix'][:, generation]
    fit_at_gen = results['best_fitness_matrix'][:, generation]

    return {
        'generation': generation,
        'label': results['label'],
        'div_mean': float(np.mean(div_at_gen)),
        'div_std': float(np.std(div_at_gen, ddof=1)),
        'div_median': float(np.median(div_at_gen)),
        'div_q25': float(np.percentile(div_at_gen, 25)),
        'div_q75': float(np.percentile(div_at_gen, 75)),
        'div_min': float(np.min(div_at_gen)),
        'div_max': float(np.max(div_at_gen)),
        'fit_mean': float(np.mean(fit_at_gen)),
        'fit_std': float(np.std(fit_at_gen, ddof=1)),
        'fit_max': float(np.max(fit_at_gen)),
    }


def compare_compositions(strict_results: dict, lax_results: dict,
                         generation: int) -> dict:
    """Compare strict vs lax at a given generation using Mann-Whitney U."""
    strict_div = strict_results['diversity_matrix'][:, generation]
    lax_div = lax_results['diversity_matrix'][:, generation]

    # Mann-Whitney U test (non-parametric, appropriate for diversity data
    # which may not be normally distributed)
    u_stat, p_value = sp_stats.mannwhitneyu(
        strict_div, lax_div, alternative='two-sided'
    )

    # Effect size: rank-biserial correlation r = 1 - 2U/(n1*n2)
    n1, n2 = len(strict_div), len(lax_div)
    r_effect = 1 - (2 * u_stat) / (n1 * n2)

    # Also compute Cohen's d for reference
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(strict_div, ddof=1) +
         (n2 - 1) * np.var(lax_div, ddof=1)) / (n1 + n2 - 2)
    )
    cohens_d = (np.mean(strict_div) - np.mean(lax_div)) / pooled_std \
        if pooled_std > 0 else 0.0

    # 95% confidence intervals (bootstrap-free, using t-distribution)
    strict_ci = sp_stats.t.interval(
        0.95, df=n1 - 1,
        loc=np.mean(strict_div),
        scale=sp_stats.sem(strict_div)
    )
    lax_ci = sp_stats.t.interval(
        0.95, df=n2 - 1,
        loc=np.mean(lax_div),
        scale=sp_stats.sem(lax_div)
    )

    return {
        'generation': generation,
        'strict_mean': float(np.mean(strict_div)),
        'strict_std': float(np.std(strict_div, ddof=1)),
        'strict_ci_low': float(strict_ci[0]),
        'strict_ci_high': float(strict_ci[1]),
        'lax_mean': float(np.mean(lax_div)),
        'lax_std': float(np.std(lax_div, ddof=1)),
        'lax_ci_low': float(lax_ci[0]),
        'lax_ci_high': float(lax_ci[1]),
        'mann_whitney_U': float(u_stat),
        'p_value': float(p_value),
        'rank_biserial_r': float(r_effect),
        'cohens_d': float(cohens_d),
    }


# ---------------------------------------------------------------------------
# Output: CSV
# ---------------------------------------------------------------------------

def save_diversity_csv(results: dict, filepath: str):
    """Save per-generation diversity for all seeds to CSV."""
    n_seeds = results['num_seeds']
    n_gens = results['diversity_matrix'].shape[1]

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['seed'] + [f'gen_{g}' for g in range(n_gens)]
        writer.writerow(header)
        for seed in range(n_seeds):
            row = [seed] + [f'{d:.6f}'
                            for d in results['diversity_matrix'][seed]]
            writer.writerow(row)
    print(f"  Saved: {filepath}")


def save_summary_csv(strict_results: dict, lax_results: dict, filepath: str):
    """Save per-generation summary statistics and comparisons to CSV."""
    n_gens = strict_results['diversity_matrix'].shape[1]

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'generation',
            'strict_div_mean', 'strict_div_std',
            'strict_div_ci_low', 'strict_div_ci_high',
            'lax_div_mean', 'lax_div_std',
            'lax_div_ci_low', 'lax_div_ci_high',
            'mann_whitney_U', 'p_value',
            'rank_biserial_r', 'cohens_d',
            'strict_best_fit_mean', 'lax_best_fit_mean',
        ])
        for gen in range(n_gens):
            comp = compare_compositions(strict_results, lax_results, gen)
            strict_fit = float(np.mean(
                strict_results['best_fitness_matrix'][:, gen]))
            lax_fit = float(np.mean(
                lax_results['best_fitness_matrix'][:, gen]))
            writer.writerow([
                gen,
                f'{comp["strict_mean"]:.6f}', f'{comp["strict_std"]:.6f}',
                f'{comp["strict_ci_low"]:.6f}', f'{comp["strict_ci_high"]:.6f}',
                f'{comp["lax_mean"]:.6f}', f'{comp["lax_std"]:.6f}',
                f'{comp["lax_ci_low"]:.6f}', f'{comp["lax_ci_high"]:.6f}',
                f'{comp["mann_whitney_U"]:.1f}', f'{comp["p_value"]:.6e}',
                f'{comp["rank_biserial_r"]:.4f}', f'{comp["cohens_d"]:.4f}',
                f'{strict_fit:.2f}', f'{lax_fit:.2f}',
            ])
    print(f"  Saved: {filepath}")


# ---------------------------------------------------------------------------
# Output: Plot
# ---------------------------------------------------------------------------

def make_plot(strict_results: dict, lax_results: dict, filepath: str):
    """Generate diversity trajectory plot with confidence bands."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_gens = strict_results['diversity_matrix'].shape[1]
    generations = np.arange(n_gens)

    # --- Panel A: Diversity trajectories ---
    ax = axes[0]
    for results, color, label in [
        (strict_results, '#2166ac', 'Strict (sel→cx→mut)'),
        (lax_results, '#b2182b', 'Lax (mut→sel→cx)'),
    ]:
        div_matrix = results['diversity_matrix']
        mean = div_matrix.mean(axis=0)
        std = div_matrix.std(axis=0, ddof=1)
        sem = std / np.sqrt(results['num_seeds'])
        ci_low = mean - 1.96 * sem
        ci_high = mean + 1.96 * sem

        ax.plot(generations, mean, color=color, linewidth=2, label=label)
        ax.fill_between(generations, ci_low, ci_high, color=color, alpha=0.2)

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Genotypic Diversity\n(mean pairwise Hamming, normalized)',
                  fontsize=11)
    ax.set_title('(a) Diversity Fingerprints', fontsize=13)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(0, n_gens - 1)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # --- Panel B: Fitness trajectories ---
    ax = axes[1]
    for results, color, label in [
        (strict_results, '#2166ac', 'Strict (sel→cx→mut)'),
        (lax_results, '#b2182b', 'Lax (mut→sel→cx)'),
    ]:
        fit_matrix = results['best_fitness_matrix']
        mean = fit_matrix.mean(axis=0)
        std = fit_matrix.std(axis=0, ddof=1)
        sem = std / np.sqrt(results['num_seeds'])
        ci_low = mean - 1.96 * sem
        ci_high = mean + 1.96 * sem

        ax.plot(generations, mean, color=color, linewidth=2, label=label)
        ax.fill_between(generations, ci_low, ci_high, color=color, alpha=0.2)

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Best Fitness (OneMax)', fontsize=11)
    ax.set_title('(b) Fitness Progression', fontsize=13)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xlim(0, n_gens - 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(strict_results: dict, lax_results: dict):
    """Print a full statistical report to stdout."""
    config = strict_results['config']
    n_seeds = strict_results['num_seeds']

    print("\n" + "=" * 78)
    print("ONEMAX COMPOSITION EXPERIMENT — STATISTICAL REPORT")
    print("=" * 78)
    print(f"\nParameters:")
    print(f"  Population size:  {config.population_size}")
    print(f"  Genome length:    {config.genome_length}")
    print(f"  Generations:      {config.max_generations}")
    print(f"  Tournament size:  {config.tournament_size}")
    print(f"  Crossover rate:   {config.crossover_rate}")
    print(f"  Mutation rate:    {config.mutation_rate} (1/L)")
    print(f"  Seeds:            {n_seeds}")

    print(f"\nComposition patterns:")
    print(f"  STRICT: selection -> crossover -> mutation")
    print(f"  LAX:    mutation -> selection -> crossover")

    # Summary at key generations
    key_gens = [0, 10, 25, 50, 75, 100]
    key_gens = [g for g in key_gens if g < strict_results['diversity_matrix'].shape[1]]

    print(f"\n{'':->78}")
    print(f"{'DIVERSITY SUMMARY':^78}")
    print(f"{'':->78}")
    print(f"{'Gen':>5}  {'Strict (mean +/- std)':>22}  {'Lax (mean +/- std)':>22}"
          f"  {'p-value':>12}  {'Cohen d':>9}")
    print(f"{'':->78}")

    for gen in key_gens:
        comp = compare_compositions(strict_results, lax_results, gen)
        sig = "***" if comp['p_value'] < 0.001 else \
              "** " if comp['p_value'] < 0.01 else \
              "*  " if comp['p_value'] < 0.05 else \
              "   "
        print(f"{gen:>5}  "
              f"{comp['strict_mean']:.4f} +/- {comp['strict_std']:.4f}    "
              f"{comp['lax_mean']:.4f} +/- {comp['lax_std']:.4f}    "
              f"{comp['p_value']:>10.2e} {sig} "
              f"{comp['cohens_d']:>8.3f}")

    # Detailed comparison at gen 50 and gen 100
    for gen in [50, config.max_generations]:
        if gen >= strict_results['diversity_matrix'].shape[1]:
            continue
        comp = compare_compositions(strict_results, lax_results, gen)
        print(f"\n--- Detailed comparison at generation {gen} ---")
        print(f"  Strict diversity: {comp['strict_mean']:.4f} +/- "
              f"{comp['strict_std']:.4f}  "
              f"95% CI [{comp['strict_ci_low']:.4f}, "
              f"{comp['strict_ci_high']:.4f}]")
        print(f"  Lax diversity:    {comp['lax_mean']:.4f} +/- "
              f"{comp['lax_std']:.4f}  "
              f"95% CI [{comp['lax_ci_low']:.4f}, "
              f"{comp['lax_ci_high']:.4f}]")
        print(f"  Mann-Whitney U:   {comp['mann_whitney_U']:.1f}")
        print(f"  p-value:          {comp['p_value']:.2e}")
        print(f"  Rank-biserial r:  {comp['rank_biserial_r']:.4f}")
        print(f"  Cohen's d:        {comp['cohens_d']:.4f}")

        # Interpret
        if comp['p_value'] < 0.001:
            print(f"  --> HIGHLY SIGNIFICANT difference (p < 0.001)")
        elif comp['p_value'] < 0.05:
            print(f"  --> Significant difference (p < 0.05)")
        else:
            print(f"  --> NOT significant (p >= 0.05)")

        d = abs(comp['cohens_d'])
        if d >= 0.8:
            print(f"  --> Large effect size (|d| >= 0.8)")
        elif d >= 0.5:
            print(f"  --> Medium effect size (|d| >= 0.5)")
        elif d >= 0.2:
            print(f"  --> Small effect size (|d| >= 0.2)")
        else:
            print(f"  --> Negligible effect size (|d| < 0.2)")

    # Fitness comparison
    print(f"\n{'':->78}")
    print(f"{'FITNESS SUMMARY':^78}")
    print(f"{'':->78}")
    print(f"{'Gen':>5}  {'Strict best (mean)':>20}  {'Lax best (mean)':>20}")
    print(f"{'':->40}")
    for gen in key_gens:
        s_fit = float(np.mean(strict_results['best_fitness_matrix'][:, gen]))
        l_fit = float(np.mean(lax_results['best_fitness_matrix'][:, gen]))
        print(f"{gen:>5}  {s_fit:>20.2f}  {l_fit:>20.2f}")

    # Final generation fitness comparison
    final_gen = config.max_generations
    if final_gen < strict_results['best_fitness_matrix'].shape[1]:
        s_final = strict_results['best_fitness_matrix'][:, final_gen]
        l_final = lax_results['best_fitness_matrix'][:, final_gen]
        u_fit, p_fit = sp_stats.mannwhitneyu(
            s_final, l_final, alternative='two-sided')
        print(f"\n  Final fitness Mann-Whitney U: {u_fit:.1f}, "
              f"p = {p_fit:.2e}")
        if p_fit < 0.05:
            print(f"  --> Significant fitness difference")
        else:
            print(f"  --> No significant fitness difference")

    print(f"\n{'=' * 78}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OneMax strict vs lax composition experiments")
    parser.add_argument('--seeds', type=int, default=30,
                        help='Number of seeds per configuration (default: 30)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating matplotlib plot')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output directory (default: results/ next to script)')
    args = parser.parse_args()

    # Determine output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outdir = args.outdir or os.path.join(script_dir, 'results')
    os.makedirs(outdir, exist_ok=True)

    config = GAConfig(
        population_size=100,
        genome_length=100,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=1.0 / 100.0,  # 1/L
        max_generations=100,
    )

    print("=" * 60)
    print("OneMax Composition Experiment")
    print("=" * 60)
    print(f"  Pop size:        {config.population_size}")
    print(f"  Genome length:   {config.genome_length}")
    print(f"  Generations:     {config.max_generations}")
    print(f"  Tournament size: {config.tournament_size}")
    print(f"  Crossover rate:  {config.crossover_rate}")
    print(f"  Mutation rate:   {config.mutation_rate}")
    print(f"  Seeds:           {args.seeds}")
    print(f"  Output dir:      {outdir}")
    print()

    t0 = time.time()

    # --- Run Experiment C: Strict composition ---
    print("--- Experiment C: Strict Composition (sel -> cx -> mut) ---")
    strict_results = run_multi_seed(
        config, strict_pipeline, 'strict', num_seeds=args.seeds)

    # --- Run Experiment D: Lax composition ---
    print("\n--- Experiment D: Lax Composition (mut -> sel -> cx) ---")
    lax_results = run_multi_seed(
        config, lax_pipeline, 'lax', num_seeds=args.seeds)

    elapsed = time.time() - t0
    print(f"\nAll runs complete in {elapsed:.1f}s")

    # --- Save CSVs ---
    print("\nSaving results...")
    save_diversity_csv(
        strict_results,
        os.path.join(outdir, 'strict_diversity.csv'))
    save_diversity_csv(
        lax_results,
        os.path.join(outdir, 'lax_diversity.csv'))
    save_summary_csv(
        strict_results, lax_results,
        os.path.join(outdir, 'comparison_summary.csv'))

    # --- Generate plot ---
    if not args.no_plot:
        print("\nGenerating plot...")
        make_plot(
            strict_results, lax_results,
            os.path.join(outdir, 'diversity_fingerprints.png'))

    # --- Print report ---
    print_report(strict_results, lax_results)

    print(f"\nTotal wall time: {elapsed:.1f}s")
    print(f"Output directory: {outdir}")


if __name__ == '__main__':
    main()

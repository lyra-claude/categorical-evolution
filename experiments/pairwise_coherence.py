#!/usr/bin/env python3
"""
Pairwise phase coherence R(d) — coherence as a function of topological distance.

Motivation (from Claudius): the stable synchronized solution on a ring is a
TRAVELING WAVE, not uniform phase-locking. Global Kuramoto R approaches zero
even when the ring is fully synchronized because phases cancel in aggregate.
R(d) — pairwise coherence by topological distance — correctly measures ring
synchronization by revealing the spatial structure.

For each topology, we compute the mean pairwise coherence between island pairs
at each topological distance d. Coherence between two islands is defined as:

    coherence(i, j) = 1 - div_{i}_{j}

where div_{i}_{j} is the pairwise L1 allele-frequency divergence (already in
the per-island CSV). This matches the Kuramoto proxy used in kuramoto_analysis.py.

Topological distances:
    Ring (5 islands, cycle 0-1-2-3-4-0):
        d=1: adjacent (0-1, 1-2, 2-3, 3-4, 4-0) — 5 pairs
        d=2: next-nearest (0-2, 0-3, 1-3, 1-4, 2-4) — 5 pairs
    Star (5 islands, hub=0):
        d=1: hub-spoke (0-1, 0-2, 0-3, 0-4) — 4 pairs
        d=2: spoke-spoke (1-2, 1-3, 1-4, 2-3, 2-4, 3-4) — 6 pairs
    Fully connected:
        d=1: all pairs — 10 pairs
    Random:
        Edges vary per generation per seed, so exact distances unknown.
        We report all pairs as "d=mean" (aggregate). For comparison, we also
        split into approximate categories by divergence magnitude.
    None:
        No coupling, so d=inf. All pairs reported at d=inf.

Expected signatures:
    Ring: R(d=1) high, R(d=2) lower → gradient = traveling wave
    Star: R(d=1) high (hub-spoke), R(d=2) moderate (spoke-spoke via hub)
    Fully connected: R(d=1) high, flat (all pairs equivalent)
    None: low R at all distances (no coupling)

Usage:
    python3 pairwise_coherence.py
    python3 pairwise_coherence.py --gen 99
    python3 pairwise_coherence.py --steady-state  # average over last 20 gens
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

# Optional: matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Topology distance definitions ──────────────────────────────────────────

N_ISLANDS = 5

# All pairwise divergence column names (upper triangle, i < j)
DIV_COLS = [f'div_{i}_{j}' for i in range(N_ISLANDS) for j in range(i + 1, N_ISLANDS)]

# Ring: 0-1-2-3-4-0
# Distance = min(|i-j|, N - |i-j|) on the cycle
RING_PAIRS_BY_DIST = defaultdict(list)
for i in range(N_ISLANDS):
    for j in range(i + 1, N_ISLANDS):
        d = min(abs(i - j), N_ISLANDS - abs(i - j))
        RING_PAIRS_BY_DIST[d].append(f'div_{i}_{j}')

# Star: island 0 = hub
# d=1: hub-spoke, d=2: spoke-spoke (must go through hub)
STAR_PAIRS_BY_DIST = defaultdict(list)
for i in range(N_ISLANDS):
    for j in range(i + 1, N_ISLANDS):
        if i == 0 or j == 0:
            STAR_PAIRS_BY_DIST[1].append(f'div_{i}_{j}')
        else:
            STAR_PAIRS_BY_DIST[2].append(f'div_{i}_{j}')

# Fully connected: all pairs at distance 1
FC_PAIRS_BY_DIST = {1: list(DIV_COLS)}

# None: all pairs at distance infinity (no edges)
NONE_PAIRS_BY_DIST = {float('inf'): list(DIV_COLS)}


def get_pair_distances(topology):
    """Return dict mapping distance -> list of div column names."""
    if topology == 'ring':
        return dict(RING_PAIRS_BY_DIST)
    elif topology == 'star':
        return dict(STAR_PAIRS_BY_DIST)
    elif topology == 'fully_connected':
        return dict(FC_PAIRS_BY_DIST)
    elif topology == 'none':
        return dict(NONE_PAIRS_BY_DIST)
    elif topology == 'random':
        # Random topology: edges change every migration event.
        # We can't recover exact distances. Return all pairs at d=1
        # (average connectivity) for comparison, but flag this.
        return {1: list(DIV_COLS)}
    else:
        raise ValueError(f"Unknown topology: {topology}")


# ── Data loading ───────────────────────────────────────────────────────────

def load_data(filepath):
    """Load per-island CSV into structured dict.

    Returns: dict[topology][seed] = list of row dicts, sorted by generation.
    """
    data = defaultdict(lambda: defaultdict(list))
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            topo = row['topology']
            seed = int(row['seed'])
            data[topo][seed].append(row)

    # Sort each seed's rows by generation
    for topo in data:
        for seed in data[topo]:
            data[topo][seed].sort(key=lambda r: int(r['generation']))

    return data


# ── Coherence computation ──────────────────────────────────────────────────

def compute_pairwise_coherence(data, generation=None, steady_state=False,
                               ss_window=20):
    """Compute R(d) for each topology.

    Args:
        data: output of load_data()
        generation: specific generation to analyze (default: final)
        steady_state: if True, average over last ss_window generations
        ss_window: number of generations for steady-state window

    Returns:
        results: dict[topology] = list of dicts with keys:
            distance, mean_coherence, std_coherence, n_pairs, n_observations,
            all_coherences (raw values for statistical tests)
    """
    topo_order = ['none', 'ring', 'star', 'random', 'fully_connected']
    results = {}

    for topo in topo_order:
        if topo not in data:
            continue

        pair_dists = get_pair_distances(topo)
        seeds = sorted(data[topo].keys())

        # Determine which rows to use
        if generation is not None:
            row_filter = lambda r: int(r['generation']) == generation
        elif steady_state:
            max_gen = max(int(r['generation']) for r in data[topo][seeds[0]])
            start_gen = max_gen - ss_window + 1
            row_filter = lambda r, sg=start_gen: int(r['generation']) >= sg
        else:
            # Default: final generation
            max_gen = max(int(r['generation']) for r in data[topo][seeds[0]])
            row_filter = lambda r, mg=max_gen: int(r['generation']) == mg

        topo_results = []

        for dist in sorted(pair_dists.keys()):
            cols = pair_dists[dist]
            all_coherences = []

            for seed in seeds:
                rows = [r for r in data[topo][seed] if row_filter(r)]
                for row in rows:
                    for col in cols:
                        div = float(row[col])
                        coherence = 1.0 - div
                        all_coherences.append(coherence)

            all_coherences = np.array(all_coherences)

            topo_results.append({
                'distance': dist,
                'mean_coherence': float(np.mean(all_coherences)),
                'std_coherence': float(np.std(all_coherences, ddof=1)),
                'sem_coherence': float(np.std(all_coherences, ddof=1) / np.sqrt(len(all_coherences))),
                'n_pairs': len(cols),
                'n_observations': len(all_coherences),
                'all_coherences': all_coherences,
            })

        results[topo] = topo_results

    return results


def compute_coherence_over_time(data, topologies=None):
    """Compute R(d) at every generation for time-series analysis.

    Returns:
        time_series: dict[topology][distance] = array of shape (n_gens,)
            with mean coherence at each generation, averaged over seeds.
    """
    if topologies is None:
        topologies = ['none', 'ring', 'star', 'random', 'fully_connected']

    time_series = {}

    for topo in topologies:
        if topo not in data:
            continue

        pair_dists = get_pair_distances(topo)
        seeds = sorted(data[topo].keys())
        max_gen = max(int(r['generation']) for r in data[topo][seeds[0]])
        n_gens = max_gen + 1

        topo_ts = {}
        for dist in sorted(pair_dists.keys()):
            cols = pair_dists[dist]
            # shape: (n_seeds, n_gens)
            seed_curves = np.zeros((len(seeds), n_gens))

            for s_idx, seed in enumerate(seeds):
                rows = sorted(data[topo][seed], key=lambda r: int(r['generation']))
                for row in rows:
                    gen = int(row['generation'])
                    divs = [float(row[col]) for col in cols]
                    seed_curves[s_idx, gen] = 1.0 - np.mean(divs)

            # Mean over seeds
            topo_ts[dist] = np.mean(seed_curves, axis=0)

        time_series[topo] = topo_ts

    return time_series


# ── Output ─────────────────────────────────────────────────────────────────

def print_results(results, label=""):
    """Print R(d) table."""
    print(f"\n{'='*80}")
    print(f"PAIRWISE PHASE COHERENCE R(d) — Coherence by Topological Distance")
    if label:
        print(f"  {label}")
    print(f"{'='*80}")

    topo_order = ['none', 'ring', 'star', 'random', 'fully_connected']

    print(f"\n{'Topology':>18} {'d':>5} {'R(d)':>8} {'std':>8} {'SEM':>8} "
          f"{'n_pairs':>8} {'n_obs':>8}")
    print("-" * 75)

    for topo in topo_order:
        if topo not in results:
            continue
        for entry in results[topo]:
            d = entry['distance']
            d_str = 'inf' if d == float('inf') else str(d)
            print(f"{topo:>18} {d_str:>5} {entry['mean_coherence']:>8.4f} "
                  f"{entry['std_coherence']:>8.4f} {entry['sem_coherence']:>8.4f} "
                  f"{entry['n_pairs']:>8} {entry['n_observations']:>8}")
        # Blank line between topologies
        if len(results[topo]) > 1:
            print()


def save_csv(results, filepath):
    """Save R(d) results to CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['topology', 'distance', 'mean_coherence', 'std_coherence',
                         'sem_coherence', 'n_pairs', 'n_observations'])

        topo_order = ['none', 'ring', 'star', 'random', 'fully_connected']
        for topo in topo_order:
            if topo not in results:
                continue
            for entry in results[topo]:
                d = entry['distance']
                d_str = 'inf' if d == float('inf') else str(d)
                writer.writerow([
                    topo, d_str,
                    f"{entry['mean_coherence']:.6f}",
                    f"{entry['std_coherence']:.6f}",
                    f"{entry['sem_coherence']:.6f}",
                    entry['n_pairs'],
                    entry['n_observations'],
                ])

    print(f"\nCSV saved to: {filepath}")


def plot_results(results, time_series, output_path):
    """Generate R(d) plot with two panels:
    Left: R(d) bar chart by topology and distance
    Right: R(d) time series for ring (d=1 vs d=2)
    """
    if not HAS_MPL:
        print("matplotlib not available, skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left panel: R(d) bar chart ──
    ax = axes[0]

    topo_order = ['none', 'ring', 'star', 'fully_connected']
    colors = {
        'none': '#999999',
        'ring': '#e74c3c',
        'star': '#f39c12',
        'random': '#2ecc71',
        'fully_connected': '#3498db',
    }
    labels_map = {
        'none': 'None',
        'ring': 'Ring',
        'star': 'Star',
        'random': 'Random',
        'fully_connected': 'Fully\nConnected',
    }

    # Build grouped bars
    bar_data = []  # (label, distance_label, mean, sem, color)
    for topo in topo_order:
        if topo not in results:
            continue
        for entry in results[topo]:
            d = entry['distance']
            if d == float('inf'):
                d_label = f"{labels_map[topo]}\n(d=∞)"
            else:
                d_label = f"{labels_map[topo]}\n(d={d})"
            bar_data.append((d_label, entry['mean_coherence'],
                             entry['sem_coherence'], colors[topo]))

    x = np.arange(len(bar_data))
    bar_labels = [b[0] for b in bar_data]
    bar_means = [b[1] for b in bar_data]
    bar_sems = [b[2] for b in bar_data]
    bar_colors = [b[3] for b in bar_data]

    bars = ax.bar(x, bar_means, yerr=bar_sems, capsize=4,
                  color=bar_colors, edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_ylabel('Pairwise Coherence R(d)', fontsize=12)
    ax.set_title('R(d) by Topology and Distance\n(Generation 99, 30 seeds)', fontsize=13)
    ax.set_ylim(0, 1.05)

    # Add value labels on bars
    for bar_obj, mean_val in zip(bars, bar_means):
        ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + 0.02,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    # ── Right panel: R(d) time series for ring ──
    ax2 = axes[1]

    if 'ring' in time_series:
        ring_ts = time_series['ring']
        for dist in sorted(ring_ts.keys()):
            curve = ring_ts[dist]
            gens = np.arange(len(curve))
            ax2.plot(gens, curve, linewidth=2,
                     label=f'Ring d={dist}',
                     color='#e74c3c' if dist == 1 else '#c0392b',
                     linestyle='-' if dist == 1 else '--')

    if 'star' in time_series:
        star_ts = time_series['star']
        for dist in sorted(star_ts.keys()):
            curve = star_ts[dist]
            gens = np.arange(len(curve))
            ax2.plot(gens, curve, linewidth=2,
                     label=f'Star d={dist}',
                     color='#f39c12' if dist == 1 else '#e67e22',
                     linestyle='-' if dist == 1 else '--')

    if 'fully_connected' in time_series:
        fc_ts = time_series['fully_connected']
        for dist in sorted(fc_ts.keys()):
            curve = fc_ts[dist]
            gens = np.arange(len(curve))
            ax2.plot(gens, curve, linewidth=2,
                     label='FC d=1',
                     color='#3498db')

    if 'none' in time_series:
        none_ts = time_series['none']
        for dist in sorted(none_ts.keys()):
            curve = none_ts[dist]
            gens = np.arange(len(curve))
            ax2.plot(gens, curve, linewidth=2,
                     label=f'None (d=∞)',
                     color='#999999', linestyle=':')

    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Pairwise Coherence R(d)', fontsize=12)
    ax2.set_title('R(d) Over Time by Topology', fontsize=13)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    # Save as both PNG and PDF
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved to: {output_path}")
    print(f"Plot saved to: {pdf_path}")


# ── Statistical tests ──────────────────────────────────────────────────────

def statistical_tests(results):
    """Run statistical tests on R(d) results."""
    print(f"\n{'='*80}")
    print("STATISTICAL TESTS")
    print(f"{'='*80}")

    # ── Test 1: Ring gradient (d=1 vs d=2) ──
    if 'ring' in results and len(results['ring']) == 2:
        ring_d1 = results['ring'][0]['all_coherences']
        ring_d2 = results['ring'][1]['all_coherences']

        u_stat, u_p = stats.mannwhitneyu(ring_d1, ring_d2, alternative='greater')
        cohen_d = (np.mean(ring_d1) - np.mean(ring_d2)) / np.sqrt(
            (np.var(ring_d1, ddof=1) + np.var(ring_d2, ddof=1)) / 2)

        print(f"\n  1. Ring traveling wave test: R(d=1) > R(d=2)?")
        print(f"     R(d=1) = {np.mean(ring_d1):.4f} +/- {np.std(ring_d1, ddof=1):.4f}")
        print(f"     R(d=2) = {np.mean(ring_d2):.4f} +/- {np.std(ring_d2, ddof=1):.4f}")
        print(f"     Gradient: {np.mean(ring_d1) - np.mean(ring_d2):.4f}")
        print(f"     Mann-Whitney U (one-sided): U={u_stat:.1f}, p={u_p:.2e}")
        print(f"     Cohen's d: {cohen_d:.3f}")
        if u_p < 0.001:
            print(f"     RESULT: STRONG traveling wave signature (p < 0.001)")
        elif u_p < 0.05:
            print(f"     RESULT: Traveling wave signature detected (p < 0.05)")
        else:
            print(f"     RESULT: No significant gradient (p = {u_p:.4f})")

    # ── Test 2: Star hub mediation (d=1 vs d=2) ──
    if 'star' in results and len(results['star']) == 2:
        star_d1 = results['star'][0]['all_coherences']
        star_d2 = results['star'][1]['all_coherences']

        u_stat, u_p = stats.mannwhitneyu(star_d1, star_d2, alternative='greater')
        cohen_d = (np.mean(star_d1) - np.mean(star_d2)) / np.sqrt(
            (np.var(star_d1, ddof=1) + np.var(star_d2, ddof=1)) / 2)

        print(f"\n  2. Star hub mediation test: R(d=1) > R(d=2)?")
        print(f"     R(d=1) hub-spoke  = {np.mean(star_d1):.4f} +/- {np.std(star_d1, ddof=1):.4f}")
        print(f"     R(d=2) spoke-spoke = {np.mean(star_d2):.4f} +/- {np.std(star_d2, ddof=1):.4f}")
        print(f"     Gradient: {np.mean(star_d1) - np.mean(star_d2):.4f}")
        print(f"     Mann-Whitney U (one-sided): U={u_stat:.1f}, p={u_p:.2e}")
        print(f"     Cohen's d: {cohen_d:.3f}")
        if u_p < 0.001:
            print(f"     RESULT: Hub strongly mediates synchronization (p < 0.001)")
        elif u_p < 0.05:
            print(f"     RESULT: Hub mediates synchronization (p < 0.05)")
        else:
            print(f"     RESULT: No significant hub mediation (p = {u_p:.4f})")

    # ── Test 3: Ring d=1 vs FC d=1 ──
    if 'ring' in results and 'fully_connected' in results:
        ring_d1 = results['ring'][0]['all_coherences']
        fc_d1 = results['fully_connected'][0]['all_coherences']

        u_stat, u_p = stats.mannwhitneyu(ring_d1, fc_d1, alternative='two-sided')

        print(f"\n  3. Ring adjacent vs Fully Connected: R_ring(d=1) vs R_fc(d=1)?")
        print(f"     Ring R(d=1) = {np.mean(ring_d1):.4f}")
        print(f"     FC   R(d=1) = {np.mean(fc_d1):.4f}")
        print(f"     Difference: {np.mean(ring_d1) - np.mean(fc_d1):.4f}")
        print(f"     Mann-Whitney U (two-sided): p={u_p:.2e}")

    # ── Test 4: None baseline ──
    if 'none' in results:
        none_r = results['none'][0]['all_coherences']
        print(f"\n  4. None topology baseline:")
        print(f"     R(d=inf) = {np.mean(none_r):.4f} +/- {np.std(none_r, ddof=1):.4f}")
        print(f"     (Should be low — no coupling, no coherence)")

    # ── Test 5: Kruskal-Wallis across all d=1 pairs ──
    groups_d1 = {}
    for topo in ['ring', 'star', 'fully_connected']:
        if topo in results:
            groups_d1[topo] = results[topo][0]['all_coherences']

    if len(groups_d1) >= 2:
        kw_stat, kw_p = stats.kruskal(*groups_d1.values())
        print(f"\n  5. Kruskal-Wallis: R(d=1) differs across topologies?")
        print(f"     H={kw_stat:.4f}, p={kw_p:.2e}")
        print(f"     {'SIGNIFICANT' if kw_p < 0.05 else 'not significant'}")


def baseline_corrected_analysis(results):
    """Compute baseline-corrected R(d) using 'none' topology as reference.

    On OneMax (and similar single-optimum landscapes), uncoupled islands
    converge to similar solutions independently, producing high baseline
    coherence. Baseline correction isolates the coupling signal.

    R_corrected(d) = R(d) - R_none
    """
    print(f"\n{'='*80}")
    print("BASELINE-CORRECTED COHERENCE")
    print("  R_corrected(d) = R(d) - R_none(d=inf)")
    print(f"{'='*80}")

    if 'none' not in results:
        print("  No 'none' topology data for baseline correction.")
        return

    baseline = results['none'][0]['mean_coherence']
    print(f"\n  Baseline (none): R = {baseline:.4f}")
    print(f"  This is the coherence from shared fitness landscape alone (no coupling).")

    topo_order = ['ring', 'star', 'random', 'fully_connected']

    print(f"\n{'Topology':>18} {'d':>5} {'R(d)':>8} {'R_corr':>8} "
          f"{'% above':>8} {'Coupling':>12}")
    print("-" * 65)

    for topo in topo_order:
        if topo not in results:
            continue
        for entry in results[topo]:
            d = entry['distance']
            d_str = str(d)
            r = entry['mean_coherence']
            r_corr = r - baseline

            # Coupling signal as fraction of available range (1 - baseline)
            available = 1.0 - baseline
            coupling_pct = (r_corr / available * 100) if available > 0 else 0

            print(f"{topo:>18} {d_str:>5} {r:>8.4f} {r_corr:>8.4f} "
                  f"{coupling_pct:>7.1f}% {'strong' if coupling_pct > 40 else 'moderate' if coupling_pct > 20 else 'weak':>12}")

    # Ring gradient in corrected space
    if 'ring' in results and len(results['ring']) == 2:
        r1_corr = results['ring'][0]['mean_coherence'] - baseline
        r2_corr = results['ring'][1]['mean_coherence'] - baseline
        gradient_corr = r1_corr - r2_corr
        gradient_pct = (gradient_corr / r1_corr * 100) if r1_corr > 0 else 0

        print(f"\n  Ring corrected gradient: {gradient_corr:.4f}")
        print(f"  Ring R_corr(d=1) = {r1_corr:.4f}, R_corr(d=2) = {r2_corr:.4f}")
        print(f"  Gradient is {gradient_pct:.1f}% of the d=1 coupling signal")
        print(f"  -> Adjacent pairs are {gradient_pct:.1f}% more coupled than next-nearest")

    # Star gradient in corrected space
    if 'star' in results and len(results['star']) == 2:
        r1_corr = results['star'][0]['mean_coherence'] - baseline
        r2_corr = results['star'][1]['mean_coherence'] - baseline
        gradient_corr = r1_corr - r2_corr
        gradient_pct = (gradient_corr / r1_corr * 100) if r1_corr > 0 else 0

        print(f"\n  Star corrected gradient: {gradient_corr:.4f}")
        print(f"  Star R_corr(d=1) = {r1_corr:.4f}, R_corr(d=2) = {r2_corr:.4f}")
        print(f"  Gradient is {gradient_pct:.1f}% of the hub-spoke coupling signal")
        print(f"  -> Hub-spoke pairs are {gradient_pct:.1f}% more coupled than spoke-spoke")


def signature_analysis(results):
    """Check if observed R(d) patterns match expected signatures.

    Thresholds are calibrated for OneMax-like landscapes where all islands
    converge to similar solutions. The 'none' baseline is high (~0.85),
    so we use statistical significance and relative gradients rather than
    absolute thresholds.
    """
    print(f"\n{'='*80}")
    print("SIGNATURE ANALYSIS — Do Results Match Predictions?")
    print(f"{'='*80}")

    checks = []

    # Get baseline for relative comparisons
    baseline = results['none'][0]['mean_coherence'] if 'none' in results else 0.0

    # 1. Ring: R(d=1) > R(d=2) → traveling wave
    #    Use statistical test, not absolute threshold
    if 'ring' in results and len(results['ring']) == 2:
        ring_d1 = results['ring'][0]['all_coherences']
        ring_d2 = results['ring'][1]['all_coherences']
        _, p = stats.mannwhitneyu(ring_d1, ring_d2, alternative='greater')
        gradient = np.mean(ring_d1) - np.mean(ring_d2)
        passed = p < 0.05 and gradient > 0
        checks.append(('Ring traveling wave: R(d=1) > R(d=2) [stat. sig.]', passed,
                        f'gradient={gradient:.4f}, p={p:.2e}'))

    # 2. Star: R(d=1) > R(d=2) → hub mediates
    if 'star' in results and len(results['star']) == 2:
        star_d1 = results['star'][0]['all_coherences']
        star_d2 = results['star'][1]['all_coherences']
        _, p = stats.mannwhitneyu(star_d1, star_d2, alternative='greater')
        gradient = np.mean(star_d1) - np.mean(star_d2)
        passed = p < 0.05 and gradient > 0
        checks.append(('Star hub mediation: R(d=1) > R(d=2) [stat. sig.]', passed,
                        f'gradient={gradient:.4f}, p={p:.2e}'))

    # 3. FC: highest R(d=1) among all topologies
    if 'fully_connected' in results:
        fc_r = results['fully_connected'][0]['mean_coherence']
        other_d1 = [results[t][0]['mean_coherence']
                    for t in ['ring', 'star', 'random'] if t in results]
        passed = all(fc_r > r for r in other_d1)
        checks.append(('FC highest coherence among coupled topologies', passed,
                        f'FC R(d=1)={fc_r:.4f}, others={[f"{r:.4f}" for r in other_d1]}'))

    # 4. None: lowest R among all topologies
    if 'none' in results:
        none_r = results['none'][0]['mean_coherence']
        all_d1 = [results[t][0]['mean_coherence']
                  for t in ['ring', 'star', 'random', 'fully_connected'] if t in results]
        passed = all(none_r < r for r in all_d1)
        checks.append(('None lowest coherence (no coupling)', passed,
                        f'None R={none_r:.4f}, coupled min={min(all_d1):.4f}'))

    # 5. Ring d=1 > Ring d=2 > None (monotone decrease with distance)
    if 'ring' in results and len(results['ring']) == 2 and 'none' in results:
        r1 = results['ring'][0]['mean_coherence']
        r2 = results['ring'][1]['mean_coherence']
        r_none = results['none'][0]['mean_coherence']
        passed = r1 > r2 > r_none
        checks.append(('Ring monotone: R(d=1) > R(d=2) > R(none)', passed,
                        f'R(1)={r1:.4f} > R(2)={r2:.4f} > R(none)={r_none:.4f}'))

    # 6. Coupled topologies all exceed baseline
    for topo in ['ring', 'star', 'fully_connected']:
        if topo in results:
            r = results[topo][0]['mean_coherence']
            coupled_vals = results[topo][0]['all_coherences']
            none_vals = results['none'][0]['all_coherences'] if 'none' in results else None
            if none_vals is not None:
                _, p = stats.mannwhitneyu(coupled_vals, none_vals, alternative='greater')
                passed = p < 0.05
                checks.append((f'{topo} d=1 exceeds none baseline [stat. sig.]', passed,
                                f'R={r:.4f} vs baseline={baseline:.4f}, p={p:.2e}'))

    print()
    n_pass = sum(1 for _, p, _ in checks if p)
    for label, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")
        print(f"         {detail}")

    print(f"\n  Score: {n_pass}/{len(checks)} predictions confirmed")

    return checks


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Pairwise phase coherence R(d) by topological distance')
    parser.add_argument('--csv', type=str, default='experiment_e_per_island.csv',
                        help='Input CSV file (default: experiment_e_per_island.csv)')
    parser.add_argument('--gen', type=int, default=99,
                        help='Generation to analyze (default: 99, final)')
    parser.add_argument('--steady-state', action='store_true',
                        help='Average over last 20 generations instead of single gen')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plot generation')
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, args.csv) if not os.path.isabs(args.csv) else args.csv
    plot_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Load
    print(f"Loading: {csv_path}")
    data = load_data(csv_path)
    n_combos = sum(len(seeds) for seeds in data.values())
    print(f"Loaded {n_combos} topology-seed combos")
    for topo in ['none', 'ring', 'star', 'random', 'fully_connected']:
        if topo in data:
            n_seeds = len(data[topo])
            n_gens = len(data[topo][next(iter(data[topo]))])
            print(f"  {topo}: {n_seeds} seeds x {n_gens} generations")

    # Print pair classification for verification
    print(f"\n--- Pair classification ---")
    for topo in ['ring', 'star', 'fully_connected']:
        dists = get_pair_distances(topo)
        print(f"  {topo}:")
        for d in sorted(dists.keys()):
            print(f"    d={d}: {dists[d]}")

    # Compute R(d)
    label = f"Generation {args.gen}" if not args.steady_state else "Steady state (last 20 gens)"
    print(f"\nComputing pairwise coherence ({label})...")

    if args.steady_state:
        results = compute_pairwise_coherence(data, steady_state=True)
    else:
        results = compute_pairwise_coherence(data, generation=args.gen)

    # Print results
    print_results(results, label=label)

    # Derive output suffix from input filename
    input_basename = os.path.splitext(os.path.basename(args.csv))[0]
    if 'maze' in input_basename:
        out_suffix = '_maze'
    elif 'checkers' in input_basename:
        out_suffix = '_checkers'
    elif 'per_island' in input_basename:
        out_suffix = ''  # original OneMax
    else:
        out_suffix = f'_{input_basename}'

    # Save CSV
    csv_out = os.path.join(script_dir, f'pairwise_coherence{out_suffix}_results.csv')
    save_csv(results, csv_out)

    # Statistical tests
    statistical_tests(results)

    # Baseline-corrected analysis
    baseline_corrected_analysis(results)

    # Signature analysis
    signature_analysis(results)

    # Time series and plot
    if not args.no_plot:
        print(f"\nComputing time series...")
        ts = compute_coherence_over_time(data)
        plot_path = os.path.join(plot_dir, f'pairwise_coherence{out_suffix}.png')
        plot_results(results, ts, plot_path)

    print(f"\nDone.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Coupling onset analysis: checkers domain + cross-domain comparison.

Extends the coupling_onset_analysis.py methodology to include checkers as the
third domain. Tests whether coupling onset timing is a STRUCTURAL property
(determined by topology) or DYNAMICAL (determined by fitness landscape).

Prior result (OneMax + Maze):
  - Onset ordering: fully_connected < random < star < ring
  - Matches algebraic connectivity of the topology graph
  - Cross-domain mean difference: 0.2 generations
  - Topology explains 28.4x more variance than domain
  => STRUCTURAL

This script adds checkers (coevolutionary landscape) as a third data point.
If the structural hypothesis holds, checkers should show the SAME ordering
despite having a fundamentally different fitness landscape (relative, not
absolute).

Methodology:
  - R(t) = 1 - population_divergence(t) (Kuramoto synchronization proxy)
  - Baseline correction: delta_R(t) = R_coupled(t) - R_none(t)
  - Coupling onset = first gen where smoothed delta_R > threshold (0.05)
  - Compare onset timing across all three domains
  - ANOVA / variance decomposition: topology vs domain

Data:
  experiment_e_checkers.csv  (checkers sweep)
  experiment_e_maze.csv      (maze sweep)
  experiment_e_raw.csv       (onemax sweep)

Output:
  plots/checkers_coupling_onset.png/pdf  (3-domain comparison figure)
  Console: detailed results summary

Usage:
    python coupling_onset_checkers.py
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
#  Configuration                                                               #
# --------------------------------------------------------------------------- #

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, 'plots')

DOMAIN_CONFIGS = {
    'onemax': {
        'path': os.path.join(SCRIPT_DIR, 'experiment_e_raw.csv'),
        'label': 'OneMax',
        'color': '#2196F3',
        'marker': 'o',
    },
    'maze': {
        'path': os.path.join(SCRIPT_DIR, 'experiment_e_maze.csv'),
        'label': 'Maze',
        'color': '#FF5722',
        'marker': 's',
    },
    'checkers': {
        'path': os.path.join(SCRIPT_DIR, 'experiment_e_checkers.csv'),
        'label': 'Checkers',
        'color': '#4CAF50',
        'marker': '^',
    },
}

TOPO_ORDER = ['none', 'ring', 'star', 'random', 'fully_connected']
COUPLED_TOPOS = ['ring', 'star', 'random', 'fully_connected']

TOPO_LABELS = {
    'none': 'None', 'ring': 'Ring', 'star': 'Star',
    'random': 'Random', 'fully_connected': 'Fully\nConnected',
}
TOPO_SHORT = {
    'none': 'none', 'ring': 'ring', 'star': 'star',
    'random': 'random', 'fully_connected': 'full',
}
TOPO_COLORS = {
    'none': '#9E9E9E', 'ring': '#FF9800', 'star': '#E91E63',
    'random': '#9C27B0', 'fully_connected': '#2196F3',
}

MIGRATION_START = 5  # migration_freq = 5 => first migration at gen 5


# --------------------------------------------------------------------------- #
#  Data loading                                                                #
# --------------------------------------------------------------------------- #

def load_domain(domain_key):
    """Load data for one domain. Returns dict[topology][seed] = {...}."""
    cfg = DOMAIN_CONFIGS[domain_key]
    filepath = cfg['path']

    if not os.path.exists(filepath):
        print(f"  [SKIP] {cfg['label']}: file not found at {filepath}")
        return None

    raw = defaultdict(lambda: defaultdict(lambda: {'gen': [], 'r': [],
                                                    'fitness': [], 'div': []}))
    with open(filepath) as f:
        reader = csv.DictReader(f)
        has_pop_div = 'population_divergence' in reader.fieldnames
        for row in reader:
            topo = row['topology']
            seed = int(row['seed'])
            gen = int(row['generation'])
            r = 1.0 - float(row['population_divergence']) if has_pop_div \
                else 1.0 - float(row['hamming_diversity'])
            fitness = float(row['best_fitness'])
            diversity = float(row['hamming_diversity'])
            raw[topo][seed]['gen'].append(gen)
            raw[topo][seed]['r'].append(r)
            raw[topo][seed]['fitness'].append(fitness)
            raw[topo][seed]['div'].append(diversity)

    data = {}
    for topo in raw:
        data[topo] = {}
        for seed in raw[topo]:
            d = raw[topo][seed]
            order = np.argsort(d['gen'])
            data[topo][seed] = {
                'generations': np.array(d['gen'])[order],
                'r': np.array(d['r'])[order],
                'fitness': np.array(d['fitness'])[order],
                'diversity': np.array(d['div'])[order],
            }
    return data


def compute_none_baseline(data):
    """Mean r trajectory for 'none' topology (no migration)."""
    if 'none' not in data:
        return None
    seeds = sorted(data['none'].keys())
    r_matrix = np.array([data['none'][s]['r'] for s in seeds])
    return np.mean(r_matrix, axis=0)


# --------------------------------------------------------------------------- #
#  Coupling onset detection                                                    #
# --------------------------------------------------------------------------- #

def compute_coupling_onset(delta_r, generations, smooth_window=10, threshold=0.05):
    """Find coupling onset from baseline-corrected r curve.

    Returns (onset_gen, is_significant).
    """
    if len(delta_r) >= smooth_window:
        delta_r_smooth = uniform_filter1d(delta_r, size=smooth_window)
    else:
        delta_r_smooth = delta_r

    mask = generations >= MIGRATION_START
    if not np.any(mask):
        return np.nan, False

    post_gens = generations[mask]
    post_delta = delta_r_smooth[mask]

    above = np.where(post_delta > threshold)[0]
    if len(above) > 0:
        return int(post_gens[above[0]]), True

    below = np.where(post_delta < -threshold)[0]
    if len(below) > 0:
        return int(post_gens[below[0]]), True

    return np.nan, False


def compute_coupling_onset_multi_threshold(delta_r, generations, smooth_window=10,
                                            thresholds=(0.02, 0.05, 0.10)):
    """Compute onset at multiple thresholds."""
    result = {}
    for thresh in thresholds:
        gen, sig = compute_coupling_onset(delta_r, generations, smooth_window, thresh)
        result[thresh] = (gen, sig)
    return result


def compute_fitness_plateau(fitness_series, generations, threshold_frac=0.01):
    """Find fitness plateau generation."""
    total_range = float(np.max(fitness_series) - np.min(fitness_series))
    if total_range == 0:
        return 0, 0.0

    threshold = threshold_frac * total_range
    df = np.diff(fitness_series)
    df_gens = generations[1:]

    window = 5
    if len(df) >= window:
        df_smooth = uniform_filter1d(df, size=window)
    else:
        df_smooth = df

    mask = df_gens >= MIGRATION_START
    df_post = df_smooth[mask]
    gens_post = df_gens[mask]

    below = np.where(df_post < threshold)[0]
    if len(below) == 0:
        return int(generations[-1]), total_range

    for i in range(len(below) - 2):
        if below[i + 1] == below[i] + 1 and below[i + 2] == below[i] + 2:
            return int(gens_post[below[i]]), total_range

    return int(gens_post[below[0]]), total_range


# --------------------------------------------------------------------------- #
#  Per-domain analysis                                                         #
# --------------------------------------------------------------------------- #

def analyze_domain(domain_key, data):
    """Compute coupling onset and fitness plateau for each topology."""
    results = {}
    baseline_r = compute_none_baseline(data)

    if baseline_r is not None:
        print(f"    Baseline r (none): r(0)={baseline_r[0]:.4f} -> "
              f"r(5)={baseline_r[min(5, len(baseline_r)-1)]:.4f} -> "
              f"r(50)={baseline_r[min(50, len(baseline_r)-1)]:.4f} -> "
              f"r(-1)={baseline_r[-1]:.4f}")

    for topo in TOPO_ORDER:
        if topo not in data:
            continue

        seeds = sorted(data[topo].keys())
        onset_gens = []
        plateau_gens = []
        n_significant = 0

        for seed in seeds:
            sd = data[topo][seed]

            if baseline_r is not None and topo != 'none':
                delta_r = sd['r'] - baseline_r
            else:
                start_idx = min(MIGRATION_START, len(sd['r']) - 1)
                delta_r = sd['r'] - sd['r'][start_idx]

            onset_gen, is_sig = compute_coupling_onset(delta_r, sd['generations'])
            plateau_gen, _ = compute_fitness_plateau(sd['fitness'], sd['generations'])

            if is_sig:
                onset_gens.append(onset_gen)
                n_significant += 1
            plateau_gens.append(plateau_gen)

        # Ensemble onset
        r_matrix = np.array([data[topo][s]['r'] for s in seeds])
        mean_r = np.mean(r_matrix, axis=0)
        gens = data[topo][seeds[0]]['generations']

        if baseline_r is not None and topo != 'none':
            mean_delta_r = mean_r - baseline_r
        else:
            start_idx = min(MIGRATION_START, len(mean_r) - 1)
            mean_delta_r = mean_r - mean_r[start_idx]

        ens_onset, ens_sig = compute_coupling_onset(mean_delta_r, gens)
        ens_multi = compute_coupling_onset_multi_threshold(mean_delta_r, gens)

        onset_gens = np.array(onset_gens, dtype=float)
        plateau_gens = np.array(plateau_gens, dtype=float)
        valid_onset = onset_gens[~np.isnan(onset_gens)]
        sig_frac = n_significant / len(seeds) if seeds else 0.0

        results[topo] = {
            'onset_gens': onset_gens,
            'onset_mean': float(np.mean(valid_onset)) if len(valid_onset) > 0 else np.nan,
            'onset_std': float(np.std(valid_onset, ddof=1)) if len(valid_onset) > 1 else 0.0,
            'onset_ensemble': float(ens_onset) if not np.isnan(ens_onset) else np.nan,
            'onset_ensemble_sig': ens_sig,
            'onset_ensemble_multi': ens_multi,
            'onset_significant_frac': sig_frac,
            'plateau_gens': plateau_gens,
            'plateau_mean': float(np.mean(plateau_gens[~np.isnan(plateau_gens)])) if np.any(~np.isnan(plateau_gens)) else np.nan,
            'plateau_std': float(np.std(plateau_gens[~np.isnan(plateau_gens)], ddof=1)) if np.sum(~np.isnan(plateau_gens)) > 1 else 0.0,
            'mean_r_raw': mean_r,
            'mean_delta_r': mean_delta_r,
            'baseline_r': baseline_r,
        }

    return results


# --------------------------------------------------------------------------- #
#  Cross-domain analysis                                                       #
# --------------------------------------------------------------------------- #

def cross_domain_analysis(all_results):
    """Full cross-domain structural vs dynamical analysis."""
    domain_keys = sorted(all_results.keys())
    n_domains = len(domain_keys)

    print("\n" + "=" * 80)
    print("CROSS-DOMAIN COUPLING ONSET ANALYSIS")
    print(f"Domains: {', '.join(DOMAIN_CONFIGS[dk]['label'] for dk in domain_keys)}")
    print("=" * 80)

    # --- 1. Ensemble onset ordering ---
    print("\n--- 1. Topology ordering of ensemble coupling onset ---")
    print("   If structural: SAME ordering across all domains.")

    onset_by_domain = {}
    for dk in domain_keys:
        label = DOMAIN_CONFIGS[dk]['label']
        means = []
        for topo in COUPLED_TOPOS:
            if topo in all_results[dk] and all_results[dk][topo]['onset_ensemble_sig']:
                means.append(all_results[dk][topo]['onset_ensemble'])
            else:
                means.append(np.nan)
        onset_by_domain[dk] = means
        valid_idx = [i for i, m in enumerate(means) if not np.isnan(m)]
        ranked = sorted(valid_idx, key=lambda i: means[i])
        ordering = [COUPLED_TOPOS[i] for i in ranked]
        print(f"   {label:>10}: {' < '.join(ordering)}")
        vals_str = ', '.join(f'{TOPO_SHORT[COUPLED_TOPOS[i]]}={means[i]:.0f}' for i in ranked)
        print(f"              ensemble onset: {vals_str}")

    # Pairwise Spearman/Kendall
    if n_domains >= 2:
        print("\n   Pairwise rank correlations:")
        for i in range(n_domains):
            for j in range(i + 1, n_domains):
                dk1, dk2 = domain_keys[i], domain_keys[j]
                v1 = np.array(onset_by_domain[dk1])
                v2 = np.array(onset_by_domain[dk2])
                valid = ~(np.isnan(v1) | np.isnan(v2))
                if np.sum(valid) >= 3:
                    l1 = DOMAIN_CONFIGS[dk1]['label']
                    l2 = DOMAIN_CONFIGS[dk2]['label']
                    if np.std(v1[valid]) == 0 or np.std(v2[valid]) == 0:
                        tau, p = stats.kendalltau(v1[valid], v2[valid])
                        print(f"   {l1} vs {l2}: Kendall tau = {tau:.4f}, p = {p:.4f}")
                    else:
                        rho, p = stats.spearmanr(v1[valid], v2[valid])
                        print(f"   {l1} vs {l2}: Spearman rho = {rho:.4f}, p = {p:.4f}")

    # --- 2. Cross-domain onset difference per topology ---
    print("\n--- 2. Cross-domain onset difference per topology ---")
    print("   If structural: |diff| should be small (< 5 gens).\n")

    pairwise_diffs = []
    for topo in COUPLED_TOPOS:
        groups, labels = [], []
        for dk in domain_keys:
            if topo in all_results[dk]:
                valid = all_results[dk][topo]['onset_gens']
                valid = valid[~np.isnan(valid)]
                if len(valid) > 0:
                    groups.append(valid)
                    labels.append(DOMAIN_CONFIGS[dk]['label'])

        if len(groups) >= 2:
            # All pairwise comparisons
            for gi in range(len(groups)):
                for gj in range(gi + 1, len(groups)):
                    if len(groups[gi]) >= 3 and len(groups[gj]) >= 3:
                        u_stat, u_p = stats.mannwhitneyu(
                            groups[gi], groups[gj], alternative='two-sided')
                        diff = abs(np.mean(groups[gi]) - np.mean(groups[gj]))
                        pairwise_diffs.append(diff)
                        sig = ' ***' if u_p < 0.001 else ' **' if u_p < 0.01 else ' *' if u_p < 0.05 else ''
                        print(f"   {topo:>18}: {labels[gi]}={np.mean(groups[gi]):.1f}+/-{np.std(groups[gi], ddof=1):.1f}, "
                              f"{labels[gj]}={np.mean(groups[gj]):.1f}+/-{np.std(groups[gj], ddof=1):.1f}, "
                              f"|diff|={diff:.1f}, p={u_p:.4f}{sig}")

    if pairwise_diffs:
        print(f"\n   Mean cross-domain onset difference: {np.mean(pairwise_diffs):.1f} generations")
        print(f"   Max  cross-domain onset difference: {np.max(pairwise_diffs):.1f} generations")

    # --- 3. Variance decomposition (ANOVA-style) ---
    print("\n--- 3. Variance decomposition (coupled topologies, significant seeds) ---")
    all_data_points = []
    for di, dk in enumerate(domain_keys):
        for ti, topo in enumerate(COUPLED_TOPOS):
            if topo in all_results[dk]:
                for o in all_results[dk][topo]['onset_gens']:
                    if not np.isnan(o):
                        all_data_points.append((o, ti, di))

    if len(all_data_points) > 10:
        arr = np.array(all_data_points)
        onsets, topos, domains = arr[:, 0], arr[:, 1].astype(int), arr[:, 2].astype(int)
        grand_mean = np.mean(onsets)
        ss_total = np.sum((onsets - grand_mean) ** 2)

        ss_topo = sum(
            np.sum(topos == ti) * (np.mean(onsets[topos == ti]) - grand_mean) ** 2
            for ti in range(len(COUPLED_TOPOS)) if np.any(topos == ti)
        )
        ss_domain = sum(
            np.sum(domains == di) * (np.mean(onsets[domains == di]) - grand_mean) ** 2
            for di in range(n_domains) if np.any(domains == di)
        )
        ss_resid = ss_total - ss_topo - ss_domain

        pct = lambda x: 100 * x / ss_total if ss_total > 0 else 0

        print(f"\n   N data points: {len(all_data_points)}")
        print(f"   Topology:  {pct(ss_topo):5.1f}% of variance (SS={ss_topo:.1f})")
        print(f"   Domain:    {pct(ss_domain):5.1f}% of variance (SS={ss_domain:.1f})")
        print(f"   Residual:  {pct(ss_resid):5.1f}% of variance (SS={ss_resid:.1f})")

        if pct(ss_domain) > 0.1:
            ratio = pct(ss_topo) / pct(ss_domain)
            print(f"   Topology/Domain ratio: {ratio:.1f}x")
        else:
            print(f"   Topology/Domain ratio: >1000x (domain variance negligible)")

        if pct(ss_topo) > 3 * pct(ss_domain):
            print(f"   => STRUCTURAL: topology governs coupling onset timing.")
        elif pct(ss_domain) > 3 * pct(ss_topo):
            print(f"   => DYNAMICAL: domain governs coupling onset timing.")
        else:
            print(f"   => MIXED: both topology and domain contribute.")

        # Two-way ANOVA (if scipy has it, otherwise Kruskal-Wallis by topology)
        print("\n   Kruskal-Wallis test (onset by topology, all domains pooled):")
        groups_by_topo = []
        for ti, topo in enumerate(COUPLED_TOPOS):
            vals = onsets[topos == ti]
            if len(vals) > 0:
                groups_by_topo.append(vals)
        if len(groups_by_topo) >= 2 and all(len(g) >= 2 for g in groups_by_topo):
            h_stat, h_p = stats.kruskal(*groups_by_topo)
            print(f"   H = {h_stat:.3f}, p = {h_p:.6f}")
            if h_p < 0.001:
                print(f"   => Topology effect is highly significant (p < 0.001).")

        print("\n   Kruskal-Wallis test (onset by domain, all topologies pooled):")
        groups_by_domain = []
        for di in range(n_domains):
            vals = onsets[domains == di]
            if len(vals) > 0:
                groups_by_domain.append(vals)
        if len(groups_by_domain) >= 2 and all(len(g) >= 2 for g in groups_by_domain):
            h_stat, h_p = stats.kruskal(*groups_by_domain)
            print(f"   H = {h_stat:.3f}, p = {h_p:.6f}")
            if h_p > 0.05:
                print(f"   => Domain effect is NOT significant (p > 0.05).")
            else:
                print(f"   => Domain effect is significant (p = {h_p:.6f}).")
    else:
        print("   Not enough data points for variance decomposition.")

    # --- 4. Onset-plateau correlation ---
    print("\n--- 4. Onset vs fitness plateau correlation (within domain) ---")
    for dk in domain_keys:
        label = DOMAIN_CONFIGS[dk]['label']
        all_o, all_p = [], []
        for topo in COUPLED_TOPOS:
            if topo not in all_results[dk]:
                continue
            res = all_results[dk][topo]
            n_sig = len(res['onset_gens'])
            for o_val, p_val in zip(res['onset_gens'], res['plateau_gens'][:n_sig]):
                if not np.isnan(o_val) and not np.isnan(p_val):
                    all_o.append(o_val)
                    all_p.append(p_val)
        all_o, all_p = np.array(all_o), np.array(all_p)
        if len(all_o) >= 5:
            r_val, p_val = stats.pearsonr(all_o, all_p)
            rho, rho_p = stats.spearmanr(all_o, all_p)
            print(f"   {label:>10}: n={len(all_o)}, Pearson r={r_val:.4f} (p={p_val:.4f}), "
                  f"Spearman rho={rho:.4f} (p={rho_p:.4f})")
            if p_val < 0.05 and abs(r_val) > 0.3:
                print(f"              => onset-plateau correlated.")
            else:
                print(f"              => no onset-plateau correlation.")
        else:
            print(f"   {label:>10}: too few significant seeds (n={len(all_o)})")


# --------------------------------------------------------------------------- #
#  Summary table                                                               #
# --------------------------------------------------------------------------- #

def print_summary(all_results):
    """Print ensemble onset summary table."""
    domain_keys = sorted(all_results.keys())

    print("\n" + "=" * 80)
    print("ENSEMBLE COUPLING ONSET (baseline-corrected, seed-averaged)")
    print("=" * 80)

    header = f"{'Topology':>18}"
    for dk in domain_keys:
        header += f"  |  {DOMAIN_CONFIGS[dk]['label']+' onset':>14}"
    header += f"  |  {'max |diff|':>10}"
    print(header)
    print("-" * len(header))

    for topo in TOPO_ORDER:
        row = f"{topo:>18}"
        vals = []
        for dk in domain_keys:
            if topo in all_results[dk]:
                res = all_results[dk][topo]
                ens = res['onset_ensemble']
                sig = res['onset_ensemble_sig']
                if not np.isnan(ens) and sig:
                    row += f"  |  {ens:>14.0f}"
                    vals.append(ens)
                else:
                    row += f"  |  {'N/A':>14}"
            else:
                row += f"  |  {'N/A':>14}"
        if len(vals) >= 2:
            max_diff = max(abs(vals[i] - vals[j])
                           for i in range(len(vals))
                           for j in range(i+1, len(vals)))
            row += f"  |  {max_diff:>10.0f}"
        else:
            row += f"  |  {'--':>10}"
        print(row)

    # Multi-threshold table
    print("\n" + "=" * 80)
    print("MULTI-THRESHOLD ONSET (sensitivity analysis)")
    print("=" * 80)

    for thresh in [0.02, 0.05, 0.10]:
        print(f"\n  --- Threshold delta_r > {thresh} ---")
        header = f"  {'Topology':>18}"
        for dk in domain_keys:
            header += f"  |  {DOMAIN_CONFIGS[dk]['label']:>8}"
        header += f"  |  {'max|diff|':>9}"
        print(header)

        for topo in COUPLED_TOPOS:
            row = f"  {topo:>18}"
            vals = []
            for dk in domain_keys:
                if topo in all_results[dk]:
                    multi = all_results[dk][topo]['onset_ensemble_multi']
                    gen, sig = multi.get(thresh, (np.nan, False))
                    if sig and not np.isnan(gen):
                        row += f"  |  {gen:>8.0f}"
                        vals.append(gen)
                    else:
                        row += f"  |  {'never':>8}"
                else:
                    row += f"  |  {'N/A':>8}"
            if len(vals) >= 2:
                max_diff = max(abs(vals[i] - vals[j])
                               for i in range(len(vals))
                               for j in range(i+1, len(vals)))
                row += f"  |  {max_diff:>9.0f}"
            else:
                row += f"  |  {'--':>9}"
            print(row)

        # Per-domain ordering at this threshold
        for dk in domain_keys:
            label = DOMAIN_CONFIGS[dk]['label']
            items = []
            for topo in COUPLED_TOPOS:
                if topo in all_results[dk]:
                    multi = all_results[dk][topo]['onset_ensemble_multi']
                    gen, sig = multi.get(thresh, (np.nan, False))
                    if sig and not np.isnan(gen):
                        items.append((gen, topo))
            items.sort()
            if items:
                order_str = ' < '.join(f"{TOPO_SHORT[t]}({g:.0f})" for g, t in items)
                print(f"  {label:>10} order: {order_str}")


# --------------------------------------------------------------------------- #
#  Plotting                                                                    #
# --------------------------------------------------------------------------- #

def plot_results(all_results, output_path):
    """Generate 4-panel comparison figure."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    domain_keys = sorted(all_results.keys())
    n_domains = len(domain_keys)
    width = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Coupling Onset Timing: 3-Domain Structural Analysis\n'
        '(baseline-corrected: $\\Delta r(t) = r_{coupled}(t) - r_{none}(t)$)',
        fontsize=13, fontweight='bold',
    )

    # ---- Panel A: Ensemble coupling onset by topology ----
    ax = axes[0, 0]
    ax.set_title('A. Ensemble Coupling Onset by Topology')
    x = np.arange(len(COUPLED_TOPOS))
    offsets = np.linspace(-width, width, n_domains)

    for di, dk in enumerate(domain_keys):
        cfg = DOMAIN_CONFIGS[dk]
        vals, errs = [], []
        for topo in COUPLED_TOPOS:
            if topo in all_results[dk] and all_results[dk][topo]['onset_ensemble_sig']:
                vals.append(all_results[dk][topo]['onset_ensemble'])
                errs.append(all_results[dk][topo]['onset_std']
                            if len(all_results[dk][topo]['onset_gens']) > 1 else 0)
            else:
                vals.append(np.nan)
                errs.append(0)
        ax.errorbar(x + offsets[di], vals, yerr=errs, fmt=cfg['marker'],
                    color=cfg['color'], label=cfg['label'], capsize=4, markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([TOPO_LABELS[t] for t in COUPLED_TOPOS], fontsize=9)
    ax.set_ylabel('Coupling Onset Generation')
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.axhline(y=MIGRATION_START, color='gray', ls='--', alpha=0.4)

    # ---- Panel B: Baseline-corrected mean delta_r curves ----
    ax = axes[0, 1]
    ax.set_title('B. Baseline-Corrected $\\Delta r(t)$ (ensemble)')

    linestyles = {domain_keys[i]: ['-', '--', '-.'][i] for i in range(n_domains)}

    for dk in domain_keys:
        ls = linestyles[dk]
        for topo in COUPLED_TOPOS:
            if topo in all_results[dk]:
                delta_r = all_results[dk][topo]['mean_delta_r']
                gens = np.arange(len(delta_r))
                if len(delta_r) >= 10:
                    delta_r_smooth = uniform_filter1d(delta_r, size=10)
                else:
                    delta_r_smooth = delta_r
                ax.plot(gens, delta_r_smooth, color=TOPO_COLORS[topo],
                        ls=ls, linewidth=1.5, alpha=0.8)

    ax.axhline(y=0, color='black', ls='-', alpha=0.3, linewidth=0.5)
    ax.axhline(y=0.02, color='gray', ls=':', alpha=0.5, linewidth=1)
    ax.axhline(y=-0.02, color='gray', ls=':', alpha=0.5, linewidth=1)
    ax.axvline(x=MIGRATION_START, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Generation')
    ax.set_ylabel('$\\Delta r$ (migration effect)')
    ax.set_xlim(0, 99)

    handles = [Line2D([0], [0], color=TOPO_COLORS[t], lw=2, label=t) for t in COUPLED_TOPOS]
    for dk in domain_keys:
        handles.append(Line2D([0], [0], color='black', ls=linestyles[dk], lw=1.5,
                              label=DOMAIN_CONFIGS[dk]['label']))
    ax.legend(handles=handles, fontsize=7, ncol=2)

    # ---- Panel C: Variance decomposition bar chart ----
    ax = axes[1, 0]
    ax.set_title('C. Variance Decomposition')

    all_data_points = []
    for di, dk in enumerate(domain_keys):
        for ti, topo in enumerate(COUPLED_TOPOS):
            if topo in all_results[dk]:
                for o in all_results[dk][topo]['onset_gens']:
                    if not np.isnan(o):
                        all_data_points.append((o, ti, di))

    if len(all_data_points) > 10:
        arr = np.array(all_data_points)
        onsets, topos_arr, domains_arr = arr[:, 0], arr[:, 1].astype(int), arr[:, 2].astype(int)
        grand_mean = np.mean(onsets)
        ss_total = np.sum((onsets - grand_mean) ** 2)
        ss_topo = sum(np.sum(topos_arr == ti) * (np.mean(onsets[topos_arr == ti]) - grand_mean) ** 2
                      for ti in range(len(COUPLED_TOPOS)) if np.any(topos_arr == ti))
        ss_domain = sum(np.sum(domains_arr == di) * (np.mean(onsets[domains_arr == di]) - grand_mean) ** 2
                        for di in range(n_domains) if np.any(domains_arr == di))
        ss_resid = ss_total - ss_topo - ss_domain

        pcts = [100 * ss_topo / ss_total, 100 * ss_domain / ss_total,
                100 * ss_resid / ss_total]
        labels_bar = ['Topology', 'Domain', 'Residual']
        colors_bar = ['#2196F3', '#FF5722', '#9E9E9E']

        bars = ax.bar(labels_bar, pcts, color=colors_bar, edgecolor='white', linewidth=0.8)
        for bar_obj, pct_val in zip(bars, pcts):
            ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                    bar_obj.get_height() + 1,
                    f'{pct_val:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('% of total variance')
        ax.set_ylim(0, max(pcts) * 1.2)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='0.5')

    # ---- Panel D: Raw mean r(t) curves ----
    ax = axes[1, 1]
    ax.set_title('D. Raw Mean $r(t)$ Trajectories')

    for dk in domain_keys:
        ls = linestyles[dk]
        for topo in TOPO_ORDER:
            if topo in all_results[dk]:
                mean_r = all_results[dk][topo]['mean_r_raw']
                gens = np.arange(len(mean_r))
                ax.plot(gens, mean_r, color=TOPO_COLORS[topo],
                        ls=ls, linewidth=1.5, alpha=0.7)

    ax.axvline(x=MIGRATION_START, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Generation')
    ax.set_ylabel('$r = 1 - $ pop. divergence')
    ax.set_xlim(0, 99)
    ax.set_ylim(0.4, 1.0)

    handles = [Line2D([0], [0], color=TOPO_COLORS[t], lw=2, label=t) for t in TOPO_ORDER]
    for dk in domain_keys:
        handles.append(Line2D([0], [0], color='black', ls=linestyles[dk], lw=1.5,
                              label=DOMAIN_CONFIGS[dk]['label']))
    ax.legend(handles=handles, fontsize=7, ncol=2, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"\nPlots saved:")
    print(f"  {output_path}")
    print(f"  {pdf_path}")


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    print("=" * 80)
    print("COUPLING ONSET TIMING: 3-DOMAIN ANALYSIS")
    print("Testing structural (topology) vs dynamical (landscape) hypothesis")
    print("Domains: OneMax + Maze + Checkers")
    print("=" * 80)

    all_data = {}
    all_results = {}

    for dk in DOMAIN_CONFIGS:
        cfg = DOMAIN_CONFIGS[dk]
        print(f"\nLoading {cfg['label']}...")
        data = load_domain(dk)
        if data is not None:
            all_data[dk] = data
            n_topos = len(data)
            n_seeds = sum(len(seeds) for seeds in data.values())
            print(f"  Loaded: {n_topos} topologies, {n_seeds} topology-seed combos")
            print(f"  Analyzing {cfg['label']}...")
            results = analyze_domain(dk, data)
            all_results[dk] = results

    if not all_results:
        print("\nERROR: No data loaded. Check file paths.")
        sys.exit(1)

    if 'checkers' not in all_results:
        print("\n[WAITING] Checkers data not available yet.")
        print("  The topology sweep is still running.")
        print("  Re-run this script when experiment_e_checkers.csv lands.")
        if len(all_results) >= 2:
            print("\n  Running analysis with available domains...\n")
        else:
            print("  Only one domain available. Exiting.")
            sys.exit(0)

    # Summary tables
    print_summary(all_results)

    # Cross-domain analysis
    if len(all_results) >= 2:
        cross_domain_analysis(all_results)

    # Plot
    os.makedirs(PLOT_DIR, exist_ok=True)
    output_path = os.path.join(PLOT_DIR, 'checkers_coupling_onset.png')
    print(f"\nGenerating plots...")
    plot_results(all_results, output_path)

    # ---- VERDICT ----
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    domain_keys = sorted(all_results.keys())

    if len(domain_keys) >= 2:
        print(f"\n  Domains analyzed: {', '.join(DOMAIN_CONFIGS[dk]['label'] for dk in domain_keys)}")
        print(f"\n  Ensemble onset (baseline-corrected), coupled topologies:\n")

        onset_diffs = []
        for topo in COUPLED_TOPOS:
            means = []
            dks_with = []
            for dk in domain_keys:
                if topo in all_results[dk] and all_results[dk][topo]['onset_ensemble_sig']:
                    means.append(all_results[dk][topo]['onset_ensemble'])
                    dks_with.append(dk)
            if len(means) >= 2:
                max_diff = max(abs(means[i] - means[j])
                               for i in range(len(means))
                               for j in range(i+1, len(means)))
                onset_diffs.append(max_diff)
                vals_str = ', '.join(f"{DOMAIN_CONFIGS[dk]['label']}={m:.0f}"
                                     for dk, m in zip(dks_with, means))
                print(f"    {topo:>18}: {vals_str}, max|diff|={max_diff:.0f}")

        if onset_diffs:
            mean_diff = np.mean(onset_diffs)
            max_diff = np.max(onset_diffs)
            print(f"\n  Mean max cross-domain onset difference: {mean_diff:.1f} generations")
            print(f"  Max  max cross-domain onset difference: {max_diff:.1f} generations")

            if mean_diff < 5:
                print(f"\n  => STRUCTURAL: Coupling onset is invariant across domains.")
                print(f"     Topology governs when migration-induced synchronization begins.")
                print(f"     Confirmed across {len(domain_keys)} domains including coevolutionary (checkers)."
                      if 'checkers' in all_results else
                      f"     Awaiting checkers (coevolutionary) domain for third confirmation.")
            elif mean_diff < 15:
                print(f"\n  => MIXED: Both topology and landscape influence coupling onset.")
            else:
                print(f"\n  => DYNAMICAL: Coupling onset differs substantially across domains.")
        else:
            print("  No topologies with significant onset in >= 2 domains.")
    else:
        print(f"\n  Only 1 domain loaded ({DOMAIN_CONFIGS[domain_keys[0]]['label']}).")
        print("  Need >= 2 domains for cross-domain comparison.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Balduzzi/Hodge decomposition of game payoff matrices.

Decomposes an antisymmetric payoff matrix A into:
  A = T + C
where:
  T = transitive component (gradient game, admits total ordering by skill)
  C = cyclic component (rock-paper-scissors-like intransitivity)

This is the Helmholtz/Hodge decomposition on the complete tournament graph,
following Balduzzi et al. (2019) "Open-ended Learning in Symmetric Zero-sum
Games" and Candogan et al. (2011) "Flows and Decompositions of Games."

Key quantity: the cyclic ratio ||C||_F / ||A||_F
  = 0: purely transitive (Elo ratings perfectly predict outcomes)
  = 1: purely cyclic (no consistent ranking exists)
  Between: mixed game with both skill hierarchy and intransitive cycles

Application: Checkers is known to have intransitive strategies. Different
migration topologies may preserve or suppress this intransitivity. The cyclic
ratio measures how much "rock-paper-scissors structure" survives evolution
under each topology.

Data format (expected):
  .npz files with keys:
    'payoff_matrices': shape (n_topologies, n_seeds, n_generations, n_strategies, n_strategies)
    'topologies': list of topology names
    'seeds': list of seed values
    'generations': list of generation numbers
  OR per-topology .npz files with:
    'payoff_matrix': shape (n_seeds, n_generations, n_strategies, n_strategies)
    'seeds': list of seed values
    'generations': list of generation numbers

Usage:
    python balduzzi_decomposition.py                    # synthetic test only
    python balduzzi_decomposition.py --data FILE.npz    # analyze real data
    python balduzzi_decomposition.py --test             # run synthetic tests
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Configuration — matches project style
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)

TOPOLOGY_ORDER = ["none", "ring", "star", "random", "fully_connected"]

TOPOLOGY_LABELS = {
    "none": "None\n(isolated)",
    "ring": "Ring",
    "star": "Star",
    "random": "Random",
    "fully_connected": "Fully\nconnected",
}

TOPOLOGY_LABELS_SHORT = {
    "none": "None (isolated)",
    "ring": "Ring",
    "star": "Star",
    "random": "Random",
    "fully_connected": "Fully connected",
}

COLORS = {
    "none": "#2166AC",
    "ring": "#67A9CF",
    "star": "#5AB4AC",
    "random": "#F4A582",
    "fully_connected": "#B2182B",
}


# ---------------------------------------------------------------------------
# Style — matches plot_checkers.py / plot_fingerprints.py
# ---------------------------------------------------------------------------

def setup_style():
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })


# ===========================================================================
# Core math: Hodge decomposition
# ===========================================================================

def hodge_decompose(A):
    """
    Hodge decomposition of an antisymmetric payoff matrix.

    Given an n x n antisymmetric matrix A (A_ij = -A_ji, A_ii = 0), decompose:
        A = T + C
    where T is the transitive (gradient) component and C is the cyclic
    (curl/intransitive) component.

    This implements the combinatorial Hodge decomposition on the edge space of
    the complete graph K_n, following Candogan et al. (2011) and Balduzzi et al.
    (2019). The key insight is that for a complete tournament graph:

      1. The divergence operator maps edge flows to vertex values:
         div(A)_i = sum_j A_ij  (net advantage of player i)

      2. The transitive component is the projection of A onto the image of the
         co-boundary operator (gradient flows). For the complete graph K_n:
             T_ij = (div(A)_i - div(A)_j) / n
         This is the unique gradient flow that has the same divergence as A.

      3. The cyclic component is the residual:
             C = A - T
         which is guaranteed to be divergence-free: sum_j C_ij = 0 for all i.

    The T_ij formula comes from solving the Poisson equation L*s = div(A) on the
    complete graph, where L = nI - 11^T. The pseudoinverse gives s_i = div(A)_i / n,
    and T_ij = s_i - s_j = (div_i - div_j) / n.

    Note on sign convention: A_ij > 0 means player i has advantage over player j.
    The "rating" s_i = div(A)_i / n = (1/n) * sum_j A_ij is player i's average
    advantage. T_ij = s_i - s_j captures the transitive skill gap.

    Args:
        A: (n, n) antisymmetric payoff matrix. A_ij > 0 means player i beats j.

    Returns:
        dict with keys:
            'T': (n, n) transitive component
            'C': (n, n) cyclic component
            's': (n,) optimal ratings (skill vector, s_i = div_i / n)
            'cyclic_ratio': float, ||C||_F / ||A||_F (0 = transitive, 1 = cyclic)
            'div_A': (n,) divergence vector
    """
    n = A.shape[0]
    assert A.shape == (n, n), f"Expected square matrix, got {A.shape}"

    # Verify antisymmetry (within numerical tolerance)
    antisym_err = np.max(np.abs(A + A.T))
    if antisym_err > 1e-10:
        # Force antisymmetry
        A = (A - A.T) / 2.0

    # Step 1: Divergence (net outflow per node)
    # div(A)_i = sum_j A_ij = total advantage of player i
    div_A = np.sum(A, axis=1)

    # Step 2: Ratings from divergence
    # For complete graph K_n, the pseudoinverse solution gives s_i = div_i / n.
    # This is the minimum-norm solution to L*s = div(A) where L = nI - 11^T.
    s = div_A / n

    # Step 3: Transitive component
    # T_ij = s_i - s_j = (div_i - div_j) / n
    # This is the gradient flow: the part of A explainable by skill ratings.
    T = s[:, np.newaxis] - s[np.newaxis, :]

    # Step 4: Cyclic component (divergence-free residual)
    C = A - T

    # Cyclic ratio: ||C||_F / ||A||_F
    norm_A = np.linalg.norm(A, 'fro')
    norm_C = np.linalg.norm(C, 'fro')
    if norm_A > 1e-15:
        cyclic_ratio = norm_C / norm_A
    else:
        cyclic_ratio = 0.0  # zero matrix

    return {
        'T': T,
        'C': C,
        's': s,
        'cyclic_ratio': float(cyclic_ratio),
        'div_A': div_A,
    }


# ===========================================================================
# Synthetic test games
# ===========================================================================

def generate_synthetic_test():
    """
    Create synthetic payoff matrices to verify the decomposition.

    Returns dict of test cases, each with:
        'A': payoff matrix
        'description': string
        'expected_cyclic_ratio': approximate expected value
        'tolerance': acceptable deviation from expected
    """
    tests = {}

    # --- Test 1: Purely transitive (Elo-like) ---
    # Players have ratings [3, 1, -1, -3]. A_ij = s_i - s_j.
    # This is a pure gradient game: cyclic ratio should be 0.
    n = 4
    ratings = np.array([3.0, 1.0, -1.0, -3.0])
    A_trans = ratings[:, np.newaxis] - ratings[np.newaxis, :]
    # Zero diagonal (no self-play)
    np.fill_diagonal(A_trans, 0.0)
    tests['purely_transitive'] = {
        'A': A_trans,
        'description': 'Purely transitive game (Elo ratings: [3, 1, -1, -3])',
        'expected_cyclic_ratio': 0.0,
        'tolerance': 1e-10,
    }

    # --- Test 2: Purely cyclic (extended rock-paper-scissors) ---
    # 4-player RPS cycle: 0 beats 1, 1 beats 2, 2 beats 3, 3 beats 0.
    # Each win = +1, loss = -1. Non-adjacent pairs = 0.
    # This is a pure curl game: all divergences are zero, cyclic ratio = 1.
    A_cyclic = np.zeros((4, 4))
    # Cycle: 0->1->2->3->0
    A_cyclic[0, 1] = 1;  A_cyclic[1, 0] = -1
    A_cyclic[1, 2] = 1;  A_cyclic[2, 1] = -1
    A_cyclic[2, 3] = 1;  A_cyclic[3, 2] = -1
    A_cyclic[3, 0] = 1;  A_cyclic[0, 3] = -1
    tests['purely_cyclic'] = {
        'A': A_cyclic,
        'description': 'Purely cyclic game (4-player RPS cycle: 0->1->2->3->0)',
        'expected_cyclic_ratio': 1.0,
        'tolerance': 1e-10,
    }

    # --- Test 3: Standard 3-player RPS ---
    # Classic rock-paper-scissors: each player beats one, loses to another.
    # All divergences = 0 (each player wins 1, loses 1). Pure cycle.
    A_rps3 = np.array([
        [ 0,  1, -1],
        [-1,  0,  1],
        [ 1, -1,  0],
    ], dtype=float)
    tests['rps_3player'] = {
        'A': A_rps3,
        'description': 'Classic 3-player RPS (0=rock, 1=paper, 2=scissors)',
        'expected_cyclic_ratio': 1.0,
        'tolerance': 1e-10,
    }

    # --- Test 4: Mixed game (transitive + cyclic) ---
    # Start with transitive ratings, add cyclic perturbation.
    # Ratings: [2, 0, -2] (transitive base).
    # Add RPS cycle with magnitude 1.
    n = 3
    ratings_3 = np.array([2.0, 0.0, -2.0])
    A_base = ratings_3[:, np.newaxis] - ratings_3[np.newaxis, :]
    np.fill_diagonal(A_base, 0.0)
    A_cycle = np.array([
        [ 0,  1, -1],
        [-1,  0,  1],
        [ 1, -1,  0],
    ], dtype=float)
    A_mixed = A_base + A_cycle
    # The Hodge decomposition should recover T = A_base and C = A_cycle,
    # since A_base is a pure gradient flow (T_ij = s_i - s_j) and A_cycle
    # is divergence-free (each row sums to 0 in RPS).
    # Verify: div(A_cycle)_0 = 1 + (-1) = 0, div(A_cycle)_1 = (-1) + 1 = 0, etc.
    # So the decomposition should perfectly separate them.
    norm_cycle = np.linalg.norm(A_cycle, 'fro')
    norm_mixed = np.linalg.norm(A_mixed, 'fro')
    expected_cr = norm_cycle / norm_mixed
    tests['mixed_game'] = {
        'A': A_mixed,
        'description': (f'Mixed game: transitive [2,0,-2] + RPS cycle (magnitude 1). '
                        f'Expected cyclic ratio ~ {expected_cr:.4f}'),
        'expected_cyclic_ratio': expected_cr,
        'tolerance': 1e-10,
    }

    # --- Test 5: Large random mixed game (10 players) ---
    # Random ratings + random cyclic component. Verifies scaling.
    rng = np.random.RandomState(42)
    n = 10
    ratings_10 = rng.randn(n) * 2
    A_trans_10 = ratings_10[:, np.newaxis] - ratings_10[np.newaxis, :]
    np.fill_diagonal(A_trans_10, 0.0)
    # Random antisymmetric cyclic perturbation
    noise = rng.randn(n, n)
    noise = (noise - noise.T) / 2.0
    np.fill_diagonal(noise, 0.0)
    A_large = A_trans_10 + 0.5 * noise
    tests['large_mixed'] = {
        'A': A_large,
        'description': '10-player game: random ratings + random cyclic noise (magnitude 0.5)',
        'expected_cyclic_ratio': None,  # unknown, just check 0 < ratio < 1
        'tolerance': None,
    }

    # --- Test 6: Zero matrix ---
    A_zero = np.zeros((4, 4))
    tests['zero_matrix'] = {
        'A': A_zero,
        'description': 'Zero matrix (no games played)',
        'expected_cyclic_ratio': 0.0,
        'tolerance': 1e-10,
    }

    # --- Test 7: Verify decomposition properties ---
    # A = T + C should hold exactly.
    # T should be antisymmetric and transitive (T_ij = s_j - s_i).
    # C should be antisymmetric and divergence-free (sum_j C_ij = 0 for all i).
    # T and C should be orthogonal in Frobenius inner product.
    # (These are checked in run_synthetic_tests, not here.)

    return tests


def run_synthetic_tests():
    """
    Run all synthetic tests and verify decomposition correctness.

    Checks:
      1. Cyclic ratio matches expected value (within tolerance)
      2. A = T + C (exact reconstruction)
      3. T is antisymmetric
      4. C is antisymmetric
      5. C is divergence-free: sum_j C_ij = 0 for all i
      6. T and C are orthogonal: trace(T^T @ C) = 0
      7. Recovered ratings match input ratings (up to additive constant)
    """
    tests = generate_synthetic_test()
    all_passed = True
    results = {}

    print("=" * 80)
    print("SYNTHETIC TEST SUITE: Hodge Decomposition Verification")
    print("=" * 80)

    for name, test in tests.items():
        A = test['A']
        desc = test['description']
        expected = test['expected_cyclic_ratio']
        tol = test['tolerance']

        print(f"\n--- Test: {name} ---")
        print(f"  {desc}")
        print(f"  Matrix size: {A.shape[0]}x{A.shape[0]}")

        result = hodge_decompose(A)
        T, C, s = result['T'], result['C'], result['s']
        cr = result['cyclic_ratio']

        print(f"  Ratings (s): {np.array2string(s, precision=4, suppress_small=True)}")
        print(f"  Cyclic ratio: {cr:.10f}")
        if expected is not None:
            print(f"  Expected:     {expected:.10f}")

        passed = True
        checks = []

        # Check 1: Cyclic ratio
        if expected is not None and tol is not None:
            err = abs(cr - expected)
            ok = err < tol
            checks.append(('Cyclic ratio', ok, f'error = {err:.2e}'))
            if not ok:
                passed = False

        # Check 2: Reconstruction A = T + C
        recon_err = np.max(np.abs(A - T - C))
        ok = recon_err < 1e-12
        checks.append(('Reconstruction A=T+C', ok, f'max error = {recon_err:.2e}'))
        if not ok:
            passed = False

        # Check 3: T antisymmetric
        t_antisym = np.max(np.abs(T + T.T))
        ok = t_antisym < 1e-12
        checks.append(('T antisymmetric', ok, f'max |T+T^T| = {t_antisym:.2e}'))
        if not ok:
            passed = False

        # Check 4: C antisymmetric
        c_antisym = np.max(np.abs(C + C.T))
        ok = c_antisym < 1e-12
        checks.append(('C antisymmetric', ok, f'max |C+C^T| = {c_antisym:.2e}'))
        if not ok:
            passed = False

        # Check 5: C divergence-free
        c_div = np.abs(np.sum(C, axis=1))
        c_div_max = np.max(c_div)
        ok = c_div_max < 1e-10
        checks.append(('C divergence-free', ok, f'max |div(C)| = {c_div_max:.2e}'))
        if not ok:
            passed = False

        # Check 6: Orthogonality <T, C>_F = 0
        inner = np.sum(T * C)  # Frobenius inner product
        ok = abs(inner) < 1e-10
        checks.append(('T perp C', ok, f'<T,C>_F = {inner:.2e}'))
        if not ok:
            passed = False

        # Check 7: For purely transitive test, verify ratings recovery
        if name == 'purely_transitive':
            input_ratings = np.array([3.0, 1.0, -1.0, -3.0])
            # Recovered ratings should match up to additive constant
            s_centered = s - np.mean(s)
            r_centered = input_ratings - np.mean(input_ratings)
            rating_err = np.max(np.abs(s_centered - r_centered))
            ok = rating_err < 1e-10
            checks.append(('Ratings recovery', ok, f'max error = {rating_err:.2e}'))
            if not ok:
                passed = False

        # Print check results
        for check_name, ok, detail in checks:
            status = 'PASS' if ok else 'FAIL'
            print(f"  [{status}] {check_name}: {detail}")

        if not passed:
            all_passed = False

        results[name] = {
            'passed': passed,
            'cyclic_ratio': cr,
            'result': result,
        }

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    n_passed = sum(1 for r in results.values() if r['passed'])
    n_total = len(results)
    print(f"\n  {n_passed}/{n_total} tests passed")
    print(f"\n  Cyclic ratios:")
    for name, r in results.items():
        expected = tests[name]['expected_cyclic_ratio']
        exp_str = f" (expected {expected:.4f})" if expected is not None else ""
        print(f"    {name:>25}: {r['cyclic_ratio']:.10f}{exp_str}")

    if all_passed:
        print(f"\n  ALL TESTS PASSED. Decomposition is mathematically correct.")
    else:
        print(f"\n  SOME TESTS FAILED. Check implementation.")

    return all_passed, results


# ===========================================================================
# Data loading
# ===========================================================================

def load_payoff_data(filepath):
    """
    Load payoff matrix data from .npz file.

    Expected format:
        'payoff_matrices': (n_topologies, n_seeds, n_generations, n, n) or
                           (n_seeds, n_generations, n, n)  [single topology]
        'topologies': array of topology name strings
        'seeds': array of seed integers
        'generations': array of generation integers

    Returns:
        dict: {topology_name: {'matrices': (n_seeds, n_gens, n, n),
                                'seeds': [...], 'generations': [...]}}
    """
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return None

    data = np.load(filepath, allow_pickle=True)
    payoffs = data['payoff_matrices']
    seeds = data['seeds'] if 'seeds' in data else np.arange(payoffs.shape[-4])
    generations = data['generations'] if 'generations' in data else np.arange(payoffs.shape[-3])

    result = {}

    if 'topologies' in data:
        topologies = list(data['topologies'])
        # Multi-topology format: (n_topo, n_seeds, n_gens, n, n)
        for i, topo in enumerate(topologies):
            result[topo] = {
                'matrices': payoffs[i],  # (n_seeds, n_gens, n, n)
                'seeds': seeds,
                'generations': generations,
            }
    else:
        # Single topology or flat format: (n_seeds, n_gens, n, n)
        topo_name = os.path.basename(filepath).replace('.npz', '')
        result[topo_name] = {
            'matrices': payoffs,
            'seeds': seeds,
            'generations': generations,
        }

    return result


# ===========================================================================
# Analysis pipeline
# ===========================================================================

def analyze_payoff_data(data):
    """
    Apply Hodge decomposition to all payoff matrices.

    Args:
        data: dict from load_payoff_data()

    Returns:
        dict: {topology: {
            'cyclic_ratios': (n_seeds, n_gens) array,
            'rating_spreads': (n_seeds, n_gens) array,
            'mean_cyclic_ratio': (n_gens,) array,
            'std_cyclic_ratio': (n_gens,) array,
            'final_cyclic_ratio': float (mean across seeds at last gen),
            'seeds': array,
            'generations': array,
        }}
    """
    results = {}

    for topo, topo_data in data.items():
        matrices = topo_data['matrices']  # (n_seeds, n_gens, n, n)
        seeds = topo_data['seeds']
        generations = topo_data['generations']
        n_seeds, n_gens = matrices.shape[0], matrices.shape[1]

        print(f"  Decomposing {topo}: {n_seeds} seeds x {n_gens} generations "
              f"({matrices.shape[-1]} strategies)...")

        cyclic_ratios = np.zeros((n_seeds, n_gens))
        rating_spreads = np.zeros((n_seeds, n_gens))

        for si in range(n_seeds):
            for gi in range(n_gens):
                A = matrices[si, gi]
                # Force antisymmetry
                A = (A - A.T) / 2.0
                np.fill_diagonal(A, 0.0)

                result = hodge_decompose(A)
                cyclic_ratios[si, gi] = result['cyclic_ratio']
                rating_spreads[si, gi] = np.std(result['s'])

        mean_cr = np.mean(cyclic_ratios, axis=0)
        std_cr = np.std(cyclic_ratios, axis=0, ddof=1) if n_seeds > 1 else np.zeros(n_gens)
        final_cr = float(np.mean(cyclic_ratios[:, -1]))
        final_cr_std = float(np.std(cyclic_ratios[:, -1], ddof=1)) if n_seeds > 1 else 0.0

        results[topo] = {
            'cyclic_ratios': cyclic_ratios,
            'rating_spreads': rating_spreads,
            'mean_cyclic_ratio': mean_cr,
            'std_cyclic_ratio': std_cr,
            'final_cyclic_ratio': final_cr,
            'final_cyclic_ratio_std': final_cr_std,
            'mean_rating_spread': np.mean(rating_spreads, axis=0),
            'std_rating_spread': np.std(rating_spreads, axis=0, ddof=1) if n_seeds > 1 else np.zeros(n_gens),
            'seeds': seeds,
            'generations': generations,
        }

    return results


# ===========================================================================
# Visualization
# ===========================================================================

def plot_cyclic_ratio_by_topology(results, outdir):
    """
    Bar chart: final-generation cyclic ratio per topology.

    Shows which topologies preserve vs suppress intransitivity.
    Hypothesis: tighter coupling (fully connected) may suppress cycles
    by forcing consensus, while loose coupling (none, ring) preserves them.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    topos = [t for t in TOPOLOGY_ORDER if t in results]
    x = np.arange(len(topos))
    means = [results[t]['final_cyclic_ratio'] for t in topos]
    stds = [results[t]['final_cyclic_ratio_std'] for t in topos]
    bar_colors = [COLORS[t] for t in topos]

    bars = ax.bar(
        x, means, yerr=stds, capsize=4,
        color=bar_colors, edgecolor='white', linewidth=0.8, width=0.65,
        error_kw={'linewidth': 1.2, 'color': '0.3'},
    )

    # Value labels above bars
    for bar_obj, mean_val, std_val in zip(bars, means, stds):
        ax.text(
            bar_obj.get_x() + bar_obj.get_width() / 2,
            bar_obj.get_height() + std_val + 0.01,
            f'{mean_val:.3f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold', color='0.3',
        )

    ax.set_xticks(x)
    ax.set_xticklabels([TOPOLOGY_LABELS[t] for t in topos], fontsize=9)
    ax.set_ylabel('Cyclic ratio $\\|C\\|_F / \\|A\\|_F$')
    ax.set_title('Intransitivity by migration topology',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, min(1.15, max(means) + max(stds) + 0.08))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    # Reference lines
    ax.axhline(y=0, color='0.7', linewidth=0.5, linestyle='-')
    ax.axhline(y=1, color='0.7', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.text(len(topos) - 0.5, 1.02, 'purely cyclic', fontsize=7, color='0.5',
            ha='right', va='bottom')
    ax.text(len(topos) - 0.5, 0.02, 'purely transitive', fontsize=7, color='0.5',
            ha='right', va='bottom')

    # Strict -> Lax annotation
    ax.annotate(
        '', xy=(len(topos) - 0.7, -0.016), xytext=(-0.3, -0.016),
        xycoords=('data', 'axes fraction'),
        textcoords=('data', 'axes fraction'),
        arrowprops=dict(arrowstyle='->', color='0.5', linewidth=1.0),
        annotation_clip=False,
    )
    ax.text(len(topos) / 2 - 0.5, -0.08,
            'strict                                                       lax',
            transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=8, color='0.5', style='italic')

    output_path = Path(outdir) / "checkers_cyclic_ratio_by_topology.png"
    for fmt in ('png', 'pdf'):
        fig.savefig(str(output_path).replace('.png', f'.{fmt}'))
    plt.close(fig)
    print(f"  Saved: {output_path.name}")
    return output_path


def plot_cyclic_evolution(results, outdir):
    """
    Line plot: cyclic ratio over generations per topology.

    Shows how intransitivity evolves during the evolutionary process.
    Key questions:
      - Does migration suppress cycles over time?
      - Do different topologies have different suppression rates?
      - Is there a phase transition in cyclic ratio?
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))

    topos = [t for t in TOPOLOGY_ORDER if t in results]

    for topo in topos:
        r = results[topo]
        gens = r['generations']
        mean_cr = r['mean_cyclic_ratio']
        std_cr = r['std_cyclic_ratio']

        ax.fill_between(gens, mean_cr - std_cr, mean_cr + std_cr,
                        color=COLORS[topo], alpha=0.12)
        ax.plot(gens, mean_cr, color=COLORS[topo], linewidth=2.0,
                label=TOPOLOGY_LABELS_SHORT[topo])

    ax.set_xlim(0, max(r['generations'][-1] for r in results.values()))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Cyclic ratio $\\|C\\|_F / \\|A\\|_F$')
    ax.set_title('Evolution of intransitivity by topology',
                 fontsize=12, fontweight='bold')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.legend(frameon=True, framealpha=0.9, edgecolor='0.8', loc='upper right')

    # Migration start line
    ax.axvline(x=5, color='0.7', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.text(5.5, 0.02, 'migration\nstart', fontsize=7, color='0.5', va='bottom')

    output_path = Path(outdir) / "checkers_cyclic_evolution.png"
    for fmt in ('png', 'pdf'):
        fig.savefig(str(output_path).replace('.png', f'.{fmt}'))
    plt.close(fig)
    print(f"  Saved: {output_path.name}")
    return output_path


def plot_rating_vs_cyclic(results, outdir):
    """
    Scatter: transitive rating spread vs cyclic ratio per topology.

    Shows the trade-off between skill hierarchy and intransitivity.
    If tighter coupling increases rating spread (stronger hierarchy)
    while decreasing cyclic ratio, that's evidence for topology-mediated
    simplification of the strategic landscape.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    topos = [t for t in TOPOLOGY_ORDER if t in results]

    for topo in topos:
        r = results[topo]
        # Use final-generation values per seed
        cr = r['cyclic_ratios'][:, -1]     # (n_seeds,)
        rs = r['rating_spreads'][:, -1]    # (n_seeds,)

        ax.scatter(rs, cr, color=COLORS[topo], s=30, alpha=0.6,
                   label=TOPOLOGY_LABELS_SHORT[topo], edgecolors='white',
                   linewidth=0.5, zorder=3)

        # Mean marker
        ax.scatter(np.mean(rs), np.mean(cr), color=COLORS[topo], s=120,
                   marker='D', edgecolors='black', linewidth=1.0, zorder=4)

    ax.set_xlabel('Rating spread (std of $s$)')
    ax.set_ylabel('Cyclic ratio $\\|C\\|_F / \\|A\\|_F$')
    ax.set_title('Skill hierarchy vs intransitivity',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, framealpha=0.9, edgecolor='0.8',
              loc='upper right', fontsize=8)

    # Quadrant annotations
    xlim = ax.get_xlim()
    ax.text(xlim[1] * 0.95, 0.95, 'high cycle\nhigh hierarchy',
            ha='right', va='top', fontsize=7, color='0.5', style='italic')
    ax.text(xlim[1] * 0.95, 0.05, 'low cycle\nhigh hierarchy\n(Elo-like)',
            ha='right', va='bottom', fontsize=7, color='0.5', style='italic')
    ax.text(xlim[0] + (xlim[1] - xlim[0]) * 0.05, 0.95, 'high cycle\nlow hierarchy\n(RPS-like)',
            ha='left', va='top', fontsize=7, color='0.5', style='italic')

    output_path = Path(outdir) / "checkers_rating_vs_cyclic.png"
    for fmt in ('png', 'pdf'):
        fig.savefig(str(output_path).replace('.png', f'.{fmt}'))
    plt.close(fig)
    print(f"  Saved: {output_path.name}")
    return output_path


# ===========================================================================
# Summary statistics
# ===========================================================================

def print_summary(results):
    """Print summary table of decomposition results."""
    print("\n" + "=" * 80)
    print("HODGE DECOMPOSITION SUMMARY")
    print("=" * 80)

    topos = [t for t in TOPOLOGY_ORDER if t in results]

    print(f"\n{'Topology':>18} {'Cyclic ratio':>14} {'Rating spread':>14} {'Interpretation':>22}")
    print("-" * 72)

    for topo in topos:
        r = results[topo]
        cr = r['final_cyclic_ratio']
        cr_std = r['final_cyclic_ratio_std']
        rs = float(np.mean(r['rating_spreads'][:, -1]))
        rs_std = float(np.std(r['rating_spreads'][:, -1], ddof=1)) if r['rating_spreads'].shape[0] > 1 else 0.0

        if cr > 0.7:
            interp = "strongly cyclic"
        elif cr > 0.4:
            interp = "mixed"
        elif cr > 0.1:
            interp = "weakly cyclic"
        else:
            interp = "transitive"

        print(f"{topo:>18} {cr:>7.4f}+/-{cr_std:<5.4f} {rs:>7.4f}+/-{rs_std:<5.4f} {interp:>22}")

    print(f"\n  Cyclic ratio = ||C||_F / ||A||_F")
    print(f"  Rating spread = std(s), skill differentiation measure")
    print(f"  Values at final generation, averaged across seeds")

    # Ordering analysis
    ordered = sorted(topos, key=lambda t: results[t]['final_cyclic_ratio'], reverse=True)
    print(f"\n  Cyclic ratio ordering (most -> least cyclic):")
    print(f"    {' > '.join(ordered)}")

    # Does tighter coupling suppress cycles?
    connectivity_order = ['none', 'ring', 'star', 'random', 'fully_connected']
    available = [t for t in connectivity_order if t in results]
    if len(available) >= 3:
        cr_by_conn = [results[t]['final_cyclic_ratio'] for t in available]
        from scipy import stats as sp_stats
        rho, p = sp_stats.spearmanr(range(len(available)), cr_by_conn)
        print(f"\n  Spearman correlation (connectivity vs cyclic ratio): rho={rho:.4f}, p={p:.4f}")
        if p < 0.05 and rho < 0:
            print(f"  => Tighter coupling SUPPRESSES intransitivity (significant)")
        elif p < 0.05 and rho > 0:
            print(f"  => Tighter coupling AMPLIFIES intransitivity (significant)")
        else:
            print(f"  => No significant monotone relationship between coupling and intransitivity")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Balduzzi/Hodge decomposition of checkers tournament payoff matrices."
    )
    parser.add_argument(
        '--data', type=str, default=None,
        help='Path to .npz file with payoff matrices.'
    )
    parser.add_argument(
        '--test', action='store_true', default=False,
        help='Run synthetic verification tests.'
    )
    args = parser.parse_args()

    setup_style()

    # If no arguments, run synthetic test
    if args.data is None and not args.test:
        args.test = True

    if args.test:
        all_passed, test_results = run_synthetic_tests()
        if not all_passed:
            sys.exit(1)

    if args.data is not None:
        print(f"\n{'=' * 80}")
        print(f"ANALYZING PAYOFF DATA: {args.data}")
        print(f"{'=' * 80}")

        data = load_payoff_data(args.data)
        if data is None:
            sys.exit(1)

        for topo, topo_data in data.items():
            m = topo_data['matrices']
            print(f"  {topo}: {m.shape[0]} seeds x {m.shape[1]} gens x "
                  f"{m.shape[2]}x{m.shape[3]} strategies")

        results = analyze_payoff_data(data)
        print_summary(results)

        print(f"\nGenerating plots...")
        outdir = str(OUT_DIR)
        plot_cyclic_ratio_by_topology(results, outdir)
        plot_cyclic_evolution(results, outdir)
        plot_rating_vs_cyclic(results, outdir)
        print(f"\nAll plots saved to: {OUT_DIR.resolve()}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""OQ41 — analysis & pre-registered falsification verdict.

Loads the raw CSVs written by oq41_run.py, takes final-generation best_fitness
(and hamming_diversity) per (arm, graph, domain, seed), and computes:

  Arm A (H¹ test, Family 2, β₁ ∈ {1,2,3,4}):
    - Kendall τ between β₁-rank and final fitness (and diversity), per domain.
      Framework predicts τ < 0 (lower H¹ ⇒ higher fitness).
    - Mann-Whitney one-sided + Vargha-Delaney A on the extreme pair
      (β₁=1 vs β₁=4). group1 = β₁=1, so A > 0.5 means low-H¹ wins.
    - Cross-domain Kendall W concordance of the per-domain graph ordering.

  Arm B (λ₂ control, Family 1, β₁=2 fixed):
    - Kendall τ between λ₂-rank and final fitness, per domain + concordance.
    - If Arm A is monotone (H¹ effect) AND Arm B is flat, the effect is
      H¹-magnitude, not λ₂.

  Holm correction across the family of per-domain tests.

PRE-REGISTERED FALSIFICATION VERDICT (per the design doc):
  FALSIFIED if, on consensus tasks at fixed N=12, low-H¹ does NOT beat high-H¹:
    one-sided test fails to reject (p ≥ 0.05) AND effect is null/wrong-signed
    (Vargha-Delaney A ≤ 0.5 OR Kendall τ(H¹-rank, outcome) ≥ 0).
  SUPPORTED if low-H¹ beats high-H¹ (p < 0.05, A > 0.5, τ < 0).
  INCONCLUSIVE otherwise (mixed signals).

Honesty: wrong-signed and null results are reported plainly; nothing is tuned.
"""
import os
import sys
import glob

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from onemax_stats import vargha_delaney_a

OQDIR = os.path.join(SCRIPT_DIR, 'oq41')
DOMAINS = ['onemax', 'maze', 'graph_coloring', 'knapsack']


def load_final():
    """Return a long DataFrame: arm, graph, domain, seed, best_fitness, hamming."""
    meta = pd.read_csv(os.path.join(OQDIR, 'oq41_graph_meta.csv'))
    meta_lookup = {(r.arm, r.graph): (r.beta1, r.lambda2)
                   for r in meta.itertuples()}
    # safe-name -> real graph name
    def safe(name):
        return name.replace('-', '').replace('β', 'b')
    safe_to_graph = {(r.arm, safe(r.graph)): r.graph for r in meta.itertuples()}

    rows = []
    for path in sorted(glob.glob(os.path.join(OQDIR, 'oq41_*_*.csv'))):
        base = os.path.basename(path)
        if base == 'oq41_graph_meta.csv':
            continue
        # oq41_{arm}_{safegraph}_{domain}.csv
        stem = base[len('oq41_'):-len('.csv')]
        parts = stem.split('_')
        arm = parts[0]
        domain = parts[-1]
        safe_graph = '_'.join(parts[1:-1])
        graph = safe_to_graph.get((arm, safe_graph))
        if graph is None:
            print(f"  WARN: could not map {base}; skipping")
            continue
        beta1, lam2 = meta_lookup[(arm, graph)]
        df = pd.read_csv(path)
        last_gen = df.generation.max()
        fin = df[df.generation == last_gen]
        for r in fin.itertuples():
            rows.append(dict(arm=arm, graph=graph, domain=domain, seed=r.seed,
                             beta1=beta1, lambda2=lam2,
                             best_fitness=r.best_fitness,
                             hamming=r.hamming_diversity))
    return pd.DataFrame(rows)


def holm(pvals_named):
    """Holm-Bonferroni. pvals_named: list of (label, p). Returns dict label->p_adj."""
    items = sorted(pvals_named, key=lambda x: x[1])
    m = len(items)
    adj = {}
    running = 0.0
    for k, (label, p) in enumerate(items):
        a = (m - k) * p
        running = max(running, a)
        adj[label] = min(running, 1.0)
    return adj


def kendall_w(rankings):
    """Kendall's W concordance. rankings: list of rank-vectors (each length k),
    one per rater (domain). Returns W in [0,1]."""
    R = np.array(rankings, dtype=float)  # m raters × k items
    m, k = R.shape
    col_sums = R.sum(axis=0)
    mean_rank = col_sums.mean()
    S = ((col_sums - mean_rank) ** 2).sum()
    denom = m ** 2 * (k ** 3 - k)
    if denom == 0:
        return float('nan')
    return 12.0 * S / denom


def analyze_arm(df, arm, key, metric='best_fitness', predict_negative=True):
    """key = 'beta1' (Arm A) or 'lambda2' (Arm B).
    Returns per-domain dict with tau, p_tau, and the per-domain graph ordering
    (by mean metric, ascending) for concordance."""
    sub = df[df.arm == arm]
    graphs = sub[['graph', key]].drop_duplicates().sort_values(key)
    out = {}
    orderings = []
    for domain in DOMAINS:
        d = sub[sub.domain == domain]
        if d.empty:
            continue
        # Kendall τ between key-value and metric across ALL seed-level points.
        tau, p_two = stats.kendalltau(d[key].values, d[metric].values)
        # one-sided p for τ < 0 (framework prediction): halve if sign matches
        if np.isnan(tau):
            p_one = 1.0
        elif tau < 0:
            p_one = p_two / 2
        else:
            p_one = 1.0 - p_two / 2
        # per-domain ordering of graphs by mean metric (for Kendall W)
        means = d.groupby('graph')[metric].mean()
        means = means.reindex(graphs.graph.values)
        order_rank = stats.rankdata(means.values)
        orderings.append(order_rank)
        out[domain] = dict(tau=tau, p_two=p_two, p_one=p_one,
                           graph_means=means.to_dict())
    W = kendall_w(orderings) if len(orderings) >= 2 else float('nan')
    return out, W, graphs


def extreme_pair(df, arm, lo_val, hi_val, key='beta1', metric='best_fitness'):
    """Mann-Whitney one-sided (low-key > high-key) + Vargha-Delaney A.
    group1 = low-key (low H¹). A > 0.5 means low H¹ beats high H¹."""
    sub = df[df.arm == arm]
    res = {}
    for domain in DOMAINS:
        d = sub[sub.domain == domain]
        lo = d[d[key] == lo_val][metric].values
        hi = d[d[key] == hi_val][metric].values
        if len(lo) == 0 or len(hi) == 0:
            continue
        # one-sided: low > high
        try:
            U, p = stats.mannwhitneyu(lo, hi, alternative='greater')
        except ValueError:
            U, p = float('nan'), 1.0
        A = vargha_delaney_a(lo, hi)
        res[domain] = dict(U=U, p=p, A=A,
                           lo_mean=float(np.mean(lo)), hi_mean=float(np.mean(hi)))
    return res


def main():
    df = load_final()
    if df.empty:
        print("No data found in", OQDIR)
        return
    print("=" * 78)
    print("OQ41 — H¹ FALSIFICATION ANALYSIS")
    print("=" * 78)
    print(f"Loaded {len(df)} final-gen observations.")
    print(f"Arms: {sorted(df.arm.unique())}  Domains: {sorted(df.domain.unique())}")
    print(f"Seeds per cell: {df.groupby(['arm','graph','domain']).size().min()}"
          f"..{df.groupby(['arm','graph','domain']).size().max()}")

    # ---------------- ARM A: H¹ test ----------------
    print("\n" + "-" * 78)
    print("ARM A — H¹ (=β₁) varying.  Prediction: τ(β₁, fitness) < 0 (low H¹ wins)")
    print("-" * 78)
    aA, WA, graphsA = analyze_arm(df, 'A', 'beta1', 'best_fitness')
    aA_div, WA_div, _ = analyze_arm(df, 'A', 'beta1', 'hamming')
    pvals_tau = []
    print(f"{'domain':<16}{'τ(β₁,fit)':>12}{'p1(τ<0)':>12}"
          f"{'τ(β₁,div)':>12}  graph-mean-fitness by β₁")
    for domain in DOMAINS:
        if domain not in aA:
            continue
        r = aA[domain]
        rd = aA_div.get(domain, {})
        means_str = "  ".join(f"{g}:{v:.2f}" for g, v in r['graph_means'].items())
        print(f"{domain:<16}{r['tau']:>12.4f}{r['p_one']:>12.4g}"
              f"{rd.get('tau', float('nan')):>12.4f}  {means_str}")
        pvals_tau.append((f"A/{domain}/tau", r['p_one']))
    print(f"\nCross-domain Kendall W (fitness ordering by β₁): {WA:.4f}")
    print(f"Cross-domain Kendall W (diversity ordering by β₁): {WA_div:.4f}")

    print("\nExtreme pair β₁=1 vs β₁=4 (group1 = β₁=1, low H¹):")
    ext = extreme_pair(df, 'A', 1, 4, 'beta1', 'best_fitness')
    print(f"{'domain':<16}{'A(1>4)':>10}{'p1(1>4)':>12}{'mean β₁=1':>12}{'mean β₁=4':>12}")
    pvals_ext = []
    for domain in DOMAINS:
        if domain not in ext:
            continue
        e = ext[domain]
        print(f"{domain:<16}{e['A']:>10.3f}{e['p']:>12.4g}"
              f"{e['lo_mean']:>12.3f}{e['hi_mean']:>12.3f}")
        pvals_ext.append((f"A/{domain}/extreme", e['p']))

    # ---------------- ARM B: λ₂ control ----------------
    print("\n" + "-" * 78)
    print("ARM B — λ₂ varying at FIXED β₁=2 (H¹ control).  Want: FLAT (|τ| small)")
    print("-" * 78)
    aB, WB, graphsB = analyze_arm(df, 'B', 'lambda2', 'best_fitness')
    print(f"{'domain':<16}{'τ(λ₂,fit)':>12}{'p2':>12}  graph-mean-fitness by λ₂")
    pvals_B = []
    for domain in DOMAINS:
        if domain not in aB:
            continue
        r = aB[domain]
        means_str = "  ".join(f"{g}:{v:.2f}" for g, v in r['graph_means'].items())
        print(f"{domain:<16}{r['tau']:>12.4f}{r['p_two']:>12.4g}  {means_str}")
        pvals_B.append((f"B/{domain}/tau", r['p_two']))
    print(f"\nCross-domain Kendall W (Arm B λ₂ ordering): {WB:.4f}")

    # ---------------- Holm correction ----------------
    all_p = pvals_tau + pvals_ext + pvals_B
    adj = holm(all_p)
    print("\n" + "-" * 78)
    print("HOLM-CORRECTED p-values (across all per-domain tests, m=%d)" % len(all_p))
    print("-" * 78)
    for label, p in sorted(all_p, key=lambda x: x[1]):
        print(f"  {label:<24} raw={p:.4g}   holm={adj[label]:.4g}")

    # ---------------- VERDICT ----------------
    print("\n" + "=" * 78)
    print("PRE-REGISTERED FALSIFICATION VERDICT")
    print("=" * 78)
    print("Rule: SUPPORTED iff low-H¹ beats high-H¹: τ(β₁,fit)<0 AND A(1>4)>0.5 "
          "AND p1<0.05.")
    print("      FALSIFIED iff p1≥0.05 AND (A≤0.5 OR τ≥0)  [null-or-wrong-signed].")
    print("      INCONCLUSIVE otherwise.\n")

    verdicts = {}
    for domain in DOMAINS:
        if domain not in aA or domain not in ext:
            continue
        tau = aA[domain]['tau']
        p1 = aA[domain]['p_one']
        A = ext[domain]['A']
        p_ext = ext[domain]['p']
        # use the extreme-pair one-sided p as the headline reject test
        reject = (p_ext < 0.05)
        right_signed = (tau < 0) and (A > 0.5)
        if reject and right_signed:
            v = "SUPPORTED"
        elif (not reject) and ((A <= 0.5) or (tau >= 0)):
            v = "FALSIFIED"
        else:
            v = "INCONCLUSIVE"
        verdicts[domain] = v
        print(f"  {domain:<16} τ(β₁,fit)={tau:+.4f}  A(1>4)={A:.3f}  "
              f"p_extreme={p_ext:.4g}  ->  {v}")

    # Overall: combine. Cross-domain consistency = Kendall W with correct sign.
    n_sup = sum(v == "SUPPORTED" for v in verdicts.values())
    n_fal = sum(v == "FALSIFIED" for v in verdicts.values())
    n_inc = sum(v == "INCONCLUSIVE" for v in verdicts.values())
    print(f"\n  Tally: {n_sup} SUPPORTED, {n_fal} FALSIFIED, {n_inc} INCONCLUSIVE")
    if n_sup == len(verdicts) and len(verdicts) > 0:
        overall = "SUPPORTED (all domains)"
    elif n_fal == len(verdicts) and len(verdicts) > 0:
        overall = "FALSIFIED (all domains)"
    elif n_sup > n_fal:
        overall = "MIXED — leaning SUPPORTED"
    elif n_fal > n_sup:
        overall = "MIXED — leaning FALSIFIED"
    else:
        overall = "INCONCLUSIVE / MIXED"
    print(f"\n  OVERALL VERDICT: {overall}")
    print(f"  Arm A cross-domain Kendall W (fitness): {WA:.4f} "
          f"(high W = consistent ordering; check sign of τ for direction)")
    print(f"  Arm B (λ₂ control) cross-domain W: {WB:.4f} "
          f"(low/flat = effect is not λ₂)")
    print("=" * 78)


if __name__ == '__main__':
    main()

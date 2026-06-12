#!/usr/bin/env python3
"""OQ41 — H¹ falsification experiment driver (Arm A + Arm B).

Pre-registered test of the β₁ → λ₂ → H¹ framework. Under the constant sheaf
F=ℝ, dim H¹(G;ℝ) = β₁, so:

  - Arm A (Family 2, β₁ ∈ {1,2,3,4}):  the H¹-VARYING arm.
  - Arm B (Family 1, β₁ = 2 fixed, λ₂ varies):  the λ₂ CONTROL (H¹ fixed).

Each graph becomes a 12-island migration topology (topology="custom", driven by
the graph's adjacency matrix), run across the 4 consensus-class domains. The
prediction (H1): lower H¹ ⇒ higher final best-fitness on consensus tasks.

This script is deliberately NOT tuned to make the framework pass. It runs the
matched designs and writes raw CSVs; oq41_analyze.py renders the verdict.

Usage:
    python oq41_run.py                 # full run: both arms, 4 domains, 30 seeds
    python oq41_run.py --smoke         # quick pipeline check (2 seeds, 1 domain)
    python oq41_run.py --arm A         # one arm only
    python oq41_run.py --volume per_edge
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import networkx as nx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', 'cais2026'))

import onemax_stats as oms
from graph_families import get_family1, get_family2, betti_1, lambda_2

OUTDIR = os.path.join(SCRIPT_DIR, 'oq41')

NUM_ISLANDS = 12
POP_SIZE = 96  # 8 per island

# Consensus-class domains. Each entry: (eval_fn, init_fn, genome_length).
# Operators/diversity default to the binary ops (all four domains are binary).
DOMAINS = ['onemax', 'maze', 'graph_coloring', 'knapsack']


def build_domain(domain, smoke=False):
    """Return (config_kwargs, eval_fn, init_fn). Configs mirror onemax_stats.main()
    Experiment-E conventions: max_generations=100, migration_freq=5, mut=1/L,
    crossover_rate=0.8, tournament_size=3, migration_rate=0.1."""
    gens = 20 if smoke else 100
    common = dict(
        population_size=POP_SIZE,
        num_islands=NUM_ISLANDS,
        tournament_size=3,
        crossover_rate=0.8,
        max_generations=gens,
        migration_freq=5,
        migration_rate=0.1,
    )
    if domain == 'onemax':
        L = 100
        common.update(genome_length=L, mutation_rate=1.0 / L)
        return common, None, None  # defaults are OneMax
    if domain == 'maze':
        from maze_domain import evaluate_maze, random_maze_population, MAZE_GENOME_LENGTH
        common.update(genome_length=MAZE_GENOME_LENGTH, mutation_rate=1.0 / MAZE_GENOME_LENGTH)
        return common, evaluate_maze, random_maze_population
    if domain == 'graph_coloring':
        from graph_coloring_domain import (evaluate_graph_coloring,
                                           random_graph_coloring_population,
                                           GRAPH_COLORING_GENOME_LENGTH)
        common.update(genome_length=GRAPH_COLORING_GENOME_LENGTH,
                      mutation_rate=1.0 / GRAPH_COLORING_GENOME_LENGTH)
        return common, evaluate_graph_coloring, random_graph_coloring_population
    if domain == 'knapsack':
        from knapsack_domain import (evaluate_knapsack, random_knapsack_population,
                                     KNAPSACK_GENOME_LENGTH)
        common.update(genome_length=KNAPSACK_GENOME_LENGTH,
                      mutation_rate=1.0 / KNAPSACK_GENOME_LENGTH)
        return common, evaluate_knapsack, random_knapsack_population
    raise ValueError(f"unknown domain {domain}")


def write_meta():
    """Record β₁ and λ₂ for every graph in both families."""
    rows = []
    for arm, family in (('A', get_family2()), ('B', get_family1())):
        for name, desc, G in family:
            A = nx.to_numpy_array(G)
            rows.append({
                'arm': arm,
                'graph': name,
                'desc': desc,
                'n': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'beta1': betti_1(G),
                'lambda2': lambda_2(A),
            })
    path = os.path.join(OUTDIR, 'oq41_graph_meta.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote graph metadata -> {path}", flush=True)


def run(arms, domains, seeds, volume_mode, smoke):
    os.makedirs(OUTDIR, exist_ok=True)
    write_meta()
    seed_list = list(range(seeds))
    arm_family = {'A': get_family2(), 'B': get_family1()}

    t0 = time.time()
    for arm in arms:
        for name, desc, G in arm_family[arm]:
            A = nx.to_numpy_array(G)
            assert A.shape[0] == NUM_ISLANDS, f"{name}: expected {NUM_ISLANDS} nodes"
            for domain in domains:
                kwargs, eval_fn, init_fn = build_domain(domain, smoke=smoke)
                config = oms.GAConfig(
                    adjacency=A,
                    volume_mode=volume_mode,
                    lean=True,
                    **kwargs,
                )
                safe = name.replace('-', '').replace('β', 'b')
                csv_path = os.path.join(OUTDIR, f'oq41_{arm}_{safe}_{domain}.csv')
                print(f"[arm {arm}] {name} (β₁={betti_1(G)}, "
                      f"λ₂={lambda_2(A):.4f}) × {domain} -> {os.path.basename(csv_path)}",
                      flush=True)
                oms.run_experiment_e(
                    seed_list, config,
                    topologies=['custom'],
                    evaluate_fn=eval_fn,
                    init_fn=init_fn,
                    incremental_csv=csv_path,
                    resume=True,
                )
    print(f"\nDONE in {time.time() - t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['A', 'B', 'both'], default='both')
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--volume', choices=['per_event', 'per_edge'], default='per_event')
    ap.add_argument('--smoke', action='store_true',
                    help='quick pipeline check: 2 seeds, 1 domain, 20 gens')
    args = ap.parse_args()

    if args.smoke:
        run(['A', 'B'], ['onemax'], seeds=2, volume_mode=args.volume, smoke=True)
        return

    arms = ['A', 'B'] if args.arm == 'both' else [args.arm]
    run(arms, DOMAINS, seeds=args.seeds, volume_mode=args.volume, smoke=False)


if __name__ == '__main__':
    main()

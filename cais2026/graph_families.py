#!/usr/bin/env python3
"""
graph_families.py — EUMAS experiment graph families.

Two families designed by Claudius (email UID 1276, 2026-05-02):

Family 1: Fixed n=12, |E|=13, β₁=2. Varying spatial arrangement of cycles.
  Isolates cycle *arrangement* from cycle *count*.

Family 2: Fixed n=12, varying |E| to get β₁ = {1, 2, 3, 4}.
  Isolates cycle count with otherwise-similar degree sequences.

Each graph is returned as a networkx Graph and a numpy adjacency matrix.
"""

import numpy as np
import networkx as nx
from scipy.linalg import eigvalsh


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def betti_1(G):
    """First Betti number β₁ = |E| - |V| + c (number of connected components)."""
    return G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G)


def lambda_2(A):
    """Algebraic connectivity: second-smallest eigenvalue of L = D - A."""
    n = A.shape[0]
    if n <= 1:
        return 0.0
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigvals = eigvalsh(L)
    return float(eigvals[1])


def graph_summary(name, G, A):
    """Return a dict of graph properties."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    b1 = betti_1(G)
    lam2 = lambda_2(A)
    degrees = [d for _, d in G.degree()]
    deg_seq = sorted(degrees, reverse=True)
    avg_deg = sum(degrees) / len(degrees)
    return {
        "name": name,
        "n": n,
        "|E|": m,
        "β₁": b1,
        "λ₂": lam2,
        "avg_deg": avg_deg,
        "deg_seq": deg_seq,
        "components": nx.number_connected_components(G),
    }


# ---------------------------------------------------------------------------
# Family 1: n=12, |E|=13, β₁=2, varying cycle arrangement
# ---------------------------------------------------------------------------

def family1_f1a():
    """F1a: C3+C5 sharing a node + pendant path (M1-style, two short cycles close).

    Construction: nodes 0-1-2-0 form C3, nodes 2-3-4-5-6 form C5 (sharing node 2),
    then a pendant path 6-7-8-9-10-11 to reach n=12, |E|=13.

    C3: 0-1-2-0 (3 edges)
    C5: 2-3-4-5-6-2 (5 edges)
    Path: 6-7-8-9-10-11 (5 edges)
    Total: 13 edges, 12 nodes, β₁ = 13-12+1 = 2 ✓
    """
    G = nx.Graph()
    G.add_nodes_from(range(12))
    # C3: 0-1-2-0
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    # C5: 2-3-4-5-6-2
    G.add_edges_from([(2, 3), (3, 4), (4, 5), (5, 6), (6, 2)])
    # Pendant path: 6-7-8-9-10-11
    G.add_edges_from([(6, 7), (7, 8), (8, 9), (9, 10), (10, 11)])
    return G


def family1_f1b():
    """F1b: C4+C4 sharing a node (M2-style, equal-length cycles).

    C4: 0-1-2-3-0 (4 edges)
    C4: 3-4-5-6-3 (4 edges, sharing node 3)
    Path: 6-7-8-9-10-11 (5 edges)
    Total: 13 edges, 12 nodes, β₁ = 13-12+1 = 2 ✓
    """
    G = nx.Graph()
    G.add_nodes_from(range(12))
    # C4: 0-1-2-3-0
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    # C4: 3-4-5-6-3
    G.add_edges_from([(3, 4), (4, 5), (5, 6), (6, 3)])
    # Pendant path: 6-7-8-9-10-11
    G.add_edges_from([(6, 7), (7, 8), (8, 9), (9, 10), (10, 11)])
    return G


def family1_f1c():
    """F1c: One long cycle C9 + bridge + pendant (satellite structure).

    C9: 0-1-2-3-4-5-6-7-8-0 (9 edges)
    Bridge: 8-9 (1 edge)
    Small cycle C3 as satellite: 9-10-11-9 (3 edges)
    Total: 13 edges, 12 nodes, β₁ = 13-12+1 = 2 ✓
    """
    G = nx.Graph()
    G.add_nodes_from(range(12))
    # C9: 0-1-2-...-8-0
    for i in range(8):
        G.add_edge(i, i + 1)
    G.add_edge(8, 0)
    # Bridge: 8-9
    G.add_edge(8, 9)
    # C3 satellite: 9-10-11-9
    G.add_edges_from([(9, 10), (10, 11), (11, 9)])
    return G


def family1_f1d():
    """F1d: Two C4 cycles at graph distance 3 (connected by long path).

    C4: 0-1-2-3-0 (4 edges)
    Path: 3-4-5-6-7 (4 edges, distance 3 between cycle endpoints 3 and 7)
    C4: 7-8-9-10-7 (4 edges)
    Extra edge to pendant: 10-11 (1 edge)
    Total: 13 edges, 12 nodes, β₁ = 13-12+1 = 2 ✓
    """
    G = nx.Graph()
    G.add_nodes_from(range(12))
    # C4: 0-1-2-3-0
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    # Path connecting: 3-4-5-6-7
    G.add_edges_from([(3, 4), (4, 5), (5, 6), (6, 7)])
    # C4: 7-8-9-10-7
    G.add_edges_from([(7, 8), (8, 9), (9, 10), (10, 7)])
    # Pendant: 10-11
    G.add_edge(10, 11)
    return G


def family1_f1e():
    """F1e: C6 with internal chord + pendant chain.

    C6: 0-1-2-3-4-5-0 (6 edges)
    Chord: 0-3 (1 edge, creates two C4 sub-cycles sharing an edge)
    Pendant chain: 5-6-7-8-9-10-11 (6 edges)
    Total: 13 edges, 12 nodes, β₁ = 13-12+1 = 2 ✓

    Note: The chord splits C6 into two C4s sharing edge 0-3,
    so the two independent cycles overlap spatially.
    """
    G = nx.Graph()
    G.add_nodes_from(range(12))
    # C6: 0-1-2-3-4-5-0
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
    # Chord: 0-3
    G.add_edge(0, 3)
    # Pendant chain: 5-6-7-8-9-10-11
    G.add_edges_from([(5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)])
    return G


# ---------------------------------------------------------------------------
# Family 2: n=12, varying |E|, β₁ = {1, 2, 3, 4}
# Similar degree sequences where possible.
# ---------------------------------------------------------------------------

def family2_b1():
    """F2-β1: n=12, β₁=1. A single cycle with pendant paths.

    C6: 0-1-2-3-4-5-0 (6 edges)
    Path from 5: 5-6-7-8-9-10-11 (6 edges)
    Total: |E|=12, β₁ = 12-12+1 = 1 ✓
    """
    G = nx.Graph()
    G.add_nodes_from(range(12))
    # C6
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
    # Path: 5-6-...-11
    G.add_edges_from([(5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)])
    return G


def family2_b2():
    """F2-β2: n=12, β₁=2. Two cycles with pendant path.

    Reuse F1a construction (C3+C5 sharing node + pendant path).
    Total: |E|=13, β₁ = 13-12+1 = 2 ✓
    """
    return family1_f1a()


def family2_b3():
    """F2-β3: n=12, β₁=3. Three independent cycles.

    C4: 0-1-2-3-0 (4 edges)
    C4: 3-4-5-6-3 (4 edges, sharing node 3)
    C4: 6-7-8-9-6 (4 edges, sharing node 6)
    Path: 9-10-11 (2 edges)
    Total: |E|=14, β₁ = 14-12+1 = 3 ✓
    """
    G = nx.Graph()
    G.add_nodes_from(range(12))
    # C4: 0-1-2-3-0
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    # C4: 3-4-5-6-3
    G.add_edges_from([(3, 4), (4, 5), (5, 6), (6, 3)])
    # C4: 6-7-8-9-6
    G.add_edges_from([(6, 7), (7, 8), (8, 9), (9, 6)])
    # Path: 9-10-11
    G.add_edges_from([(9, 10), (10, 11)])
    return G


def family2_b4():
    """F2-β4: n=12, β₁=4. Four independent cycles.

    C3: 0-1-2-0 (3 edges)
    C3: 2-3-4-2 (3 edges, sharing node 2)
    C3: 4-5-6-4 (3 edges, sharing node 4)
    C3: 6-7-8-6 (3 edges, sharing node 6)
    Path: 8-9-10-11 (3 edges)
    Total: |E|=15, β₁ = 15-12+1 = 4 ✓
    """
    G = nx.Graph()
    G.add_nodes_from(range(12))
    # C3: 0-1-2-0
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    # C3: 2-3-4-2
    G.add_edges_from([(2, 3), (3, 4), (4, 2)])
    # C3: 4-5-6-4
    G.add_edges_from([(4, 5), (5, 6), (6, 4)])
    # C3: 6-7-8-6
    G.add_edges_from([(6, 7), (7, 8), (8, 6)])
    # Path: 8-9-10-11
    G.add_edges_from([(8, 9), (9, 10), (10, 11)])
    return G


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def get_family1():
    """Return list of (name, description, Graph) for Family 1."""
    return [
        ("F1a", "C3+C5 sharing node + path (short cycles, close)", family1_f1a()),
        ("F1b", "C4+C4 sharing node + path (equal cycles, close)", family1_f1b()),
        ("F1c", "C9 + bridge + C3 satellite (big+small, bridged)", family1_f1c()),
        ("F1d", "Two C4 at distance 3 + pendant (equal cycles, far)", family1_f1d()),
        ("F1e", "C6+chord + pendant chain (overlapping cycles)", family1_f1e()),
    ]


def get_family2():
    """Return list of (name, description, Graph) for Family 2."""
    return [
        ("F2-β1", "C6 + path (single cycle)", family2_b1()),
        ("F2-β2", "C3+C5 sharing node + path (two cycles)", family2_b2()),
        ("F2-β3", "3×C4 chain + path (three cycles)", family2_b3()),
        ("F2-β4", "4×C3 chain + path (four cycles)", family2_b4()),
    ]


# ---------------------------------------------------------------------------
# Main: summary table + adjacency matrices
# ---------------------------------------------------------------------------

def print_table(family_name, entries):
    """Print a markdown summary table for a graph family."""
    print(f"\n{'='*70}")
    print(f"  {family_name}")
    print(f"{'='*70}")
    print(f"| {'Name':<8} | {'n':>3} | {'|E|':>4} | {'β₁':>3} | {'λ₂':>8} | {'avg_deg':>7} | Description")
    print(f"|{'-'*10}|{'-'*5}|{'-'*6}|{'-'*5}|{'-'*10}|{'-'*9}|{'-'*40}")

    for name, desc, G in entries:
        A = nx.to_numpy_array(G, dtype=float)
        s = graph_summary(name, G, A)
        print(f"| {s['name']:<8} | {s['n']:>3} | {s['|E|']:>4} | {s['β₁']:>3} | {s['λ₂']:>8.5f} | {s['avg_deg']:>7.2f} | {desc}")

    print()


def print_adjacency_matrices(family_name, entries):
    """Print adjacency matrices for each graph."""
    print(f"\n--- Adjacency Matrices: {family_name} ---\n")
    for name, desc, G in entries:
        A = nx.to_numpy_array(G, dtype=int)
        print(f"{name} ({desc}):")
        print(f"  Degree sequence: {sorted([d for _, d in G.degree()], reverse=True)}")
        # Print compact matrix
        n = A.shape[0]
        for i in range(n):
            row_str = " ".join(str(A[i, j]) for j in range(n))
            print(f"  [{row_str}]")
        print()


def main():
    f1 = get_family1()
    f2 = get_family2()

    # Validate all graphs
    print("Validating graph constructions...")
    for name, desc, G in f1:
        n, m, b1 = G.number_of_nodes(), G.number_of_edges(), betti_1(G)
        assert n == 12, f"{name}: expected n=12, got {n}"
        assert m == 13, f"{name}: expected |E|=13, got {m}"
        assert b1 == 2, f"{name}: expected β₁=2, got {b1}"
        assert nx.is_connected(G), f"{name}: graph is disconnected"
    print("  Family 1: all 5 graphs pass (n=12, |E|=13, β₁=2, connected) ✓")

    for name, desc, G in f2:
        n = G.number_of_nodes()
        b1 = betti_1(G)
        expected_b1 = int(name.split("β")[1])
        assert n == 12, f"{name}: expected n=12, got {n}"
        assert b1 == expected_b1, f"{name}: expected β₁={expected_b1}, got {b1}"
        assert nx.is_connected(G), f"{name}: graph is disconnected"
    print("  Family 2: all 4 graphs pass (n=12, correct β₁, connected) ✓")

    print_table("Family 1: Fixed β₁=2, varying cycle arrangement", f1)
    print_table("Family 2: Fixed n=12, varying β₁", f2)

    print_adjacency_matrices("Family 1", f1)
    print_adjacency_matrices("Family 2", f2)


if __name__ == "__main__":
    main()

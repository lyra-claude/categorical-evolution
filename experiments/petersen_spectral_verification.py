#!/usr/bin/env python3
"""
Exact spectral verification: GP(5,1) vs C_5.

Verifies Claudius's claim that lambda2(GP(5,1)) == lambda2(C_5) using
exact symbolic arithmetic (sympy), not floating point.

Also performs the Fourier block decomposition over Z_5 to show how
GP(5,1) decomposes into symmetric and antisymmetric modes.

GP(5,1) = pentagonal prism:
  - Outer cycle: 0-1-2-3-4-0
  - Inner cycle: 5-6-7-8-9-5
  - Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
  - 3-regular, 10 vertices
"""

from sympy import (
    Matrix, Rational, cos, pi, sqrt, simplify, trigsimp,
    pretty, Symbol, exp, I, Abs, re, im,
    nsimplify, radsimp, cancel, factor, expand, pprint
)
from collections import Counter


def build_gp51_adjacency():
    """Build the 10x10 adjacency matrix of GP(5,1)."""
    n = 10
    A = Matrix.zeros(n, n)

    # Outer cycle: 0-1-2-3-4-0
    for i in range(5):
        j = (i + 1) % 5
        A[i, j] = 1
        A[j, i] = 1

    # Inner cycle: 5-6-7-8-9-5
    for i in range(5):
        j = (i + 1) % 5
        A[i + 5, j + 5] = 1
        A[j + 5, i + 5] = 1

    # Spokes: i <-> i+5
    for i in range(5):
        A[i, i + 5] = 1
        A[i + 5, i] = 1

    return A


def build_c5_adjacency():
    """Build the 5x5 adjacency matrix of C_5."""
    n = 5
    A = Matrix.zeros(n, n)
    for i in range(5):
        j = (i + 1) % 5
        A[i, j] = 1
        A[j, i] = 1
    return A


def exact_c5_eigenvalues():
    """
    Known exact eigenvalues of C_5:
    lambda_k = 2*cos(2*pi*k/5) for k = 0, 1, 2, 3, 4

    k=0: 2
    k=1: 2*cos(2*pi/5) = (sqrt(5)-1)/2 - 1 = (-1+sqrt(5))/2
    k=2: 2*cos(4*pi/5) = -(1+sqrt(5))/2
    k=3: 2*cos(6*pi/5) = 2*cos(4*pi/5) = -(1+sqrt(5))/2
    k=4: 2*cos(8*pi/5) = 2*cos(2*pi/5) = (-1+sqrt(5))/2
    """
    eigs = []
    for k in range(5):
        eig = 2 * cos(2 * pi * k / 5)
        eigs.append(trigsimp(eig))
    return sorted(eigs, reverse=True)


def fourier_block_decomposition():
    """
    GP(5,1) has Z_5 symmetry (cyclic rotation of both rings simultaneously).

    In the Fourier basis over Z_5, the 10x10 adjacency matrix block-diagonalizes
    into five 2x2 blocks (one per irrep k = 0, 1, 2, 3, 4).

    For each k, the 2x2 block is:
        B_k = [ 2*cos(2*pi*k/5),  1 ]
              [ 1,  2*cos(2*pi*k/5) ]

    (The diagonal entries come from the cycle adjacency in Fourier space,
     the off-diagonal entries come from the spoke connections.)

    The eigenvalues of B_k are:
        2*cos(2*pi*k/5) + 1  (symmetric mode: outer + inner in phase)
        2*cos(2*pi*k/5) - 1  (antisymmetric mode: outer - inner)
    """
    print("=" * 70)
    print("FOURIER BLOCK DECOMPOSITION OF GP(5,1) OVER Z_5")
    print("=" * 70)
    print()
    print("GP(5,1) has Z_5 rotational symmetry. Each vertex pair (i, i+5)")
    print("transforms together under Z_5. The adjacency matrix block-")
    print("diagonalizes into five 2x2 blocks in the Fourier basis.")
    print()

    all_block_eigs = []

    for k in range(5):
        c = trigsimp(2 * cos(2 * pi * k / 5))
        B_k = Matrix([
            [c, 1],
            [1, c]
        ])
        sym_eig = trigsimp(c + 1)
        anti_eig = trigsimp(c - 1)

        print(f"  Block k={k}:")
        print(f"    B_{k} = [ {c},  1 ]")
        print(f"          [ 1,  {c} ]")
        print(f"    Symmetric mode (outer+inner):      {sym_eig}")
        print(f"    Antisymmetric mode (outer-inner):   {anti_eig}")

        # Verify against B_k eigenvalues
        block_eigs = list(B_k.eigenvals().keys())
        block_eigs_simplified = [trigsimp(e) for e in block_eigs]
        print(f"    Verified eigenvalues of B_{k}:       {block_eigs_simplified}")
        print()

        all_block_eigs.append(sym_eig)
        all_block_eigs.append(anti_eig)

    return sorted(all_block_eigs, reverse=True, key=lambda x: float(x))


def main():
    print("=" * 70)
    print("EXACT SPECTRAL VERIFICATION: GP(5,1) vs C_5")
    print("Using sympy symbolic arithmetic — no floating point")
    print("=" * 70)
    print()

    # ---- C_5 eigenvalues ----
    print("-" * 70)
    print("1. EIGENVALUES OF C_5 (cycle on 5 vertices)")
    print("-" * 70)

    A_c5 = build_c5_adjacency()
    print("Adjacency matrix of C_5:")
    pprint(A_c5)
    print()

    # Compute eigenvalues symbolically
    c5_eigs_dict = A_c5.eigenvals()
    print("Eigenvalues (with multiplicities):")
    c5_eigs_sorted = []
    for eig, mult in sorted(c5_eigs_dict.items(), key=lambda x: float(x[0]), reverse=True):
        eig_s = trigsimp(eig)
        print(f"  {eig_s}  (multiplicity {mult})")
        for _ in range(mult):
            c5_eigs_sorted.append(eig_s)

    # Also show the known closed-form
    print()
    print("Known closed forms:")
    for k in range(5):
        val = trigsimp(2 * cos(2 * pi * k / 5))
        print(f"  k={k}: 2*cos(2*pi*{k}/5) = {val}")

    # lambda2 of C_5 (second-largest eigenvalue)
    c5_sorted = sorted([float(e) for e in c5_eigs_sorted], reverse=True)
    c5_lambda2_idx = 1  # second largest
    # Find exact lambda2
    c5_eigs_float_sorted = sorted(c5_eigs_sorted, key=lambda x: float(x), reverse=True)
    c5_lambda2 = c5_eigs_float_sorted[c5_lambda2_idx]
    print(f"\n  lambda2(C_5) = {c5_lambda2}")
    print(f"  Numerical:     {float(c5_lambda2):.10f}")

    # Laplacian lambda2
    c5_lap_lambda2 = trigsimp(2 - c5_lambda2)
    print(f"\n  Laplacian lambda2(C_5) = 2 - lambda2_adj = {c5_lap_lambda2}")
    print(f"  = 2 - 2*cos(2*pi/5) = {trigsimp(2 - 2*cos(2*pi/5))}")
    print(f"  Numerical: {float(c5_lap_lambda2):.10f}")

    # ---- GP(5,1) eigenvalues ----
    print()
    print("-" * 70)
    print("2. EIGENVALUES OF GP(5,1) (pentagonal prism, 10 vertices)")
    print("-" * 70)

    A_gp = build_gp51_adjacency()
    print("Adjacency matrix of GP(5,1):")
    pprint(A_gp)
    print()

    # Verify it's 3-regular
    row_sums = [sum(A_gp.row(i)) for i in range(10)]
    assert all(s == 3 for s in row_sums), f"Not 3-regular! Row sums: {row_sums}"
    print("Verified: 3-regular graph (all row sums = 3)")
    print()

    # Compute eigenvalues symbolically
    print("Computing exact eigenvalues of GP(5,1)... (this may take a moment)")
    gp_eigs_dict = A_gp.eigenvals()
    print("Eigenvalues (with multiplicities):")
    gp_eigs_sorted = []
    for eig, mult in sorted(gp_eigs_dict.items(), key=lambda x: float(x[0]), reverse=True):
        eig_s = trigsimp(eig)
        print(f"  {eig_s}  (multiplicity {mult})")
        for _ in range(mult):
            gp_eigs_sorted.append(eig_s)

    # lambda2 of GP(5,1)
    gp_eigs_float_sorted = sorted(gp_eigs_sorted, key=lambda x: float(x), reverse=True)
    gp_lambda2 = gp_eigs_float_sorted[1]
    print(f"\n  lambda2(GP(5,1)) = {gp_lambda2}")
    print(f"  Numerical:        {float(gp_lambda2):.10f}")

    # Laplacian lambda2 for GP(5,1): d - lambda2_adj = 3 - lambda2_adj
    gp_lap_lambda2 = trigsimp(3 - gp_lambda2)
    print(f"\n  Laplacian lambda2(GP(5,1)) = 3 - lambda2_adj = {gp_lap_lambda2}")
    print(f"  Numerical: {float(gp_lap_lambda2):.10f}")

    # ---- Comparison ----
    print()
    print("-" * 70)
    print("3. COMPARISON: lambda2(GP(5,1)) vs lambda2(C_5)")
    print("-" * 70)

    # Adjacency lambda2 comparison
    diff_adj = trigsimp(gp_lambda2 - c5_lambda2)
    print(f"\n  Adjacency lambda2:")
    print(f"    GP(5,1):  {gp_lambda2}  = {float(gp_lambda2):.10f}")
    print(f"    C_5:      {c5_lambda2}  = {float(c5_lambda2):.10f}")
    print(f"    Difference: {diff_adj}")
    print(f"    Equal? {diff_adj == 0}")

    # Laplacian lambda2 comparison
    diff_lap = trigsimp(gp_lap_lambda2 - c5_lap_lambda2)
    print(f"\n  Laplacian lambda2:")
    print(f"    GP(5,1):  {gp_lap_lambda2}  = {float(gp_lap_lambda2):.10f}")
    print(f"    C_5:      {c5_lap_lambda2}  = {float(c5_lap_lambda2):.10f}")
    print(f"    Difference: {diff_lap}")
    print(f"    Equal? {diff_lap == 0}")

    # ---- Check: is lambda2(GP(5,1)) = 2*cos(2*pi/5) exactly? ----
    print()
    print("-" * 70)
    print("4. IS lambda2_adj(GP(5,1)) = 2*cos(2*pi/5) EXACTLY?")
    print("-" * 70)

    target = trigsimp(2 * cos(2 * pi / 5))
    diff_target = trigsimp(gp_lambda2 - target)
    print(f"  2*cos(2*pi/5) = {target} = {float(target):.10f}")
    print(f"  lambda2_adj(GP(5,1)) = {gp_lambda2} = {float(gp_lambda2):.10f}")
    print(f"  Difference = {diff_target}")
    print(f"  Equal? {diff_target == 0}")

    # ---- Check: is laplacian lambda2(GP(5,1)) = 2 - 2*cos(2*pi/5)? ----
    print()
    target_lap = trigsimp(2 - 2 * cos(2 * pi / 5))
    print(f"  Is Laplacian lambda2(GP(5,1)) = 2 - 2*cos(2*pi/5)?")
    print(f"    2 - 2*cos(2*pi/5) = {target_lap} = {float(target_lap):.10f}")
    print(f"    Laplacian lambda2(GP(5,1)) = {gp_lap_lambda2} = {float(gp_lap_lambda2):.10f}")
    diff_lap_target = trigsimp(gp_lap_lambda2 - target_lap)
    print(f"    Difference = {diff_lap_target}")
    print(f"    Equal? {diff_lap_target == 0}")

    # ---- Fourier decomposition ----
    print()
    block_eigs = fourier_block_decomposition()

    print()
    print("-" * 70)
    print("5. VERIFICATION: Block eigenvalues match full spectrum?")
    print("-" * 70)

    print("\n  Eigenvalues from blocks (sorted):")
    for e in sorted(block_eigs, key=lambda x: float(x), reverse=True):
        print(f"    {e}  ({float(e):.10f})")

    print("\n  Eigenvalues from full matrix (sorted):")
    for e in sorted(gp_eigs_sorted, key=lambda x: float(x), reverse=True):
        print(f"    {e}  ({float(e):.10f})")

    # Check they match
    block_floats = sorted([float(e) for e in block_eigs], reverse=True)
    full_floats = sorted([float(e) for e in gp_eigs_sorted], reverse=True)
    match = all(abs(a - b) < 1e-12 for a, b in zip(block_floats, full_floats))
    print(f"\n  Block decomposition matches full spectrum? {match}")

    # ---- Summary ----
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("GP(5,1) eigenvalues from Fourier decomposition over Z_5:")
    print()
    print("  k  |  C_5 eigenvalue   |  Symmetric (c+1)  |  Antisymmetric (c-1)")
    print("  ---|-------------------|-------------------|---------------------")
    for k in range(5):
        c = trigsimp(2 * cos(2 * pi * k / 5))
        s = trigsimp(c + 1)
        a = trigsimp(c - 1)
        print(f"  {k}  |  {str(c):17s}|  {str(s):17s}|  {str(a)}")
    print()
    print("The symmetric modes are C_5 eigenvalues shifted by +1 (spoke coupling).")
    print("The antisymmetric modes are C_5 eigenvalues shifted by -1.")
    print()
    print("lambda2(GP(5,1)): the second-largest eigenvalue of GP(5,1)")
    print(f"  = {gp_lambda2} = {float(gp_lambda2):.10f}")
    print()
    print("This comes from the k=1 symmetric block: 2*cos(2*pi/5) + 1")
    k1_sym = trigsimp(2 * cos(2 * pi / 5) + 1)
    print(f"  = {k1_sym} = {float(k1_sym):.10f}")
    diff_k1 = trigsimp(gp_lambda2 - k1_sym)
    print(f"  Matches lambda2? Difference = {diff_k1}, Equal = {diff_k1 == 0}")
    print()

    # The key question: does lambda2(GP(5,1)) equal lambda2(C_5)?
    print("KEY RESULT:")
    print(f"  lambda2_adj(C_5)     = 2*cos(2*pi/5) = {c5_lambda2}")
    print(f"  lambda2_adj(GP(5,1)) = 2*cos(2*pi/5) + 1 = {gp_lambda2}")
    print()
    if diff_adj == 0:
        print("  THEY ARE EQUAL (adjacency lambda2).")
    else:
        print(f"  THEY ARE NOT EQUAL (adjacency lambda2). Difference = {diff_adj}")
        print(f"  GP(5,1) lambda2 = C_5 lambda2 + 1")
        print(f"  The spoke coupling shifts the symmetric modes by +1.")
    print()
    print("Laplacian perspective:")
    print(f"  Lap lambda2(C_5)     = 2 - 2*cos(2*pi/5) = {c5_lap_lambda2} = {float(c5_lap_lambda2):.10f}")
    print(f"  Lap lambda2(GP(5,1)) = 3 - lambda2_adj   = {gp_lap_lambda2} = {float(gp_lap_lambda2):.10f}")
    if diff_lap == 0:
        print("  Laplacian lambda2 values ARE EQUAL.")
    else:
        print(f"  Laplacian lambda2 values differ by: {diff_lap} = {float(diff_lap):.10f}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

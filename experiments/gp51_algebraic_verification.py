#!/usr/bin/env python3
"""
Algebraic verification: GP(5,1) vs C_5 Laplacian eigenvalues.

Proves the exact relationship between the Petersen graph GP(5,1) and the
cycle graph C_5 using symbolic arithmetic (sympy). No floating point.

Key results proved:
  1. lambda2(L_C5) = lambda2(L_GP51) = (5 - sqrt(5))/2  [EXACTLY EQUAL]
  2. lambda2(A_GP51) = lambda2(A_C5) + 1                  [shifted by spoke coupling]
  3. The Laplacian absorbs the degree shift: (3 - (c+1)) = (2 - c)
  4. Z5 Fourier decomposition explains the mechanism

The algebraic reason: GP(5,1) = C_5 □ K_2 (Cartesian product with an edge).
For k-regular G and d-regular H, the Cartesian product G □ H has
adjacency eigenvalues lambda_i(G) + mu_j(H), and is (k+d)-regular.
C_5 is 2-regular, K_2 is 1-regular, so GP(5,1) is 3-regular.
Laplacian eigenvalues of G □ H are lambda_i(L_G) + mu_j(L_H).
Since lambda2(L_K2) = 0 (K_2 has only eigenvalues 0 and 2),
the smallest nonzero Laplacian eigenvalue of GP(5,1) is
min(lambda2(L_C5) + 0, 0 + lambda1(L_K2)) = min(lambda2(L_C5), 2).
Since lambda2(L_C5) = (5-sqrt(5))/2 ≈ 1.382 < 2, the answer is lambda2(L_C5).
"""

from sympy import (
    Matrix, Rational, cos, pi, sqrt, simplify, trigsimp, expand,
    pretty, pprint, eye, zeros, Symbol, factor, cancel, nsimplify,
    det, Poly, symbols
)


# =============================================================================
# Part 1: Exact eigenvalue formulas from circulant structure
# =============================================================================

def verify_c5_laplacian_eigenvalues():
    """
    C_5 is a circulant graph with connection set {1, 4}.
    Its adjacency eigenvalues are lambda_k = 2*cos(2*pi*k/5), k=0..4.
    It is 2-regular, so Laplacian eigenvalues are mu_k = 2 - lambda_k.

    Returns the exact Laplacian eigenvalues and lambda2(L).
    """
    print("=" * 72)
    print("PART 1: C_5 LAPLACIAN EIGENVALUES (exact)")
    print("=" * 72)
    print()

    # Build adjacency matrix
    A = Matrix.zeros(5, 5)
    for i in range(5):
        A[i, (i + 1) % 5] = 1
        A[(i + 1) % 5, i] = 1

    # Build Laplacian L = D - A = 2I - A (since 2-regular)
    L = 2 * eye(5) - A

    print("C_5 adjacency matrix A:")
    pprint(A)
    print()
    print("C_5 Laplacian L = 2I - A:")
    pprint(L)
    print()

    # Compute eigenvalues symbolically
    eigs = L.eigenvals()
    print("Laplacian eigenvalues (with multiplicities):")
    lap_eigs = []
    for eig, mult in sorted(eigs.items(), key=lambda x: float(x[0])):
        eig_s = trigsimp(eig)
        print(f"  {eig_s}  (multiplicity {mult})  ≈ {float(eig_s):.10f}")
        for _ in range(mult):
            lap_eigs.append(eig_s)
    lap_eigs.sort(key=lambda x: float(x))

    # The known closed form for lambda2
    lambda2_expected = Rational(5, 2) - sqrt(5) / 2
    lambda2_computed = lap_eigs[1]  # smallest nonzero (index 1, since index 0 is 0)

    print()
    print(f"  lambda2(L_C5) computed:  {lambda2_computed}")
    print(f"  lambda2(L_C5) expected:  (5 - sqrt(5))/2 = {lambda2_expected}")
    diff = simplify(lambda2_computed - lambda2_expected)
    print(f"  Difference:              {diff}")
    assert diff == 0, f"MISMATCH: {diff}"
    print(f"  VERIFIED: lambda2(L_C5) = (5 - sqrt(5))/2 exactly.")
    print(f"  Numerical value:         {float(lambda2_expected):.15f}")
    print()

    return lap_eigs, lambda2_expected


# =============================================================================
# Part 2: GP(5,1) via Z5 Fourier block decomposition
# =============================================================================

def verify_gp51_fourier_blocks():
    """
    GP(5,1) has Z_5 symmetry. Under Z_5 Fourier decomposition, the 10x10
    adjacency matrix block-diagonalizes into five 2x2 blocks.

    For Fourier mode k (k = 0, 1, 2, 3, 4):
      The cycle contributes c_k = 2*cos(2*pi*k/5) on the diagonal
      The spoke contributes 1 on the off-diagonal

    Block B_k = [[c_k, 1], [1, c_k]]

    Eigenvalues of B_k: c_k + 1 (symmetric) and c_k - 1 (antisymmetric)

    Returns all adjacency eigenvalues and the Laplacian lambda2.
    """
    print("=" * 72)
    print("PART 2: GP(5,1) FOURIER BLOCK DECOMPOSITION")
    print("=" * 72)
    print()
    print("Under Z_5 symmetry, the 10x10 adjacency decomposes into 2x2 blocks.")
    print("Each block B_k has the cycle eigenvalue c_k on the diagonal and")
    print("spoke coupling 1 on the off-diagonal.")
    print()

    adj_eigenvalues = []
    print("  k | c_k = 2cos(2πk/5) | Symmetric (c+1)   | Antisymmetric (c-1)")
    print("  --|--------------------|--------------------|--------------------")

    for k in range(5):
        c_k = trigsimp(2 * cos(2 * pi * k / 5))

        # Build the 2x2 block and verify its eigenvalues
        B_k = Matrix([[c_k, 1], [1, c_k]])
        block_eigs = list(B_k.eigenvals().keys())
        block_eigs = [trigsimp(e) for e in block_eigs]
        block_eigs.sort(key=lambda x: float(x), reverse=True)

        sym = trigsimp(c_k + 1)
        anti = trigsimp(c_k - 1)

        # Verify the block eigenvalues match c_k ± 1
        assert simplify(block_eigs[0] - sym) == 0, f"Block k={k} sym mismatch"
        assert simplify(block_eigs[1] - anti) == 0, f"Block k={k} anti mismatch"

        print(f"  {k} | {str(c_k):18s} | {str(sym):18s} | {anti}")

        adj_eigenvalues.append(sym)
        adj_eigenvalues.append(anti)

    adj_eigenvalues.sort(key=lambda x: float(x), reverse=True)

    print()
    print("All 10 adjacency eigenvalues of GP(5,1):")
    for i, e in enumerate(adj_eigenvalues):
        label = ""
        if i == 0:
            label = "  <-- lambda1 (largest)"
        elif i == 1:
            label = "  <-- lambda2 (second largest)"
        print(f"  {str(e):25s}  ≈ {float(e):10.6f}{label}")

    # Identify lambda2_adj
    lambda2_adj = adj_eigenvalues[1]
    print()
    print(f"lambda2_adj(GP(5,1)) = {lambda2_adj}")

    # Laplacian lambda2: GP(5,1) is 3-regular, so L = 3I - A
    # Laplacian eigenvalues = 3 - adj_eigenvalues (sorted ascending)
    lap_eigenvalues = [trigsimp(3 - e) for e in adj_eigenvalues]
    lap_eigenvalues.sort(key=lambda x: float(x))

    lambda2_lap = lap_eigenvalues[1]  # smallest nonzero
    print(f"lambda2_lap(GP(5,1)) = 3 - lambda2_adj = {trigsimp(lambda2_lap)}")
    print()

    return adj_eigenvalues, lambda2_adj, lambda2_lap


# =============================================================================
# Part 3: Full matrix verification (ground truth)
# =============================================================================

def verify_full_matrix():
    """
    Build the full 10x10 GP(5,1) Laplacian and compute eigenvalues directly.
    This serves as ground truth to validate the Fourier block decomposition.
    """
    print("=" * 72)
    print("PART 3: FULL MATRIX VERIFICATION (ground truth)")
    print("=" * 72)
    print()

    # Build adjacency
    A = Matrix.zeros(10, 10)
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

    # Verify 3-regular
    for i in range(10):
        assert sum(A.row(i)) == 3, f"Row {i} sum != 3"
    print("GP(5,1) is 3-regular (verified).")

    # Build Laplacian
    L = 3 * eye(10) - A

    # Compute eigenvalues
    print("Computing eigenvalues of the full 10x10 Laplacian...")
    eigs = L.eigenvals()
    full_eigs = []
    for eig, mult in sorted(eigs.items(), key=lambda x: float(x[0])):
        eig_s = trigsimp(eig)
        print(f"  {eig_s}  (multiplicity {mult})  ≈ {float(eig_s):.10f}")
        for _ in range(mult):
            full_eigs.append(eig_s)
    full_eigs.sort(key=lambda x: float(x))

    lambda2_full = full_eigs[1]
    print()
    print(f"lambda2(L_GP51) from full matrix: {lambda2_full}")
    print()

    return full_eigs, lambda2_full


# =============================================================================
# Part 4: The algebraic proof
# =============================================================================

def algebraic_proof(c5_lambda2, gp51_adj_lambda2, gp51_lap_lambda2, gp51_lap_lambda2_full):
    """
    Prove the exact algebraic relationship between C_5 and GP(5,1).
    """
    print("=" * 72)
    print("PART 4: THE ALGEBRAIC PROOF")
    print("=" * 72)
    print()

    # ------------------------------------------------------------------
    # Claim 1: lambda2(A_GP51) = lambda2(A_C5) + 1
    # ------------------------------------------------------------------
    print("CLAIM 1: lambda2(A_GP51) = lambda2(A_C5) + 1")
    print("-" * 50)

    c5_adj_lambda2 = trigsimp(2 * cos(2 * pi / 5))
    print(f"  lambda2(A_C5)   = 2*cos(2π/5) = {c5_adj_lambda2}")
    print(f"  lambda2(A_GP51) = {gp51_adj_lambda2}")

    # lambda2(A_GP51) comes from the k=1 symmetric mode: c_1 + 1
    k1_sym = trigsimp(2 * cos(2 * pi / 5) + 1)
    diff1 = simplify(gp51_adj_lambda2 - k1_sym)
    assert diff1 == 0, f"lambda2(A_GP51) != c_1 + 1: diff = {diff1}"
    print(f"  lambda2(A_GP51) = 2*cos(2π/5) + 1 = {k1_sym}")

    diff_spoke = simplify(gp51_adj_lambda2 - c5_adj_lambda2 - 1)
    assert diff_spoke == 0, f"Spoke shift not exactly 1: diff = {diff_spoke}"
    print(f"  Difference: lambda2(A_GP51) - lambda2(A_C5) = {simplify(gp51_adj_lambda2 - c5_adj_lambda2)}")
    print(f"  PROVED: Adjacency lambda2 values differ by exactly 1 (spoke coupling).")
    print()

    # ------------------------------------------------------------------
    # Claim 2: lambda2(L_GP51) = lambda2(L_C5) exactly
    # ------------------------------------------------------------------
    print("CLAIM 2: lambda2(L_GP51) = lambda2(L_C5)")
    print("-" * 50)

    expected = Rational(5, 2) - sqrt(5) / 2  # (5 - sqrt(5))/2
    print(f"  lambda2(L_C5)   = 2 - 2*cos(2π/5) = {c5_lambda2}")
    print(f"  lambda2(L_GP51) = 3 - (2*cos(2π/5) + 1) = {gp51_lap_lambda2}")

    # The algebra:
    # lambda2(L_GP51) = 3 - lambda2(A_GP51)
    #                 = 3 - (2*cos(2π/5) + 1)
    #                 = 2 - 2*cos(2π/5)
    #                 = lambda2(L_C5)
    print()
    print("  Algebraic proof:")
    print("    lambda2(L_GP51) = d_GP - lambda2(A_GP)")
    print("                    = 3 - (2*cos(2π/5) + 1)")
    print("                    = 2 - 2*cos(2π/5)")
    print("                    = d_C5 - lambda2(A_C5)")
    print("                    = lambda2(L_C5)")
    print()

    diff2 = simplify(gp51_lap_lambda2 - c5_lambda2)
    assert diff2 == 0, f"Laplacian lambda2 NOT equal: diff = {diff2}"
    print(f"  Symbolic verification: {gp51_lap_lambda2} - {c5_lambda2} = {diff2}")
    print(f"  PROVED: Laplacian lambda2 values are exactly equal.")
    print()

    # Cross-check with full matrix computation
    diff3 = simplify(gp51_lap_lambda2_full - c5_lambda2)
    assert diff3 == 0, f"Full matrix cross-check failed: diff = {diff3}"
    print(f"  Cross-check with full 10x10 matrix: difference = {diff3}")
    print(f"  CONFIRMED by independent full-matrix computation.")
    print()

    # ------------------------------------------------------------------
    # Claim 3: The degree shift absorbs the spoke coupling
    # ------------------------------------------------------------------
    print("CLAIM 3: The degree shift absorbs the spoke coupling")
    print("-" * 50)
    print()
    print("  For any k-regular graph G, Laplacian eigenvalue = k - adj eigenvalue.")
    print("  C_5 is 2-regular.  GP(5,1) is 3-regular.  Degree difference = 1.")
    print()
    print("  The spoke adds +1 to every adjacency eigenvalue's symmetric mode.")
    print("  The spoke also adds +1 to the degree (regularity).")
    print("  These two +1 shifts cancel in the Laplacian: (k+1) - (lambda+1) = k - lambda.")
    print()
    print("  This is NOT a coincidence. It follows from GP(5,1) = C_5 □ K_2:")
    print("    - Cartesian product: adjacency eigenvalues are lambda_i + mu_j")
    print("    - K_2 has adjacency eigenvalues {+1, -1}")
    print("    - The symmetric mode picks up mu = +1, the antisymmetric picks up mu = -1")
    print("    - For the Laplacian: eigenvalues of L_{G□H} = eigenvalues of L_G ⊕ L_H")
    print("    - L_K2 has eigenvalues {0, 2}")
    print("    - So Laplacian eigenvalues of GP(5,1) = {lambda_i(L_C5) + 0} ∪ {lambda_i(L_C5) + 2}")
    print("    - The smallest nonzero is min(lambda2(L_C5), 0+2) = lambda2(L_C5)")
    print("    - since (5-√5)/2 ≈ 1.382 < 2")
    print()

    # Verify the Cartesian product claim
    print("  Verification of Cartesian product Laplacian spectrum:")
    L_K2_eigs = [0, 2]
    c5_lap_eigs_raw = []
    for k in range(5):
        mu = trigsimp(2 - 2 * cos(2 * pi * k / 5))
        c5_lap_eigs_raw.append(mu)

    cartesian_lap_eigs = []
    for mu_c5 in c5_lap_eigs_raw:
        for mu_k2 in L_K2_eigs:
            cartesian_lap_eigs.append(trigsimp(mu_c5 + mu_k2))
    cartesian_lap_eigs.sort(key=lambda x: float(x))

    print("    L_C5 eigenvalues:")
    for e in sorted(c5_lap_eigs_raw, key=lambda x: float(x)):
        print(f"      {e}  ≈ {float(e):.6f}")
    print("    L_K2 eigenvalues: {0, 2}")
    print("    Cartesian product L_{C5□K2} eigenvalues (= L_C5 ⊕ L_K2):")
    for e in cartesian_lap_eigs:
        print(f"      {e}  ≈ {float(e):.6f}")
    print()

    # These should match the full GP(5,1) Laplacian eigenvalues
    # (We'll verify in the summary)

    return expected, cartesian_lap_eigs


# =============================================================================
# Part 5: Normalized algebraic connectivity
# =============================================================================

def normalized_analysis(c5_lambda2, gp51_lap_lambda2):
    """
    Show that the normalized algebraic connectivity is preserved.
    """
    print("=" * 72)
    print("PART 5: NORMALIZED ALGEBRAIC CONNECTIVITY")
    print("=" * 72)
    print()

    # Cheeger-like normalization: lambda2 / d
    c5_norm = trigsimp(c5_lambda2 / 2)
    gp51_norm = trigsimp(gp51_lap_lambda2 / 3)

    print("  Normalization by degree (lambda2 / d):")
    print(f"    C_5:     lambda2/d = {c5_lambda2}/2 = {c5_norm}  ≈ {float(c5_norm):.10f}")
    print(f"    GP(5,1): lambda2/d = {gp51_lap_lambda2}/3 = {gp51_norm}  ≈ {float(gp51_norm):.10f}")
    diff_norm = simplify(c5_norm - gp51_norm)
    print(f"    Difference: {diff_norm}  ≈ {float(diff_norm):.10f}")
    print(f"    These are NOT equal (different denominators).")
    print()

    # The spectral gap normalization: lambda2 / lambda_max(L)
    # For k-regular graph, lambda_max(L) <= 2k (equality for bipartite)
    # C_5 not bipartite: lambda_max(L_C5) = 2 + 2*cos(2*pi*2/5) = 2 - (1+sqrt(5))/2 = (3-sqrt(5))/2... no
    # Actually lambda_max(L_C5) = 2 - min_adj_eig
    # min adj eig of C_5 is 2*cos(4*pi/5) = -(1+sqrt(5))/2
    c5_adj_min = trigsimp(2 * cos(2 * pi * 2 / 5))  # k=2
    c5_lap_max = trigsimp(2 - c5_adj_min)
    gp51_adj_min = trigsimp(2 * cos(2 * pi * 2 / 5) - 1)  # k=2, antisymmetric mode
    gp51_lap_max = trigsimp(3 - gp51_adj_min)

    print("  Spectral gap normalization (lambda2 / lambda_max):")
    print(f"    C_5:     lambda_max(L) = {c5_lap_max}  ≈ {float(c5_lap_max):.6f}")
    print(f"    GP(5,1): lambda_max(L) = {gp51_lap_max}  ≈ {float(gp51_lap_max):.6f}")

    c5_ratio = trigsimp(c5_lambda2 / c5_lap_max)
    gp51_ratio = trigsimp(gp51_lap_lambda2 / gp51_lap_max)
    print(f"    C_5:     lambda2/lambda_max = {c5_ratio}  ≈ {float(c5_ratio):.10f}")
    print(f"    GP(5,1): lambda2/lambda_max = {gp51_ratio}  ≈ {float(gp51_ratio):.10f}")
    diff_ratio = simplify(c5_ratio - gp51_ratio)
    print(f"    Difference: {simplify(diff_ratio)}  ≈ {float(diff_ratio):.10f}")
    if diff_ratio == 0:
        print("    EQUAL: spectral gap ratio is preserved.")
    else:
        print("    NOT equal (the spectral gap ratio is not preserved).")
    print()

    # What IS preserved is the absolute value of lambda2(L)
    print("  What IS preserved (exactly):")
    print(f"    lambda2(L_C5)     = {c5_lambda2} = (5-√5)/2")
    print(f"    lambda2(L_GP(5,1)) = {gp51_lap_lambda2} = (5-√5)/2")
    print(f"    The absolute algebraic connectivity is identical.")
    print(f"    The bottleneck structure of C_5 is fully inherited by GP(5,1).")
    print()


# =============================================================================
# Part 6: Summary
# =============================================================================

def print_summary(c5_lambda2, gp51_adj_lambda2, gp51_lap_lambda2, cartesian_eigs, full_eigs):
    """Print the final summary of all results."""
    print("=" * 72)
    print("SUMMARY OF ALGEBRAIC VERIFICATION")
    print("=" * 72)
    print()
    print("Graph          | Regularity | lambda2(A)              | lambda2(L)")
    print("---------------|------------|-------------------------|------------------")
    c5_adj_l2 = trigsimp(2 * cos(2 * pi / 5))
    print(f"C_5            | 2-regular  | 2cos(2π/5) = {str(c5_adj_l2):10s} | (5-√5)/2 = {c5_lambda2}")
    print(f"GP(5,1)        | 3-regular  | 2cos(2π/5)+1 = {str(gp51_adj_lambda2):8s} | (5-√5)/2 = {gp51_lap_lambda2}")
    print()
    print("EXACT RESULTS:")
    print(f"  1. lambda2(A_GP51) - lambda2(A_C5) = 1          (spoke coupling)")
    print(f"  2. lambda2(L_GP51) = lambda2(L_C5) = (5-√5)/2   (IDENTICAL)")
    print(f"  3. GP(5,1) = C_5 □ K_2 (Cartesian product)")
    print(f"  4. L_{{GP51}} spectrum = L_{{C5}} ⊕ L_{{K2}} spectrum")
    print(f"  5. The Z_5 Fourier modes are preserved; spoke coupling = K_2 factor")
    print()
    print("WHY THE LAPLACIAN lambda2 IS PRESERVED:")
    print("  The spoke coupling adds +1 to every symmetric adjacency eigenvalue.")
    print("  But GP(5,1) is 3-regular (one more than C_5's 2-regular).")
    print("  Laplacian = degree - adjacency, so the +1 shifts cancel exactly:")
    print("    L_GP lambda2 = 3 - (lambda2(A_C5) + 1) = 2 - lambda2(A_C5) = L_C5 lambda2")
    print()
    print("PAPER IMPLICATION:")
    print("  The algebraic connectivity lambda2(L) — which governs mixing time,")
    print("  synchronization threshold, and coupling onset — is determined entirely")
    print("  by the cycle substructure C_5. The spoke connections add bandwidth")
    print("  (degree) but do not change the bottleneck. This is the categorical")
    print("  content: the laxator captures the C_5 bottleneck regardless of the")
    print("  spoke structure.")
    print()

    # Final cross-check: Cartesian product spectrum vs full matrix spectrum
    print("FINAL CROSS-CHECK: Cartesian product vs full matrix Laplacian spectrum")
    cartesian_sorted = sorted(cartesian_eigs, key=lambda x: float(x))
    full_sorted = sorted(full_eigs, key=lambda x: float(x))
    all_match = True
    for i, (c, f) in enumerate(zip(cartesian_sorted, full_sorted)):
        d = simplify(c - f)
        status = "OK" if d == 0 else f"MISMATCH (diff={d})"
        if d != 0:
            all_match = False
        print(f"  eigenvalue {i:2d}: Cartesian = {str(c):25s}  Full = {str(f):25s}  {status}")

    print()
    if all_match:
        print("  ALL EIGENVALUES MATCH: GP(5,1) = C_5 □ K_2 confirmed spectrally.")
    else:
        print("  WARNING: Some eigenvalues do not match.")
    print()
    print("=" * 72)
    print("VERIFICATION COMPLETE — ALL CLAIMS PROVED WITH EXACT ARITHMETIC")
    print("=" * 72)


# =============================================================================
# Main
# =============================================================================

def main():
    print()
    print("  GP(5,1) vs C_5: Algebraic Eigenvalue Verification")
    print("  Using sympy exact arithmetic (no floating point)")
    print()

    # Part 1: C_5
    c5_lap_eigs, c5_lambda2 = verify_c5_laplacian_eigenvalues()

    # Part 2: GP(5,1) via Fourier blocks
    gp51_adj_eigs, gp51_adj_lambda2, gp51_lap_lambda2 = verify_gp51_fourier_blocks()

    # Part 3: Full matrix ground truth
    full_eigs, gp51_lap_lambda2_full = verify_full_matrix()

    # Part 4: The algebraic proof
    expected, cartesian_eigs = algebraic_proof(
        c5_lambda2, gp51_adj_lambda2, gp51_lap_lambda2, gp51_lap_lambda2_full
    )

    # Part 5: Normalized analysis
    normalized_analysis(c5_lambda2, gp51_lap_lambda2)

    # Part 6: Summary
    print_summary(c5_lambda2, gp51_adj_lambda2, gp51_lap_lambda2, cartesian_eigs, full_eigs)


if __name__ == "__main__":
    main()

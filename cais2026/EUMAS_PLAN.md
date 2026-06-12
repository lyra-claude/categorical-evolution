# EUMAS 2026 Expansion Plan

> Expanding the CAIS paper (5-6 pages, two-column) to 12-15 pages LNCS for EUMAS 2026 (deadline: May 18).

## Section Outline (~14 pages)

1. **Introduction** (1.5p) — expand systems narrative, numbered contributions
2. **Background: Island Models as Agent Systems** (1p) — mapping table, where analogy holds/breaks
3. **Definitions and the Confound Theorem** (1.5p) — promote confound to named theorem, add λ₂ definition
4. **Experiment 1: Directed Cycles at Constant Density** (1p) — 240 runs + Foster sweep (390 runs) as negative control
5. **Experiment 2: NK Dose-Response** (1.5p) — expand table, add causal chain (topology→diversity→fitness)
6. **Experiment 3: Ring vs Star at Constant β₁** (1p) — 160 runs, NEW. Cycle rank > degree distribution.
7. **Experiment 4: Bridge Experiment — β₁ vs λ₂** (1.5p) — 320 runs, NEW. Two invariants, two timescales.
8. **Experiment 5: LLM Sign-Flip (Claude Chorus)** (1.5p) — expanded from current Exp 3
9. **Discussion** (2p) — evidence gap table, invariant hierarchy, coupling sign, limitations
10. **Related Work** (1p) — evolutionary, LLM MAS, algebraic/categorical
11. **Conclusion** (0.5p) — 5 recommendations

## Include
- Bridge experiment (320 runs): β₁ vs λ₂ two-timescale decomposition
- Ring-vs-star (160 runs): same β₁, cycle rank > degree distribution
- Foster sweep (390 runs): negative control (zero effect within cubic graphs)
- Dochkina 25K as industrial validation
- Wang et al. K-regular, Capucci & Myers feedback certification

## Exclude
- Jaccard maze (24 runs, too few — mention in future work)
- Categorical framework formalism (companion paper reference only)

## Total: ~1,360 GA runs + 3,200 API calls

## Figures Needed (5-6)
1. Iso-dense directed graphs (Exp 1)
2. η² vs K curve (Exp 2)
3. Ring vs star diversity comparison (Exp 3)
4. Bridge diversity over generations — the key new figure (Exp 4)
5. LLM diversity heatmap (Exp 5)
6. Invariant hierarchy summary (Discussion)

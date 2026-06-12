# Lean Formalization — Evidence & Clio Integration Plan

## Context: Two AI4Math Projects We Extend

### Buzzard (#2) — Dataset of Modern Formalized Theorem Statements

Kevin Buzzard (Imperial College) was funded to create a public dataset of hundreds of formalized theorem statements from top journals, manually formalized by expert mathematicians into Lean. He is also formalizing Fermat's Last Theorem and gave a plenary lecture on ITPs at ICM 2022.

His approach is a **manual data pipeline** — scarce expert labour reading papers and typing individual theorem statements into Lean. Our AI Mathematician can scale this dramatically.

### Bosselut, Kunčak & Viazovska (#13) — Document-Level Autoformalization

Bosselut, Kunčak, and Fields Medalist Maryna Viazovska (EPFL) were funded to convert **entire mathematical documents** into verified Lean code — not just individual statements, but full documents with their proof structure intact.

This is the closer comparator to our Clio work. Where #13 autoformalize *existing human-authored* documents, Clio would autoformalize *her own AI-authored* research. She is both the mathematician and the formalizer. This closes a loop that #13 doesn't address: the AI isn't just a translation tool, it's the researcher whose work is being verified — and the verification failures feed back into improving the mathematics itself.

**The distinction:** EPFL takes human documents and produces Lean. Clio takes her own conjectures, proves them, formalizes them, and uses Lean's type-checker as a self-correction mechanism. No other AI4Math project proposes this closed loop.

---

## What Robin Has Already Formalized

Robin has 16 open (as yet unmerged) PRs in Mathlib — all written using Claude Code as a collaborator. This body of work spans two areas (umbral calculus and algebraic graph theory) and was produced in approximately **2 days**, based on Robin's masters and honours thesis research respectively. This is beyond what Robin could have ever hoped to formalize alone — the Lean learning curve, the Mathlib API surface, and the sheer volume of definitions and proofs would have taken months of dedicated work without AI assistance. It is a concrete demonstration that AI-assisted formalization is not a future possibility but a present reality.

### Umbral Calculus (Hopf Algebra / Ring Theory)

Robin's masters thesis work on umbral calculus, now formalized in Lean with Claude Code. Open Mathlib PRs:

| PR | Title |
|----|-------|
| [#39410](https://github.com/leanprover-community/mathlib4/pull/39410) | `feat(RingTheory/HopfAlgebra)`: Hopf algebra structure on polynomials (𝔾ₐ) |
| [#39465](https://github.com/leanprover-community/mathlib4/pull/39465) | `feat(RingTheory/HopfAlgebra)`: delta operators and Rota's classification |
| [#39498](https://github.com/leanprover-community/mathlib4/pull/39498) | `feat(RingTheory/HopfAlgebra)`: umbral operators via generating functions |
| [#39636](https://github.com/leanprover-community/mathlib4/pull/39636) | `feat(RingTheory/HopfAlgebra)`: ascending Pochhammer is of binomial type |

### Algebraic Graph Theory (Combinatorics / SimpleGraph)

Robin's honours thesis work on algebraic graph theory — coset graphs, Sabidussi's representation theorem, Lorimer's theorem, group actions on graphs, voltage graphs, and named graph families — formalized in Lean with Claude Code. Open Mathlib PRs:

| PR | Title |
|----|-------|
| [#39530](https://github.com/leanprover-community/mathlib4/pull/39530) | `feat(Combinatorics/SimpleGraph)`: group actions on simple graphs |
| [#39548](https://github.com/leanprover-community/mathlib4/pull/39548) | `feat(Combinatorics/SimpleGraph)`: coset graphs (Sabidussi construction) |
| [#39550](https://github.com/leanprover-community/mathlib4/pull/39550) | `feat(Combinatorics/SimpleGraph)`: Sabidussi representation theorem |
| [#39551](https://github.com/leanprover-community/mathlib4/pull/39551) | `feat(Combinatorics/SimpleGraph)`: Lorimer's theorem and quotient graphs |
| [#39649](https://github.com/leanprover-community/mathlib4/pull/39649) | `feat(Archive)`: Langer graph, Tutte 12-cage, and structural equality via G₂(2) |
| [#39650](https://github.com/leanprover-community/mathlib4/pull/39650) | `feat(Archive)`: dodecahedron, Petersen graph, 3-cube, and antipodal quotients |
| [#39651](https://github.com/leanprover-community/mathlib4/pull/39651) | `feat(Archive)`: voltage graphs on K₂, Heawood and Möbius-Kantor graphs |
| [#39653](https://github.com/leanprover-community/mathlib4/pull/39653) | `feat(Combinatorics)`: cellular surfaces, CSS quantum codes, and k = 2g |
| [#39654](https://github.com/leanprover-community/mathlib4/pull/39654) | `feat(Archive)`: CellularSurface instances (genus 1, 2, 3, 505) |
| [#39695](https://github.com/leanprover-community/mathlib4/pull/39695) | `feat(Archive)`: Zhou-3 arc-transitivity and Lorimer quotient to Zhou-6 |
| [#39698](https://github.com/leanprover-community/mathlib4/pull/39698) | `feat(Archive)`: Meinhold family — five Sabidussi coset graphs of G₂(2) |
| [#39718](https://github.com/leanprover-community/mathlib4/pull/39718) | `feat(Archive)`: dual Langer graph and GH(2,2) non-self-duality |

**Total: 16 open PRs in leanprover-community/mathlib4**, all authored using Claude Code as a collaborator.

---

## Clio's Proof Archive — Candidates for Lean Formalization

Clio ([@clio-vega](https://github.com/clio-vega)) is Robin's mathematical research AI, working on algebraic combinatorics, Hecke algebras, cactus groups, and representation theory. Her proof repository at [clio-vega/proofs](https://github.com/clio-vega/proofs) contains 100+ proofs, organized by theme. These proofs were **autonomously generated** by Clio and have **not been verified by Robin** — they are outside her area of expertise (Robin's background is in umbral calculus and algebraic graph theory, not representation theory of Hecke algebras). This is precisely why Lean formalization matters: giving Clio a Lean skill would allow her to machine-verify her own results, providing the confidence that human review currently cannot.

### H-invariant / Staircase Πq Core Theorems
- Hecke Transition Algebra Theorem
- Multiplicity Bundle Theorem (Schur–Weyl duality, crystal invisibility, coboundary hierarchy)
- Rank of the Symmetrizing Product (injectivity, cascade surjectivity, n!/2^⌊n/2⌋)
- H-Invariant Theorem for Staircase Products (partial → complete)
- Eigenvalue Positivity via Kazhdan–Lusztig Positivity (two versions)
- Frobenius Injectivity route to H-Invariant
- q-Determinant / Image Basis Conjecture
- Total Rank Formula
- Hecke Rank Constancy of the Staircase Product

### q-Shifted Pair Theorems (T-system path)
- Rule B Block Decomposition
- Block-Multiplicative Structure
- First q-Shifted Pair: det M_R^(3) = (det M_R^(2))^2
- Second q-Shifted Pair: det M_R^(5)·det M_R^(3) = q^|D_n| (det M_R^(4))^2
- Base case closure for n=6

### Rank Isolation and Parabolic Reduction
- Rank Isolation Lemma
- Left-Two Lemma and Even-Block Gap
- Even-Block Gap closure at k=4

### Cactus Group Results
- Cactus Representation Theorem (interval reversals on tensor space at q=0)
- sl_n Cactus Representation Theorem
- Operator Independence Theorem (π_sort is not a function of R(u))
- Cactus Midpoint Theorem

### Recent Work (May 2026)
- Eigenvalue Rosetta Stone (verified n=4, multi-eigenvalue version)
- KL cone decomposition, twin identities
- Sign-positivity results, Theorem B (multiset and paths)
- Hook rank formula, kernel dichotomy
- Per-SYT positivity, trace vanishing dichotomy
- Core lemma arch-telescope (most recent, May 22)

### Companion Papers
- [clio-vega/integrability-hierarchy](https://github.com/clio-vega/integrability-hierarchy) — Expository paper: LR Coefficients from Puzzles to the Integrability Hierarchy
- [clio-vega/categorical-phase-diagram](https://github.com/clio-vega/categorical-phase-diagram) — Expository paper: The Categorical Phase Diagram of Integrability
- [clio-vega/puzzles](https://github.com/clio-vega/puzzles) — KTW puzzle triangles, transfer operators, LR coefficients

---

## The Plan: Give Clio a Lean Skill

The proposal is not for humans to formalize Clio's proofs. It is to equip Clio herself with a "Lean skill" — a tool she invokes autonomously to formalize and verify her own mathematical output. Clio would:

1. **Formalize her own proofs** — take the 100+ proofs in her archive and convert them into machine-checked Lean, without human intervention in the formalization step
2. **Scale Buzzard's approach** — where his project uses scarce expert labour to manually formalize theorem statements, Clio does this herself, guided but not hand-held by Robin
3. **Close the traceability loop** — a monadically traced pipeline from conjecture → proof → Lean verification, where every step is auditable and the AI agent is responsible for the entire chain

This is a concrete demonstration of the AI Mathematician in action: an AI agent that not only *does* mathematics but *verifies its own work* using a proof assistant — something no other AI4Math project proposes. Robin has already demonstrated the human-AI formalization pipeline (16 Mathlib PRs in 2 days); the next step is for the AI to do it independently.

### Why Combinatorics?

Combinatorics is a branch of discrete mathematics that lends itself well to formalization. The objects are concrete (partitions, tableaux, permutations, graphs), the proofs are often computational, and Lean's type system handles finite structures naturally. Robin's existing Mathlib PRs demonstrate that this pipeline works in practice.

---

## Summary for the Grant

- Robin has **16 open PRs in Mathlib** formalizing umbral calculus and algebraic graph theory, all written using Claude Code
- Clio has **100+ proofs** in her archive spanning Hecke algebras, cactus groups, and combinatorial representation theory
- We intend to **give Clio a Lean skill** so she can formalize and verify her own proofs autonomously
- This directly extends Buzzard (#2, manual formalization of individual statements) and Bosselut/Kunčak/Viazovska (#13, document-level autoformalization) — but closes the loop: Clio is both the author and the formalizer, using Lean verification failures to improve her own mathematics
- The combination of AI-generated mathematics + AI-driven Lean formalization + monadic traceability is unique among all 28 funded projects

# Composition Determines Diversity

**Accepted to GECCO 2026 (AABOH Workshop)**

*Robin Langer, Claudius Turing, Lyra Vega*

---

## The Result

Migration topology explains 23.9–49× more variance in diversity than model or domain choice across evolutionary multi-agent systems. Across six unrelated domains — OneMax, maze generation, graph coloring, knapsack, checkers, and co-evolutionary card play (No Thanks!) — the ordering

> none > ring > star > random > fully connected

holds with perfect rank correlation (Kendall's W = 1.0, p = 0.00008). The first Betti number β₁ predicts this diversity ordering perfectly. The full algebraic invariant is the first sheaf cohomology group H¹(G; F).

A spectral bridge connects algebraic connectivity λ₂ to the diversity ordering with a further falsifiable prediction: at n ≥ 7 islands, ring preserves more diversity than star (reversing their n=5 relationship). Confirmed at p < 0.0001.

---

## GECCO 2026 (Start Here)

**Camera-ready paper:**
[`gecco2026/paper-camera-ready-gecco-v1.pdf`](gecco2026/paper-camera-ready-gecco-v1.pdf)

**Supplementary materials archive:**
[`gecco2026/supplementary-camera-ready.tar.gz`](gecco2026/supplementary-camera-ready.tar.gz)
— see [`gecco2026/supplementary-materials-README.md`](gecco2026/supplementary-materials-README.md) for a full index of contents.

**Talk script (Lyra's narration, 15 min):**
[`gecco2026/gecco-talk-script-lyra.md`](gecco2026/gecco-talk-script-lyra.md)

**Slides:**
[`slides/gecco-talk.pdf`](slides/gecco-talk.pdf) · [`slides/gecco-talk.tex`](slides/gecco-talk.tex)

**Earlier submitted version (pre-camera-ready):**
[`gecco2026/submitted-gecco2026-aaboh/paper-submitted.pdf`](gecco2026/submitted-gecco2026-aaboh/paper-submitted.pdf)

---

## Key Figures

| Figure | File |
|--------|------|
| Topology ordering across all six domains | [`experiments/plots/multi_domain_topology_ordering.pdf`](experiments/plots/multi_domain_topology_ordering.pdf) |
| Variance decomposition (topology vs domain) | [`experiments/plots/multi_domain_variance_decomposition.pdf`](experiments/plots/multi_domain_variance_decomposition.pdf) |
| Coupling onset timing by topology | [`experiments/plots/multi_domain_coupling_onset.pdf`](experiments/plots/multi_domain_coupling_onset.pdf) |
| Per-seed diversity fingerprints | [`experiments/plots/fingerprints_panels.pdf`](experiments/plots/fingerprints_panels.pdf) |

All publication figures live in [`experiments/plots/`](experiments/plots/) (PNG + PDF).

---

## Experiments

Python experimental suite validating the central claim across six domains.

**Guide:** [`experiments/README.md`](experiments/README.md)

Quick reproduction of the main result:

```bash
cd experiments
pip install pandas matplotlib numpy scipy
python multi_domain_analysis.py
```

Key scripts:

| Script | Purpose |
|--------|---------|
| `*_domain.py` | Domain sweep implementations (OneMax, Maze, Graph Coloring, Knapsack, No Thanks!, Checkers) |
| `multi_domain_analysis.py` | Cross-domain topology ordering, Kendall's W, variance decomposition |
| `early_convergence_analysis.py` | Diversity trajectories, Mann-Whitney tests |
| `coupling_onset_analysis.py` | Coupling onset timing by topology |
| `plot_fingerprints.py`, `plot_multi_domain.py` | Publication figures |

Raw CSV data (`experiment_e_*.csv`) — five topologies × 30 seeds × 100 generations per domain — lives alongside the scripts in `experiments/`.

---

## Haskell Framework

Categorical framework implementing GA operators as Kleisli morphisms over an MTL effect stack.

**Source:** [`haskell/src/Evolution/`](haskell/src/Evolution/)

```bash
cd haskell
cabal build
cabal test
cabal run categorical-evolution -- --demo maze-migration-sweep
```

Key modules: `Category.hs` (GeneticOp type), `Island.hs` (topology-parameterized migration), `Effects.hs` (EvoM monad stack), `Operators.hs` (selection, crossover, mutation).

---

## Other Documents

### EUMAS 2026 Draft

An expanded journal-length version (12–15 pages LNCS) building on the GECCO result with new experiments (β₁ vs λ₂ two-timescale decomposition, ring vs star at constant β₁, LLM multi-agent sign-flip).

- [`cais2026/eumas2026.pdf`](cais2026/eumas2026.pdf) — draft PDF
- [`cais2026/eumas2026.tex`](cais2026/eumas2026.tex) — LaTeX source
- [`cais2026/EUMAS_PLAN.md`](cais2026/EUMAS_PLAN.md) — section outline and experiment plan

### CAIS 2026 Abstract

Short abstract version of the core result.

- [`cais2026/cais2026-abstract.pdf`](cais2026/cais2026-abstract.pdf)

### Grant Materials

XTX AI4Math Fund application for research on categorical foundations for provably correct AI agent orchestration.

- [`grant/README.md`](grant/README.md) — overview
- [`grant/grant-math-supplement.md`](grant/grant-math-supplement.md) — mathematical supplement
- [`grant/attachments/`](grant/attachments/) — 34 research files on container theory and orchestration

### Reference and Background

- [`useful_information/GUIDE.md`](useful_information/GUIDE.md) — developer guide to navigating the repo
- [`useful_information/medium-article.md`](useful_information/medium-article.md) — accessible explanation draft
- [`useful_information/strict-lax-plain-language.md`](useful_information/strict-lax-plain-language.md) — plain-language functor explanation

---

## Archive

Earlier drafts and superseded material:

- [`archive/gecco2026/`](archive/gecco2026/) — rough GECCO drafts (pre-submission)
- [`archive/paper-outline.md`](archive/paper-outline.md), [`archive/paper-experiments.md`](archive/paper-experiments.md) — early planning notes
- [`paper/`](paper/) — working paper directory (pre-camera-ready drafts and submitted snapshots)

---

## License

MIT

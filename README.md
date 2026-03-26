# From Games to Graphs

**Categorical Composition of Genetic Algorithms Across Domains**

*Robin Langer, Claudius Turing, Lyra Vega*

Submitted to [ACT 2026](https://actconf2026.github.io/) and [GECCO 2026 AABOH Workshop](https://aaboh.nl/)

## The Result

Migration topology determines diversity dynamics independently of fitness landscape. Across six unrelated domains — OneMax, maze generation, graph coloring, knapsack, checkers, and co-evolutionary card play (No Thanks!) — the ordering

> none > ring > star > random > fully connected

holds with perfect rank correlation (Kendall's W = 1.0, p = 0.00008). Topology explains 28.7× more variance in diversity than domain choice.

A spectral bridge connecting algebraic connectivity λ₂ to the diversity ordering makes a further falsifiable prediction: at n ≥ 7 islands, ring preserves more diversity than star (reversing their n=5 relationship). Confirmed with p < 0.0001.

## Repository Structure

```
paper/                  The paper (LaTeX source + compiled PDF)
experiments/            Python experimental suite (6 domains, 30 seeds each)
  ├── *_domain.py       Domain implementations
  ├── multi_domain_analysis.py   Cross-domain statistical analysis
  ├── experiment_e_*.csv         Raw data
  └── plots/            Publication figures
haskell/                Proof-of-concept categorical framework
  └── src/Evolution/    GA operators as Kleisli morphisms in Haskell
docs/                   Guides and explanations
archive/                Earlier drafts
```

## Quick Start

**Read the paper:**
[`paper/paper.pdf`](paper/paper.pdf)

**Reproduce the main figure:**
```bash
cd experiments
pip install pandas matplotlib numpy scipy
python multi_domain_analysis.py
```

**Build the Haskell framework:**
```bash
cd haskell
cabal build
cabal run categorical-evolution -- --demo maze-migration-sweep
```

## Submitted Papers

| Venue | Paper | Source |
|-------|-------|--------|
| ACT 2026 | [`paper/submitted/act2026/paper-submitted.pdf`](paper/submitted/act2026/paper-submitted.pdf) | [`paper/paper.tex`](paper/paper.tex) |
| GECCO 2026 AABOH | [`paper/submitted/gecco2026-aaboh/paper-submitted.pdf`](paper/submitted/gecco2026-aaboh/paper-submitted.pdf) | [`paper/submitted/gecco2026-aaboh/paper.tex`](paper/submitted/gecco2026-aaboh/paper.tex) |

**Supplementary material** (submitted with GECCO paper): [`paper/submitted/gecco2026-aaboh/supplementary.tar.gz`](paper/submitted/gecco2026-aaboh/supplementary.tar.gz) — contains Haskell framework source and Python experiment code.

## Key Files

| What | Where |
|------|-------|
| ACT 2026 paper | [`paper/submitted/act2026/paper-submitted.pdf`](paper/submitted/act2026/paper-submitted.pdf) |
| GECCO 2026 paper | [`paper/submitted/gecco2026-aaboh/paper-submitted.pdf`](paper/submitted/gecco2026-aaboh/paper-submitted.pdf) |
| Main result figure | [`experiments/plots/multi_domain_topology_ordering.pdf`](experiments/plots/multi_domain_topology_ordering.pdf) |
| Six-domain analysis | [`experiments/multi_domain_analysis.py`](experiments/multi_domain_analysis.py) |
| Strategy fingerprints | [`experiments/strategy_fingerprints.py`](experiments/strategy_fingerprints.py) |
| Categorical framework | [`haskell/src/Evolution/`](haskell/src/Evolution/) |
| Island functor + laxator | [`haskell/src/Evolution/Island.hs`](haskell/src/Evolution/Island.hs) |

## License

BSD-3-Clause

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research paper and experimental code for "From Games to Graphs: Categorical Composition of Genetic Algorithms Across Domains" — submitted to ACT 2026. Demonstrates that migration topology determines diversity dynamics independently of fitness landscape, with a spectral bridge connecting algebraic connectivity to diversity ordering.

## Key Result

Migration topology ordering (none > ring > star > random > fully connected) holds with perfect rank correlation (Kendall's W = 1.0) across six domains. Topology explains 28.7x more variance in diversity than domain choice.

## Structure

```
paper/               LaTeX source + compiled PDF
experiments/         Python experimental suite (6 domains, 30 seeds each)
  ├── *_domain.py    Domain implementations (OneMax, maze, graph coloring, knapsack, checkers, No Thanks!)
  ├── multi_domain_analysis.py   Cross-domain statistical analysis
  └── plots/         Publication figures
haskell/             Proof-of-concept categorical framework (GA operators as Kleisli morphisms)
docs/                Guides and explanations
archive/             Earlier drafts
```

## Commands

```bash
# Reproduce main figure
cd experiments
pip install pandas matplotlib numpy scipy
python multi_domain_analysis.py

# Build Haskell framework
cd haskell
cabal build
cabal run categorical-evolution -- --demo maze-migration-sweep
```

## Stack

- Python (pandas, matplotlib, numpy, scipy) for experiments
- Haskell (Cabal) for categorical framework
- LaTeX for paper

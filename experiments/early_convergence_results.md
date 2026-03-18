# Early Convergence Analysis

**Date:** 2026-03-18
**Hypothesis:** Ring topology establishes its diversity advantage before generation 30.
**Method:** Mann-Whitney tests + Cohen's d at gen 10, 30, 99 across 6 domains.

## Ring vs Star: Cohen's d Over Time

| Domain | Gen 10 | Gen 30 | Gen 99 | Trend |
|--------|--------|--------|--------|-------|
| Maze | -0.064 | -0.099 | 0.643* | LATE (growing) |
| OneMax | 0.739 | 32.373 | 0.526 | EARLY (stable) |
| Graph Coloring | 0.158 | 0.261 | 0.509 | LATE (growing) |
| Knapsack | -0.029 | 0.278 | 0.158 | MID (by gen 30) |
| No Thanks! | 0.060 | 0.338 | 0.475 | LATE (growing) |
| Checkers | -0.227 | 0.285 | 0.577* | LATE (growing) |

## Ring vs FC: Cohen's d Over Time

| Domain | Gen 10 | Gen 30 | Gen 99 | p (Gen 10) | p (Gen 99) |
|--------|--------|--------|--------|------------|------------|
| Maze | 0.659* | 2.455*** | 2.989*** | 0.0224 | 0.0000 |
| OneMax | 2.611 | 12.358 | 2.985 | 0.3333 | 0.3333 |
| Graph Coloring | 1.168*** | 2.107*** | 2.670*** | 0.0001 | 0.0000 |
| Knapsack | 0.825** | 1.528*** | 0.923*** | 0.0023 | 0.0002 |
| No Thanks! | 0.387 | 1.537*** | 2.045*** | 0.2009 | 0.0000 |
| Checkers | 0.361 | 1.510*** | 1.874*** | 0.4918 | 0.0000 |

## Coupling Onset Generation

First generation where topology diversity drops below none-baseline by >1 SE.

| Domain | Ring | Star | Random | FC |
|--------|------|------|--------|----|
| Maze | 6 | 7 | 6 | 6 |
| OneMax | 8 | 6 | 6 | 6 |
| Graph Coloring | 6 | 6 | 6 | 6 |
| Knapsack | 6 | 6 | 6 | 6 |
| No Thanks! | 6 | 7 | 6 | 6 |
| Checkers | 5 | 6 | 0 | 6 |

## Topology Ordering Stability

| Domain | Gen 10 | Gen 30 | Gen 99 | Stable? |
|--------|--------|--------|--------|---------|
| Maze | None > Star > Ring > Random > FC | None > Star > Ring > Random > FC | None > Ring > Star > Random > FC | Early only |
| OneMax | None > Ring > Star > Random > FC | None > Ring > Random > Star > FC | None > Ring > Star > Random > FC | NO |
| Graph Coloring | None > Ring > Star > Random > FC | None > Ring > Star > Random > FC | None > Ring > Star > Random > FC | YES (all) |
| Knapsack | None > Star > Ring > Random > FC | None > Ring > Star > Random > FC | None > Ring > Star > Random > FC | By gen 30 |
| No Thanks! | None > Ring > Star > Random > FC | None > Ring > Star > Random > FC | None > Ring > Star > Random > FC | YES (all) |
| Checkers | None > Star > Random > Ring > FC | None > Ring > Star > Random > FC | None > Ring > Star > Random > FC | By gen 30 |

## Key Finding

- **2/6** domains show identical topology ordering at gen 10, 30, and 99.
- **4/6** domains have ordering established by gen 30.

**Conclusion:** The topology ordering is MOSTLY established by generation 30, 
though some domains show later shifts. The compositional structure has a strong 
early effect, but domain-specific dynamics can modulate the ordering over time.

## Plots

- Per-domain trajectory plots: `plots/early_convergence_<domain>.png`
- Combined effect size plot: `plots/early_convergence_combined.png`
- Coupling onset bar chart: `plots/coupling_onset_all_domains.png`

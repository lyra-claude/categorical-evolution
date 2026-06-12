# Supplementary Materials

> Data, code, and reproduction instructions for "From Games to Graphs: A Category-Theoretic Framework for Evolutionary Computation."

## Contents

This directory serves as an index to the supplementary materials distributed across the repository. All experimental data, analysis code, and the Haskell framework that generates it are included in full.

### Experimental Data

All raw CSV data from the six-domain topology sweep lives in `experiments/`:

| File | Domain | Rows | Description |
|------|--------|------|-------------|
| `experiment_e_onemax.csv` | OneMax | 15,000 | 50-bit bitstring optimization |
| `experiment_e_maze.csv` | Maze | 15,000 | 6x6 grid maze generation |
| `experiment_e_graph_coloring.csv` | Graph Coloring | 15,000 | 20-vertex 4-coloring |
| `experiment_e_knapsack.csv` | Knapsack | 15,000 | 50-item 0/1 knapsack |
| `experiment_e_nothanks.csv` | No Thanks! | 15,000 | Co-evolutionary card game |
| `experiment_e_checkers.csv` | Checkers | 15,000 | Co-evolutionary 8x8 checkers |
| `experiment_e_maze_n7.csv` | Maze (n=7) | varies | Spectral theorem validation |
| `experiment_e_per_island.csv` | OneMax | varies | Per-island diversity data |

Each CSV contains 5 topologies x 30 seeds x 100 generations. Columns include `topology`, `seed`, `generation`, `hamming_diversity`, `population_divergence`, `best_fitness`, per-island metrics, and pairwise divergences.

### Analysis Code

See [`experiments/README.md`](../experiments/README.md) for a complete guide to every Python script, including usage instructions and output locations.

### Haskell Framework

The core GA framework is a Haskell library in `src/Evolution/`. It implements genetic operators as Kleisli morphisms with an MTL effect stack. See the [main README](../README.md) for an architectural overview.

Key modules:
- `src/Evolution/Category.hs` -- `GeneticOp` type and Kleisli composition
- `src/Evolution/Effects.hs` -- `EvoM` monad stack (Reader + State + Writer)
- `src/Evolution/Operators.hs` -- Selection, crossover, mutation as morphisms
- `src/Evolution/Island.hs` -- Island model with topology-parameterized migration
- `src/Evolution/Examples/` -- Domain implementations (BitString, Maze, Checkers, etc.)

### Plots

Generated figures live in `experiments/plots/`. Key figures referenced in the paper:

- `multi_domain_topology_ordering.{png,pdf}` -- Topology ordering across all domains
- `multi_domain_coupling_onset.{png,pdf}` -- Coupling onset timing by topology
- `multi_domain_variance_decomposition.{png,pdf}` -- Variance decomposition (topology vs domain)
- `early_convergence_*.png` -- Early convergence dynamics per domain
- `fingerprints_panels.{png,pdf}` -- Per-seed diversity traces by topology

## Reproducing Results

### Prerequisites

**Haskell** (for the GA framework):
```bash
# GHC 9.6+ and Cabal
cabal build
cabal test
```

**Python** (for experiments and analysis):
```bash
pip install numpy scipy pandas matplotlib networkx
# sympy is needed only for algebraic verification scripts
pip install sympy
```

### Running the Sweep Experiments

Each domain sweep runs 5 topologies x 30 seeds x 100 generations. The Python scripts in `experiments/` re-implement the GA logic from the Haskell framework to enable large-scale parallel sweeps.

```bash
cd experiments/

# Individual domains (each takes ~5-30 minutes depending on domain):
python onemax_stats.py              # OneMax (Experiment E in the paper)
python maze_domain.py               # Maze
python graph_coloring_domain.py     # Graph Coloring
python knapsack_domain.py           # Knapsack
python nothanks_domain.py           # No Thanks! (co-evolutionary)
python checkers_domain.py           # Checkers (co-evolutionary, slowest)
```

Output CSVs are written to the `experiments/` directory.

### Running the Analysis

After generating (or using the provided) CSV data:

```bash
cd experiments/

# Multi-domain analysis (generates summary stats + figures)
python multi_domain_analysis.py

# Early convergence analysis
python early_convergence_analysis.py

# Coupling onset timing
python coupling_onset_analysis.py

# Domain-specific plots
python plot_fingerprints.py --domain all
python plot_multi_domain.py
python plot_checkers.py
```

All plots are saved to `experiments/plots/`.

### Spectral Verification

The algebraic connectivity claims can be verified symbolically:

```bash
cd experiments/
python gp51_algebraic_verification.py    # GP(5,1) vs C_5 Laplacian eigenvalues
python petersen_spectral_verification.py  # Fourier block decomposition over Z_5
python anti_ramanujan_sweep.py           # lambda_2 vs diversity correlation
```

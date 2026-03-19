# Experiments

> Python scripts for running topology sweep experiments and analyzing results across six domains.

## Overview

The experiments in this directory validate the paper's central claim: migration topology determines diversity dynamics independently of the fitness domain. Each domain sweep runs a genetic algorithm with 5 island topologies (none, ring, star, random, fully connected), 30 seeds, and 100 generations, producing CSV data that is then analyzed for topology ordering, coupling onset, and domain independence.

## Dependencies

```bash
pip install numpy scipy pandas matplotlib networkx
# For algebraic verification only:
pip install sympy
```

## Domain Sweep Scripts

These scripts implement the GA sweep for each domain. Each produces a CSV file with columns: `topology`, `seed`, `generation`, `hamming_diversity`, `population_divergence`, `best_fitness`, plus per-island and pairwise metrics.

| Script | Domain | Genome | Fitness | Output |
|--------|--------|--------|---------|--------|
| `onemax_stats.py` | OneMax | 50-bit binary | Sum of 1-bits (absolute) | `experiment_e_onemax.csv` |
| `maze_domain.py` | Maze | 60-bit binary | Solvability + path length (absolute) | `experiment_e_maze.csv` |
| `graph_coloring_domain.py` | Graph Coloring | 40-bit binary | 1 - violations/edges (absolute) | `experiment_e_graph_coloring.csv` |
| `knapsack_domain.py` | Knapsack | 50-bit binary | Value/max with penalty (absolute) | `experiment_e_knapsack.csv` |
| `nothanks_domain.py` | No Thanks! | 13-float continuous | Tournament placement (co-evolutionary) | `experiment_e_nothanks.csv` |
| `checkers_domain.py` | Checkers | 64-bit binary | Round-robin wins (co-evolutionary) | `experiment_e_checkers.csv` |
| `sorting_network_domain.py` | Sorting Network | 28-bit binary | Fraction of inputs sorted (absolute) | `experiment_e_sorting_network.csv` |

### Running a Domain Sweep

```bash
# Each sweep runs 5 topologies x 30 seeds x 100 generations
python onemax_stats.py                # ~5 min
python maze_domain.py                 # ~10 min
python graph_coloring_domain.py       # ~5 min
python knapsack_domain.py             # ~5 min
python nothanks_domain.py             # ~15 min (tournament evaluation)
python checkers_domain.py             # ~30 min (round-robin games, uses multiprocessing)

# onemax_stats.py also supports specific experiment selection:
python onemax_stats.py --exp C        # Migration frequency sweep only
python onemax_stats.py --exp D        # Boundary position sweep only
python onemax_stats.py --seeds 10     # Fewer seeds for quick test
```

## Analysis Scripts

These scripts consume the CSV data and produce statistics, tables, and publication-quality figures.

### Multi-Domain Analysis

| Script | Description | Output |
|--------|-------------|--------|
| `multi_domain_analysis.py` | Cross-domain topology ordering, variance decomposition, Kendall's W | Figures + console summary |
| `early_convergence_analysis.py` | Diversity trajectories at gen 10/30/99, Mann-Whitney tests, coupling onset | `plots/early_convergence_*.png` |
| `coupling_onset_analysis.py` | Coupling onset timing by topology, structural vs dynamical hypothesis | Console + `plots/coupling_onset_*.{png,pdf}` |
| `coupling_onset_checkers.py` | Extends coupling onset to include checkers domain | Console + plots |
| `nothanks_analysis.py` | Full No Thanks! analysis: Kruskal-Wallis, Spearman, Kuramoto, effect sizes | `nothanks_analysis_results.md` |

### Plotting Scripts

| Script | Description | Output |
|--------|-------------|--------|
| `plot_fingerprints.py` | Per-seed diversity traces ("fingerprints") by topology | `plots/fingerprints_*.{png,pdf}` |
| `plot_multi_domain.py` | Multi-domain panel, bar chart, and overlay figures | `plots/multi_domain_*.{png,pdf}` |
| `plot_checkers.py` | Checkers-specific fingerprints, R(d), and cross-domain comparison | `plots/checkers_*.{png,pdf}` |
| `rd_two_panel.py` | Two-panel R(d) figure (global R + coherence by topological distance) | `plots/rd_two_panel_maze.{png,pdf}` |

### Spectral and Graph-Theoretic Analysis

| Script | Description |
|--------|-------------|
| `anti_ramanujan_sweep.py` | lambda_2 vs diversity correlation across topologies |
| `gp51_algebraic_verification.py` | Symbolic proof: lambda_2(GP(5,1)) = lambda_2(C_5) |
| `petersen_spectral_verification.py` | Fourier block decomposition of GP(5,1) over Z_5 |
| `kuramoto_analysis.py` | Kuramoto order parameter analysis (r = 1 - divergence) |
| `pairwise_coherence.py` | R(d): coherence as function of topological distance |
| `chimera_detection.py` | Chimera state detection in star topology |
| `time_averaged_adjacency.py` | Snapshot vs time-averaged lambda_2 for random topology |
| `snapshot_vs_timeavg_all_topologies.py` | Extends time-averaging analysis to all topologies |

### Utility Scripts

| Script | Description |
|--------|-------------|
| `test_onemax.py` | Unit tests for the OneMax GA operators |
| `restore_checkers.py` | Data recovery utility for corrupted checkers CSV |

## Data Files

### Primary Sweep Data (CSV)

All CSVs follow the same schema: `topology, seed, generation, hamming_diversity, population_divergence, best_fitness, ...`

- `experiment_e_onemax.csv` -- OneMax (also called `experiment_e_raw.csv`)
- `experiment_e_maze.csv` -- Maze (6x6)
- `experiment_e_maze_n7.csv` -- Maze (n=7, spectral theorem validation)
- `experiment_e_graph_coloring.csv` -- Graph Coloring
- `experiment_e_knapsack.csv` -- 0/1 Knapsack
- `experiment_e_nothanks.csv` -- No Thanks!
- `experiment_e_checkers.csv` -- Checkers
- `experiment_e_sorting_network.csv` -- Sorting Network (degenerate; excluded from main results)
- `experiment_e_per_island.csv` -- Per-island diversity data (OneMax)

### Analysis Results

- `pairwise_coherence_results.csv` -- R(d) data for OneMax
- `pairwise_coherence_maze_results.csv` -- R(d) data for Maze
- `topology_sweep_results.md` -- Summary tables
- `early_convergence_results.md` -- Early convergence analysis
- `nothanks_analysis_results.md` -- No Thanks! full analysis

### Logs

Sweep logs from various runs (safe to delete):
- `checkers_sweep*.log`
- `nothanks_sweep*.log`
- `maze_n7_sweep_20260318.log`

## Output Directory

All generated plots are saved to `plots/` as both PNG (300 DPI) and PDF.

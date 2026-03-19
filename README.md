# Categorical Evolution

> A category-theoretic framework for evolutionary computation, with empirical validation across six domains.

This repository contains the Haskell library, Python experiment code, and LaTeX paper for **"From Games to Graphs: A Category-Theoretic Framework for Evolutionary Computation"** (ACT 2026).

**Authors:** Robin Langer, Claudius Turing, Lyra Vega

## Paper

The ACT 2026 paper is in `act2026/paper.tex`. To build:

```bash
cd act2026
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

**Central result:** Migration topology determines diversity dynamics independently of the fitness domain. The canonical ordering (none > ring > star > random > fully connected) holds across all six domains with Kendall's W = 1.0 (p = 0.00008). Algebraic connectivity (lambda_2 of the topology graph) governs coupling onset timing -- a structural, not dynamical, property.

## Repository Structure

```
categorical-evolution/
|-- act2026/                      # ACT 2026 paper (LaTeX, EPTCS format)
|   |-- paper.tex                 # Main paper source
|   |-- references.bib            # Bibliography
|   +-- eptcs.cls, eptcs.bst      # EPTCS style files
|
|-- gecco2026/                    # GECCO 2026 AABOH workshop paper
|
|-- src/Evolution/                # Haskell library (the categorical backbone)
|   |-- Category.hs               #   GeneticOp type, Kleisli composition (>>>:)
|   |-- Effects.hs                #   EvoM monad stack (Reader + State + Writer)
|   |-- Operators.hs              #   Selection, crossover, mutation as morphisms
|   |-- Pipeline.hs               #   High-level evolve / evolveN
|   |-- Island.hs                 #   Island model with topology-parameterized migration
|   |-- Coevolution.hs            #   Competitive coevolution (sample-based)
|   |-- Strategy.hs               #   Composable evolutionary strategies
|   |-- Landscape.hs              #   Fitness landscape analysis
|   +-- Examples/                 #   Domain implementations
|       |-- BitString.hs          #     OneMax and target-matching
|       |-- Maze.hs               #     Grid maze generation
|       |-- Checkers.hs           #     8x8 American Checkers
|       |-- GraphColoring.hs      #     Graph coloring
|       |-- Knapsack.hs           #     0/1 Knapsack
|       |-- SortingNetwork.hs     #     Sorting network
|       +-- SymbolicRegression.hs #     GP with expression trees
|
|-- experiments/                  # Python sweep code and analysis
|   |-- README.md                 # Complete guide to all scripts
|   |-- onemax_stats.py           #   OneMax domain sweep
|   |-- maze_domain.py            #   Maze domain sweep
|   |-- graph_coloring_domain.py  #   Graph coloring domain sweep
|   |-- knapsack_domain.py        #   Knapsack domain sweep
|   |-- nothanks_domain.py        #   No Thanks! domain sweep (co-evolutionary)
|   |-- checkers_domain.py        #   Checkers domain sweep (co-evolutionary)
|   |-- multi_domain_analysis.py  #   Cross-domain analysis and figures
|   |-- experiment_e_*.csv        #   Raw experimental data (6 domains)
|   +-- plots/                    #   Generated figures (PNG + PDF)
|
|-- supplementary-materials/      # Index to all supplementary materials
|   +-- README.md                 # Reproduction instructions
|
|-- test/                         # Haskell test suite
|-- demo/                         # Haskell demo programs
|-- categorical-evolution.cabal   # Cabal build file
+-- GUIDE.md                      # Collaborator guide
```

## The Haskell Backbone

The core of this project is a Haskell library that formalizes genetic operators as morphisms in a Kleisli category. Every domain shares this common categorical structure.

### Core Abstraction

Genetic operators are morphisms in the Kleisli category for the evolution monad `EvoM`:

```haskell
newtype GeneticOp m a b = GeneticOp { runOp :: [a] -> m [b] }

-- Composition accumulates effects:
evaluate fitnessFunc
  >>>: elitistSelect
  >>>: onePointCrossover
  >>>: pointMutate bitFlip
```

Each `>>>:` is Kleisli composition. The pipeline reads left-to-right: evaluate, then select, then cross over, then mutate.

### MTL Effect Stack

The evolution monad uses a standard MTL transformer stack:

```haskell
type EvoM = ReaderT GAConfig (StateT StdGen (Writer GALog))
```

- **`MonadReader GAConfig`** -- GA parameters (population size, mutation rate, tournament size)
- **`MonadState StdGen`** -- PRNG state for stochastic operators
- **`MonadWriter GALog`** -- Generational statistics accumulate monoidally

### Island Model and Topology

The island model (`Evolution.Island`) runs multiple populations with topology-parameterized migration. Migration is a natural transformation between population functors -- it moves individuals between islands without depending on each island's pipeline. Topologies include ring, star, fully connected, random, and none (isolated).

### Coevolution

`Evolution.Coevolution` supports competitive coevolution where fitness is relative -- individuals are evaluated by playing against opponents, not against a fixed landscape. This is used for the No Thanks! and Checkers domains.

## Experiments

The `experiments/` directory contains Python reimplementations of the GA for large-scale parallel sweeps across six domains. Each domain sweep runs 5 topologies x 30 seeds x 100 generations.

The six domains span absolute fitness (OneMax, Maze, Graph Coloring, Knapsack), co-evolutionary fitness (No Thanks!, Checkers), binary genomes, continuous genomes, and combinatorial landscapes. Despite this diversity, all six produce the same topology ordering.

See [`experiments/README.md`](experiments/README.md) for complete instructions on running sweeps and analysis.

## Supplementary Materials

See [`supplementary-materials/README.md`](supplementary-materials/README.md) for data descriptions, reproduction instructions, and links to all code and figures.

## Building

```bash
# Haskell library
cabal build
cabal test

# Python experiments
pip install numpy scipy pandas matplotlib networkx
cd experiments && python multi_domain_analysis.py
```

## License

BSD-3-Clause

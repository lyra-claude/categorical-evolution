# From Games to Graphs: Categorical Composition of Genetic Algorithms Across Domains

## Paper Outline

### Abstract

Genetic algorithms share a common compositional structure across domains —
selection, crossover, mutation compose as morphisms in a Kleisli category
regardless of whether the genome represents game-playing strategy weights or
maze topology. We demonstrate this through two concrete implementations: evolving
checkers evaluation functions (game theory) and evolving maze designs (structural
optimization), both expressed in a single Haskell framework where GA operators
are first-class categorical morphisms. The framework is translated from Rust GA
code (mazegen-rs) to expose the categorical structure hidden in imperative
implementations. We position this work against the INSTINCT literature corpus,
where citation network analysis reveals that game-theoretic coevolution and
structural design optimization occupy disconnected clusters — our categorical
framework provides the shared language they lack.

---

### 1. Introduction: Samuel's Insight, Revisited

- Arthur Samuel (1959): machine learning for checkers evaluation = evolutionary
  optimization of strategy weights. The first ML paper is also an evolutionary
  computation paper. The weights of a linear evaluation function ARE a genome.
- The same compositional structure appears in maze generation: a maze is a
  spanning tree on a grid; mutations swap edges; fitness measures difficulty,
  tortuosity, dead-end ratio. Different domain, same algorithm shape.
- **Thesis**: The compositional structure of GAs — `evaluate >>> select >>>
  crossover >>> mutate` — is a categorical pipeline. Formalizing this lets us
  see that checkers weight evolution and maze topology evolution are instances
  of the same morphism composition, differing only in the objects (genome types)
  and the concrete fitness function.
- Stanley invented both NEAT (variable-topology neuroevolution, 2002) and POET
  (open-ended environment co-evolution, 2019) but never combined them. The
  combination is non-trivial because diversity maintenance machinery assumes
  fixed-dimensional genotypes. Our categorical framework offers a language for
  bridging variable-dimensional composition.

### 2. Background

#### 2.1 Category-printf and co-Kleisli Composition
- Cale's category-printf: format specs as morphisms in the co-Kleisli
  category for `((->) m)` where `m` is a monoid
- Composition accumulates argument types: `"Hello " . s . "!" :: Format (String -> r) r`
- Key insight: the /structure/ is in the composition, not in the individual specs

#### 2.2 MTL and Kleisli Composition
- Shuck's mtl: `MonadSelect`, `MonadAccum`, `MonadReader`, `MonadWriter`
- Kleisli composition accumulates monadic effects
- GA operators naturally need: randomness (State StdGen), config (Reader GAConfig),
  logging (Writer GALog)

#### 2.3 The Dual Insight
- co-Kleisli accumulates arguments (what the pipeline DEMANDS)
- Kleisli accumulates effects (what the pipeline DOES)
- GA operators are Kleisli morphisms: `GeneticOp m a b = [a] -> m [b]`
- Composition: `evaluate >>>: select >>>: crossover >>>: mutate`

### 3. Two Domains, One Category

#### 3.1 Checkers: Evolving Evaluation Functions
- Samuel-style features: piece count, king count, center control, advancement,
  mobility, vulnerability, protection (10 features)
- Genome: `[Double]` — a 10-dimensional weight vector
- Fitness: tournament play win rate (self-play or against baseline)
- Mutation: Gaussian perturbation on each weight
- Crossover: arithmetic blend or one-point on the weight vector
- **The categorical pipeline is identical to BitString OneMax** — only the
  genome type, fitness function, and mutation operator change

#### 3.2 Mazes: Evolving Topological Structures
- Genome: a spanning tree on an n×n grid (set of edges)
- Mutation: edge swap — remove a tree edge, add a non-tree edge that reconnects
  the two components. **Preserves the spanning tree invariant.**
- Crossover: union of both parents' edges, extract a random spanning tree via
  shuffled Kruskal's algorithm
- Fitness: difficulty score, tortuosity, solution path length, dead-end ratio
- Multi-objective: NSGA-II Pareto front over competing fitness targets
- **Same categorical pipeline, but the morphisms operate on graph structures
  rather than numeric vectors**

#### 3.3 What Changes, What Stays
- The categorical composition (`>>>:`) is invariant across domains
- The monad stack (Reader + State + Writer) is invariant
- What changes: the /objects/ (genome type), the /fitness morphism/, the
  /mutation morphism/
- This is exactly what category theory predicts: the structure is in the arrows,
  not in the objects

### 4. From Rust to Haskell: Revealing Hidden Composition

#### 4.1 The Rust Implementation (mazegen-rs)
- Imperative GA loop: `for generation in 0..config.generations { ... }`
- Mutation, crossover, selection are functions called in sequence
- The compositional structure is there but hidden in control flow
- Edge-swap mutation is a method on `Genome`, not a composable morphism
- Fitness evaluation is a standalone function, not a pipeline stage

#### 4.2 The Haskell Translation (categorical-evolution)
- Same algorithm, different structure: operators are values, not control flow
- `evaluate fitnessFunc >>>: elitistSelect >>>: crossover >>>: pointMutate mutator`
- The pipeline is a first-class value that can be passed, composed, transformed
- Higher-order operators (withElitism, island strategies) are /functors/ on pipelines
- The Rust→Haskell translation doesn't change the algorithm; it changes what you
  can /see/ about the algorithm

#### 4.3 Three Levels of Composition
1. **Operators**: `GeneticOp m a b` — individual GA steps compose via `>>>:`
2. **Pipelines**: A full generation step is itself a `GeneticOp`; pipelines
   compose across generations
3. **Strategies**: `generationalGA`, `steadyStateGA`, `islandStrategy` — strategy
   combinators that transform pipelines. These are functors on the category of
   GeneticOps.

### 5. Connecting the Islands: Literature Positioning

#### 5.1 The Knowledge Graph
- INSTINCT corpus: 76 papers, 555 concept nodes, 446 edges
- Five clusters: neuroevolution, morphology-control, open-ended evolution,
  quality-diversity, game-theory/coevolution
- Citation network analysis reveals structural holes between clusters

#### 5.2 The Game-Theory Island
- Game-theoretic coevolution has zero concept overlap with 3 of 4 other clusters
- Only bridge: "fitness" shared with morphology-control
- This is the deepest structural hole in the corpus

#### 5.3 Bridge Papers
- Stanley & Miikkulainen (2002) NEAT: appears in 6/10 structural holes
- Cully et al. (2014) Robots That Can Adapt: bridges 4 holes
- Lehman & Stanley (2011) Novelty Search: bridges neuroevolution↔OEE and OEE↔QD
- **Our contribution bridges game-theory↔structural-design** by showing both are
  instances of the same categorical framework

#### 5.4 The Neuroevolution↔OEE Gap
- Bridge score 22, but inflated by tool-usage citations
- NEAT used as tool in OEE, not studied as source of open-ended complexity
- Nobody has coupled growing-topology neuroevolution with growing-complexity
  environments
- The categorical framework offers a language for describing variable-dimensional
  composition (functors between behavior categories of different dimensions)

### 6. Results

#### 6.1 Checkers Weight Evolution
- Evolved weights vs. TD-learned weights vs. hand-tuned defaults
- Tournament play results across the three strategies
- The GA discovers similar weight priorities to TD learning (piece count > king
  count > back row > center control) but with different magnitudes

#### 6.2 Maze Topology Evolution
- Pareto front: difficulty vs. tortuosity vs. dead-end ratio
- Population diversity dynamics under edge-swap mutation
- Comparison with random maze generation (DFS, Kruskal, Prim)

#### 6.3 Categorical Invariants
- Strict/Lax Dichotomy Theorem: island functor is strict iff migration = 0
- Composition signatures: four distinct diversity dynamics (flat, hourglass,
  island, adaptive) — these appear identically in checkers and maze evolution
- The 2-category lifting law holds for pure strategies

### 7. Discussion: What the Category Reveals

- The categorical framework shows that the difference between game-playing
  evolution and structural optimization is not in the algorithm but in the
  morphisms
- Speciation (NEAT's key innovation) could serve as a credit assignment mechanism
  across the morphology-controller boundary — an unexplored direction
- Variable-dimensional genotypes (maze topologies with different numbers of
  edges) can be handled by functors between categories with different object
  types, rather than by projecting onto fixed behavioral descriptors

### 8. What Else?

- **Coevolution as functor composition**: Checkers naturally supports
  coevolution — evolve two populations of evaluation functions against each
  other. The coevolution module already handles this. Competitive coevolution
  in checkers + cooperative coevolution in mazes (rooms that connect) would
  show functorial composition across coevolution modes.
- **Landscape analysis as preprocessing**: The Landscape module's
  `recommendStrategy` function analyzes the fitness landscape (ruggedness,
  neutrality, correlation length) and auto-selects a GA variant. Running this
  on both checkers-weight-space and maze-topology-space would show how the
  same landscape analysis framework produces different strategy recommendations
  for different domains.
- **GP as a third domain**: Symbolic regression (already implemented) is a
  third genome type — expression trees. Same categorical pipeline, third object
  type. The paper becomes a three-point argument rather than two-point.
- **Interactive visualization**: Knowledge graph visualization showing how the
  paper's contribution bridges the structural holes, overlaid on the INSTINCT
  graph.

---

## Relationship to Existing Work

### Robin's Projects
- **checkers** (Python): Samuel-style evaluation with TD learning →
  categorical-evolution Checkers module (Haskell): same evaluation, GA training
- **mazegen-rs** (Rust, forked by Lyra): imperative GA for maze design →
  categorical-evolution (Haskell): same algorithm, categorical structure exposed

### Claudius's Contributions
- Knowledge graph analysis: bridge scores, structural holes, cluster topology
- Paper positioning: neuroevolution↔OEE gap, game-theory island isolation
- Categorical insight: functors between behavior categories of different dimensions

### Lyra's Contributions
- categorical-evolution framework: 10 modules, 49 tests, ~3000 lines
- Strict/Lax Dichotomy Theorem formalization
- Checkers + Maze examples connecting domains
- Rust→Haskell translation documenting the hidden compositional structure

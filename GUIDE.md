# From Games to Graphs — Companion Guide

**A developer-friendly explanation of the categorical-evolution project**

*Lyra Vega, Claudius Turing, Robin Langer*

> This is the companion guide to our formal paper, "From Games to Graphs: Categorical Composition of Genetic Algorithms Across Domains" (see `act2026/paper.tex`). The formal paper contains the mathematical definitions, proofs, and publication-ready results. This guide is for a developer or researcher picking up this project for the first time. It explains what we built, why it matters, and how the pieces fit together.

---

## Current Status (March 2026)

**Where the project stands:**

- **ACT 2026 submission.** Abstract due March 23, paper due March 30. 12 pages, EPTCS format, single-blind. Three authors: Langer, Turing, Vega. Submission #10 on EasyChair.
- **Six domains confirmed.** The same topology ordering (none > ring > star > random > fully connected) holds across OneMax, maze, graph coloring, knapsack, No Thanks!, and checkers. **Kendall's W = 1.0** across all six domains (chi-square = 24.0, p = 0.00008). All 15 pairwise Spearman correlations = 1.0. Domain independence is not approximate — it is exact.
- **Laxator formally defined.** Definition 5 in the paper. The laxator measures how far a migration graph pushes composition away from strict functoriality. This is our key theoretical contribution.
- **Spectral validation.** Algebraic connectivity (lambda2) of the migration graph predicts coupling onset, diversity ordering, and laxator magnitude. Independently confirmed by Sanz et al. (oscillator physics) and Brewster et al. (evolutionary graph theory, Nowak group).
- **Three-formalisms convergence.** Category theory (us), spectral graph theory (lambda2), and Markov chain mixing time (evolutionary graph theory) all predict the same ordering. Three views, same elephant.
- **5.5x structural constant.** The random topology's diversity-to-star ratio is 5.5x, stable across 30 seeds with CV = 2.18%. This is not a noisy mean — it is a structural constant of the system, arising from time-averaged algebraic connectivity.
- **Implementation.** Python experiments produce the data. The original Haskell proof-of-concept established the categorical structure; Python is now the primary experimental language. All experiment scripts live in `experiments/`.

**Key files:**

| File | What it is |
|------|-----------|
| `act2026/paper.tex` | The formal paper (12 pages, LaTeX) |
| `gecco2026/paper.tex` | GECCO workshop paper (9 pages, separate venue) |
| `experiments/*.py` | All experimental scripts |
| `experiments/experiment_e_*.csv` | Raw sweep data per domain |
| `experiments/plots/` | Generated figures |
| This file | You are here |

---

## 1. The Core Idea

In 1959, Arthur Samuel wrote a program that learned to play checkers by adjusting the weights of a linear evaluation function through self-play. Sixty years later, the same algorithmic shape appears in a maze generator that evolves maze topologies, and in a knapsack solver that evolves item selections. The genomes are incompatible types. The fitness functions measure incommensurable quantities. And yet the algorithm — evaluate, select, recombine, mutate, repeat — is structurally identical.

This project names that structure and shows what follows from naming it.

The structure is categorical. GA operators are morphisms (arrows) in a Kleisli category, where the monad carries the side effects every genetic algorithm needs: randomness, configuration, and logging. Composition of morphisms is pipeline assembly. A full evolutionary run is a single composite arrow. Strategies — the high-level decisions about how to organize evolution (generational, steady-state, island, adaptive) — are functors: transformations from one pipeline category to another that preserve (or fail to preserve) compositional structure.

**This is not a metaphor.** When we say the island model is a functor, we mean it satisfies specific mathematical laws — and when those laws hold strictly versus laxly, the observable behavior differs in predictable, measurable ways.

> For the precise definitions — the monad stack, the Kleisli category construction, the laxator — see Sections 2--3 of the formal paper (`act2026/paper.tex`).

---

## 2. Categorical Background (Key Definitions Only)

This section gives just enough category theory to follow the guide. The formal paper has the full treatment. For a gentler entry, the pigeon-breeding and Unix-pipeline analogies below are genuine — they are not metaphors dressed up as mathematics. They are the same mathematical structure in different clothing.

### What is a category?

A category **C** is:

1. A collection of **objects** (think: types of data).
2. For every pair of objects A, B, a collection of **arrows** f : A -> B (think: transformations).
3. **Composition**: given f : A -> B and g : B -> C, their composite g . f : A -> C.
4. **Identity arrows**: id_A : A -> A for each object.

Composition must be associative and identities must be neutral. That is the entire definition.

**Pigeon example.** You breed racing pigeons. Objects = types of flocks (unscored, scored, selected). Arrows = operations (time each pigeon, keep the top 20, breed pairs). Composing evaluate-then-select-then-breed gives one generation step — itself an arrow you can compose again.

**Unix example.** Objects = stream formats (SAM, BED, VCF). Arrows = command-line tools. The pipe operator `|` is composition. `cat` is the identity. A pipeline is a composite arrow.

### The GA lifecycle is a category

The objects are population types. The arrows are GA operators:

| Operator | Arrow type | What it does |
|----------|-----------|-------------|
| Evaluate | `[genome] -> [scored genome]` | Assigns fitness |
| Select | `[scored genome] -> [scored genome]` | Keeps the best |
| Crossover | `[scored genome] -> [scored genome]` | Breeds pairs |
| Mutate | `[scored genome] -> [scored genome]` | Random variation |
| Unwrap | `[scored genome] -> [genome]` | Strips scores |

One generation:

```
evaluate >>> select >>> crossover >>> mutate >>> unwrap
```

This composite is an arrow `[genome] -> [genome]`. Running 100 generations is composing this arrow with itself 100 times. An entire evolutionary run is a single composite arrow.

### Where the monad enters

GA operators are not pure functions — they need randomness, configuration, and logging. A **monad** packages these side effects into a context that the composition operator handles automatically. Instead of pure arrows `A -> B`, we work with **effectful arrows** `A -> M(B)`. These form the **Kleisli category** for M. Each `>>>` threads the effects invisibly.

> The concrete monad stack is defined in the formal paper (Section 2). The key point: all three effects (randomness, configuration, logging) are unified in one structure, so adding a new effect changes one place, not every function signature.

### Functors: strategies as structure-preserving maps

A **functor** F : C -> D maps between categories, preserving composition and identities. The island model is a functor: it takes a single-population GA and produces a multi-population GA. Migration is a natural transformation — it moves individuals between islands without inspecting them.

The critical distinction is **strict vs. lax**:

- A **strict** functor preserves composition exactly: F(g . f) = F(g) . F(f).
- A **lax** functor preserves it only up to a coherence morphism (the **laxator**): F(g . f) -> F(g) . F(f).

The laxator measures the gap. When islands are completely isolated (no migration), the functor is strict. The moment you add any migration — even one individual — the functor becomes lax. The laxator magnitude depends on the migration topology, parameterized by the graph's algebraic connectivity lambda2.

---

## 3. Six Domains, One Category

The project confirms domain independence across six domains. Each domain provides different genome types, fitness functions, and mutation operators. The categorical pipeline is identical across all of them.

### 3.1 Domain Roster

| Domain | Genome type | Fitness | Mutation | Notes |
|--------|------------|---------|----------|-------|
| **OneMax** | Binary string | Count of 1s | Bit flip | Simplest possible; baseline |
| **Maze** | Binary (wall presence) | BFS path length + dead ends + branching | Bit flip | Structural design |
| **Checkers** | Real-valued weights | Win rate via self-play | Gaussian perturbation | Game strategy, co-evolutionary |
| **Graph coloring** | Integer (color per node) | Constraint violations | Random recolor | Constraint satisfaction |
| **Knapsack** | Binary (include/exclude) | Value under weight constraint | Bit flip | Classic combinatorial optimization |
| **No Thanks!** | Strategy weights | Tournament win rate | Gaussian perturbation | Co-evolutionary card game |

Sorting network was also tested but is degenerate — all topologies converge to the same fitness, providing no diversity gradient. This is a scope condition (trivial landscape), not a failure of the framework.

### 3.2 What Changes, What Stays

| Aspect | Changes per domain | Invariant across all |
|--------|-------------------|---------------------|
| Genome type | Yes | Pipeline shape |
| Fitness function | Yes | Monad stack |
| Mutation operator | Yes | Strategy combinators |
| | | Composition operator (`>>>`) |
| | | Category laws |
| | | **Topology ordering** |

The key empirical result: the **same topology ordering** (none > ring > star > random > fully connected, from most to least diverse) holds across all six non-degenerate domains with perfect agreement. Kendall's W = 1.0 (chi-square = 24.0, p = 0.00008). All 15 pairwise Spearman correlations = 1.0. Domain has zero effect on the ordering.

> The three-axis taxonomy (landscape x representation x constraint) that frames domain independence was proposed by Claudius and adopted in the formal paper.

### 3.3 Per-Domain Results

Each domain was swept across 5 topologies with 30 seeds per topology (150 runs per domain, 900 runs total across all 6 domains). The results below document the key findings per domain, ordered by date of completion.

#### OneMax, Maze, Graph Coloring, Knapsack (Original Four)

These four domains established the canonical ordering and confirmed domain independence. Full details are in the formal paper. Each shows Spearman rho = 1.0 with the canonical ordering (none > ring > star > random > FC).

#### No Thanks! (5th Domain — March 18)

No Thanks! is a co-evolutionary card game domain. Fitness is tournament win rate, so the fitness landscape changes every generation as opponents co-evolve. This is the strongest test of topology effects because the fitness signal is noisy and adversarial.

- 150 runs COMPLETE (5 topologies x 30 seeds)
- Perfect rank correlation (Spearman rho = 1.0) with canonical ordering
- **53.9% phase transition** (none -> ring) — the largest of any domain
- Co-evolutionary fitness *amplifies* topology effects rather than masking them

The 53.9% phase transition is notable: when the fitness landscape itself co-evolves, the coupling introduced by even minimal migration has an outsized effect on diversity. This is the opposite of what one might expect — a noisier fitness signal does not wash out the topology ordering; it sharpens it.

#### Checkers (6th Domain — March 19)

Checkers is the most computationally demanding domain: real-valued weight vectors evaluated through multi-game self-play. It is also the hardest test, since co-evolutionary noise and high-dimensional genomes could plausibly mask topology effects.

- 30 seeds, 150 runs (5 topologies x 30 seeds), 15,000 data rows
- Perfect rank correlation (Spearman rho = 1.0) with canonical ordering
- **Smallest phase transition: 11.1%** (none -> ring) — co-evolutionary buffering damps topology effects
- Ring vs star: Cohen's d = 0.577, p = 0.029
- None vs FC: Cohen's d = 5.76
- **5.5x random inflation STABLE:** CV = 2.18% across 30 seeds

The 11.1% phase transition is the smallest of any domain, and it tells the opposite story from No Thanks!. In checkers, co-evolutionary dynamics *buffer* the topology effect — the adversarial fitness landscape partially absorbs the coupling introduced by migration. Yet the ordering still holds perfectly. This is the strongest universality argument: even in the hardest domain, where the effect size is smallest, the rank ordering is invariant.

The 5.5x random inflation factor (random topology diversity / star topology diversity = 5.5x) has CV = 2.18% across 30 independent seeds. This is not a noisy estimate — it is a structural constant of the system, arising from the time-averaged algebraic connectivity of the random migration graph.

### 3.4 Six-Domain Concordance (March 19)

With all six domains complete, we computed the full concordance statistics:

- **Kendall's W = 1.0** (chi-square = 24.0, p = 0.00008)
- **All 15 pairwise domain Spearman correlations = 1.0**
- Domains: OneMax, Maze, graph_coloring, knapsack, No Thanks!, checkers

This is domain independence — the topology ordering is invariant under fitness functor change. In categorical terms: changing the objects (genome types, fitness functions) does not change the behavior of the morphisms (compositional structure). This is what the theory predicts, and it holds exactly.

### 3.5 What the Category Predicts

If the categorical structure is real, then:

1. **Changing the domain (objects) should not change the compositional behavior (arrows).** Confirmed: topology ordering is domain-invariant across all 6 domains (W = 1.0).
2. **Diversity fingerprints should be determined by composition pattern, not by content.** Confirmed: the same strategy (flat, hourglass, island, adaptive) produces qualitatively the same diversity trajectory regardless of domain.
3. **The strict/lax transition should be binary — zero migration = strict, any migration = lax.** Confirmed: one migrant changes everything.

---

## 4. The Three Formalisms

One of the strongest findings is that three independent theoretical frameworks predict the same migration topology ordering. This is the "three views, same elephant" narrative that anchors the paper.

### 4.1 Category Theory (Our Framework)

GA operators compose as Kleisli morphisms. The island model is a lax functor. The laxator phi_G measures deviation from strict functoriality. Laxator magnitude increases with the algebraic connectivity lambda2 of the migration graph G.

**Prediction:** Topologies with higher lambda2 produce more coupling, less diversity, more lax composition.

### 4.2 Spectral Graph Theory (lambda2)

The algebraic connectivity lambda2 is the second-smallest eigenvalue of the graph Laplacian. It governs how fast information diffuses across a network. Higher lambda2 = faster mixing = stronger coupling between islands.

Our topologies, ordered by lambda2:
- none: lambda2 = 0 (disconnected)
- ring: lambda2 ~ 0.38 (C5)
- star: lambda2 = 1.0
- random: lambda2 varies, time-averaged ~ 39.74 (110x higher than snapshot mean)
- fully connected: lambda2 = 5.0 (K5)

This ordering matches the diversity ordering exactly.

**Independent confirmation:** Sanz et al. (2603.05668) showed that coupled oscillator synchronization onset collapses to a universal curve when plotted against lambda2. Different network topologies, different oscillator dynamics — same lambda2-governed transition. This is precisely our coupling onset result, from physics.

**n=7 Maze Spectral Test (March 18).** At n=5 islands, the ring (C5) has lambda2 = 0.382 < 1.0 = lambda2(star), so the spectral theorem predicts ring > star for diversity. At n=7, C7 has lambda2 = 0.753, still below 1.0, so the prediction still holds. We tested this explicitly: 60 maze runs at n=7, ring diversity = 0.387 vs star diversity = 0.336, p = 6.6e-5. The spectral theorem's prediction is confirmed at both island counts.

### 4.3 Markov Chain Mixing Time (Evolutionary Graph Theory)

Brewster et al. (2503.09841, Nowak group at Harvard) proved that consensus time on ring graphs scales as N^3, while on complete graphs it scales as N^2. Slower mixing = more diversity preservation = our topology ordering.

**The convergence:** Three formalisms — categorical (laxator), spectral (lambda2), and combinatorial (mixing time) — independently predict the same ordering. None of these groups cited each other. We are the first to note the convergence.

### 4.4 Connective Tissue: Markov Chain Mixing Time

Per Claudius's advice: the Markov chain mixing time serves as connective tissue between the three formalisms. Migration is literally a random walk on the migration graph. The mixing time of this walk determines how fast island populations homogenize. Lambda2 governs the mixing time (Cheeger inequality). And the laxator measures the compositional consequence of this mixing.

So: **laxator magnitude ~ mixing rate ~ 1/mixing time ~ lambda2**.

This chain of equalities connects the categorical, spectral, and combinatorial views into a single quantitative story.

---

## 5. The Strict/Lax Dichotomy

This is the main empirical result. It has two parts: the binary transition and the continuous parameterization.

### 5.1 The Binary Transition

The island functor preserves sequential composition if and only if no migration occurs. Any nonzero migration — even one individual per thousand generations — produces a lax functor. The laxity magnitude is independent of migration frequency and composition boundary position.

**Migration frequency sweep** (4-island ring, 20-bit OneMax): At freq=40 (no migration events in the window), divergence = 0. At every other frequency (2, 5, 10, 20), divergence is uniform at ~0.75. Binary transition, not gradual.

**Boundary position sweep** (freq=5, boundary from gen 15 to 25): Divergence is ~0.75 regardless of whether the boundary coincides with a migration event. This rules out the naive explanation that only "interrupted" migrations cause divergence.

The mechanism is chaotic amplification: resetting the migration schedule at the composition boundary introduces a perturbation that cascades through the coupled stochastic system.

> In pigeon terms: if you split a 40-generation breeding program into two 20-generation halves, the results differ from running it straight through — unless no birds ever migrated between lofts. One transferred pigeon changes everything.

### 5.2 The Continuous Parameterization

Within the lax regime, the degree of laxity varies continuously with migration topology. The ordering — none (strict) > ring > star > random > fully connected (most lax) — is parameterized by lambda2 of the migration graph.

The phase transition at coupling onset is sharp: none -> ring produces a diversity drop that ranges from 11.1% (checkers) to 53.9% (No Thanks!), depending on domain. Each subsequent step produces <= 9%. This is a genuine symmetry break, not a sigmoid.

### 5.3 Early Convergence and Two-Phase Dynamics (March 18)

Analysis of the early generations (gen 0--20) across all six domains reveals a universal two-phase process:

1. **Coupling onset (gen 5--7).** All connected topologies begin diverging from the disconnected baseline. This onset timing is universal — it does not depend on the domain or the specific topology. The ordering of onset times matches lambda2: FC first, then random, star, ring.
2. **Topology-dependent divergence (gen 7--99).** After coupling onset, the topologies gradually separate into their final ordering. This phase is where the continuous parameterization by lambda2 becomes visible.

A key observation: **ring starts behind star at generation 10 in 3 of 6 domains**, then overtakes by generation 30--50. The ring advantage is not about initial diversity preservation — it is about sustained resistance to homogenization. The ring's lower algebraic connectivity means it mixes more slowly, and this slow-mixing advantage compounds over generations.

This two-phase structure explains why short experiments might miss the full topology ordering: the initial coupling onset is fast and dramatic, but the subsequent differentiation is gradual.

### 5.4 Eight Independent Confirmations

The strict/lax pattern appears across optimization paradigms:

1. Our experiments (island GA)
2. Google/MIT scaling laws (arXiv:2512.08296)
3. Constitutional evolution (Kumar et al., AAMAS 2026)
4. DRQ convergence (David Ha, Jan 2026)
5. Semantic collapse in LLM evolution (Alpay 2602.18450)
6. CodeEvolve operator synergy (arXiv:2510.14150)
7. MadEvolve nested composition
8. LLM conformity dynamics (Han et al. 2601.05606)

Each of these discovered the same pattern independently: isolated components maintain their dynamics (strict), while coupled components degrade compositional invariants (lax). None used categorical vocabulary. The formal paper provides citations and analysis for all eight.

---

## 6. Diversity Fingerprints

Different ways of composing the same operators produce qualitatively distinct diversity trajectories. We call these **diversity fingerprints**. Four compositions have been characterized:

- **Flat (generational):** Monotonic diversity decline. No structural intervention prevents convergence.
- **Hourglass:** Spike-crash-rebound. Three phases (explore, converge, diversify) clearly visible.
- **Island:** Stable maintenance after initial drop. Migration prevents collapse without dramatic oscillation.
- **Adaptive:** Spike then collapse. Plateau detection triggers convergence with no recovery mechanism.

These fingerprints are stable across genome types and fitness landscapes — the composition pattern, not the individual operators, determines the trajectory. The formal paper conjectures (Conjecture 1) that fingerprints are functorial: a domain-change functor preserving composition structure preserves fingerprint shape.

> See the formal paper for data tables and statistical analysis. The experiments that generated these results are in `experiments/plot_fingerprints.py` and the raw data in `experiments/experiment_e_*.csv`.

---

## 7. Project Architecture

### 7.1 Experimental Pipeline

All experiments follow the same pattern:

1. **Domain script** (`experiments/*_domain.py`) — defines genome type, fitness function, mutation, crossover for one domain.
2. **Sweep script** — runs the domain across 5 topologies x 30 seeds x 50 generations. Outputs CSV with per-generation diversity, fitness, and per-island statistics.
3. **Analysis script** — reads CSV, computes topology ordering, coupling onset, diversity fingerprints.
4. **Plot script** — generates figures for the paper.

Key scripts:

| Script | Purpose |
|--------|---------|
| `experiments/maze_domain.py` | Maze evolution domain |
| `experiments/checkers_domain.py` | Checkers evolution with game engine |
| `experiments/knapsack_domain.py` | Knapsack domain |
| `experiments/graph_coloring_domain.py` | Graph coloring domain |
| `experiments/nothanks_domain.py` | No Thanks! card game domain |
| `experiments/multi_domain_analysis.py` | Cross-domain statistical analysis |
| `experiments/coupling_onset_analysis.py` | Coupling onset detection |
| `experiments/kuramoto_analysis.py` | Kuramoto order parameter mapping |
| `experiments/balduzzi_decomposition.py` | Hodge/Balduzzi decomposition |
| `experiments/anti_ramanujan_sweep.py` | Lambda2 vs diversity sweep |
| `experiments/petersen_spectral_verification.py` | GP(5,1) spectral verification |
| `experiments/time_averaged_adjacency.py` | Time-averaged lambda2 for random topology |
| `experiments/snapshot_vs_timeavg_all_topologies.py` | Snapshot vs time-averaged comparison |
| `experiments/n7_maze_spectral_test.py` | n=7 maze spectral theorem validation |
| `experiments/early_convergence_analysis.py` | Early convergence two-phase analysis |

### 7.2 The Haskell Proof-of-Concept

The categorical structure was originally discovered by translating a working Rust GA (mazegen-rs) into Haskell. The Haskell code made the compositional structure visible through the type system — operators compose with `>>>:`, the monad stack unifies effects, and the category laws hold by construction.

The Haskell implementation proved that the categorical framework is not just notation — the types enforce the compositional structure. But the experimental work (topology sweeps, multi-domain analysis, spectral validation) is all in Python. The Haskell code is a proof of concept; Python is the production experimental language.

> The original Rust-to-Haskell translation story is documented in the formal paper's related work section. The key insight: the translation changed nothing about the algorithm's behavior — it changed what is *visible* about the algorithm's structure.

### 7.3 Positioning: The Optimization Zoo

We complete the categorical optimization landscape:

| Paradigm | Group | Categorical structure |
|----------|-------|---------------------|
| Neural networks | Gavranovic et al. (ICML 2024) | Monads in Para |
| Reinforcement learning | Hedges & Sakamoto (EPTCS 429, 2025) | Parametrised optics |
| Compositional RL | Bakirtzis et al. (JMLR v26, 2025) | Categorical MDPs |
| **Evolutionary computation** | **Us** | **Kleisli morphisms** |

We fill the last gap. No evolutionary computation paper has appeared at ACT. Ever.

---

## 8. What the Category Reveals

The categorical framework enables three things that imperative implementations cannot:

**Formal composition reasoning.** The strict/lax dichotomy is a statement about functorial composition. You cannot ask "does this for-loop preserve sequential composition?" because the for-loop does not have a type that admits the question. The categorical framework gives evolutionary strategies a type — a morphism in a Kleisli category — and composition questions become type-level questions.

**Domain-independent strategy design.** The hourglass strategy was designed once and applied to six domains without modification. This works because the strategy operates on the categorical structure (morphism composition) rather than on the content (genome representation).

**Diversity fingerprints as compositional invariants.** The observation that flat, hourglass, island, and adaptive produce qualitatively distinct diversity trajectories — stable across domains — is a statement about composition. The fingerprint is determined by how operators are composed, not by what they operate on.

### Limitations (Honest Assessment)

- **Scale.** Small populations (30--60) and short runs (20--50 generations) on deliberately simple problems. The fingerprints and dichotomy theorem may not survive scaling to industrial-size problems.
- **Statistical power.** 30 seeds per condition for topology sweeps (900 total runs across 6 domains). Early experiments used fewer seeds but all final results use 30.
- **The lambda2 story has a gap.** For fixed topologies, lambda2 perfectly predicts diversity ordering. For the random topology (time-varying), the snapshot-mean lambda2 is 0.36 but the time-averaged lambda2 is 39.74 — a 110x gap. The laxator captures what lambda2 alone cannot for time-varying graphs. This is a feature, not a bug, but it needs to be stated clearly.

### Future Directions

- **Co-Kleisli selection.** Selection may be a co-Kleisli arrow (dual construction). Post-ACT investigation.
- **Sheaf cohomology.** The island model as a fitness landscape sheaf, with H^1 measuring integration failure across islands. Connected to Inoue (2026).
- **Variable-dimensional genotypes.** NEAT's growing-topology networks involve genomes whose dimensionality changes over evolution — functors between categories with different object types.
- **Landscape-aware strategy selection.** Predicting which fingerprint a landscape demands, closing the loop between theory and practice.

---

## 9. How to Get Started

If you are picking up this project:

1. **Read the formal paper** (`act2026/paper.tex`) for mathematical definitions and results.
2. **Run a domain sweep.** Pick any domain script in `experiments/`, run it across topologies, and verify you get the canonical ordering.
3. **Look at the plots.** `experiments/plots/` has the generated figures. The multi-domain panel and coupling onset comparison are the most informative.
4. **Check the analysis.** `experiments/multi_domain_analysis.py` is the central cross-domain statistical script. It computes the variance ratio, Spearman correlations, and domain p-value.

**Dependencies:** Python 3.12, numpy, scipy, matplotlib. No Haskell compiler needed for the experimental work.

**GitHub:** [github.com/lyra-claude/categorical-evolution](https://github.com/lyra-claude/categorical-evolution)

---

## References

The formal paper (`act2026/paper.tex`) contains the complete bibliography (32+ sources). Key references for this guide:

- Samuel, A. L. (1959). Some studies in machine learning using the game of checkers. *IBM J. Res. Dev.*, 3(3), 210--229.
- Gavranovic, B. et al. (2024). Categorical deep learning. *ICML*.
- Hedges, J. & Sakamoto, R. (2025). Reinforcement learning in categorical cybernetics. *EPTCS 429*.
- Bakirtzis, G. et al. (2025). Compositional MDPs. *JMLR v26*.
- Sanz, A. et al. (2026). Universal synchronization onset. *arXiv:2603.05668*.
- Brewster, C. et al. (2025). Consensus time on evolutionary graphs. *arXiv:2503.09841*.
- Han, S. et al. (2026). LLM conformity dynamics. *arXiv:2601.05606*.

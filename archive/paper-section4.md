# Section 4: From Rust to Haskell — Revealing Hidden Composition

The categorical framework did not emerge from abstract theory imposed on a toy problem. It was discovered by translating a working imperative genetic algorithm — `mazegen-rs`, a Rust maze evolution system — into Haskell. The translation changed nothing about the algorithm's behavior. It changed what is *visible* about the algorithm's structure.

This section presents the translation as a case study in how category theory surfaces compositional structure that imperative code hides in control flow.

## 4.1 The Rust Implementation

The Rust GA in `mazegen-rs` follows a standard imperative pattern. The evolution loop (approximately 75 lines) manually sequences operations:

```rust
for gen in 1..=config.generations {
    let mut next_genomes: Vec<Genome> = Vec::new();

    // Elitism
    for i in 0..elite {
        next_genomes.push(genomes[i].clone());
    }

    // Fill rest with offspring
    while next_genomes.len() < config.pop_size {
        let a = tournament_select(&fitnesses, config.tournament_size, &mut rng);
        let b = tournament_select(&fitnesses, config.tournament_size, &mut rng);
        let mut child = genomes[a].crossover(&genomes[b], &mut rng);

        if rng.gen::<f64>() < config.mutation_rate {
            child.mutate(&mut rng);
        }

        let f = evaluate(&child, &config.target);
        next_fitnesses.push(f);
        next_genomes.push(child);
    }

    // Sort by fitness
    // ... index-based sorting of parallel arrays ...
}
```

Three things are hidden in this code:

1. **Selection returns an index**, not an individual. The caller must look up `genomes[a]` — the connection between selection and crossover is implicit, mediated by an integer.

2. **Composition is manual**. The sequence `select → crossover → mutate → evaluate` exists only as ordered statements in a loop body. There is no type representing "a pipeline of GA operations."

3. **Effects are threaded explicitly**. Randomness (`&mut rng`), configuration (`config.tournament_size`), and evaluation target (`config.target`) are passed as separate arguments to every function. Adding logging would require modifying every function signature.

The code works. It evolves mazes with controllable fitness targets. But the compositional structure — the fact that this is a pipeline of operations that compose associatively — is not represented anywhere in the program. The programmer knows it; the type system does not.

## 4.2 The Haskell Translation

The same algorithm in the categorical framework:

```haskell
generationStep :: ([a] -> Double) -> (a -> EvoM a) -> Int -> GeneticOp EvoM [a] [a]
generationStep fitFunc mutFunc gen =
  evaluate fitFunc
    >>>: logGeneration gen
    >>>: elitistSelect
    >>>: onePointCrossover
    >>>: pointMutate mutFunc
    >>>: pointwise individual
```

The entire generation step is six lines. Each line is a morphism; `>>>:` is Kleisli composition in the `EvoM` monad. The type signature states exactly what happens: a function from populations of `[a]` to populations of `[a]`, carried in the `EvoM` monad.

## 4.3 What the Translation Reveals

### Selection becomes a morphism, not a lookup

Rust:
```rust
fn tournament_select(fitnesses: &[f64], k: usize, rng: &mut impl Rng) -> usize
```

Haskell:
```haskell
tournamentSelect :: GeneticOp EvoM (Scored a) (Scored a)
```

In Rust, selection returns an index. The caller must look up the genome by index, creating an implicit data dependency between two parallel arrays (`genomes` and `fitnesses`). In Haskell, selection is a morphism from scored populations to scored populations. There are no indices. The `Scored a` type bundles each individual with its fitness score, eliminating the parallel-array coupling.

### The pipeline becomes a first-class value

In Rust, the generation step is not a value — it's a block of code inside a `for` loop. It cannot be passed to another function, composed with other generation steps, or inspected.

In Haskell, `generationStep` is a value of type `GeneticOp EvoM [a] [a]`. It is a first-class morphism in the Kleisli category for `EvoM`. This means:

- It can be composed with other strategies: `sequential(explore, converge)`
- It can be lifted into an island functor: `islandStrategy config step`
- It can be the target of landscape analysis: `recommendStrategy` examines the fitness landscape and returns a `Strategy` value

None of these operations are possible when the generation step is trapped inside a loop.

### Effects unify through the monad

Rust threads three concerns separately:
- Randomness: `&mut rng` on every function
- Configuration: `config.tournament_size`, `config.mutation_rate` as explicit arguments
- Logging: not present (would require modifying every function)

Haskell's `EvoM` monad (`ReaderT GAConfig (WriterT GALog (State StdGen))`) unifies all three:
- `randomDouble` draws from the state layer
- `ask` reads configuration from the reader layer
- `tell` writes to the log in the writer layer

Operators compose without knowing which effects the other operators use. Adding a new effect (e.g., `WriterT DiversityLog`) would change the monad stack once, not every function signature.

### Category laws become enforceable

The Rust code provides no way to verify that the pipeline is well-formed. Reordering operations (e.g., mutating before selecting) is a logic error but not a type error.

The Haskell type system enforces composition. Each `GeneticOp m a b` has an input type `a` and an output type `b`. Composing `f :: GeneticOp m a b` with `g :: GeneticOp m c d` requires `b ~ c` — a type mismatch is a compile error. The identity law (`idOp >>>: f = f`) and associativity law (`(f >>>: g) >>>: h = f >>>: (g >>>: h)`) hold by the monad laws for `EvoM`.

## 4.4 The Same Algorithm, Different Visibility

The Rust and Haskell implementations produce the same results. Given the same random seed, initial population, and parameters, they evolve the same populations through the same trajectory. The mutation operator (edge swap preserving spanning tree invariant), the crossover operator (Kruskal's on union of parent edges), the selection operator (tournament) — these are identical in semantics.

What changes is visibility:

| Aspect | Rust | Haskell |
|--------|------|---------|
| Pipeline | Implicit (loop body) | Explicit (`GeneticOp m a b`) |
| Selection output | Index (`usize`) | Individual (`Scored a`) |
| Scoring | Parallel arrays | Co-located (`Scored a`) |
| Composition | Manual sequencing | Type-checked `>>>:` |
| Effects | Threaded explicitly | Unified in `EvoM` |
| Reuse | Copy-paste the loop | Compose morphisms |

The Haskell version does not add any new capability. It makes the existing capability *composable*. And composability is what enables the three-level tower (operators → pipelines → strategies), the island strategy functor, and the dichotomy theorem. None of these results are about Haskell as a language. They are about making composition explicit enough to reason about formally.

## 4.5 Lessons for the GA Community

The GA literature overwhelmingly uses imperative implementations (Python, C++, Java). The standard pattern — a `for` loop with selection, crossover, mutation, and evaluation called in sequence — is universal. Our translation suggests this pattern hides significant structure:

1. **Selection as index-returning function** conflates what is fundamentally a population-to-population morphism with an array lookup. This makes it harder to compose selection strategies or reason about their interaction with other operators.

2. **The generation step as loop body** prevents formal reasoning about sequential, parallel, and adaptive strategy compositions. When the step is a loop body, the only way to combine strategies is to write a new loop.

3. **Manual effect threading** makes it harder to add instrumentation (logging, diversity tracking) after the fact. In the categorical framework, adding `logGeneration` to the pipeline is one line.

We do not claim that GA practitioners should switch to Haskell. We claim that the categorical structure we describe is present in *every* GA implementation — the types are there whether or not the language can express them. Recognizing this structure enables formal reasoning about composition (as in the dichotomy theorem) even when working in imperative languages.

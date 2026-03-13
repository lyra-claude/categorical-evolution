# 2.1 A Minimal Category Theory for Evolutionary Biologists

You already think categorically. When you describe a genetic algorithm as "evaluate, then select, then recombine, then mutate," you are describing a composition of arrows in a category. When you observe that island migration works the same way regardless of whether the organisms are bitstrings or neural networks, you are observing a natural transformation. This section names the structure you are already using. Once named, it becomes something you can reason about formally — and the reasoning yields results (Section 6.3) that are not obvious from the algorithmic description alone.

## What is a category?

A category **C** consists of:

1. A collection of **objects** (think: types of data).
2. For every pair of objects A, B, a collection of **arrows** (or **morphisms**) f : A → B (think: transformations from A-data to B-data).
3. For every object A, an **identity arrow** id_A : A → A that does nothing.
4. A **composition operation**: given f : A → B and g : B → C, their composite g ∘ f : A → C applies f then g.

Composition must satisfy two laws:

- **Identity**: composing with the identity does nothing. id ∘ f = f and f ∘ id = f.
- **Associativity**: parenthesization doesn't matter. (h ∘ g) ∘ f = h ∘ (g ∘ f).

That is the entire definition. A category is a collection of things and transformations between them, where transformations compose associatively and there is a do-nothing transformation for each thing.

**Example: Unix pipelines.** The objects are stream formats (SAM, BED, VCF, plain text). The arrows are command-line tools: `samtools view` transforms SAM → BAM, `bedtools coverage` transforms BAM × BED → coverage table, `sort` transforms text → sorted text. The pipe operator `|` is composition. The identity is `cat`. A pipeline like

```
samtools view -F 4 aligned.bam | bedtools coverage -a genes.bed | sort -k5 -nr
```

is a composite arrow from SAM input to sorted coverage output. The pipeline is itself an arrow — you could pipe its output into another tool. **Composition closes over the same type.**

## The GA lifecycle is a category

In a genetic algorithm, the objects are population types:

- `[genome]` — a list of unevaluated genomes (e.g., bitstrings, weight vectors, trees)
- `[scored genome]` — genomes paired with fitness values
- `[genome]` again, after selection and recombination strip the scores

The arrows are GA operators:

| Operator | Arrow type | What it does |
|----------|-----------|--------------|
| Evaluate | `[genome] → [scored genome]` | Score each individual against the fitness function |
| Select | `[scored genome] → [scored genome]` | Choose who reproduces (tournament, roulette, elitist) |
| Crossover | `[scored genome] → [scored genome]` | Recombine pairs of parents |
| Mutate | `[scored genome] → [scored genome]` | Randomly perturb each individual |
| Unwrap | `[scored genome] → [genome]` | Discard fitness scores, keep genomes |

One generation is the composition:

```
evaluate ∘ unwrap ∘ mutate ∘ crossover ∘ select ∘ evaluate
```

or equivalently, reading left to right (the convention we adopt in this paper):

```
evaluate >>> select >>> crossover >>> mutate >>> unwrap
```

This composite is itself an arrow `[genome] → [genome]`. Running 100 generations is composing this arrow with itself 100 times. An entire evolutionary run is a single composite arrow.

The key property is **closure**: a pipeline of five operators is itself an operator. A strategy that runs one pipeline for 50 generations and then switches to another pipeline is itself a strategy. An island model containing four independent GAs with periodic migration is itself a GA. At every level of organization, the composite has the same type as its parts.

This is not a metaphor. It is the same mathematical property that makes Unix pipelines composable: the output of any stage is a valid input to any subsequent stage, so you can freely rearrange, add, or remove stages without breaking the type contract.

## Where the monad enters

There is one complication. GA operators are not pure functions — they have **side effects**:

- **Randomness.** Selection, crossover, and mutation all need random numbers. Each random draw changes the state of the random number generator, and the next operator must see the updated state.
- **Configuration.** Tournament size, mutation rate, population size — these parameters are read by every operator but never modified.
- **Logging.** Each generation produces statistics (best fitness, average fitness, diversity). These accumulate into a history over the course of the run.

A pure function `[scored genome] → [scored genome]` cannot draw random numbers. A function that takes a random seed and returns an updated seed alongside its result — `([scored genome], seed) → ([scored genome], seed)` — can, but now every operator must explicitly thread the seed through, and the types no longer compose cleanly.

A **monad** solves this by packaging the side effects into a context that the composition operator handles automatically. Instead of pure arrows `A → B`, we work with **effectful arrows** `A → M(B)`, where `M` is a monad that carries the effects. For this paper, `M` carries randomness, configuration, and logging simultaneously.

To see the difference concretely, consider tournament selection — an operator that picks the fittest individual from a random sample. Without a monad, every operator must manually accept and return all the bookkeeping state:

```
function tournament_select(population, config, rng_state, log):
    k = config.tournament_size              -- read from config
    contestants = sample(population, k, rng_state)
    rng_state = advance(rng_state)          -- update random state
    winner = argmax(contestants, fitness)
    log = log + "selected individual #..."  -- append to log
    return (winner, config, rng_state, log) -- thread everything through
```

Every operator carries the same boilerplate: accept `(config, rng_state, log)`, thread them through, return them alongside the actual result. Composing two operators means manually wiring the output state of one into the input state of the next. Adding a new concern (say, diversity tracking) changes every operator's signature.

With a monad, the same operator becomes:

```
function tournament_select(population):
    k = read_config().tournament_size       -- monad provides config
    contestants = random_sample(population, k)  -- monad handles rng
    winner = argmax(contestants, fitness)
    write_log("selected individual #...")   -- monad accumulates log
    return winner                           -- just the result
```

The operator expresses only its own logic. The monad threads the random state, propagates configuration, and accumulates the log — invisibly, at composition time. This is not syntactic sugar; it is a change in the mathematical structure. The operator is now an arrow `A → M(B)` in a different category, and composition in that category handles the plumbing.

These effectful arrows `A → M(B)` form their own category — the **Kleisli category** for `M`. Composition in the Kleisli category is defined as: given `f : A → M(B)` and `g : B → M(C)`, their composite applies `f`, extracts the `B` from inside `M(B)`, and feeds it to `g`, producing `M(C)`. The monad's internal machinery threads the random seed, propagates the configuration, and accumulates the log. The individual operators never see this plumbing.

The GA pipeline

```
evaluate >>> select >>> crossover >>> mutate >>> unwrap
```

is a composition in the Kleisli category. Each `>>>` threads the effects — randomness, configuration, logging — through invisibly. The result is a single effectful arrow `[genome] → M([genome])` that, when executed with a configuration and a random seed, produces a new population and a log of what happened.

## Functors and natural transformations

Two more concepts complete the toolkit needed for this paper.

A **functor** F : **C** → **D** is a mapping between categories. It sends each object in **C** to an object in **D** and each arrow in **C** to an arrow in **D**, preserving composition and identities. Think: a systematic translation from one domain to another that respects the structure of both.

In the island model (Section 3), each island runs its own evolutionary pipeline. The island model itself is a functor: it takes a single-population GA (an arrow in the "single-population" category) and produces a multi-population GA (an arrow in the "island" category). The functor preserves composition — if you compose two single-population strategies and then lift them to islands, you get the same result as lifting each one separately and composing the island strategies.

A **natural transformation** η : F → G between two functors F and G is a family of arrows η_A : F(A) → G(A), one for every object A, satisfying the **naturality condition**: for every arrow f : A → B,

```
         η_A
  F(A) ───────→ G(A)
   │               │
   │ F(f)          │ G(f)
   ↓               ↓
  F(B) ───────→ G(B)
         η_B
```

The square commutes: going right-then-down gives the same result as going down-then-right.

**Migration is a natural transformation.** Consider two island populations, each modeled as a list functor. Migration moves individuals from one island to another. The naturality condition says:

```
             migrate
  Island₁(genome) ─────────→ Island₂(genome)
       │                            │
       │ map f                      │ map f
       ↓                            ↓
  Island₁(phenotype) ──────→ Island₂(phenotype)
              migrate
```

Migrating genomes and then computing phenotypes gives the same result as computing phenotypes and then migrating. This is true because migration moves individuals between populations without inspecting or modifying them — it is a **structural** operation on populations, not a **content** operation on individuals. In biological terms: gene flow is indifferent to the genotype-phenotype map.

This is not a vacuous observation. The naturality of migration is what enables the Strict/Lax Dichotomy Theorem (Section 6.3): the island functor preserves composition strictly if and only if migration is zero. Any nonzero migration — even one individual per thousand generations — introduces the same qualitative deviation from strict composition. The magnitude of deviation depends on the connectivity topology (ring, fully connected, hierarchical), not on the migration rate. This echoes a classical result in population genetics: Wright's F_ST depends on the product Nm (effective population size × migration rate) and on the spatial structure, not on N or m alone.

## Summary of categorical vocabulary

| Category theory | In this paper | Biological intuition |
|----------------|---------------|---------------------|
| Object | Population type | What kind of data each individual carries |
| Arrow (morphism) | GA operator | A step in the evolutionary lifecycle |
| Composition | Pipeline assembly (`>>>`) | "Do this, then do that" |
| Identity | Pass-through operator | A generation where nothing happens |
| Monad | Effect context (`M`) | The bookkeeping: randomness, config, logging |
| Kleisli arrow | Effectful GA operator | An operator that uses random numbers, reads config, writes logs |
| Kleisli composition | Effectful pipeline (`>>>`) | Compose operators while threading all bookkeeping automatically |
| Functor | Island model | Lift a single-population GA to a multi-population GA |
| Natural transformation | Migration | Move individuals between islands without inspecting them |

The remainder of Section 2 introduces the specific categorical constructions used in our framework (co-Kleisli composition in 2.2, monad transformer layers in 2.3). Readers comfortable with the vocabulary above may skip directly to Section 3, where we apply the framework to two concrete domains.

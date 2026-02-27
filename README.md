# categorical-evolution

**Category-theoretic genetic algorithms using MTL effects.**

A Haskell library that composes genetic operators (selection, mutation, crossover) as morphisms in a Kleisli category, with MTL monad transformers providing the effect context.

## The Idea

### category-printf: composing format specs

[`category-printf`](https://github.com/cgibbard/category-printf) uses the co-Kleisli category for the comonad `(->) m` (functions from a monoid) to build type-safe format specifications:

```haskell
type Format m = Cokleisli ((->) m)

-- Composition accumulates argument types:
"Hello, " . i . "! You are " . s . " years old."
-- :: Format String String (String -> Int -> String)
```

Each format spec is a morphism. Composition (`.`) in the co-Kleisli category accumulates the arguments that `sprintf` will demand. The monoid `m` (String, Text, etc.) is the "output medium."

### categorical-evolution: composing genetic operators

We use the Kleisli category for the evolution monad `EvoM` to compose genetic operators:

```haskell
newtype GeneticOp m a b = GeneticOp { runOp :: [a] -> m [b] }

-- Composition accumulates effects:
evaluate fitnessFunc
  >>>: elitistSelect
  >>>: onePointCrossover
  >>>: pointMutate bitFlip
-- :: GeneticOp EvoM [Bool] (Scored [Bool])
```

Each operator is a morphism. Composition (`>>>:`) in the Kleisli category accumulates the monadic effects that `runEvoM` will resolve. The monad `EvoM` (randomness, config, logging) is the "effect medium."

### The parallel

| | **category-printf** | **categorical-evolution** |
|---|---|---|
| **Category** | Co-Kleisli `((->) m)` | Kleisli `EvoM` |
| **Morphisms** | Format specifications | Genetic operators |
| **Composition** | Accumulates argument types | Accumulates monadic effects |
| **"Medium"** | Monoid (String, Text) | Monad (Reader, State, Writer) |
| **Runner** | `sprintf`, `printfWith` | `runEvoM`, `evolve` |

Both libraries share the insight that **interesting computations can be decomposed into composable pieces within a category**, where the categorical structure handles the bookkeeping (argument threading vs. effect propagation).

## The MTL Connection

The evolution monad uses a standard MTL transformer stack:

```haskell
type EvoM = ReaderT GAConfig (StateT StdGen (Writer GALog))
```

Each layer maps to a GA concern:

- **`MonadReader GAConfig`** — GA parameters (population size, mutation rate, tournament size). Operators read config without threading it manually.
- **`MonadState StdGen`** — Randomness. Mutation, selection, and crossover all need random numbers; the PRNG state threads through automatically.
- **`MonadWriter GALog`** — Evolutionary history. Each generation logs its statistics (best/avg/worst fitness, diversity). History accumulates monoidal.

### MonadSelect as fitness-based selection

[`sjshuck/mtl`](https://github.com/sjshuck/mtl) adds `MonadSelect r`:

```haskell
class MonadSelect r m | m -> r where
  select :: ((a -> r) -> a) -> m a
```

This says: "given a ranking function `(a -> r)`, pick the best `a`." Tournament selection in a GA does exactly this — rank individuals by fitness, pick the winner. Our `tournamentSelect` is the concrete instantiation of this abstract pattern.

### MonadAccum as evolutionary history

`MonadAccum w` provides `look` (see accumulated state) and `add` (append to it). Evolutionary history is exactly this: each generation appends its statistics to a monoidal log. Our `logGeneration` operator uses `MonadWriter` (a close relative of `MonadAccum`) to accumulate `GALog` entries.

## Usage

```haskell
import Evolution.Category
import Evolution.Effects
import Evolution.Operators
import Evolution.Pipeline
import Evolution.Examples.BitString

-- Run OneMax: evolve bitstrings to maximize number of 1-bits
main :: IO ()
main = runOneMax 20
```

### Custom pipelines

```haskell
-- Build a custom evolutionary pipeline by composing operators:
myPipeline :: ([a] -> Double) -> (a -> EvoM a) -> Int -> GeneticOp EvoM [a] [a]
myPipeline fitFunc mutFunc gen =
  evaluate fitFunc
    >>>: logGeneration gen
    >>>: tournamentSelect        -- or: rouletteSelect, elitistSelect
    >>>: onePointCrossover       -- or: uniformCrossover
    >>>: pointMutate mutFunc
    >>>: pointwise individual    -- unwrap Scored
```

Each `>>>:` is Kleisli composition. The pipeline reads left-to-right, like prose: "evaluate, then log, then select, then cross over, then mutate."

### Island model

Run multiple populations in parallel with periodic migration:

```haskell
import Evolution.Island

let model = IslandModel
      { numIslands    = 4
      , migrationRate = 0.1   -- 10% of population migrates
      , migrationFreq = 10    -- every 10 generations
      , topology      = Ring  -- or FullyConnected
      }
    islands = uniformIslands oneMaxFitness bitFlip populations
    result = evolveIslands model oneMaxFitness config gen islands
```

Migration is a natural transformation between population functors — it moves
individuals between islands without caring about each island's pipeline.

### Competitive coevolution

Two populations evaluate against each other, creating an evolutionary arms race:

```haskell
import Evolution.Coevolution

-- Population A tries to match B, B tries to differ from A
let matchScore a b = fromIntegral $ length $ filter id $ zipWith (==) a b
    differScore b a = fromIntegral $ length $ filter id $ zipWith (/=) b a
    result = coevolve matchScore differScore
                      bitFlip bitFlip
                      defaultCoevoConfig config gen
                      populationA populationB
```

This is the same structure as arena tournaments in morphological evolution:
creatures don't have intrinsic fitness, only relative fitness against opponents.
Intransitive dynamics emerge naturally.

### Strategy composition

Compose entire evolutionary algorithms the same way you compose operators:

```haskell
import Evolution.Strategy

-- Race a generational GA against a steady-state GA
let strat = race
      (generationalGA oneMaxFitness bitFlip (AfterGens 50))
      (steadyStateGA oneMaxFitness bitFlip (AfterGens 50))

-- Adaptive: try short run, fall back if not converged
let strat = adaptive
      (\r -> fitness (resultBest r) >= 18.0)
      (generationalGA fitFunc mutFunc (AfterGens 10))
      (steadyStateGA fitFunc mutFunc (AfterGens 40))

-- Sequential: explore broadly, then refine
let strat = sequential
      (generationalGA fitFunc mutFunc (AfterGens 25))
      (generationalGA fitFunc mutFunc (Plateau 10))
```

Strategies form a monoid under `sequential` with `idStrategy` as identity.
Termination conditions (`StopWhen`) form a Boolean algebra via `StopOr`/`StopAnd`.

### Landscape analysis

Characterize a fitness landscape, then auto-select the best strategy:

```haskell
import Evolution.Landscape

-- Analyze via random walk
let mutGenome = singlePointMutation bitFlip
profile <- analyzeLandscape fitFunc mutGenome startPoint 500

-- ruggedness profile ==> 0.06 (smooth)
-- correlationLen profile ==> 16.8 (long-range correlation)

-- Auto-select strategy based on landscape
let strat = recommendStrategy profile fitFunc mutFunc (AfterGens 50)
```

Smooth landscapes get generational GA. Rugged landscapes get race(gen, steady-state).
Neutral landscapes get steady-state for sustained exploration.

## Modules

| Module | Description |
|---|---|
| `Evolution.Category` | Core types: `GeneticOp`, `Scored`, composition operators |
| `Evolution.Effects` | `EvoM` monad stack, random utilities, configuration |
| `Evolution.Operators` | Selection, crossover, mutation as categorical morphisms |
| `Evolution.Pipeline` | High-level `evolve` / `evolveN`, `generationStep` |
| `Evolution.Island` | Island model with ring/fully-connected migration |
| `Evolution.Coevolution` | Competitive coevolution with sample-based evaluation |
| `Evolution.Strategy` | Composable evolutionary strategies with combinators |
| `Evolution.Landscape` | Fitness landscape analysis and auto-strategy selection |
| `Evolution.Examples.BitString` | OneMax and target-matching examples |
| `Evolution.Examples.SymbolicRegression` | Genetic programming with expression trees |

## Building

```bash
cabal build
cabal test
cabal run    # (if you add an executable)
```

## Acknowledgements

- [Cale Gibbard](https://github.com/cgibbard) for `category-printf` — the co-Kleisli composition trick
- [Sam Shuck](https://github.com/sjshuck) for the `mtl` fork with `MonadSelect` and `MonadAccum`
- The conceptual bridge: categories are everywhere, and the same compositional patterns that make format strings elegant also make evolutionary algorithms elegant

## License

BSD-3-Clause

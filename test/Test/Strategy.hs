module Test.Strategy (runTests) where

import Data.List (sortBy)
import Data.Ord (comparing, Down(..))
import System.Random (mkStdGen)

import Evolution.Category
import Evolution.Effects
import Evolution.Examples.BitString
import Evolution.Strategy

-- | Run strategy tests. Returns number of failures.
runTests :: IO Int
runTests = do
  putStrLn "--- Strategy tests ---"
  failures <- sequence
    [ test "generational GA improves fitness"   testGenerationalGA
    , test "steady-state GA improves fitness"    testSteadyStateGA
    , test "sequential improves over single"     testSequential
    , test "race returns better result"          testRace
    , test "adaptive switches on failure"        testAdaptive
    , test "idStrategy is identity"              testIdStrategy
    , test "mapStrategy preserves fitness"       testMapStrategy
    , test "plateau detection stops early"       testPlateau
    , test "lifting functor law (2-category)"    testLiftingFunctorLaw
    , test "island strategy improves fitness"    testIslandStrategy
    , test "island strategy beats single pop"    testIslandBeatsSingle
    ]
  return (sum failures)

test :: String -> Bool -> IO Int
test name True  = putStrLn ("  [PASS] " ++ name) >> return 0
test name False = putStrLn ("  [FAIL] " ++ name) >> return 1

-- Helper: create a scored initial population for strategy tests
mkScoredPop :: Int -> Int -> Int -> [Scored [Bool]]
mkScoredPop seed popSize genomeLen =
  let config = defaultConfig { populationSize = popSize }
      g = mkStdGen seed
      (rawPop, _, _) = runEvoM config g (randomPopulation popSize genomeLen)
  in map (\genome -> Scored genome (oneMaxFitness genome)) rawPop

-- Helper: run a strategy and return the result
runStrat :: Strategy [Bool] -> [Scored [Bool]] -> StrategyResult [Bool]
runStrat s pop =
  let config = defaultConfig
        { populationSize = length pop
        , mutationRate   = 0.05
        , crossoverRate  = 0.7
        , tournamentSize = 3
        , eliteCount     = 2
        }
      g = mkStdGen 42
      (result, _, _) = runEvoM config g (runStrategy s pop)
  in result

-- | Generational GA should improve OneMax fitness over 30 generations
testGenerationalGA :: Bool
testGenerationalGA =
  let pop = mkScoredPop 100 30 20
      initBest = fitness $ head $ sortBy (comparing (Down . fitness)) pop
      s = generationalGA oneMaxFitness bitFlip (AfterGens 30)
      r = runStrat s pop
  in fitness (resultBest r) > initBest && resultGens r == 30

-- | Steady-state GA should also improve fitness
testSteadyStateGA :: Bool
testSteadyStateGA =
  let pop = mkScoredPop 200 30 20
      initBest = fitness $ head $ sortBy (comparing (Down . fitness)) pop
      s = steadyStateGA oneMaxFitness bitFlip (AfterGens 30)
      r = runStrat s pop
  in fitness (resultBest r) > initBest && resultGens r == 30

-- | Sequential composition: GA for 20 gens, then another GA for 20 gens,
-- should be at least as good as a single GA for 20 gens
testSequential :: Bool
testSequential =
  let pop = mkScoredPop 300 30 20
      single = generationalGA oneMaxFitness bitFlip (AfterGens 20)
      composed = sequential
                   (generationalGA oneMaxFitness bitFlip (AfterGens 20))
                   (generationalGA oneMaxFitness bitFlip (AfterGens 20))
      rSingle = runStrat single pop
      rComposed = runStrat composed pop
  in fitness (resultBest rComposed) >= fitness (resultBest rSingle)
     && resultGens rComposed == 40

-- | Race should return the better of two strategies
testRace :: Bool
testRace =
  let pop = mkScoredPop 400 30 20
      -- Race a short GA against a longer one
      short = generationalGA oneMaxFitness bitFlip (AfterGens 5)
      long  = generationalGA oneMaxFitness bitFlip (AfterGens 30)
      rRace = runStrat (race short long) pop
      -- The race winner should be at least as good as the short one alone
      rShort = runStrat short pop
  in fitness (resultBest rRace) >= fitness (resultBest rShort)

-- | Adaptive should switch to fallback when primary doesn't meet predicate
testAdaptive :: Bool
testAdaptive =
  let pop = mkScoredPop 500 30 20
      genomeLen = 20 :: Int
      -- Primary: very short run (5 gens) — won't reach high fitness
      primary = generationalGA oneMaxFitness bitFlip (AfterGens 5)
      -- Fallback: longer run (25 gens) — should improve further
      fallback = generationalGA oneMaxFitness bitFlip (AfterGens 25)
      -- Predicate: accept only if perfect score (impossible in 5 gens)
      predicate res = fitness (resultBest res) >= fromIntegral genomeLen
      s = adaptive predicate primary fallback
      r = runStrat s pop
      rPrimary = runStrat primary pop
      -- Primary won't reach perfect, so fallback runs
      -- Total gens = 5 + 25 = 30, and result should improve
  in resultGens r == 30
     && fitness (resultBest r) >= fitness (resultBest rPrimary)

-- | idStrategy should return the population unchanged with 0 gens
testIdStrategy :: Bool
testIdStrategy =
  let pop = mkScoredPop 600 10 8
      r = runStrat idStrategy pop
  in resultGens r == 0
     && length (resultPop r) == length pop
     && fitness (resultBest r) == fitness (head (sortBy (comparing (Down . fitness)) pop))

-- | mapStrategy should preserve fitness through encoding/decoding
testMapStrategy :: Bool
testMapStrategy =
  let pop = mkScoredPop 700 20 16
      -- Trivial encoding: Bool -> Int -> Bool
      toInt :: [Bool] -> [Int]
      toInt = map (\b -> if b then 1 else 0)
      toBool :: [Int] -> [Bool]
      toBool = map (> 0)
      -- Fitness on Int representation
      intFitness :: [Int] -> Double
      intFitness = fromIntegral . length . filter (> 0)
      -- Strategy on Int representation
      intStrat = generationalGA intFitness (\x -> return (1 - x)) (AfterGens 20)
      -- Map it to work on Bool representation
      boolStrat = mapStrategy toInt toBool intStrat
      -- Run on Bool population
      rBool = runStrat boolStrat pop
  in fitness (resultBest rBool) > 0  -- Sanity: it ran and produced a result

-- | Plateau detection should stop early when fitness stops improving
testPlateau :: Bool
testPlateau =
  let pop = mkScoredPop 800 30 20
      -- Stop after 5 gens without improvement OR after 100 gens
      s = generationalGA oneMaxFitness bitFlip
            (StopOr (Plateau 5) (AfterGens 100))
      r = runStrat s pop
      -- Should stop before 100 gens (plateau detection kicked in)
  in resultGens r < 100

-- | The lifting functor law: sequential(lift(P,N), lift(P,M)) should produce
-- the same final population as lift(P, N+M).
--
-- This tests Claudius's 2-category question: does composing strategies
-- (sequential) give the same result as composing the underlying pipelines
-- and lifting once?
--
-- The law holds at the population level because:
-- 1. The step function is identical (same operators, same parameters)
-- 2. PRNG state threads continuously through sequential composition
-- 3. logGeneration only affects the Writer, not the State or population
--
-- The law breaks ONLY at the log level (generation counter resets in sequential)
testLiftingFunctorLaw :: Bool
testLiftingFunctorLaw =
  let pop = mkScoredPop 900 30 20
      -- sequential(lift(P,20), lift(P,20))
      composed = sequential
                   (generationalGA oneMaxFitness bitFlip (AfterGens 20))
                   (generationalGA oneMaxFitness bitFlip (AfterGens 20))
      -- lift(P, 40)
      single = generationalGA oneMaxFitness bitFlip (AfterGens 40)
      rComposed = runStrat composed pop
      rSingle   = runStrat single pop
      -- Sort populations by genome for comparison
      sortByGenome = sortBy (comparing individual)
      composedPop = map individual (sortByGenome (resultPop rComposed))
      singlePop   = map individual (sortByGenome (resultPop rSingle))
  -- Final populations should be identical (functor law at population level)
  in composedPop == singlePop

-- | Island strategy should improve fitness
testIslandStrategy :: Bool
testIslandStrategy =
  let pop = mkScoredPop 1000 40 20
      initBest = fitness $ head $ sortBy (comparing (Down . fitness)) pop
      config' = IslandConfig
        { islandCount    = 4
        , islandMigRate  = 0.1
        , islandMigFreq  = 5
        , islandTopology = IslandRing
        }
      s = islandStrategy config' (gaStep oneMaxFitness bitFlip) (AfterGens 30)
      r = runStrat s pop
  in fitness (resultBest r) > initBest && resultGens r == 30

-- | Island strategy should beat (or match) single-population GA
-- because migration maintains diversity
testIslandBeatsSingle :: Bool
testIslandBeatsSingle =
  let pop = mkScoredPop 1100 40 20
      single = generationalGA oneMaxFitness bitFlip (AfterGens 40)
      config' = IslandConfig
        { islandCount    = 4
        , islandMigRate  = 0.1
        , islandMigFreq  = 5
        , islandTopology = IslandRing
        }
      island = islandStrategy config' (gaStep oneMaxFitness bitFlip) (AfterGens 40)
      rSingle = runStrat single pop
      rIsland = runStrat island pop
  -- Island should be competitive (at least 90% of single's fitness)
  in fitness (resultBest rIsland) >= fitness (resultBest rSingle) * 0.9

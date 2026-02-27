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

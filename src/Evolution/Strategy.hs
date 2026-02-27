{-# LANGUAGE ScopedTypeVariables #-}

-- | The category of evolutionary strategies.
--
-- == From operators to strategies
--
-- 'Evolution.Category' gives us composable /operators/ — single
-- population transformations like selection, crossover, and mutation.
-- Those compose within a generation.
--
-- This module lifts the abstraction: /strategies/ are composable
-- /algorithms/, each running for multiple generations. Strategies form
-- a category:
--
--   * Objects: scored population types
--   * Morphisms: strategies (algorithms transforming populations)
--   * Composition: 'sequential' (run one, then the other)
--   * Identity: 'idStrategy' (do nothing)
--
-- == Combinators as categorical constructions
--
-- * 'sequential' — composition in the strategy category
-- * 'race' — product (run both, take the best)
-- * 'adaptive' — coproduct with discriminator (try one, fall back to other)
-- * 'mapStrategy' — functor between strategy categories on different types
--
-- == Termination conditions
--
-- 'StopWhen' forms a Boolean algebra over stopping conditions:
-- 'StopOr' and 'StopAnd' let you combine conditions freely.
module Evolution.Strategy
  ( -- * Termination conditions
    StopWhen(..)
    -- * Strategy results
  , StrategyResult(..)
    -- * Strategy type
  , Strategy(..)
  , mkStrategy
  , idStrategy
    -- * Core strategies
  , generationalGA
  , steadyStateGA
    -- * Combinators
  , sequential
  , race
  , adaptive
    -- * Strategy functor
  , mapStrategy
  ) where

import Control.Monad.Reader
import Control.Monad.State.Strict
import Data.List (sortBy)
import Data.Ord (comparing, Down(..))
import System.Random

import Evolution.Category
import Evolution.Effects
import Evolution.Operators

-- | Termination condition for a strategy. Forms a Boolean algebra
-- under 'StopOr' (join) and 'StopAnd' (meet).
data StopWhen
  = AfterGens Int              -- ^ Stop after N steps
  | FitnessAbove Double        -- ^ Stop when best fitness exceeds threshold
  | Plateau Int                -- ^ Stop when no improvement for N consecutive steps
  | StopOr StopWhen StopWhen   -- ^ Stop when either condition is met
  | StopAnd StopWhen StopWhen  -- ^ Stop when both conditions are met
  deriving (Show, Eq)

-- Internal state for checking termination.
data StopState = StopState
  { ssGens      :: !Int     -- ^ Steps so far
  , ssBest      :: !Double  -- ^ Best fitness seen
  , ssNoImprove :: !Int     -- ^ Consecutive steps without improvement
  }

-- | Check whether a termination condition is satisfied.
checkStop :: StopWhen -> StopState -> Bool
checkStop (AfterGens n)   ss = ssGens ss >= n
checkStop (FitnessAbove t) ss = ssBest ss >= t
checkStop (Plateau n)      ss = ssNoImprove ss >= n
checkStop (StopOr a b)     ss = checkStop a ss || checkStop b ss
checkStop (StopAnd a b)    ss = checkStop a ss && checkStop b ss

-- | Result of running a strategy.
data StrategyResult a = StrategyResult
  { resultPop   :: [Scored a]   -- ^ Final scored population
  , resultBest  :: Scored a     -- ^ Best individual found across all steps
  , resultGens  :: Int          -- ^ Total steps taken
  } deriving (Show)

-- | An evolutionary strategy: a composable algorithm that transforms
-- scored populations over multiple generations.
--
-- Strategies form a monoid under 'sequential' composition with
-- 'idStrategy' as identity.
newtype Strategy a = Strategy
  { runStrategy :: [Scored a] -> EvoM (StrategyResult a)
  }

-- | Identity strategy: returns the population unchanged.
-- This is the identity morphism in the strategy category.
idStrategy :: Strategy a
idStrategy = Strategy $ \pop ->
  return $ StrategyResult pop (bestOf pop) 0

-- | Create a strategy from a step function and termination condition.
--
-- The step function takes a step counter and the current scored population,
-- and returns the next scored population. The strategy iterates the step
-- function until the termination condition is satisfied.
mkStrategy :: (Int -> [Scored a] -> EvoM [Scored a]) -> StopWhen -> Strategy a
mkStrategy step stop = Strategy $ \pop0 -> do
  let best0 = bestOf pop0
      ss0 = StopState 0 (fitness best0) 0
  go ss0 best0 pop0
  where
    go ss overallBest pop
      | checkStop stop ss = return $ StrategyResult pop overallBest (ssGens ss)
      | otherwise = do
          pop' <- step (ssGens ss) pop
          let bestNow = bestOf pop'
              newBestFit = max (ssBest ss) (fitness bestNow)
              improved = fitness bestNow > ssBest ss
              overallBest' = if fitness bestNow > fitness overallBest
                             then bestNow else overallBest
              ss' = StopState
                { ssGens      = ssGens ss + 1
                , ssBest      = newBestFit
                , ssNoImprove = if improved then 0 else ssNoImprove ss + 1
                }
          go ss' overallBest' pop'

-- | Standard generational GA as a strategy.
--
-- Each step runs: log -> elitist select -> crossover -> mutate -> re-evaluate.
-- This wraps the existing operator pipeline from "Evolution.Operators".
generationalGA :: ([a] -> Double) -> (a -> EvoM a) -> StopWhen -> Strategy [a]
generationalGA fitFunc mutFunc stop = mkStrategy step stop
  where
    step gen pop = runOp pipeline pop
      where
        pipeline = logGeneration gen
               >>>: elitistSelect
               >>>: onePointCrossover
               >>>: pointMutate mutFunc
               >>>: pointwise reeval
        reeval s = Scored (individual s) (fitFunc (individual s))

-- | Steady-state GA as a strategy.
--
-- Instead of replacing the entire population each generation, this
-- replaces individuals one pair at a time: select two parents,
-- cross and mutate to produce two offspring, replace the two worst.
-- Each step does @popSize / 2@ replacements (one "generation equivalent").
--
-- Steady-state GAs typically maintain more population diversity than
-- generational GAs, at the cost of slower convergence on easy problems.
steadyStateGA :: forall a. ([a] -> Double) -> (a -> EvoM a) -> StopWhen -> Strategy [a]
steadyStateGA fitFunc mutFunc stop = mkStrategy step stop
  where
    step :: Int -> [Scored [a]] -> EvoM [Scored [a]]
    step _gen pop0 = do
      cfg <- ask
      let replacements = max 1 (populationSize cfg `div` 2)
      doReplacements replacements pop0

    doReplacements :: Int -> [Scored [a]] -> EvoM [Scored [a]]
    doReplacements 0 pop = return pop
    doReplacements n pop = do
      pop' <- singleReplace pop
      doReplacements (n - 1) pop'

    singleReplace :: [Scored [a]] -> EvoM [Scored [a]]
    singleReplace pop = do
      cfg <- ask
      let k = tournamentSize cfg
          cxRate = crossoverRate cfg
      -- Tournament select two parents
      p1 <- tournamentPick k pop
      p2 <- tournamentPick k pop
      -- Crossover
      r <- randomDouble
      (c1g, c2g) <- if r < cxRate
        then do
          let g1 = individual p1
              g2 = individual p2
              len = min (length g1) (length g2)
          pt <- randomInt 1 (max 1 (len - 1))
          let (h1, t1) = splitAt pt g1
              (h2, t2) = splitAt pt g2
          return (h1 ++ t2, h2 ++ t1)
        else return (individual p1, individual p2)
      -- Mutate each gene (respecting mutation rate)
      let rate = mutationRate cfg
      c1' <- mapM (\g -> do
        r' <- randomDouble
        if r' < rate then mutFunc g else return g) c1g
      c2' <- mapM (\g -> do
        r' <- randomDouble
        if r' < rate then mutFunc g else return g) c2g
      -- Score offspring
      let s1 = Scored c1' (fitFunc c1')
          s2 = Scored c2' (fitFunc c2')
      -- Replace two worst (sorted ascending by fitness)
      let sorted = sortBy (comparing fitness) pop
      return (s1 : s2 : drop 2 sorted)

    tournamentPick :: Int -> [Scored b] -> EvoM (Scored b)
    tournamentPick k pop = do
      contestants <- replicateM k (randomChoice pop)
      return $ bestOf contestants

-- | Sequential composition: run the first strategy to completion,
-- then run the second starting from the first's final population.
--
-- This is composition in the strategy category.
-- @sequential idStrategy s = s@ and @sequential s idStrategy = s@.
--
-- Generation counts are summed. The overall best is the better of
-- the two strategies' bests.
sequential :: Strategy a -> Strategy a -> Strategy a
sequential s1 s2 = Strategy $ \pop -> do
  r1 <- runStrategy s1 pop
  r2 <- runStrategy s2 (resultPop r1)
  let best = betterOf (resultBest r1) (resultBest r2)
  return $ StrategyResult (resultPop r2) best (resultGens r1 + resultGens r2)

-- | Race: run both strategies on the same initial population with
-- independent random seeds, return whichever finds the better result.
--
-- This is the categorical product: both morphisms are applied,
-- and the best outcome is projected out.
race :: Strategy a -> Strategy a -> Strategy a
race s1 s2 = Strategy $ \pop -> do
  -- Split the PRNG for independent runs
  g <- get
  let (g1, g2) = split g
  -- Run first strategy
  put g1
  r1 <- runStrategy s1 pop
  -- Run second strategy
  put g2
  r2 <- runStrategy s2 pop
  -- Return the better result
  return $ if fitness (resultBest r1) >= fitness (resultBest r2) then r1 else r2

-- | Adaptive: run the primary strategy, then check a predicate on the result.
-- If the predicate returns 'True', accept the result.
-- If 'False', run the fallback strategy starting from the primary's
-- final population.
--
-- This is a conditional coproduct: the predicate discriminates between
-- accepting the identity (keep result) or applying the fallback morphism.
--
-- Common predicates:
--
-- @
--   -- Switch if best fitness is below threshold
--   adaptive (\\r -> fitness (resultBest r) >= 8.0) primary fallback
--
--   -- Switch if the primary ran too long without converging
--   adaptive (\\r -> resultGens r < 50) primary fallback
-- @
adaptive :: (StrategyResult a -> Bool) -> Strategy a -> Strategy a -> Strategy a
adaptive predicate primary fallback = Strategy $ \pop -> do
  r1 <- runStrategy primary pop
  if predicate r1
    then return r1
    else do
      r2 <- runStrategy fallback (resultPop r1)
      let best = betterOf (resultBest r1) (resultBest r2)
      return $ StrategyResult (resultPop r2) best (resultGens r1 + resultGens r2)

-- | Map a strategy from one representation to another.
--
-- Given decoding and encoding functions between types @a@ and @b@,
-- transform a @Strategy a@ into a @Strategy b@. This is a functor
-- between strategy categories:
--
-- @
--   mapStrategy id id s = s                                        -- identity
--   mapStrategy (f . g) (h . k) = mapStrategy g k . mapStrategy f h -- composition
-- @
--
-- Use this to apply a strategy designed for one genome representation
-- to a different one. For example, map a bitstring strategy to work
-- on integer-encoded genomes via binary encoding/decoding.
mapStrategy :: (b -> a) -> (a -> b) -> Strategy a -> Strategy b
mapStrategy decode encode s = Strategy $ \popB -> do
  let popA = map (\scored -> Scored (decode (individual scored)) (fitness scored)) popB
  r <- runStrategy s popA
  let popB' = map (\scored -> Scored (encode (individual scored)) (fitness scored)) (resultPop r)
      bestB = Scored (encode (individual (resultBest r))) (fitness (resultBest r))
  return $ StrategyResult popB' bestB (resultGens r)

-- Helpers (not exported)

bestOf :: [Scored a] -> Scored a
bestOf = head . sortBy (comparing (Down . fitness))

betterOf :: Scored a -> Scored a -> Scored a
betterOf a b = if fitness a >= fitness b then a else b

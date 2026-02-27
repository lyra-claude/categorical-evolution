# Composable Evolutionary Strategies: When Composition Structure Is Readable from Population Dynamics

## Abstract

Evolutionary algorithms are routinely composed — run one strategy, then another;
race two strategies in parallel; adaptively switch mid-run. Yet the compositional
structure of these algorithms is invisible in standard frameworks: there is no
formal language for asking whether the order of composition matters, or whether
composing strategies inside an island model produces the same dynamics as composing
their island-lifted counterparts. We present a categorical framework in which
evolutionary strategies are morphisms, composition is sequential execution, and
the island model is a 2-functor that lifts strategies into parallel populations
with migration. Our main empirical finding is that **composition structure is
readable from population dynamics**: four qualitatively different compositions
(flat, hourglass, island, adaptive) produce distinct diversity trajectories on
the same fitness landscape. Our main theoretical contribution is a **dichotomy
theorem** that simultaneously classifies island strategy composition as strict or
lax and establishes that lax composition produces position-invariant divergence
— the magnitude of the divergence between composed-then-lifted and
lifted-then-composed strategies is determined by the spectral gap of the
evolutionary Markov chain, not by the number or position of affected migration
events.


## 1. Introduction

Evolutionary algorithms do not run in isolation. Practitioners compose them:
an exploratory phase with high mutation feeds into a convergence phase with
strong selection; multiple sub-populations evolve independently and exchange
migrants; an adaptive controller monitors progress and switches strategies
mid-run. These compositions are the operational reality of evolutionary
computation, but they are typically described in ad hoc procedural terms —
"run algorithm A for 20 generations, then switch to algorithm B" — with no
formal account of what composition preserves, what it breaks, and what
structure it reveals.

This paper provides that formal account.

We model evolutionary strategies as morphisms in a category whose objects are
population types and whose composition is sequential execution: run the first
strategy to completion, pass its final population to the second. Standard
combinators — sequential, race, adaptive switching — correspond to categorical
constructions (composition, product, coproduct with discriminator). The island
model, which lifts a strategy into a collection of parallel sub-populations
with periodic migration, becomes a 2-functor parameterized by migration rate,
frequency, and topology.

This framework is not merely a relabeling exercise. It makes predictions, and
those predictions are testable. Two in particular:

**Prediction 1: Trajectory readability.** If strategies are genuinely different
morphisms, their compositions should produce qualitatively different population
dynamics. This is obvious in principle but has not been systematically
demonstrated. We show that four compositions — a flat generational GA, an
hourglass strategy (explore → bottleneck → diversify), an island model, and
an adaptive switcher — produce diversity trajectories that are visually
distinguishable and qualitatively characteristic of their compositional
structure (Section 3). The hourglass's three-phase trajectory — a diversity
spike during exploration, a sharp crash at the bottleneck, and a rebound during
diversification — is the paper's signature figure. Notably, genotypic and
phenotypic diversity decouple at composition boundaries: the bottleneck
minimizes genotypic diversity while *maximizing* phenotypic diversity, acting
as a filter that kills structural redundancy while preserving functional
variation.

**Prediction 2: The strict/lax dichotomy.** The island functor should interact
with sequential composition in a structured way. Either composing strategies
and then lifting to islands produces the same result as lifting and then
composing (strict functoriality), or it does not (lax functoriality), and the
discrepancy should be characterizable. We prove a dichotomy theorem (Section 5):

- *Strict case*: When migration rate is zero or migration frequency exceeds
  the total strategy length, the island functor preserves composition exactly.
  Composed-then-lifted equals lifted-then-composed at the population level.

- *Lax case*: Under any non-zero inter-island coupling, the functor is
  uniformly lax. The population divergence between composed-then-lifted and
  lifted-then-composed saturates to a characteristic level D* that depends on
  migration rate and frequency but is **independent of composition boundary
  position and the number of affected migration events**. D* is determined
  asymptotically by the spectral gap of the evolutionary Markov chain over
  population states.

The second clause is the surprising result. One might expect that a composition
boundary that disrupts a single migration event would produce less divergence
than one that phase-shifts an entire migration schedule. It does not. The
evolutionary dynamics amplify any perturbation — whether a single displaced
event or a dozen — to the same saturation level, because the underlying
Markov chain is ergodic. The system's memory of the perturbation's size is
erased by mixing. This connects evolutionary computation to ergodic theory
and has a stark practical consequence: **you cannot reason about composed
island strategies by reasoning about islands independently**, no matter how
infrequent migration is. A single migration event per thousand generations is
categorically equivalent to continuous migration, in the precise sense that both
produce the same characteristic divergence under composition.

The paper proceeds as follows. Section 2 introduces the categorical framework:
the category of genetic operators, the category of strategies, and the island
functor. Section 3 presents our first experiment — four strategy compositions
on symbolic regression, tracking genotypic and phenotypic diversity trajectories
per generation. Section 4 presents our second experiment — a migration frequency
sweep and boundary position sweep that reveal the strict/lax dichotomy and
boundary invariance empirically. Section 5 states and proves the dichotomy
theorem. Section 6 surveys related work on island models, adaptive operator
selection, and categorical approaches to computation. Section 7 discusses
implications, limitations, and open questions.

All experiments use the `categorical-evolution` Haskell library, which
implements the framework described here. Code and data are available at
[repository URL].

# Introduction: Four Groups, One Insight

*(Revised draft — aligned to actual paper.tex on feat/act2026-paper. Retains
narrative structure from v1; replaces all mismatched claims.)*

---

Evolutionary algorithms are routinely described, but rarely formalized. A
practitioner report reads: "run selection, then crossover, then mutation;
repeat; stop when fitness plateaus." This is a procedure, not a structure.
It tells us what to do but not what the operations preserve, what composes
cleanly, or what happens when the same operators are lifted into a
multi-population system with migration. The absence of formal structure
has a cost: we cannot predict how pipeline organization shapes evolutionary
dynamics. We can only run experiments and observe.

This paper provides the formal account.

## The Convergence

Over the past three years, four independent research groups — working on
four different optimization paradigms — have arrived at the same structural
insight: the way an optimization pipeline *composes* its components
determines its dynamics more than the components themselves.

**Neural networks.** Gavranović et al. showed that deep learning
architectures are constructions in **Para**, where gradient descent is a
functor and backpropagation its composition law. How layers compose governs
both expressiveness and trainability.

**Reinforcement learning.** Hedges and Sakamoto formalized RL within
categorical cybernetics, expressing the Bellman equation as a natural
transformation. Compositional structure of the agent–environment loop
determines convergence.

**Compositional RL.** Bakirtzis et al. formalized MDPs categorically,
deriving safety guarantees from composition structure — the closest
analogue to our approach.

**Evolutionary computation: the remaining gap.** One major paradigm
remains outside this landscape. Despite five decades of study, GA
formalization has remained set-theoretic or statistical. The compositional
structure of a GA pipeline — selection, crossover, mutation composed in
sequence — has been treated as an implementation detail rather than a
mathematical object. We close this gap.

Part of the barrier is linguistic. The Rosetta Stone table in the paper
translates between practitioner and category-theory vocabulary: what
practitioners call "reliability failure" is non-associative composition;
"compound error" is laxator norm; "modular system" is strict functor. The
concepts are the same. The formalisms diverge. Our framework is the bridge.

## The Framework

We model GA operators as Kleisli morphisms over a population monad
T = Reader × Writer × State, capturing configuration, logging, and
randomness. Operators compose via Kleisli composition; pipelines are
their composites. This is not a relabeling exercise — it makes a
categorical structure visible that was always there, and that structure
makes precise, testable predictions.

The island model, which lifts a GA strategy into a collection of parallel
sub-populations with periodic migration, becomes a functor
I_G : **Strat** → **Island_G** parameterized by migration graph G. The
critical question: is this functor strict or lax?

**Strict** when islands evolve independently: I_G(σ₁ ∘ σ₂) = I_G(σ₁) ∘ I_G(σ₂).
Composition before lifting equals lifting before composition.

**Lax** when islands are coupled by migration: the functor preserves
composition only up to the *laxator*, a natural transformation
φ_G : I_G(σ₁) ∘ I_G(σ₂) → I_G(σ₁ ∘ σ₂) measuring the deviation.

This is not a free parameter. The laxator magnitude **grows with the
algebraic connectivity λ₂(G)** of the migration graph (Remark 1 in the
paper). Since λ₂ governs mixing time, this connects categorical structure
to spectral graph theory: topology determines the strict-to-lax continuum,
which determines diversity dynamics.

*(Throughout we restrict to connected topologies, λ₂ > 0. The
disconnected case reduces to n independent runs outside the island model.)*

## The Predictions

This categorical structure yields two families of testable predictions.

**Prediction 1: Topology ordering is domain-independent.**
If diversity dynamics are governed by categorical structure rather than
fitness-landscape specifics, then the same topology ordering should appear
across unrelated domains. We study five migration topologies: none, ring,
star, random, and fully connected (FC). The categorical framework predicts
the ordering none > ring > star > random > FC — strictly monotone in λ₂.

We verify this across **six unrelated domains**: OneMax (synthetic
bitstring), maze generation, graph coloring, knapsack, checkers (game
strategy), and No Thanks! (co-evolutionary card play). The ordering is
identical in every domain. Kendall's W = 1.0, p = 0.00008. The
diversity fingerprint is a functorial invariant of migration topology,
not a property of the fitness landscape.

**Prediction 2: The n = 7 spectral boundary.**
The spectral ordering is not monotone in island count. At n = 5 islands,
the ring topology has λ₂(C₅) ≈ 1.38 > 1.0 = λ₂(K₁,₄) (star), so ring
preserves more diversity than star. At n = 7, λ₂(C₇) ≈ 0.80 < 1.0 = λ₂(K₁,₆),
reversing the prediction: star should outperform ring at n = 7.

This is a *prospective* prediction — we derived it from the spectral
theorem before running the n = 7 experiment. The experiment confirms it
with p < 0.0001 (p = 6.6 × 10⁻⁵).

## Why This Is Surprising

The domain-independence result is non-obvious. Checkers and OneMax share
no structure. Their fitness landscapes are qualitatively different — one
co-evolutionary, game-theoretic, and high-dimensional; the other synthetic,
smooth, and bit-additive. That the same five-topology ordering holds in
both, to rank-correlation p = 0.00008, argues that the ordering is not an
artifact of the particular optimization problem. It is a property of how
migration topology structures the composition of evolutionary dynamics.

The n = 7 reversal is non-obvious in a different way. Practitioners
choosing between ring and star topologies for a 7-island system would have
no principled basis to prefer one over the other. The spectral theorem
gives one: compute λ₂, compare to 1.0, and the categorical structure
predicts which topology preserves more diversity. The prediction holds.

## Positioning

Five independent convergences now support the core claim — that categorical
structure predicts system-level optimization behavior:

1. Us (category theory / Kleisli morphisms)
2. Li et al. (homological algebra)
3. Sanz (physics / spectral universality)
4. Brewster & Nowak (evolutionary graph theory)
5. Wilson (causal category theory)

Five traditions, five formalisms, the same elephant. Zhang et al.'s
"Monadic Context Engineering" (arXiv:2512.22431) bridges CT and agent
systems via monads, but is sequential with no topology — complementary
rather than competing, and worth citing.

## Conjecture

We do not prove a theorem about strict/lax dichotomy — we propose
Conjecture 1: the diversity fingerprint is functorial. That is, a
domain-change functor preserving composition structure preserves
fingerprint shape. The empirical evidence is strong (six domains,
W = 1.0), but the conjecture generalizes beyond our experiments.
The ACT audience will understand the distinction between a conjecture
with supporting evidence and a proved theorem.

---

*Notes for integration into paper.tex:*
- *This replaces the four-composition/hourglass framing from v1*
- *"Dichotomy theorem" → Conjecture 1 throughout*
- *"2-functor" → "lax (monoidal) functor" throughout*
- *Boundary invariance / ergodic mixing claims from v1: removed*
- *Section outline updated: Intro → Framework → Two Domains → Topology
  Sweep → Spectral Analysis → Discussion*

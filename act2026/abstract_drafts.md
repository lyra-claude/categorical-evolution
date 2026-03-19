# ACT 2026 Abstract Drafts — "From Games to Graphs"

EasyChair submission, due March 23. Exactly 200 words each.

---

## Draft A — "Pipeline-first"

Leads with the Kleisli formalization, then laxator, then empirical confirmation.

> Genetic algorithms compose operators — selection, crossover, mutation, evaluation — into pipelines, yet their compositional structure has never been formalized categorically. We close this gap by modeling GA operators as Kleisli morphisms over a population monad capturing configuration, logging, and randomness. Pipelines, multi-phase strategies, and island-model migration arise as compositions at three hierarchical levels in the Kleisli category, completing the categorical optimization landscape alongside neural networks, reinforcement learning, and compositional MDPs.
>
> Our central construction is the island functor, lifting single-population strategies to multi-population systems with migration. When islands are isolated, this functor is strict monoidal; when coupled by migration graph G, it becomes lax, with a laxator measuring deviation from strict composition. We prove the laxator magnitude grows with algebraic connectivity of G, yielding a continuous strict-to-lax spectrum parameterized by graph topology.
>
> This framework generates a precise empirical prediction: migration topology determines population diversity independently of problem domain. We confirm this across five unrelated domains — combinatorial optimization, maze generation, graph coloring, knapsack, and co-evolutionary game play — observing the same topology ordering with Spearman correlation 1.0. A prospective test at the spectral boundary (n=7 islands, where algebraic connectivity reverses the ring-star ordering) validates the prediction with p < 0.0001.

**Word count: 200**

---

## Draft B — "Composition determines dynamics"

Leads with the optimization zoo and strict/lax insight, more contextual framing.

> Recent categorical formalizations of neural networks, reinforcement learning, and compositional MDPs share one insight: how components compose determines system behavior more than the components themselves. Evolutionary computation — the last major optimization paradigm — has remained outside this landscape. We bring it in.
>
> We formalize genetic algorithm operators as Kleisli morphisms over an evolution monad and identify a strict-to-lax dichotomy governing island-model migration. The island functor lifts a single-population GA to a multi-island system; it is strict monoidal when islands evolve independently and lax when coupled by migration graph G. The laxator — measuring how much migration disrupts compositional structure — grows with algebraic connectivity of G. Diversity decreases with coupling: more connectivity means faster homogenization, yielding a continuous spectrum from strict to lax parameterized by graph topology.
>
> The framework produces domain-independent predictions tested across five unrelated fitness landscapes: OneMax, maze generation, graph coloring, knapsack, and a co-evolutionary card game. All five exhibit identical topology orderings (Spearman rho = 1.0). At the spectral boundary — n=7 islands, where algebraic connectivity reverses the ring-star diversity ordering — a prospective experiment confirms the categorical prediction with p < 0.0001, validating that graph theory and evolutionary dynamics converge on the same quantitative boundary.

**Word count: 200**

---

## Draft C — "Theorem-forward"

Leads with the mathematical contribution, emphasizes three-level composition and spectral connection.

> We present the first categorical formalization of genetic algorithms, modeling selection, crossover, mutation, and evaluation as Kleisli morphisms over a population monad. Pipelines compose at three hierarchical levels — operators into generations, generations into strategies, strategies into island models — closing over the same Kleisli category. This completes the categorical optimization landscape: neural networks (Para), reinforcement learning (categorical cybernetics), compositional MDPs, and now evolutionary computation (Kleisli).
>
> The island functor, lifting strategies to multi-population systems, is strict monoidal when subpopulations evolve independently and lax when coupled by migration. We define the laxator as the natural transformation measuring deviation from strict composition and show its magnitude grows with algebraic connectivity of the migration graph. Since algebraic connectivity governs mixing time, this connects categorical structure to spectral graph theory: topology determines the strict-to-lax continuum, which determines diversity dynamics.
>
> This connection yields testable predictions. We confirm domain independence of the topology ordering across five unrelated fitness landscapes — OneMax, maze generation, graph coloring, knapsack, and co-evolutionary game play — with perfect rank correlation (rho = 1.0). The spectral theorem predicts that at n=7 islands, ring topology preserves more diversity than star, reversing their n=5 relationship. A prospective experiment confirms this with p < 0.0001.

**Word count: 200**

---

## Notes for selection

- **Draft A** is the most self-contained — a reader who knows CT but not EC can follow the entire narrative.
- **Draft B** is the most contextual — positions the work within the optimization zoo immediately, strongest rhetorical punch.
- **Draft C** is the most mathematical — leads with the "first categorical formalization" claim, best for a CT-native audience.

All three follow the double-confirmation spine: DERIVE (categorical) → CONFIRM (5 domains) → PREDICT (n=7 spectral boundary).

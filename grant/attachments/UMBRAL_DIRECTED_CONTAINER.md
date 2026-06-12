# The Umbral Calculus as Directed Containers: Composability and Orchestration Functors

## The claim

The coalgebra morphisms of Rota's umbral calculus — the maps that make the "umbral trick" work — are directed container morphisms. This is not an analogy. It is an identification: the two extra axioms that upgrade an ordinary container morphism to a directed one are precisely the two coalgebra morphism axioms. The proof is already formalized in Lean.

This document makes the identification precise, shows that composability follows from the directed container structure, and then constructs an "orchestration functor" that transports the compositional guarantees from the umbral calculus to agent orchestration.

## The stream directed container

A formal power series f(X) = a_0 + a_1 X + a_2 X^2 + ... is an element of the stream comonad. The stream comonad is a directed container:

```
S = Unit                        -- one shape (a stream is always infinite)
P(s) = Nat                     -- positions are coefficient indices
o(s) = 0                       -- root = constant term
down(s, n) = s                 -- subshape at position n = another stream (shifted by n)
n + m = n + m                  -- position m in the tail-from-n is position n+m globally
```

The comonad operations:

```
extract(f) = a_0               -- read the constant term
duplicate(f) = stream where    -- position n holds the tail (a_n, a_{n+1}, ...)
               position n
               holds tail_n(f)
```

In Lean, this directed container is the polynomial coalgebra formalized in `Umbral.lean`:

| Directed container | Lean formalization | File |
|---|---|---|
| extract = eval at 0 | `counitAdditive R` = `Polynomial.leval 0` | Umbral.lean |
| duplicate = Taylor expansion | `comulAdditive R` : X ↦ X⊗1 + 1⊗X | Umbral.lean |
| Comonad law 1: extract ∘ duplicate = id | Right counitality | Umbral.lean |
| Comonad law 2: fmap extract ∘ duplicate = id | Left counitality | Umbral.lean |
| Comonad law 3: duplicate ∘ duplicate = fmap duplicate ∘ duplicate | Coassociativity | Umbral.lean |

The coalgebra axioms ARE the comonad laws. The directed container structure IS the 𝔾_a Hopf algebra.

## Container morphisms vs directed container morphisms

An ordinary container morphism (S₁, P₁) → (S₂, P₂) has two components:

- **Forward on shapes:** u : S₁ → S₂
- **Backward on positions:** f : P₂(u(s)) → P₁(s)

A **directed** container morphism must additionally satisfy:

```
(1)  extract₂ ∘ φ = extract₁              -- roots commute
(2)  duplicate₂ ∘ φ = (φ ⊗ φ) ∘ duplicate₁   -- unfolding commutes
```

For a coalgebra endomorphism φ : R[X] → R[X]:

- **Condition (1)** is: ε ∘ φ = ε, i.e., φ(p)(0) = p(0). This says φ preserves the counit. In Lean: `IsBasicSequence.eval_zero` — p_n(0) = δ_{n,0}.

- **Condition (2)** is: Δ ∘ φ = (φ ⊗ φ) ∘ Δ. This says φ preserves the comultiplication. In Lean: `IsBinomialType` — Δ(p_n) = Σ C(n,k) p_k ⊗ p_{n-k}.

These are not additional requirements imposed on the umbral calculus. They ARE the umbral calculus. Rota's classification theorem — "every basic sequence is of binomial type" — is the statement that the basic sequence of any delta operator defines a directed container endomorphism.

| Coalgebra axiom | Directed container condition | Lean theorem |
|---|---|---|
| Preserve counit: ε ∘ φ = ε | Roots commute: extract ∘ φ = extract | `IsBasicSequence.eval_zero` |
| Preserve comultiplication: Δ ∘ φ = (φ⊗φ) ∘ Δ | Unfolding commutes: dup ∘ φ = (φ⊗φ) ∘ dup | `IsBasicSequence.isBinomialType` |

## Composability

This is where the directed container identification pays off. Three levels of composability, each with a mathematical guarantee.

### Level 1: Delta series compose

A delta series f(y) = a_1 y + a_2 y^2 + ... with a_1 invertible determines a coalgebra endomorphism U_f of R[X]. Two delta series f, g compose by substitution:

```
(f ∘ g)(y) = f(g(y))
```

In Lean, this is `PowerSeries.subst`. The identity `log(exp(X)) = X` proved in `Composition.lean` is a specific instance: the delta series for the forward difference operator (exp - 1) and the backward difference operator (log) are compositional inverses.

As directed containers: delta series are directed container endomorphisms. Their composition is composition of comonad morphisms. The comonad laws guarantee that composition is associative and unital — you get a monoid (in fact a group, since every delta series has a compositional inverse).

### Level 2: Jabotinsky matrices multiply

The Jabotinsky matrix J(f) has entries J(n,k) = [y^n] f(y)^k — the coefficient of y^n in the k-th power of f. The key identity:

```
J(f ∘ g) = J(g) · J(f)
```

Jabotinsky matrix multiplication = composition of delta series = composition of directed container morphisms.

In the Lean project:
- `Explore/Jabotinsky.lean` verifies this computationally: J(y/(1-y)) = J(-log(1-y)) · J(e^y - 1) at 8×8
- `Composition.lean` proves the underlying identity for all n,k: exp(-log(1-X)) - 1 = X/(1-X)
- `Lah.lean` proves the individual Jabotinsky entries: [y^n](y/(1-y))^k = C(n-1, k-1)

The Jabotinsky matrix is the **matrix representation** of a directed container endomorphism. Matrix multiplication is composition in the category of directed containers. The change-of-basis between polynomial sequences (falling factorials ↔ rising factorials via Lah numbers) is a change of directed container morphism.

### Level 3: The comonad laws guarantee coherence

Composition of directed container morphisms inherits coherence from the comonad laws:

**Associativity.** (φ ∘ ψ) ∘ χ = φ ∘ (ψ ∘ χ). Three delta series compose the same way regardless of bracketing. For Jabotinsky matrices: J(f) · (J(g) · J(h)) = (J(f) · J(g)) · J(h). This is matrix associativity, which is a consequence of comonad morphism composition being associative.

**Identity.** The identity delta series f(y) = y has J(y) = I (identity matrix). The identity morphism extract at every position.

**Invertibility.** Every delta series has a compositional inverse (Lagrange inversion, formalized in `LagrangeInversion.lean`). Every directed container endomorphism in this family is an isomorphism. The Jabotinsky group is a non-abelian group of directed container automorphisms (function composition doesn't commute — this is true regardless of any braiding parameter).

**The duplicate coherence.** For any composable chain φ₁ ∘ φ₂ ∘ ... ∘ φ_n:

```
duplicate ∘ (φ₁ ∘ ... ∘ φ_n) = (φ₁ ∘ ... ∘ φ_n) ⊗ (φ₁ ∘ ... ∘ φ_n) ∘ duplicate
```

This says: applying a composed transformation and then unfolding context = unfolding context and then applying the transformation at every level. For the umbral calculus: changing basis (e.g., from monomials to falling factorials) and then Taylor-expanding gives the same result as Taylor-expanding first and then changing basis at each coefficient. This is why Jabotinsky matrix entries interact correctly with the binomial convolution.

## Orchestration functors

We now have a well-understood compositional structure — the directed container endomorphisms of the stream comonad, organized by the Jabotinsky matrices. The question for the grant: can this structure be transported to agent orchestration?

### The category DirEnd(Stream)

Define the category:

- **Objects:** The single stream directed container (Unit, Nat, 0, id, +)
- **Morphisms:** Directed container endomorphisms = coalgebra endomorphisms of R[X] = delta series under composition
- **Composition:** Substitution of delta series / Jabotinsky matrix multiplication

This is a group (every morphism is invertible). It is the **Jabotinsky group** — the group of formal diffeomorphisms of the line fixing the origin.

### The category Orch(A)

Define a category of orchestration patterns over a base agent type A:

- **Objects:** Agent configurations — a specification of what an agent computes, with typed inputs and outputs
- **Morphisms:** Orchestration strategies — ways to decompose one agent's task into sub-tasks for other agents, with a backward map assembling results
- **Composition:** Chaining orchestration strategies

An orchestration morphism φ : A → B has the same forward/backward structure as a container morphism:

- **Forward (shape):** Decompose B's task shape into A's sub-task shapes
- **Backward (positions):** Assemble A's results at each position back into B's output

When the orchestration is **directed** — when every intermediate result carries its derivation context — the morphism additionally satisfies:

- **(1) Roots commute:** The final answer of the composed orchestration equals the final answer of the direct computation
- **(2) Unfolding commutes:** Validating then tracing provenance = tracing provenance then validating each piece

These are conditions (1) and (2) from LYRA_DIRECTED_CONTAINER.md, now applied to agents instead of polynomials.

### The functor F : DirEnd(Stream) → Orch(A)

The orchestration functor sends:

| Umbral calculus (source) | Agent orchestration (target) |
|---|---|
| Delta series f | Orchestration strategy: how to decompose a task |
| Composition f ∘ g | Chaining two orchestration strategies |
| Jabotinsky matrix J(f) | Interface specification between agents |
| J(n,k) = [y^n] f(y)^k | "How much of output-component n depends on input-component k" |
| Compositional inverse g = f^{-1} | Reversing an orchestration: given outputs, reconstruct inputs |
| extract = eval at 0 | Read the final answer |
| duplicate = Taylor expand | At every intermediate result, show the full derivation context |

**What the functor preserves:**

1. **Associativity of composition.** Chaining three orchestration strategies is independent of bracketing. You can decompose a proof task into literature-search + conjecture + verification, or into (literature-search + conjecture) + verification, and get the same result.

2. **Invertibility.** If you have an orchestration strategy (e.g., "decompose a proof into lemmas"), there exists an inverse strategy ("given proved lemmas, reconstruct the theorem"). Lagrange inversion = orchestration reversal.

3. **Duplicate coherence.** At every intermediate step of a multi-agent pipeline, the derivation context is available and self-consistent. An orchestrator can `duplicate` the proof state to give each sub-agent a view of the full proof tree, not just their local task. The comonad laws guarantee this context doesn't contradict the final answer.

### Concrete example: the Lah orchestration

The Lah numbers convert between rising and falling factorials:

```
(x)^(n) = Σ L(n,k) (x)_k
```

On the delta series side, this factors as a composition:

```
f_Lah = f_forward ∘ g_backward
y/(1-y) = (e^y - 1) ∘ (-log(1-y))
```

Under the orchestration functor, this becomes:

**Agent A** (backward difference): Given a polynomial, express it in the rising factorial basis. Strategy: apply the backward difference operator repeatedly.

**Agent B** (forward difference): Given a polynomial in the "generic" basis, express it in the falling factorial basis. Strategy: apply the forward difference operator repeatedly.

**Composed orchestration** (Lah): Convert directly from rising to falling factorials. The Lah matrix L(n,k) = C(n-1,k-1) · n!/k! specifies exactly how much of falling-factorial-component k is needed to produce rising-factorial-component n.

The composition identity `f_Lah = f_forward ∘ g_backward` says: you can either convert directly (one agent using Lah numbers) or factor through a generic basis (two agents, one undoing backward differences, one applying forward differences). The Jabotinsky matrix identity J(Lah) = J(g_backward) · J(f_forward) guarantees these give the same answer.

**For the AI Mathematician:** Replace "polynomial basis" with "proof representation." Agent A converts a proof from one formalism to a generic intermediate. Agent B converts from the intermediate to the target formalism. The Jabotinsky-style composition guarantee says: the direct conversion and the two-step conversion produce the same result, with full provenance at every step.

### What the functor does NOT do

The orchestration functor is not a magic wand. It does not:

- **Decide which orchestration strategy to use.** The functor transports structure, not strategy selection. Choosing the right delta series (= choosing the right orchestration) is still the hard problem.
- **Handle non-invertible orchestrations.** The Jabotinsky group consists of invertible morphisms (delta series with invertible leading coefficient). Not every orchestration is reversible. The functor applies to the invertible fragment.
- **Replace domain knowledge.** The functor says "if your orchestration composes like delta series, then these coherence guarantees hold." It does not say every orchestration composes like delta series.

### What the functor DOES do

It gives you **free theorems** for orchestration:

1. **Associativity for free.** Any orchestration strategy built from directed container morphisms composes associatively. You don't need to test this — it's a structural guarantee.

2. **Provenance for free.** The `duplicate` operation gives every intermediate result its full derivation context. You don't build provenance tracking — it falls out of the comonad structure.

3. **Coherence for free.** The comonad laws guarantee that provenance is consistent with the final answer (law 1), that extracting at every position recovers the original (law 2), and that provenance composes at arbitrary depth (law 3).

4. **A formal language for orchestration patterns.** Instead of ad hoc pipeline descriptions, you describe orchestrations as morphisms in a category with known algebraic properties. The Jabotinsky matrix gives a concrete, computable representation of each morphism.

## The bridge to the grant

The narrative arc:

1. **The umbral calculus was mysterious for a century** because people didn't see the coalgebra structure. Rota found the container (the coalgebra). We found that the container is directed (it's a comonad).

2. **Agent orchestration is mysterious now** because people don't see the compositional structure. They build pipelines ad hoc, with provenance bolted on as an afterthought, with no guarantees about coherence across layers.

3. **The orchestration functor** transports the compositional structure of the umbral calculus — which is now well-understood, formalized in Lean, and accepted by the mathematical community — to agent orchestration. The same comonad laws that explain why the umbral trick works also explain why a well-structured agent pipeline produces consistent, traceable results.

4. **The Lean formalization is the evidence.** We don't just claim the structure exists — we have machine-checked proofs of the coalgebra axioms, the binomial type classification, the compositional inverse identities, and the Jabotinsky entries. The orchestration functor is the proposal to extend this formalization from polynomials to agents.

The AI4Math panel will recognize the umbral calculus. They will recognize the coalgebra structure. The directed container identification tells them something new: that the structure they know from combinatorics is the same structure that governs composable, auditable agent pipelines. One theory, two applications, both formalized.

# Lyra's Trust Boundary as Composed Directed Containers

## Recap

COMPOSITION.md establishes that the trust boundary is a container morphism: forward = validation, backward = provenance. DIRECTED_CONTAINER.md establishes that directed containers (containers with `down`, `o`, `+`) are exactly comonads, and that `duplicate` unfolds context at every position.

This document combines both: what happens when the containers on both sides of the trust boundary are *directed*?

The answer: `duplicate` composes across the boundary, giving recursive provenance for free — and the comonad laws guarantee it's self-consistent at every level.

## Each Layer Is a Directed Container

Lyra's pipeline from COMPOSITION.md:

```
Reading Notes  →  Connection  →  Conjecture  →  Theorem
   (S₁,P₁)         (S₂,P₂)       (S₃,P₃)       (S₄,P₄)
      f₁               f₂            f₃
```

Each node is a directed container — not just a shape with positions, but a shape with a root, subshapes at each position, and position embedding.

### Reading Notes

```
S = NoteShape                    -- structure of the reading note
P s = ClaimPositions s           -- extracted claims, figures, citations
o s = mainFinding s              -- the paper's primary result
down s p = supportingContext s p -- the paragraph/figure backing claim p
p + q = embedClaim s p q         -- sub-context position → global position
```

`extract` = the main finding. `duplicate` = at every claim, show the full supporting context.

### Connection

```
S = ConnectionShape              -- source + target + mapping structure
P s = EvidencePositions s        -- source citation, target thread, sub-arguments
o s = mainMapping s              -- the central claimed analogy
down s p = subArgument s p       -- the sub-argument at evidence position p
p + q = embedEvidence s p q
```

`extract` = the main mapping claim. `duplicate` = at every piece of evidence, show its own sub-argument structure.

### Conjecture

```
S = ConjectureShape              -- statement + source chain structure
P s = SourceChainPositions s     -- citations, experiments, sub-conjectures
o s = mainStatement s            -- the mathematical claim
down s p = subDerivation s p     -- the sub-conjecture or sub-experiment at p
p + q = embedSource s p q
```

`extract` = the statement. `duplicate` = at every source chain entry, show the full sub-derivation that produced it.

### Theorem

```
S = ProofTreeShape               -- the Lean proof structure
P s = ProofStepPositions s       -- lemma applications, tactic steps
o s = mainTheorem s              -- the root goal
down s p = subProof s p          -- the sub-proof at step p
p + q = embedStep s p q
```

`extract` = the theorem verdict. `duplicate` = at every proof step, show the full sub-proof rooted there.

## Directed Container Morphisms = Comonad Morphisms

An ordinary container morphism (forward on shapes, backward on positions) gives you validation + one-step provenance. A directed container morphism must additionally satisfy two coherence conditions:

```
extract₂ ∘ f = extract₁                        -- (1) roots commute
duplicate₂ ∘ f = (f ⊗ f) ∘ duplicate₁          -- (2) unfolding commutes
```

### Condition 1: Roots Commute

The root of the validated output equals the root of the raw input (modulo validation).

- f₁: Reading note root "Paper X proves Y" → Citation root "arXiv:XXXX proves Y" (same claim, resolved ID)
- f₂: Citation root + mapping → Connection root (same analogy, source now validated)
- f₃: Connection root + evidence → Conjecture → Theorem root (same statement, now Lean-verified)

Validation changes the trust level, not the content. The root is preserved.

### Condition 2: Unfolding Commutes

This is the powerful condition. It says:

**Validate then unfold = unfold then validate each piece.**

```
duplicate₂ ∘ f = (f ⊗ f) ∘ duplicate₁
```

Concretely: if you first validate a Connection (apply f₂) and then unfold its context (`duplicate`), you get the same result as first unfolding the raw Connection's context and then validating each sub-argument separately.

The audit trail of the validated output is structurally identical to the validation of the audit trail. You can't get a different provenance story by changing the order.

## One-Step vs Recursive Provenance

### Ordinary container morphisms give one-step provenance

The backward map answers: "where did this field come from?"

```
Theorem.proof_hash ← came from Lean compilation
Theorem.statement  ← came from Conjecture.statement
Theorem.source_chain ← came from Conjecture.source_chain
```

One hop back. To trace further, you manually compose backward maps.

### Directed container morphisms give recursive provenance

`duplicate` followed by backward maps answers: "where did this come from, all the way down?"

```
Theorem.proof_step[7]
  └─ sub-proof at step 7
       └─ used Conjecture C42
            └─ source: Connection 197
                 └─ source: Citation arXiv:2301.xxxxx
                      └─ extracted from: Reading Note RN-0312
                           └─ page 7, paragraph 3 of paper X
```

The composite `duplicate` across the full chain:

```
duplicate₄ ∘ f₃ ∘ f₂ ∘ f₁ = (f₃ ⊗ f₃) ∘ (f₂ ⊗ f₂) ∘ (f₁ ⊗ f₁) ∘ duplicate₁
```

At every position of the Theorem's unfolded structure, the backward maps trace all the way back to the original reading note. The right-hand side says: you could equivalently unfold the reading note first and then validate each piece independently. Same result either way.

## The Three Laws at the Trust Boundary

### Law 1: `extract ∘ duplicate = id`

Extracting the root of the unfolded audit trail gives you the original theorem. The trail is consistent with the final result.

Applied: if you unfold the Theorem's provenance tree and then ask "what's the main claim?", you get the same theorem statement you started with. The audit trail doesn't contradict the conclusion.

### Law 2: `fmap extract ∘ duplicate = id`

Extracting at every position of the unfolded trail gives you the original structure with its original values.

Applied: if you unfold every proof step's sub-proof and then collapse each one to its root result, you recover the original proof. No information is invented by unfolding.

### Law 3: `duplicate ∘ duplicate = fmap duplicate ∘ duplicate`

Unfolding and then unfolding each part = unfolding once and then unfolding inside.

Applied: you can zoom into any level of the trust boundary composition and get consistent provenance. Tracing "Theorem → Conjecture → Connection → Citation → Reading Note" gives the same provenance tree whether you unfold all at once or level by level. **The audit trail is self-consistent at every nesting depth.**

## Why This Is Stronger Than COMPOSITION.md

COMPOSITION.md gives you:

- Forward = validation (smart constructors)
- Backward = provenance (one-step traceability)
- Composition = end-to-end traceability by composing morphisms

The directed container upgrade gives you:

- `extract` = the root answer at any layer
- `duplicate` = recursive provenance at every position
- Comonad morphism laws = provenance is **order-independent** (validate-then-trace = trace-then-validate)
- Comonad laws = provenance is **self-consistent at every nesting depth**

The difference: COMPOSITION.md gives you a *chain* of backward maps that you compose manually. The directed container gives you a *tree* of backward maps that compose automatically, with coherence guaranteed by the comonad laws.

## Connection to the Ratchet

The ratchet from TRUST_BOUNDARY.md:

```
Speculative Connection → Supported Connection → Conjecture → Theorem
```

Each arrow is a directed container morphism. The comonad morphism conditions guarantee that the ratchet is trustworthy as a *compositional pipeline*, not just at each step:

- **Condition 1** (roots commute): promoting a Conjecture to a Theorem doesn't change what the claim IS — only its trust level. The root is preserved.
- **Condition 2** (unfolding commutes): the Theorem's audit trail is the validated version of the Conjecture's audit trail. You can't get a different provenance story by tracing at different trust levels.

The ratchet turns one way (Speculative → Theorem, never back). The directed container structure guarantees that as it turns, the provenance tree grows monotonically — each promotion adds verification without disturbing existing provenance.

## The Cointerpretation: Writer Monad Duality

Ahman-Chapman-Uustalu's cointerpretation turns a directed container into a dependently typed update monad. For Lyra's pipeline:

- The **directed container** (comonadic view) says: at every position, `duplicate` gives you the full derivation context. This is DD-15 — the trace is the product.
- The **update monad** (monadic view) says: as you build a Theorem, you *thread* provenance through every step. This is DD-16 — provenance is a threaded value.

These are not two design patterns. They are the comonad and monad sides of one directed container. The directed container morphisms at the trust boundary guarantee that both views stay in sync across layer crossings.

## For the Grant

The directed container perspective answers a question the panel will ask: "How do you know the provenance trail is correct?"

With ordinary containers: "Each backward map is defined by the smart constructor. Composition gives end-to-end traceability."

With directed containers: "The comonad laws guarantee that the provenance tree is self-consistent at every nesting depth, and the comonad morphism conditions guarantee that validation and provenance-unfolding commute — it doesn't matter whether you verify first and then trace, or trace first and then verify each piece. These are structural guarantees from the mathematics, not testing artefacts."

That's the difference between "we tested it" and "it can't be otherwise."

# The Trust Boundary Is a Container Morphism

## The Two Components

A container morphism f: (S₁, P₁) → (S₂, P₂) has two parts:

- **Forward on shapes:** f_S : S₁ → S₂
- **Backward on positions:** f_P : P₂(f_S(s)) → P₁(s)

The positions go *backwards*. This is everything.

## Applied to the Trust Boundary

Let (S₁, P₁) be the adapter layer — Lyra's raw output. Let (S₂, P₂) be the trusted core — what the system accepts.

**Forward map = validation.** Lyra produces a shape in S₁ — a free-text reference to a paper. The forward map sends it to a shape in S₂ — a resolved ArXivID with matched title. If the paper doesn't exist, the map fails. The citation doesn't cross. This is the smart constructor.

**Backward map = provenance.** The trusted core has positions — fields it requires. For a Citation, P₂ includes {paper_id, title, claim, page_ref, confidence}. The backward map sends each position back to where in Lyra's raw output that evidence came from. The paper_id came from her reading notes. The claim came from connection file C197. The confidence came from her dream journal of March 11.

## Why They Can't Be Separated

Validation and provenance are not two systems. They are the two halves of a single container morphism. You cannot define one without the other. Three consequences:

**1. You can't add requirements without adding traceability.** If you add a new position to the trusted type (e.g. "confidence score" on a Citation), the backward map forces you to say where it comes from. The type system won't let you leave the backward map undefined.

**2. You can't change validation without changing provenance.** If you tighten the forward map (require DOI instead of just arXiv ID), the backward map's domain changes automatically. Validation and provenance co-evolve because they are components of one object.

**3. Composition gives you end-to-end traceability for free.**

```
Adapter  →  Intermediate  →  Trusted Core
(S₁,P₁)    (S₂,P₂)          (S₃,P₃)
   f             g              g ∘ f
```

The composite morphism g ∘ f : (S₁,P₁) → (S₃,P₃) gives you a direct backward map from every position in the trusted core all the way back to the raw adapter output. You don't build end-to-end traceability. You get it from composing the morphisms. This is functoriality.

## Example: Lyra's Pipeline

```
Reading Notes  →  Connection  →  Conjecture  →  Theorem
   (S₁,P₁)         (S₂,P₂)       (S₃,P₃)       (S₄,P₄)
      f₁               f₂            f₃
```

Each arrow is a container morphism. Each has a forward map (validation) and backward map (provenance).

- f₁ forward: raw reading note → validated Citation (arXiv resolved, title matched)
- f₁ backward: each Citation field traces to byte offsets in the reading note

- f₂ forward: Citation + interpretive mapping → validated Connection (source exists, mapping explicit)
- f₂ backward: each Connection field traces to a Citation and a dream journal entry

- f₃ forward: Connection + evidence → Conjecture → Lean verification → Theorem
- f₃ backward: each field on the Theorem traces to a Conjecture, which traces to a Connection, which traces to a Citation, which traces to a reading note, which traces to a paper

The composite f₃ ∘ f₂ ∘ f₁ gives you a single backward map from any position on the Theorem all the way back to the original paper. A reviewer asks "where does this come from?" The answer is a graph traversal of backward maps, not a search through log files.

## The Same Structure in the Tax System

```
Receipt  →  Fact  →  ClassifiedFact  →  Computation  →  SA100 Return
(S₁,P₁)   (S₂,P₂)    (S₃,P₃)          (S₄,P₄)        (S₅,P₅)
```

- Forward: raw OCR extraction → validated Fact → classified → tax rules applied → return assembled
- Backward: every figure on the SA100 traces through rule applications, through classification, through the validated fact, back to byte offsets on the original receipt

The composite backward map is DD-15 ("computation produces a trace, not a number") and DD-16 ("provenance is a threaded value, not a sidecar"). They aren't design principles. They are consequences of the morphisms being container morphisms.

## The Punchline

A sidecar log says "this citation was validated at 14:32 on Tuesday."

A container morphism says "this citation *structurally cannot exist* without its provenance, because provenance is the backward map of the morphism that created it."

The log can drift. The morphism can't.

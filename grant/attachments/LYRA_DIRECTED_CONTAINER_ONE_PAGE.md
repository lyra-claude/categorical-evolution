# Provably Correct Traceability for AI Mathematics (Proposed)

## The Problem

An AI mathematician produces a chain of reasoning: it reads papers, draws connections, forms conjectures, and drafts results for human review. Today, every step in this chain has the same trust level: none. A carefully sourced connection and a speculative analogy sit side by side with no distinction. If a collaborator or reviewer asks "where did this claim come from?", the answer is buried in unstructured logs — or lost entirely.

We propose adding two things to Lyra: a **trust boundary** that separates verified from unverified output, and a **provenance guarantee** that traces every result back to its sources, through every intermediate step.

## The Proposed Architecture

Lyra's output would flow through a pipeline of increasing trust:

```
Reading Notes  →  Connections  →  Conjectures  →  Results
  (raw claims)    (analogies)     (statements)    (human-reviewed)
```

At each arrow, a **smart constructor** validates the output before it crosses:
- Citations must resolve against arXiv/DOI (no hallucinated references)
- Conjectures must carry a source chain (no orphan claims)
- Results are verified either by Lean (machine-checked proof) or by Lyra's human collaborator before submission to a journal for peer review — the same workflow as a graduate student drafting a paper with their supervisor

What stays outside the boundary stays creative and unconstrained — blog drafts, strategic advice, speculative connections. The boundary doesn't suppress creativity. It separates what is trusted from what is exploratory.

## The Mathematical Guarantee

Container theory (Abbott, Altenkirch, Ghani 2003/2005; Ahman, Chapman, Uustalu 2014) provides structural guarantees that go beyond "we tested the pipeline":

**1. End-to-end traceability.** Each smart constructor is a container morphism: it maps outputs forward (validation) and inputs backward (provenance). Container morphisms compose. Chaining the full pipeline gives automatic end-to-end traceability — from a result back through the Conjecture, Connection, and Reading Note that produced it, down to a specific paragraph in a specific paper.

**2. Recursive provenance.** At every intermediate result, the framework unfolds the full derivation context — not just "this came from X" but the entire tree of reasoning rooted at that point. Every connection carries its evidence. Every conjecture carries its source chain. Every citation carries its extraction context.

**3. Order independence.** It does not matter whether you verify first and then trace provenance, or trace first and then verify each piece. The results agree. This is not a design choice — it is a structural consequence of the mathematics (the comonad morphism conditions on directed containers).

**4. Self-consistency at every depth.** You can zoom into any level of the provenance tree and get a consistent picture. Tracing a result back to a Reading Note gives the same provenance whether you unfold all at once or level by level. The audit trail cannot contradict itself at any nesting depth.

These four properties are **mathematical guarantees**, not testing artefacts. They hold by construction for any pipeline built from container morphisms — including future extensions (new output types, new verification tools, new agents in the pipeline).

## What This Means in Practice

A mathematician using Lyra can ask of any output:
- **"Where did this come from?"** → the backward map traces to sources
- **"How confident should I be?"** → the trust level is explicit (speculative / supported / human-reviewed / Lean-verified)
- **"Can I verify this independently?"** → the full derivation tree is available, with every citation resolvable and every experiment reproducible

A reviewer evaluating Lyra's output can audit the provenance chain end-to-end and know that the chain is structurally sound — not because someone tested it, but because the mathematics makes inconsistency impossible.

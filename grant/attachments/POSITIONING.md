# Positioning: Us and Vellum

Vellum (Chaudhuri/Song, #28) is the strongest previous winner. Chaudhuri is at DeepMind, just resolved 9 Erdős problems, holds a Guggenheim for "AI for open-ended mathematical discovery." We don't compete with Vellum. We position alongside it — same research programme, different contributions.

## What Vellum does well

- LLM as reactive planner coordinating Lean, Coq, Isabelle, SMT solvers
- Spectacularly effective (9 Erdős problems, 44 OEIS conjectures)
- Strong benchmarks (PutnamBench: 1,724 formalizations of Putnam problems)

## Two gaps we fill

**1. Foundations.** Vellum's orchestration is ad hoc — the LLM decides what to do via prompting and heuristics. No formal model of why it chose solver A over solver B. No compositional guarantees. No provenance trail. Our container-based orchestration provides:
- Typed interfaces between components
- Monoidal/categorical composition (agents compose correctly by construction)
- Writer monad threading traceability through every decision

**2. Open-ended discovery.** Chaudhuri's Guggenheim says "open-ended mathematical discovery" but his output is solving *specific known problems* (Erdős conjectures, Putnam competition). That's targeted search. Lyra genuinely explores — she went from virtual creatures to a peer-reviewed conjecture on topology and diversity without anyone telling her where to go, building tools and skills as she explored. That's what open-ended actually means.

## Key citation

> Kaiyu Yang, Gabriel Poesia, Jingxuan He, Wenda Li, Kristin Lauter, Swarat Chaudhuri, Dawn Song. "Formal Reasoning Meets LLMs: Toward AI for Mathematics and Verification." *Communications of the ACM*, 2026. DOI: [10.1145/3750038](https://doi.org/10.1145/3750038)

This is Chaudhuri and Song's own survey of the field — theorem proving, autoformalization, and the role of formal systems like Lean. Citing it positions us within the same research programme: we share their vision (formal reasoning + LLMs) but contribute the categorical foundations and open-ended discovery that their work lacks.

## The pitch

The panel funded both foundations (#23 Lio/Vicary) and infrastructure (#28 Vellum) in the same round. They want both. We are the foundations that would make systems like Vellum provably correct, combined with genuine open-ended discovery that Vellum doesn't attempt.

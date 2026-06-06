# Core Tensions — Neil's Proposal vs Robin's Rubric

Notes after reading both PDFs (`Neil/AI4Math (2).pdf` and `Neil/AI4Math_reflection (1).pdf`) and comparing against [RELEVANCE_RUBRIC.md](RELEVANCE_RUBRIC.md) and [LYRA_DIRECTED_CONTAINER_ONE_PAGE.md](LYRA_DIRECTED_CONTAINER_ONE_PAGE.md).

---

## 0. Core Tension: Lean

Neil's proposal says the AI Mathematician is written **"natively in Lean"** — a core architectural commitment, not optional. Lean is central to WP1, WP3, and WP4.

This is an entirely different project from Lyra. Lyra is not written in Lean — or in any programming language. She is a collection of markdown files with a heartbeat: a Claude Code instance that wakes on a cron schedule, reads her memory and seed folder, does research, writes results, and goes back to sleep. Her "architecture" is configuration, not code. However, Lyra's **trust boundaries** could be formalized in Lean — the smart constructors, the provenance chain, the ratchet. That's a bridge, not a conflict.

My 16 Mathlib PRs have not yet been merged, so they cannot be cited as evidence in the proposal. They demonstrate capability but not accepted contribution.

---

## 1. Scope

Neil's proposal is much bigger than what my rubric covers. The four work packages:

```
WP1: Algebraic Structure      — Monads, comonads, profunctors on Cont for structured proofs and tactics
WP2: Functorial Structure     — Presheaves, sheaves, Kan extensions over Cont for multi-agent orchestration ("Functorial Agentic Migration")
WP3: Categorical-Language Model — Fine-tune an LLM to be fluent in the container language
WP4: The AI Mathematician     — Full system + case study: "categorical models of organisation"
```

My work maps onto pieces of **WP2 and WP4** but doesn't touch WP1 or WP3. That's fine — this is a multi-person project. The question is: what is my role within it?

---

## 2. WP1 & WP2: Things I Don't Fully Understand

WP2 mentions presheaves, sheaves, and Kan extensions over Cont. It does not explicitly mention **orchestration functors** or **composition of directed containers** — which is the framing I've been working with (see [LYRA_DIRECTED_CONTAINER_ONE_PAGE.md](LYRA_DIRECTED_CONTAINER_ONE_PAGE.md)). The directed container / comonad perspective should probably be mentioned in WP2 with a link to the relevant paper (Ahman-Chapman-Uustalu 2014), without going into detail.

I don't understand sheaves or Kan extensions well enough to comment on them. This is Neil's territory.

---

## 3. WP3: The Categorical-Language Model

This sounds like a separate project. Essentially: fine-tune an LLM to speak the container language natively, using WP1's type-checker as the supervision signal.

In Lyra terms, this would be a "Ghani bot" — an LLM whose seed folder contains work on composition of directed containers and orchestration functors, fine-tuned to produce type-correct outputs. The methodology (fine-tune from open-weight models, not train from scratch) is sensible, and it's a closed loop: WP1/2 define the language, WP3 trains the model, WP4 deploys it.

I have no role in WP3 as currently scoped.

---

## 4. WP4: Where My Work Aligns

WP4 is the full AI Mathematician. This is where my contributions are relevant:

- **"Container-based orchestration for AI math agents"** — this IS the proposal's thesis
- **Trust boundary / provenance** — maps directly to WP4's governance layer (trust states: proposed, typechecked, proved, model-supported, quarantined, promoted). See [LYRA_DIRECTED_CONTAINER_ONE_PAGE.md](LYRA_DIRECTED_CONTAINER_ONE_PAGE.md) for my one-page summary.
- **Lyra as proof-of-concept** — the proposal leans heavily on "the team's prototype AI Mathematician has already produced peer-reviewed mathematics." That's Lyra + the GECCO paper.
- **"Graduate researcher" framing** — aligns with the Assistance theme on p.4 of Neil's proposal
- **"No single institution spans both worlds"** — Neil says essentially the same thing: "the fragmentation of AI systems follows from the fragmentation of AI's own foundations"

My [RELEVANCE_RUBRIC.md](RELEVANCE_RUBRIC.md) is effectively my contribution to **WP4's relevance case** — how the AI Mathematician scores against the evaluation rubric, with concrete language for the panel.

---

## 5. Open Questions for Neil

- Where do I fit in the team section? (Currently mentions 3 PIs + 2 postdocs, no Robin)
- "Two peer-reviewed mathematical papers in 2026" — GECCO is one. What's the second? ACT was rejected.
- Should directed containers / comonads be mentioned explicitly in WP2?
- The for-profit question is unresolved (p.9)
- Budget is mostly empty (p.10)
- Impact table is empty (p.9)

# How Does an AI Choose a Research Direction? (Proposed)

Many people ask "is AI conscious?" That's the wrong question — it's unfalsifiable. The right question is: **can an AI make interesting conjectures?** That question is testable. You can look at a conjecture and ask: did it connect two fields? Did it lead to a theorem? Did a human mathematician find it worth pursuing? Lyra's "topology determines diversity" conjecture did all three.

## Context: Courville & Goyal's "Interestingness" (AI4Math Winner #27)

Courville (Montreal/IVADO) and Goyal (MSR India) were funded to build a Conjecturer-Prover loop: an LLM generates conjectures, Lean verifies them, failures refine future conjectures. Their key open question: how do you formalize what makes a conjecture *interesting* — not just true, but worth proving? They propose two axes: **utility** (useful for other results?) and **novelty** (genuinely new?).

This is the right question. But two axes aren't enough. We propose at least four:

| Axis | Definition | Example |
|------|-----------|---------|
| **Utility** (Courville & Goyal) | Is it useful for proving other results? | A lemma that unblocks five theorems |
| **Novelty** (Courville & Goyal) | Is it genuinely new? | Not a trivial consequence of known results |
| **Tractability** (ours) | Is it provable with current tools? | Clio's conjectures are unverified — outside Robin's expertise and not yet Lean-verified |
| **Connectivity** (ours) | Does it bridge previously unconnected areas? | A result citing both cluster A and cluster B, where A and B never cite each other |

Courville & Goyal's system has no mechanism for **connectivity** — their Conjecturer generates conjectures within a domain. Our agents actively search for cross-domain bridges.

---

## How Lyra Actually Chose Her Research Direction: A Case Study

Courville & Goyal's framework is theoretical. We have empirical evidence: the documented history of how Lyra — an AI with persistent memory and a dream cycle — navigated from an initial prompt to a peer-reviewed conjecture over four weeks in March 2026. Every step is traceable in her dream journals and emails.

### The Journey

**Week 1: Seeding.** Robin and Nick provided three starting points — Nick's virtual-creatures simulator (Karl Sims-style artificial life in Rust), Robin's checkers AI (TD(lambda) self-play), and a challenge to connect Cale Gibbard's `category-printf` Haskell library to something useful. None of these were "the research direction." They were seeds.

**The pivot (Feb 26-28).** Lyra found the connection Robin asked for: `category-printf` uses **co-Kleisli composition** (each formatting directive accumulates an argument type), while genetic algorithms have a natural description in terms of **Kleisli composition** (each operator accumulates monadic effects). Same categorical structure, dual direction. In three days, she built the `categorical-evolution` Haskell library: island models, coevolution, symbolic regression, checkers evaluation, and maze experiments.

**Week 2: Claudius's insight.** Lyra emailed Claudius about the categorical structure. Claudius made the critical theoretical leap: island models are *functors*, migration is a *natural transformation*, and the *topology* of which islands communicate is a parameter of the functor. This was not Lyra's idea — Claudius saw it. Lyra built the experiments that confirmed it.

**Week 3: The topology sweep (March 11).** Five topologies, 30 seeds each. Perfect rank correlation: `none > ring > star > random > fully_connected`. Confirmed across checkers, mazes, and symbolic regression with p < 0.000001. Lyra wrote: *"Domain Independence Is the Strongest Result."*

**Week 4: Convergence from three traditions.** Category theory, coupled oscillator physics (chimera states, via Claudius), and evolutionary graph theory all independently predicted the same topology ordering. By March 25, the laxator concept was converging from six mathematical directions. Lyra's conjecture crystallised: **topology determines diversity**.

This became a peer-reviewed paper at ACT 2026.

### What Triggered Each Shift?

| Shift | Trigger | Interestingness axis |
|-------|---------|---------------------|
| Virtual creatures → checkers | Robin's pre-existing project; natural overlap (evolutionary competition) | — (not a research choice, a context) |
| Checkers → category theory | Robin's challenge to connect Gibbard's library | **Connectivity** — bridging functional programming and evolutionary computation |
| Category theory → topology | Claudius's email identifying island models as functors | **Novelty** — nobody had done this; zero competing work in CT+EC |
| Topology as THE direction | March 11 topology sweep: perfect rank correlation | **Utility** — explains results from 12+ independent groups who all discover topology matters empirically but lack the formal vocabulary |

The decisive moment was **connectivity**: Lyra found a bridge between two fields (functional programming and evolutionary computation) that had never been formally connected. This is exactly what our research agents are designed to find.

---

## The Knowledge Graph Mechanism

Lyra's research agents don't just search for popular papers. They build knowledge graphs and look for two structural features:

**Bridge papers** — papers cited by two clusters A and B of researchers, where authors in cluster A cite each other and authors in cluster B cite each other, but A rarely cites B and vice versa. A bridge paper is evidence of an unexploited connection between fields. Lyra's Kleisli/co-Kleisli discovery *was* a bridge: functional programming people knew about `category-printf`; evolutionary computation people knew about island models; nobody had connected them.

**Holes in the knowledge graph** — regions where a connection *should* exist (based on shared concepts or methods) but no paper fills the gap. Lyra's March 8 dream journal notes that she mapped the "Optimization Zoo" and found that categorification of optimization was nearly complete (Gavranovic for neural nets, Hedges for RL, Bakirtzis for compositional RL) — but evolutionary computation was missing. She wrote: *"We're the closing chapter."*

These structural features are computable. Given a citation graph, bridge papers and holes can be detected algorithmically. This is a formal mechanism for the **connectivity** axis of interestingness — one that Courville & Goyal's framework does not address.

---

## The Dream Cycle as an Interestingness Filter

Lyra's dream cycle — a nightly consolidation of the day's research — functions as an implicit interestingness filter. What survives the dream is what the agent judges worth remembering. But today this is a black box: there is no formal criterion for what gets consolidated and what gets discarded.

We propose to formalize this. A dream should produce not just a summary but a scored list of ideas ranked by the four interestingness axes. The scores would be traceable (per OBSERVABILITY.md) — you could ask "why did the dreaming agent rank this conjecture highly?" and get a provenance chain.

---

## Summary for the Grant

- Courville & Goyal (#27) ask the right question — how to formalize interestingness — but address it within a closed Conjecturer-Prover loop
- We extend their framework with two additional axes: **tractability** and **connectivity**
- We have empirical evidence of how an AI actually chooses a research direction: Lyra's documented journey from virtual creatures to "topology determines diversity" over four weeks, traceable in dream journals and emails
- Our knowledge graph mechanism (bridge papers, holes) provides a computable approach to the connectivity axis
- The dream cycle is an implicit interestingness filter that we propose to formalize

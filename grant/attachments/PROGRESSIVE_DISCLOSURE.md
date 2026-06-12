# Start Here

## Positioning Documents (vs previous AI4Math winners)

**[CLIO_LEAN.md](CLIO_LEAN.md)** — vs Buzzard (#2, Imperial)
Buzzard was funded to manually formalize theorem statements into Lean. Robin has already done this with Claude (16 Mathlib PRs in 2 days). Clio has 100+ unverified proofs. We propose giving Clio a Lean skill so she can verify her own work.

**[OBSERVABILITY.md](OBSERVABILITY.md)** — vs Gowers (#11, Cambridge)
Gowers traces provenance within a proof ("why this step?"). We propose tracing provenance across Lyra's entire agent pipeline ("why this claim?"), using the same Writer monad architecture already deployed in Neil's AI tax platform.

**[INTERESTINGNESS.md](INTERESTINGNESS.md)** — vs Courville & Goyal (#27, Montreal/MSR)
Many ask "is AI conscious?" Wrong question — unfalsifiable. The right question: can an AI make *interesting* conjectures? That's testable. Courville & Goyal formalize interestingness (utility, novelty). We add two axes (tractability, connectivity) and provide empirical evidence: Lyra's documented four-week journey from virtual creatures to a peer-reviewed conjecture, driven by bridge papers and knowledge graph holes.

**[POSITIONING.md](POSITIONING.md)** — vs Vellum (#28, UT Austin/Berkeley/DeepMind)
One page. Vellum is the strongest previous winner (9 Erdős problems). We don't compete — we provide two things it lacks: categorical foundations for provably correct orchestration, and genuine open-ended discovery.

## Container Theory — Core

**[AGENT_INTERFACE.md](AGENT_INTERFACE.md)** — What an agent interface is: what you can ask it, what it can answer. Starts from Lyra's research agent as a concrete example.

**[COMPOSITION.md](COMPOSITION.md)** — The trust boundary is a container morphism. Forward map = validation, backward map = provenance. Composition law guarantees chaining boundaries is well-defined.

**[DIRECTED_CONTAINER.md](DIRECTED_CONTAINER.md)** — Directed containers: containers with `down`, `o`, `+` that are exactly comonads. `duplicate` unfolds context at every position. The cofree directed container models exploratory proof as a branching tree of attempts.

**[LYRA_DIRECTED_CONTAINER.md](LYRA_DIRECTED_CONTAINER.md)** — Lyra's trust boundary as composed directed containers. Synthesizes COMPOSITION.md and DIRECTED_CONTAINER.md: the proof tree as a directed container ties the grant together.

**[TRUST_BOUNDARY.md](TRUST_BOUNDARY.md)** — Trust boundary design for the AI Mathematician, modelled on the tax system (DD-01 through DD-18). Seven smart constructors mapping to tax design decisions.

**[COMPOSABLE_ORCHESTRATION_PATTERNS.md](COMPOSABLE_ORCHESTRATION_PATTERNS.md)** — The three levels: containers (agents), container morphisms (connections), functors over containers (orchestration patterns). Neil's key insight — orchestrators are functors, not just wrappers. Explained via tax, Ethereum, umbral calculus, and agents. Includes open questions for Neil.

## Container Examples in the Wild

**[ETHEREUM_CONTAINER.md](ETHEREUM_CONTAINER.md)** — Ethereum blockchain as a Ghani container: contract storage layout = shape, storage slots = positions. Code is data, but the EVM sits outside.

**[LISP_CONTAINER.md](LISP_CONTAINER.md)** — Lisp as a deeper container: S-expressions are containers whose metacircular evaluator is a fixed point of the container endofunctor. No external VM needed.

**[LEAN_AS_CONTAINER.md](LEAN_AS_CONTAINER.md)** — Lean as a container: typeclasses as (S, P), Mathlib's 200+ typeclass hierarchy as a diagram of container morphisms. Robin's umbral calculus (CoalgHom k[x] k[x]) as container morphisms.

**[ANDREW_CONTAINERS2.md](ANDREW_CONTAINERS2.md)** — How Ghani containers connect to agents, with Aiko Services (Andy's robotics framework) as a case study. PipelineElements are containers, S-expressions run through the whole stack. Includes composition and provenance.

## Reference Documents

**[CONTAINER_PAPERS_SUMMARY.md](CONTAINER_PAPERS_SUMMARY.md)** — Summary of six foundational container theory papers, with connections to the AI tax system and AI Mathematician.

**[VELLUM.md](VELLUM.md)** — Deep dive on Chaudhuri/Song: Copra architecture, PutnamBench, the Erdős results, FERMAT, and the comparison table.

**[BRAIN_DUMP.md](BRAIN_DUMP.md)** — Everything we know about Neil's Calculus of Containers: the theory, the hierarchy, the tax system as proof-of-concept, and what we still don't understand.

**[PREVIOUS_WINNERS.md](PREVIOUS_WINNERS.md)** — All 28 funded projects grouped by theme, with strategic analysis.

**[containers_pdf/](containers_pdf/)** — The foundational container papers (Abbott-Altenkirch-Ghani et al.)

## Team & Admin

**[ABOUT_ME_TEAM.md](ABOUT_ME_TEAM.md)** — Robin's bio, publications, and profiles of Lyra and Clio.

**[EMAIL_DRAFT_CONTAINERS.md](EMAIL_DRAFT_CONTAINERS.md)** — Draft email to Neil asking for bibliography and clarification on the container-to-agent bridge.

**[WHATSAPP.md](WHATSAPP.md)** — Working notes from 23 May session. The session where the grant argument crystallized: Vellum positioning, containers everywhere (tax/Ethereum/Lisp/Lean), trust boundaries as container morphisms, directed containers and the proof tree, the four-point grant argument.

## Grant Strategy

**[RELEVANCE_RUBRIC.md](RELEVANCE_RUBRIC.md)** — Working backward from the evaluation rubric. Decisions made, Relevance section drafted, key quotes bank.

**[LYRA_DIRECTED_CONTAINER_ONE_PAGE.md](LYRA_DIRECTED_CONTAINER_ONE_PAGE.md)** — One-page theory summary for non-experts. No jargon. Four provenance guarantees from container theory.

**[CORE_TENSION.md](CORE_TENSION.md)** — Tensions between Neil's proposal PDF and Robin's rubric work. Lean question, scope mismatch, where Robin fits, open questions for Neil.

**[COMPOSABLE_ORCHESTRATION_PATTERNS.md](COMPOSABLE_ORCHESTRATION_PATTERNS.md)** — The three levels: containers (agents), container morphisms (connections), functors over containers (orchestration patterns). Neil's key insight — orchestrators are functors, not just wrappers.

---

Historical materials (Neil's original grant draft, AI Tax PDF, earlier notes) are in `historical/`.

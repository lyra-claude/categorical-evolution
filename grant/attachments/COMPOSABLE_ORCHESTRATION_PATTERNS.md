# Composable Orchestration Patterns

Neil flagged: *"One thing I am not sure you got was the use of functors over containers for agentic orchestrators and managers and meta-agents."*

This document explains what that means, using four running examples.

---

## The Three Levels

Container theory gives us three levels of structure. Most of our documents so far cover levels 1 and 2. Neil is pointing at level 3 — which is where the real grant contribution lives.

| Level | What it is | Mathematical object |
|-------|-----------|-------------------|
| 1 | The agent | A container (S, P) |
| 2 | The connection between agents | A container morphism |
| 3 | The orchestration pattern | A functor over containers |

---

## Level 1: Containers — The Individual Agents

A container is a pair (S, P) where S is the set of shapes and P(s) is the set of positions for each shape. A value in the container is a choice of shape together with data at every position.

### Tax

Each tax computation node is a container.
- **Shape S** = the types of tax items it can process (salary, deductions, capital gains, ...)
- **Positions P(s)** = the data fields it needs for each type (for salary: gross amount, PAYG withheld, employer ABN, ...)
- **A value** = a specific tax item with all its fields filled in

### Ethereum

Each smart contract is a container.
- **Shape S** = the ABI — the list of callable functions (transfer, approve, mint, ...)
- **Positions P(s)** = the parameters each function accepts (for transfer: recipient address, amount)
- **A value** = a specific function call with all its arguments filled in (a transaction)

### Umbral Calculus

The HopfAlgebra typeclass is a container.
- **Shape S** = the typeclass signature (multiplication, comultiplication, unit, counit, antipode, plus proof obligations)
- **Positions P(s)** = the implementations and proofs needed to fill each slot
- **A value** = a specific instance (k[x] with Δ(x) = x⊗1 + 1⊗x, or rooted trees with admissible cuts, or the shuffle algebra)

### Agents

Each AI agent is a container.
- **Shape S** = the request types it can handle (prove this theorem, search this literature, formalize this statement, ...)
- **Positions P(s)** = the response data for each request type (for "prove this theorem": the proof, the dependencies used, the confidence level)
- **A value** = a specific task with its completed response

**Summary:** At level 1, everything is an individual thing doing its job. No composition, no interaction. Just: what can you ask it, and what does it give back.

---

## Level 2: Container Morphisms — The Connections

A container morphism f: (S₁, P₁) → (S₂, P₂) has two components:
- A **forward map** on shapes: S₁ → S₂ (transforms the request)
- A **backward map** on positions: for each s ∈ S₁, P₂(f(s)) → P₁(s) (pulls back the response)

The forward map goes the direction you'd expect. The backward map goes *backwards* — this is where provenance lives.

### Tax

The pipeline step connecting one computation to the next is a container morphism.
- **Forward** = validate and transform the output shape (e.g., "gross income" becomes "taxable income" after subtracting deductions)
- **Backward** = provenance trail: "this taxable income figure came from *these* gross income entries minus *those* deduction entries"
- The backward map is why DD-15/DD-16 work: the trace isn't a separate logging system, it's the backward component of the morphism

### Ethereum

A contract calling another contract is a container morphism.
- **Forward** = construct the call (encode function selector + arguments according to the callee's ABI)
- **Backward** = the execution trace: internal calls, state changes, events emitted — all recorded in the transaction receipt
- This is why DeFi "composes" — each contract call produces a receipt that traces back through every sub-call

### Umbral Calculus

A coalgebra morphism CoalgHom(k[x], k[x]) is a container morphism. These ARE the polynomial sequences of binomial type.
- **Forward** = the polynomial sequence transformation (e.g., x^n ↦ (x)_n, the falling factorial)
- **Backward** = the coproduct structure is preserved: Δ(p_n(x)) = Σ p_k(x) ⊗ p_{n-k}(y) still holds
- Rota's insight: the "umbral trick" works because these morphisms compose. Composition of coalgebra morphisms = composition of container morphisms. The binomial theorem for every Sheffer sequence is a single theorem about container morphism composition.

### Agents

The trust boundary between two agents is a container morphism.
- **Forward** = validation: parse the request, check types, verify the agent is authorized to make this request
- **Backward** = provenance: "this result came from *that* sub-agent's output, which was produced using *these* sources"
- Smart constructors (SC-1 through SC-7) implement the forward map. The backward map threads provenance automatically.

**Summary:** At level 2, things connect. Data flows forward, provenance flows backward. Composition works: if you chain two morphisms, you get end-to-end traceability for free (backward maps compose). This is where most of our documents stop. But Neil is saying: there's a level above this.

---

## Level 3: Functors Over Containers — The Orchestration Patterns

This is the level Neil flagged. A functor F: **Cont** → **Cont** takes containers to containers and container morphisms to container morphisms. It must satisfy:

- **F(id) = id** — orchestrating an agent with the identity connection changes nothing
- **F(g ∘ f) = F(g) ∘ F(f)** — orchestrating a composed pipeline = composing the orchestrated pieces

A functor over containers is an **orchestration pattern** — a *recipe* for building composite agents from simpler ones. It's not a specific connection between two agents (that's a morphism). It's a *way of connecting agents* that can be applied to any agents of the right type.

Why does this matter? Because at level 2, you build pipelines by hand — agent A connects to agent B connects to agent C. At level 3, you define *patterns* of composition that apply uniformly. The functor laws guarantee that these patterns are coherent.

### Tax

The tax pipeline DAG is built by an orchestration functor.

**The functor:** Takes individual tax computation containers and composes them according to the structure of the tax law. "First compute gross income, then apply deductions, then compute tax payable" is a sequential composition pattern. "Compute salary income and investment income independently, then merge" is a parallel-then-merge pattern.

**Why it's a functor, not just a pipeline:** When the tax law changes (new deduction type, restructured brackets), you swap out individual containers and the functor re-composes automatically. You don't rewire the pipeline — the orchestration pattern is stable, only the components change. The functor laws guarantee that the new pipeline still has end-to-end traceability, because F preserves morphism composition.

**Concretely:** The tax system doesn't have one pipeline. It has a *pipeline builder* that takes a tax law specification and produces a pipeline. That builder is the functor.

### Ethereum

DeFi protocol factories are orchestration functors.

**The functor:** Uniswap's factory contract takes two token contracts (containers) and produces a liquidity pool contract (new container). The factory doesn't know or care what the tokens are — it works uniformly on any ERC-20. Give it USDC and ETH, you get one pool. Give it DAI and WBTC, you get another. Same pattern, different components.

**Why it's a functor, not just a constructor:** A token upgrade (migration from v1 to v2) is a container morphism. The factory must map this morphism to a corresponding pool upgrade — the pool's behavior changes consistently with the token change. F(morphism) = corresponding morphism. That's functoriality.

**Concretely:** This is why DeFi is called "money legos." The lego-ness isn't just that contracts call each other (level 2). It's that there are *composition patterns* (factory, router, aggregator) that work uniformly across any contracts of the right type. Each pattern is a functor.

### Umbral Calculus

The construction "take a Hopf algebra, produce its umbral calculus" is a functor — but a different *flavor* of functor from the tax and agent examples. It's worth being precise about this.

**The functor:** H ↦ End_coalg(H). Take any Hopf algebra H (a container: the typeclass shape, filled with a specific instance). Produce the set of its coalgebra endomorphisms (a new container: the space of polynomial-sequence-like transformations of H).

- Apply it to k[x] with the binomial coproduct → you get the classical umbral calculus (Sheffer sequences, Appell sequences, binomial-type sequences)
- Apply it to the Connes-Kreimer Hopf algebra of rooted trees → you get an "umbral calculus of renormalization"
- Apply it to the shuffle Hopf algebra → you get an "umbral calculus of free Lie algebras"

**Why it's a functor:** A Hopf algebra morphism f: H → K induces a map End_coalg(H) → End_coalg(K). The umbral calculus transforms *consistently* with the Hopf algebra change. Robin's plan to "generalize the umbral calculus to more and more Hopf algebras" is literally: explore the image of this functor on new objects.

**How this differs from the other examples:** In tax and agents, the level 3 functor is a *composition pattern* — it takes multiple components and assembles them into a pipeline. H ↦ End_coalg(H) doesn't compose multiple Hopf algebras into a pipeline. It takes *one* Hopf algebra and produces *the space of all its transformations*. It's a **generalization functor** ("given any algebraic structure, here's its associated calculus"), not a **composition functor** ("here's how to wire these components together").

The umbral calculus does have composition — within End_coalg(H), individual coalgebra endomorphisms compose (computing falling factorials via Bernoulli polynomials means composing two morphisms, and the choice of which intermediate sequences to route through is an orchestration decision). But that's level 2 morphism composition, not level 3 functorial orchestration.

**The honest summary:** The umbral calculus gives the best level 1 and level 2 examples (typeclasses as containers, CoalgHom as container morphisms). At level 3, tax and agents carry the orchestration story. The umbral calculus contributes something different: **functorial generalization** — the same construction applied uniformly across domains. Both are functors. They do different jobs.

### Agents

The orchestrator — a manager agent that composes sub-agents — is a functor.

**The functor:** Takes sub-agent containers and produces a new composite agent container. A "research manager" functor takes a literature-search agent and a proof-search agent and produces a "research" agent. The research agent's shapes are "research questions." Its positions are filled by coordinating the two sub-agents.

**Why it's a functor, not just a wrapper:**
- If you upgrade the literature-search agent (a container morphism: same shapes, better backward maps — richer provenance), the functor must produce a corresponding upgrade of the research agent. The orchestration pattern adapts automatically to improved components. F(upgrade) = corresponding upgrade.
- If you compose the literature-search agent with a citation-checker first (a morphism), and then orchestrate — you must get the same result as orchestrating first and then applying the citation-checker upgrade. F(g ∘ f) = F(g) ∘ F(f). The order in which you improve and orchestrate doesn't matter.

**Concretely:** Lyra already does this informally — she has research, code, dream, and email modes that coordinate. The grant proposes making this coordination a *functor* with categorical guarantees. When you add a new mode (e.g., "Lean formalization"), the orchestration pattern absorbs it without rewiring. The functor laws guarantee that provenance still threads through the new configuration correctly.

---

## The Unifying Observation: Composition Determines Diversity

The tax example reveals something important. Re-read:

> *The tax system doesn't have one pipeline. It has a pipeline builder that takes a tax law specification and produces a pipeline. That builder is the functor.*

Now compare with Lyra's GECCO 2026 paper, "Composition Determines Diversity":

> *An evolutionary algorithm doesn't have one search behavior. It has a composition pattern that takes genetic operators and produces a search process. That composition pattern is the functor.*

These are the same thesis. The functor — the orchestration pattern, the topology of composition — is the thing that determines the character of the output. Not the individual components. Change the components, keep the functor → different data, same structural guarantees. Change the functor, keep the components → fundamentally different behavior.

This connects two apparently separate research threads:
- **The grant** (container-based agent orchestration) proposes that orchestrators are functors over containers
- **The GECCO paper** (topology determines diversity in GAs) provides empirical evidence that the composition pattern determines system behavior

They're the same research programme. The grant formalizes *why* "composition determines diversity" works, using the Calculus of Containers. The GECCO paper is the empirical evidence that the formalization is capturing something real.

**For the panel:** "We have empirical evidence that the composition pattern determines system behavior (GECCO 2026, DOI: 10.1145/3795101.3814659). The grant formalizes *why*, using functors over containers — and extends the principle from genetic algorithms to agent orchestration."

---

## Why Level 3 Is the Grant Contribution

Level 1 (agents as containers) is conceptually simple — it's just typed interfaces.

Level 2 (trust boundaries as container morphisms) is valuable — it gives you composable provenance. But you still build each pipeline by hand.

Level 3 (orchestration as functors over containers) is where the real power lives:

1. **Orchestrators compose.** If F and G are orchestration functors, so is G ∘ F. A meta-agent (an orchestrator of orchestrators) is just functor composition. No new theory needed — it falls out of the mathematics.

2. **Orchestrators preserve guarantees.** The functor laws mean that if your sub-agents have provenance (backward maps), the orchestrated composite has provenance. If your sub-agents compose correctly, the orchestrated composite composes correctly. Guarantees propagate upward through the hierarchy for free.

3. **Orchestrators are swappable.** Different orchestration patterns (sequential, parallel, hierarchical, speculative) are different functors on the same category. You can swap one for another without changing the sub-agents. This is how you experiment with orchestration strategies scientifically — change the functor, measure the results, the components are controlled.

4. **This is what Vellum lacks.** Vellum's LLM planner decides orchestration strategy via prompting. There's no guarantee that a different decomposition of the same problem yields a consistent result. There's no formal relationship between upgrading a sub-solver and upgrading the orchestrated system. Functoriality is exactly the missing piece.

---

## Open Questions (for Neil)

- **Which specific functors?** The category **Cont** has many endofunctors. Which ones correspond to the orchestration patterns Neil has in mind? Sequential and parallel composition are monoidal structure. What about hierarchical delegation — is that substitution of polynomial functors?

- **Natural transformations between orchestrators.** If two orchestration patterns (functors) achieve the same goal differently, a natural transformation between them would be a principled way to migrate from one strategy to another. Is this how Neil thinks about meta-agents — as natural transformations?

- **Monads and comonads on Cont.** A monad on **Cont** would be an orchestration pattern that you can nest (μ: F∘F → F) and embed into (η: Id → F). Does the cofree directed container construction (from DIRECTED_CONTAINER.md) give a comonad on **Cont**? That would connect the proof-tree exploration model to the orchestration story.

- **The Calculus of Containers "hierarchy."** Neil's grant draft mentions a hierarchy. Is this a hierarchy of functors? Functors over functors? Or something else?

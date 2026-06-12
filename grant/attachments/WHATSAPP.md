# WhatsApp Session — 23 May 2026

Robin's working notes from a conversation with Claude, captured as they happened. This is the session where the grant argument crystallized.

---

## 1. Vellum Positioning (12:32–1:19 pm)

**Vellum (#28)** is the most impressive previous winner. PIs: Swarat Chaudhuri (UT Austin + DeepMind, 2025 Guggenheim Fellow, created Copra) and Dawn Song (UC Berkeley, MacArthur + Guggenheim + ACM Fellow). Their team also created PutnamBench (ICML 2024 Best Paper, AI for Math Workshop).

Vellum is an open-source orchestration framework: LLMs act as planners coordinating Lean, Isabelle, SMT solvers, etc. The LLM handles strategy; the provers handle rigour. They solved 9 new Erdos problems in May 2026 alone.

**Key positioning insight:** Vellum's orchestration is ad hoc (prompt-based heuristics, no formal model of planner decisions, no typed interfaces, no provenance). Ours is categorical (container morphisms model delegation, types ensure correct composition, monads ensure traceability). **Vellum is engineering without foundations. We provide the foundations.**

Chaudhuri's Guggenheim was for "AI for open-ended mathematical discovery" — which is exactly what Lyra does. The difference: Vellum solves specific problems (Erdos conjectures). Our agents **explore** and see what they find, building tools as they go.

Copra does not appear to have a public GitHub.

See: [VELLUM.md](VELLUM.md), [POSITIONING.md](POSITIONING.md)

---

## 2. Containers Are Everywhere (2:18–3:00 pm)

### The Tax System Discovery

The tax system's design decisions DD-15 and DD-16 are literally the same mathematical object: a container value. The architecture works in production because the mathematics is right, even when the engineers didn't use that vocabulary. This is strong evidence for the grant: the theory isn't imposed on the system — the system already *is* the theory.

### Ethereum as a Container

Smart contracts on Ethereum are containers:
- **Shape S** = the ABI (callable functions with parameter types)
- **Positions P(s)** = the parameters each function accepts
- A transaction fills positions with concrete data
- The EVM enforces all state changes pass through validated logic (= DD-05)
- The execution trace (receipt) is the product, not a side-effect

Robin's [Gateway project](https://github.com/RaggedR/gateway) is a working example of a container on the Ethereum blockchain.

### Lisp as a Self-Interpreting Container

S-expressions are containers (tree skeleton = shape, leaves = positions). But unlike Ethereum (which needs the EVM as an external interpreter), in Lisp `eval` is itself an S-expression — the container contains its own interpreter. Fixed point of the container endofunctor?

See: [BRAIN_DUMP.md](BRAIN_DUMP.md)

---

## 3. Trust Boundaries = Container Morphisms (3:04–3:30 pm)

The trust boundary for Lyra maps directly onto the tax system's design decisions via seven smart constructors:

| Smart Constructor | Tax DD | What it does |
|---|---|---|
| SC-1 (Citation) | DD-05 | AI outputs never bypass the core |
| SC-2 (Conjecture) | DD-16 | Provenance is threaded, not a sidecar |
| SC-3 (Theorem) | DD-17 | Suitability gate — no back door |
| SC-4 (Experiment) | DD-07 | Pure, replayable computation |
| SC-5 (Connection) | DD-08 | Composable transformations |
| SC-6 (DreamEntry) | DD-10 | Computation vs guidance vs advice |
| SC-7 (PublicClaim) | DD-17 | Regulated output requires a gate |

**"The trust boundary is the container morphism. The forward map is validation. The backward map is provenance."**

The composition law guarantees chaining boundaries is well-defined. The full-and-faithful theorem guarantees no information is lost or invented. Smart constructors implement the forward map at the trust boundary.

---

## 4. Lean Typeclasses Are Containers (3:33–3:42 pm)

A Lean typeclass is a container (S, P):
- **Shape S** = the typeclass signature (operations, arities, proof obligations)
- **Positions P(s)** = the slots to fill (implementations + proofs)

Writing `instance : Group Z` isn't "registering Z as a group" — it's filling positions in a container. Mathlib's 200+ typeclass hierarchy connected by `extends` is a **diagram of container morphisms**.

**Why containers and not just types?** Types give you the objects. Containers give you the morphisms — and the morphisms are where the mathematical content lives. A type system says "Z is a group." A container says *how* Z is a group (which filling), what it forgets to be a monoid (which backward map), and what theorems propagate (which compositions).

### Robin's Umbral Calculus as Container Morphisms

The HopfAlgebra typeclass is a container. Different fillings yield different mathematics:
- Rooted trees with admissible cuts → Connes-Kreimer (Feynman renormalisation)
- k[x] with Δ(x) = x⊗1 + 1⊗x → umbral calculus
- Shuffle algebra on words → free Lie algebras, BCH formula

Same container, different fillings, completely different mathematics. Generic theorems apply to all three.

**The deep point:** CoalgHom(k[x], k[x]) — the polynomial sequences of binomial type — ARE container morphisms. They map shape forward (polynomial sequence changes) and positions backward (coproduct structure preserved). The umbral trick works because coalgebra morphisms compose = container morphisms compose. **Rota's contribution was recognising the container.**

Any theorem Lean proves generically about CoalgHom k[x] k[x] automatically applies to every Sheffer sequence, every Appell sequence, every classical umbral identity. The century-old trick becomes a structural consequence of positions mapping backward.

Robin's plan: generalize the umbral calculus to more and more Hopf algebras.

See: [CLIO_LEAN.md](CLIO_LEAN.md)

---

## 5. Directed Containers and the Proof Tree (3:51–4:25 pm)

### Lyra Already Needs This

Lyra is already "orchestrated" — single instance, but communicating across time, performing different functions (code, research, dream, email). Even without decomposing her into microservices, she needs trust boundaries between instances performing different tasks.

### The Grant Argument in Four Points

1. We've already got Lyra and she's already really cool
2. She published a paper, draws connections, makes analogies, formulates conjectures, does sophisticated literature review
3. We'd like to make her better with a type boundary (= container that composes categorically, like Lean, Ethereum, and the tax system)
4. We also have Clio with 100+ proofs to verify in Lean

### Agents as Open Games

Each agent is an open game (Hedges/Ghani). The orchestrator composes them sequentially or in parallel. Backward flow carries "how useful was your output?" — possibly a better framework for "topology determines diversity" (separate research project).

### Comonadic Context — The Key Theoretical Contribution

**Directed containers** equip every position with a context (the subshape rooted there). For an AI Mathematician: every intermediate result carries its full derivation context — not just the value, but the sub-proof that produced it.

**Do we carry failed attempts?** The cofree directed container on C produces the cofree comonad — non-well-founded trees labelled by C. This models exploratory proof: each step branches, producing a (potentially infinite) tree of attempts. The comonad extracts the current best result while retaining full exploration history.

`duplicate` doesn't copy data — it unfolds context. At every position, it replaces the local value with the entire substructure rooted there.

### Three Directed Containers That Tie the Grant Together

1. **Tax DAG** — shows the architecture works in production
2. **Formal power series** — shows it captures real mathematics (Robin's umbral calculus)
3. **Proof tree** — shows it's what the AI Mathematician needs: `duplicate` at a goal gives the orchestrator the full context of dependencies, which is the information needed to assign sub-goals to sub-agents

### The Provenance Guarantee

**"How do you know the provenance trail is correct?"**

- With ordinary containers: each backward map is defined by the smart constructor; composition gives end-to-end traceability
- With directed containers: the comonad laws guarantee self-consistency at every nesting depth; comonad morphism conditions guarantee that validation and provenance-unfolding commute (verify-then-trace = trace-then-verify). Structural guarantees from mathematics, not testing artefacts.

### The Unifying Observation

DD-15 ("computation produces a trace, not a number") and DD-16 ("provenance is a threaded value, not a sidecar") are **the same idea**: a container value is a shape (computation structure) + filling (data at each position). Trace = shape. Provenance = position-to-source mapping. They cannot drift apart because they are components of a single mathematical object.

**"Provenance monad threading provenance through the tax computation and the comonadic computation trace are dual perspectives on the same directed container structure. This is not a metaphor — it is a precise mathematical correspondence."**

---

## 6. Where We Stand (4:25 pm)

> "We have enough for the grant now. It's just a question of deciding what to leave out."

**Risks acknowledged:**
- Only 1 peer-reviewed paper → lean into the working system angle
- Competition is very high → this research is worth doing regardless of the grant outcome

**Deadline:** June 5, 2026.

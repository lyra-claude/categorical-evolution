# Container Theory Papers: Summary for AI Tax and AI Mathematician

This document summarises six papers on container theory and related ideas, with
connections to two projects:

- **AI Tax**: The concentric-layer tax platform described in `starting-simple-ai-tax.pdf`
- **AI Mathematician**: Agentic orchestration for mathematical research (task decomposition,
  multi-agent coordination, traceability of reasoning)

The summaries assume familiarity with Haskell (GADTs, monads, `Writer`) but not
deep category theory. I define categorical terms where they first appear.

---

## 1. Abbott, Altenkirch, Ghani -- "Categories of Containers" (2003)

### Key technical idea

A **container** is a pair `(S, P)` where `S` is a type of *shapes* and `P : S -> Type` is a
family of *positions*. The associated functor maps any type `X` to
`Sigma s:S. (P s -> X)` -- choose a shape, then fill every position with data. In
Haskell pseudocode:

```haskell
data Container s p x = MkC s (p -> x)   -- p depends on s via a type family
```

Lists are the container `(Nat, Fin)` -- shape is the length, positions are indices.
Binary trees are `(TreeShape, LeafPositions)`. The paper proves that the functor
`T : Cont -> [Set, Set]` sending containers to their associated functors is **full and
faithful**: every polymorphic function between container functors is *uniquely*
determined by a container morphism, which is a pair `(u, f)` where `u` maps shapes
forward and `f` maps positions backward. Containers are closed under products,
coproducts, composition, and fixed points, so all strictly positive types
(the recursive Haskell types you actually write) are containers.

### Connection to AI Tax

**Shapes as computation skeletons, positions as data slots.**
The tax engine's `Computation` type (DD-15) -- a DAG of `RuleApplication` nodes --
is a shape. The positions are the slots where input data (`Fact` values) are placed.
The container perspective makes DD-15 and DD-16 look inevitable: when you
separate shape from data, provenance is literally a function from positions to
sources. You don't *thread* provenance as an afterthought; it *is* the position map.

- **DD-05 (AI never bypasses the core):** A container morphism's backward map
  forces every position in the output to trace back to a position in the input.
  There is no way to "invent" data at a position -- it must come from somewhere
  upstream. This is the mathematical version of "AI outputs must be parsed by a
  smart constructor." The type system prevents phantom data.
- **DD-16 (provenance is a threaded value):** In a container, the position map
  *is* the provenance. A `Computation` value pairs a shape (the rule-application
  DAG) with a filling (the actual monetary values). The filling function maps
  each position to the `Fact` that produced it. This is not metadata bolted on;
  it is the definition of what a container element *is*.
- **Concentric layers:** The container morphism between the Interface layer
  and the Adapter layer is a pair `(u, f)`: `u` maps user-facing request shapes to
  internal query shapes (forward), and `f` maps internal responses back to
  user-facing responses (backward). The trust boundary at the Fact Builder
  (DD-04) is exactly the point where the position map must pass through
  a smart constructor.

### Connection to agent orchestration

Container morphisms give a precise model of **task decomposition**. An orchestrator
agent takes a high-level task shape (e.g., "prove Theorem X") and maps it to
a compound shape of sub-tasks (the `u` map). When sub-agents return results,
the backward map `f` assembles them into the positions of the original shape.
The full-and-faithful theorem guarantees that every polymorphic (i.e., data-agnostic)
orchestration strategy is uniquely captured this way. No orchestration structure
is lost when you work in the container language.

### Concrete example

A "prove this conjecture" task has shape `ProofGoal` with positions
`{hypotheses, lemmas_needed, counterexample_search}`. The orchestrator maps
this to sub-agent shapes: `LiteratureSearch`, `SmallCaseComputation`,
`ProofAttempt`. The backward map assembles sub-agent outputs into the
original positions. The composition theorem guarantees that chaining
two such decompositions (e.g., a sub-agent further decomposes `ProofAttempt`)
gives a valid composite container morphism.

---

## 2. Abbott, Altenkirch, Ghani -- "Containers: Constructing Strictly Positive Types" (2005)

**Note:** The PDF in the directory appears to contain the wrong paper (an unrelated
signal processing paper on MMSE equalization). The content described here is
reconstructed from the 2003 paper's Section 7 on strictly positive types, which
covers the same material.

### Key technical idea

Every strictly positive type -- built from constants, type variables, sums, products,
exponentials by constants, and fixed points -- corresponds to a container. The
paper gives an explicit translation table:

| Type expression | Container |
|----------------|-----------|
| Constant K | `(K, const Void)` -- K shapes, no positions |
| Variable X | `((), const ())` -- one shape, one position |
| `U + V` | coproduct of containers |
| `U * V` | product of containers |
| `K => U` | `(K -> S_U, Sigma k. P_U (f k))` |
| `mu X. F(X)` | initial algebra in `Cont` |

This means that for any recursive Haskell datatype you could define, there is
an equivalent container, and you can reason about polymorphic functions on it
using only shape-and-position maps.

### Connection to AI Tax

- **DD-04 (smart constructors for all domain types):** Every domain type in the
  tax system (`Money`, `TaxYear`, `Fact`, `Computation`) is a strictly positive type
  and therefore a container. The smart constructor discipline is the
  container-theoretic principle that values are built by choosing a *valid* shape
  and then filling positions -- the shape constrains what positions exist. A
  `Fact` with no source document has the wrong shape; the type system refuses it.
- **DD-03 (phantom types for jurisdiction and tax year):** Phantom type parameters
  act as shape refinements. `Allowance UK Y2025` is a sub-container of `Allowance`
  where the shape set is restricted. This is the container-theoretic version of
  indexing shapes by jurisdiction.
- **DD-08 (scenarios as composable transformations):** `Scenario y = TaxInput y -> TaxInput y`
  is a container endomorphism: a shape map plus a backward position map.
  Composition of scenarios is composition of container morphisms. The paper's
  closure under composition guarantees that combining scenarios always
  produces a valid transformation.

### Connection to agent orchestration

The strictly-positive-type result tells us that any well-founded task decomposition
tree an orchestrator might build is itself a container. This means we can
*reason generically* about orchestration strategies without knowing the specific
mathematical domain. An "AI Mathematician" that decomposes problems into
sub-problems, each with typed inputs and outputs, is building elements of a
container. The fixed-point construction (mu and nu) means we can handle both
finitely terminating proof searches (mu) and ongoing research processes (nu,
greatest fixed point -- potentially infinite streams of conjectures).

---

## 3. Altenkirch, Ghani et al. -- "Indexed Containers" (2015)

### Key technical idea

An **indexed container** over an index set `I` is a pair `(S, P)` where `S : Set`
and `P : S -> I -> Set`. Whereas an ordinary container stores data of a single
type `X` at its positions, an indexed container stores data of *sort* `i` at
positions tagged with `i in I`. The extension maps an `I`-indexed family
`A : I -> Set` to `Sigma s:S. Pi i:I. P s i -> A i`.

In Haskell terms, if ordinary containers generalise `data F x = ...` (one type
parameter), indexed containers generalise `data F (x :: I -> Type) (j :: I) = ...`
(a family indexed by `I`, with positions requiring data of specific sorts).

The paper shows indexed containers form a **relative monad**, and that they are
closed under parameterised initial algebras (using indexed W-types), enabling
mutual and nested inductive families. The key example is well-scoped lambda
terms: `ScLam n` (terms with at most `n` free variables), where the index `n`
tracks the variable count and the `lam` constructor increments it.

### Connection to AI Tax

This paper is the most directly relevant to the tax system's architecture.

- **The concentric layers are an indexed container.** Let `I = {Interface, Adapter, Core}`.
  The tax pipeline has shapes (pipeline stages) and positions indexed by layer.
  A `Fact` position requires Adapter-layer data; a `RuleApplication` position
  requires Core-layer data. The indexed container structure enforces that data
  of the wrong layer cannot leak into the wrong position. This is DD-05
  expressed as a typing discipline on positions.
- **DD-15 (computation produces a trace):** The `Computation` DAG is an indexed
  container where shapes are rule-application trees and positions are indexed
  by the *kind* of value they hold (input fact, intermediate result, final
  figure). The index prevents confusion between, say, a gross income figure
  and a tax liability figure even though both are `Money` values.
- **DD-09 (time as explicit dimension):** Adding a `TaxYear` index to the container
  makes the time dimension explicit at the type level. A `Rules y` value
  is a shape parameterised by `y : TaxYear`, and positions in that shape
  can only be filled with `Fact y` values of the matching year. The indexed
  container structure makes it a *type error* to apply 2024 rules to 2025 data.
- **Mutual inductive families for regulatory regimes:** Phase 3b envisions
  multiple regulatory regimes (HMRC, FCA, AML). Each regime has its own
  `Fact` subtypes and `Rule` library, but they share the kernel. Indexed containers
  model this as a *mutual* inductive family: the `Fact` type is indexed by
  regime, and the kernel operates generically over the index.

### Connection to agent orchestration

Indexed containers model **multi-sorted task decomposition**. In an AI Mathematician,
different positions in a proof attempt require different *sorts* of input: some positions
need lemma proofs, others need computational evidence, others need literature
references. The index `I` is the sort of contribution. An orchestrator assigns
sub-agents to positions by matching their capability (sort) to the position's
index. The relative monad structure gives a notion of *binding* -- just as `lam`
in `ScLam` binds a variable, a proof step can introduce a local definition that
subsequent steps can reference. The Kleisli composition of the relative monad
captures *sequential* agent coordination where later agents depend on earlier results.

### Concrete example

Consider a tax computation that needs both an income figure and a deduction figure
to compute tax owed. As an indexed container with `I = {Income, Deduction, Tax}`:
- Shape: `ComputeTax` (one shape)
- Positions: `P ComputeTax Income = ()`, `P ComputeTax Deduction = ()`,
  `P ComputeTax Tax = Void` (tax is output, not input)

A filling provides an `Income`-sorted value and a `Deduction`-sorted value.
The type system prevents filling the `Income` position with a `Deduction` value.

---

## 4. Ghani, Hedges, Winschel, Zahn -- "Compositional Game Theory" (2018)

### Key technical idea

**Open games** are morphisms in a symmetric monoidal category. An open game
`G : (X, S) -> (Y, R)` has four components: a set of strategy profiles, a **play**
function (forward: strategies + observations -> choices), a **coplay** function
(backward: strategies + observations + utility -> coutility), and a **best response**
relation. The key insight is that open games are equivalent to families of
**lenses** -- pairs of functions `(X -> Y, X * R -> S)` with a forward map and a
backward map, exactly like container morphisms.

Games compose **sequentially** (categorical composition, chaining forward and
backward maps) and **in parallel** (monoidal product). The paper proves that
the Nash equilibria of a composite game are determined compositionally from the
equilibria of its components. String diagrams visualise the information flow:
forward (observations/choices) flows left-to-right, backward (utility) flows
right-to-left.

### Connection to AI Tax

- **DD-08 (scenarios as composable transformations):** What-if scenarios in Phase 2
  are exactly open games where the "player" is the user, the "strategy" is the
  scenario choice (e.g., "contribute extra to pension"), and the "utility" is
  the net tax benefit. The compositional structure means that combining
  independent scenarios ("extra pension AND marriage allowance transfer")
  is just the monoidal product of two open games. The paper's framework
  guarantees that the equilibrium (optimal strategy) of the composite is
  determined by the components.
- **Coplay as explanation/audit trail:** The backward-flowing coplay function
  carries utility information from the output back to inputs. In the tax
  system, this is the explanation flow: given a final tax figure (output), the
  coplay traces backward through the computation to explain *why* each input
  contributed to that figure. This is DD-15 and DD-16 seen through a
  game-theoretic lens.
- **DD-10 (computation/guidance/advice separation):** The distinction between
  computation, guidance, and advice maps onto open-game components:
  computation is the play function (deterministic forward map), guidance is
  the coplay (backward explanation), and advice would be the best-response
  relation (what you *should* do). The paper's framework keeps these cleanly
  separated, just as DD-10 demands.

### Connection to agent orchestration

Open games are a natural model for **multi-agent coordination**. Each agent is
an open game; the orchestrator composes them sequentially (agent B uses agent A's
output) or in parallel (agents work simultaneously on independent sub-problems).
The backward flow carries "how useful was your output?" information, enabling:

- **Credit assignment:** The coplay function distributes utility backward through
  the agent composition, telling each agent how much its contribution mattered.
- **Equilibrium as coherence:** A system of agents is in "equilibrium" when no
  single agent can improve the overall outcome by changing its strategy
  unilaterally. This is a formal notion of *coherent multi-agent behaviour*.
- **String diagrams as agent workflows:** The paper's string diagram language
  gives a visual, formal syntax for agent pipelines. An AI Mathematician
  orchestrator could define workflows as string diagrams that compose
  conjecture-generation, proof-search, and verification agents.

### Concrete example

The P2 scenario engine:
- Agent A: generates candidate scenarios (pension, ISA, marriage allowance)
- Agent B: evaluates each scenario by running the tax computation
- Agent C: ranks scenarios by net benefit

As an open game: A is a decision node, B is a function node (deterministic
computation), C is a counit (collapses to ranking). The composite game's
equilibrium gives the optimal scenario recommendation.

---

## 5. Videla -- "Container Morphisms for Composable Interactive Systems" (2024)

### Key technical idea

Videla reinterprets containers as **APIs**: a container `(a, a')` describes an
interface where `a` is the type of *requests* and `a' : a -> Type` is the
type of *responses* (depending on the request). A container morphism
`(f, f') : (a, a') => (b, b')` is an **API transformer**: `f` maps surface-level
requests to underlying requests, and `f'` maps underlying responses back to
surface-level responses. Composition of morphisms chains API layers.

The paper builds monads on containers (e.g., `MaybeAll` for fallible APIs),
the **Kleene star** for repeated requests (sessions/protocols), and
state/costate pairs for executing API transformers as real programs.
The `Lift` construction turns a monad in `Type` (like `IO`) into a comonad
on containers, enabling effectful APIs to be described purely and then
executed. Implemented in Idris 2 with mechanised proofs.

### Connection to AI Tax

This paper is the **most practically relevant** -- it describes exactly the kind
of layered system the tax platform is.

- **Concentric layers as composed container morphisms:** The tax platform's
  Interface -> Adapter -> Core pipeline is literally a chain of API transformers:
  ```
  CLI : (String >< String) => MaybeAll Internal
  WEB : HTTP => MaybeAll Internal
  toQuery : Internal => DB
  ```
  The `MaybeAll` monad handles parse failures (not every string is a valid
  tax query), exactly matching DD-04's smart constructor discipline. The
  composition `app = CLI; toQuery` gives the end-to-end system.

- **DD-14 (Python <-> inner core boundary protocol):** The subprocess + JSON
  protocol between Python and Haskell is a container morphism. The Python
  side's request type is `JSONRequest`, the Haskell side's is `TypedQuery`.
  The forward map parses JSON into typed queries (smart constructor); the
  backward map serialises typed results to JSON. The container framework
  guarantees that composition of such boundaries is associative -- adding
  a third language layer wouldn't break the protocol.

- **Kleene star for sessions:** The sequential product `a >> b` and Kleene star
  `a*` model multi-step interactions. A tax filing session is:
  `Ingest >> Classify* >> Compute >> Reconcile*`
  The `*` on Classify and Reconcile means "repeat until done." The type
  system enforces the protocol: you can't Compute before Ingesting.

- **State/Costate as trust boundary:** A client (State) provides requests;
  a server (Costate) handles them. The `run` function pairs them. The
  trust boundary (DD-04, DD-05) is the pairing point: the Costate
  (trusted core) never sees raw user input, only typed requests that
  passed through the State's smart constructors.

- **Lift IO for effectful adapters:** `Lift IO DB` wraps a database API with
  IO effects. The counit gives a way to test with a pure in-memory mock.
  This matches the tax system's testing strategy: the inner core is pure
  (DD-07), the adapter uses `Lift IO` for OCR, LLM calls, etc.

### Connection to agent orchestration

- **Agents as API transformers:** Each agent exposes an API (accepts tasks,
  returns results). An orchestrator composes agents as container morphisms.
  The composition guarantees type safety across the agent boundary: agent B's
  request type must match agent A's response type (after transformation).
- **Kleene star for iterative reasoning:** A proof agent that tries multiple
  approaches uses `ProofAttempt*` -- zero or more attempts until `Done`.
  The Kleene star's type ensures each attempt consumes the previous result.
- **MaybeAll for fallible agents:** An agent that might fail (e.g., a literature
  search that finds nothing) uses `MaybeAll`. The Kleisli composition
  `parse >=> router` sequences fallible agents cleanly.

### Concrete example

The tax platform's P1 pipeline as container morphisms in Idris-style pseudocode:
```
Ingest    : UserDoc => MaybeAll RawExtraction
FactBuild : RawExtraction => MaybeAll Fact
Classify  : Fact => ClassifiedFact
Compute   : ClassifiedFact* => Computation
Render    : Computation => SA100

pipeline : UserDoc => MaybeAll SA100
pipeline = Ingest >=> FactBuild >=> (map Classify) >=> Compute >=> Render
```

---

## 6. Ahman, Chapman, Uustalu -- "When Is a Container a Comonad?" (2014)

### Key technical idea

A **directed container** is a container `(S, P)` equipped with three additional
operations:
- `down : Pi s. P s -> S` -- each position determines a **subshape**
- `o : Pi s. P s` -- each shape has a **root** position
- `(+) : Pi s. Pi p. P (s down p) -> P s` -- positions in a subshape
  translate to positions in the global shape

Subject to five laws (a generalisation of monoid laws). The paper's main theorem:
**a container is a comonad if and only if it is a directed container.** Directed
containers interpret fully faithfully into comonads on Set. The category `DCont`
is the pullback of `Cont -> [Set,Set]` along the forgetful functor
`Comonads(Set) -> [Set,Set]`.

In Haskell terms: a comonad `W` where `W X = Sigma s:S. (P s -> X)` must have
`extract` (pick the root) and `duplicate` (replace each data point with the
substructure rooted there). The directed container axioms are exactly what make
`extract` and `duplicate` well-defined and law-abiding.

Examples: non-empty lists (with suffixes as subshapes), streams, zippers.
Lists *cannot* be directed containers (the empty list has no root). The
cofree directed container on `C` gives the cofree comonad on `[[C]]`.
The cointerpretation (`DCont^op -> Monads(Set)`) yields "dependently typed
update monads" -- monads `T X = Pi s. P s -> X` that resemble `Reader`/`Writer`
hybrids with state-dependent updates.

### Connection to AI Tax

- **DD-15 and DD-16 through the comonadic lens:** The `Computation` DAG is a
  directed container. Each node (position) determines a sub-computation
  (subshape via `down`). The root position is the final tax liability. The
  translation operation `(+)` maps positions in a sub-computation to positions
  in the global computation. This is precisely what DD-16's "given any value,
  the chain of inputs and rules behind it is reachable" means -- you `down`
  to the subshape at that position, and its positions are the inputs.

- **`extract` is the final figure, `duplicate` is the full trace:**
  `extract` on the Computation gives the value at the root -- the final tax
  liability (a single number). `duplicate` replaces each node with the entire
  sub-computation rooted there -- this *is* the computation trace (DD-15).
  The comonad laws guarantee that extracting the root of the duplicated
  structure recovers the original, and that duplicating twice is the same as
  duplicating and then duplicating each sub-part. This is a correctness
  guarantee on the audit trail: the trace is consistent with the final number.

- **Writer monad via cointerpretation:** The cointerpretation functor turns a
  directed container into a "dependently typed update monad" `T X = Pi s. P s * X`.
  When `S = 1` (a single state), this degenerates to `(P * , +)` -- a writer
  monad on the free monoid over `P`. The tax system's provenance writer
  monad (DD-16: "provenance is a threaded value") is literally this
  construction. The directed container perspective reveals that the Writer
  monad for provenance and the comonadic computation trace are *dual views
  of the same structure*.

- **Zippers as focused computation:** The focussed container construction
  turns any container `C` into a directed container whose shapes are pairs
  `(s, p)` -- a shape and a designated position. This is the Huet zipper.
  For the Computation DAG, a zipper lets you focus on any intermediate
  figure and see both the sub-computation producing it and the context
  (which other computations depend on it). This is exactly what the
  Explanation Renderer needs: focus on box 1 of the SA100 and show both
  its derivation and its role in the total.

### Connection to agent orchestration

- **Comonadic context for agents:** A directed container equips every
  position with a *context* (the subshape rooted there). In an AI
  Mathematician, this means every intermediate result carries its full
  derivation context -- not just the value, but the sub-proof that
  produced it. An orchestrator can `duplicate` the proof state to give
  each sub-agent a view of the full proof tree, not just their local task.

- **Cofree comonad as proof trace:** The cofree directed container on a
  container `C` produces the cofree comonad -- non-well-founded trees
  labelled by `C`. This models an *exploratory* proof process: each step
  branches into further steps, producing a (potentially infinite) tree
  of attempts. The comonad structure lets you extract the current best
  result while retaining the full exploration history.

- **Update monads for agent state:** The cointerpretation gives update
  monads where the "state" is a shape and "updates" are positions. An
  agent's internal state evolves as it processes tasks. The directed
  container laws guarantee that composing two updates (two task
  completions) is associative and has a unit (the "no-op" task at the root).

### Concrete example

The tax `Computation` as a directed container:
```
S = DAGShape            -- set of valid computation DAG shapes
P s = NodePositions s   -- positions (nodes) in DAG of shape s
down s p = subDAG s p   -- sub-computation rooted at node p
o s = rootNode s        -- the final-liability node
p + p' = embedPos s p p' -- position p' in sub-DAG at p, mapped to global position

extract :: Computation a -> a
extract c = c ! rootNode (shape c)   -- read the final tax figure

duplicate :: Computation a -> Computation (Computation a)
duplicate c = c { at each position p, replace value with subDAG(p) }
-- every node now carries its full sub-computation: the audit trail
```

---

## Cross-cutting themes

### The shape/position decomposition unifies DD-15 and DD-16

The tax system's two most distinctive design decisions -- "computation produces
a trace, not a number" (DD-15) and "provenance is a threaded value, not a
sidecar" (DD-16) -- are not two separate ideas. They are the *same* idea seen
from the container perspective: a value in a container is a shape (the computation
structure) together with a filling (the data at each position). The "trace" is
the shape. The "provenance" is the position-to-source mapping. They cannot
drift apart because they are components of a single mathematical object.

### Container morphisms are trust boundaries

Every boundary in the tax system (DD-04, DD-05, DD-14) is a container morphism:
a forward map on shapes (translating requests/inputs) paired with a backward map
on positions (translating responses/outputs). The composition law guarantees
that chaining boundaries is well-defined. The full-and-faithful theorem guarantees
that no information is lost or invented at a boundary. Smart constructors are
the code that implements the forward map of a container morphism at the trust
boundary.

### Open games are the right model for scenario evaluation

The P2 what-if engine and the P3a financial planning engine are compositional
games in the sense of Ghani-Hedges. Scenarios compose via the monoidal product.
The backward flow (coplay) carries the explanation of *why* a scenario is
beneficial, which is exactly what the Explanation Renderer needs.

### Directed containers connect computation traces to writer monads

The Ahman-Chapman-Uustalu cointerpretation reveals that the `Writer Provenance`
monad threading provenance through the tax computation and the comonadic
`Computation` trace are dual perspectives on the same directed container structure.
This is not a metaphor -- it is a precise mathematical correspondence.

### For the AI Mathematician

The container framework offers a typed discipline for agentic orchestration:

1. **Task shapes** describe what kind of work needs doing (prove, compute, search literature)
2. **Positions** describe what inputs each task needs, indexed by sort (lemma, computation, reference)
3. **Container morphisms** are agent implementations: they translate high-level tasks to low-level operations and assemble results backward
4. **The Kleene star** handles iterative agents that retry or refine
5. **Directed containers** give every intermediate result a full derivation context (the comonadic `duplicate`)
6. **Open games** model multi-agent coordination where agents' strategies interact

The mathematical guarantee is compositionality: agents built from these
primitives compose correctly by construction, just as the tax system's pipeline
composes correctly by construction.

---

## Reading order recommendation

1. Start with **Videla (2024)** -- the most concrete and implementation-oriented.
   Read the Idris code examples. This paper connects containers to software
   engineering practice.
2. Then **Abbott-Ghani (2003)** -- the foundational definitions. Focus on
   Sections 3 (basic properties) and 7 (strictly positive types).
3. Then **Ahman-Chapman-Uustalu (2014)** -- directed containers and comonads.
   Focus on Section 3 (directed containers) and Section 6 (cointerpretation
   into monads).
4. Then **Ghani-Hedges (2018)** -- compositional game theory. Focus on
   Sections III-V (open games and composition).
5. Then **Altenkirch-Ghani (2015)** -- indexed containers. This is the most
   technically demanding but gives the framework for multi-sorted, multi-layer
   systems.
6. Skip the 2005 paper unless the correct PDF is located -- its content is
   subsumed by the 2003 paper's Section 7.

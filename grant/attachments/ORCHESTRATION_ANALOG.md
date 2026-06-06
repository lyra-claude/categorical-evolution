# Orchestration as a Functor on Directed Containers

## Agent Orchestration

An orchestrator is a **functor** between directed containers. It maps:

- A high-level intent (S_intent, P_intent) to a concrete pipeline (S_pipeline, P_pipeline)
- The functor preserves `down`, `o`, and `+` — meaning sub-task decomposition and provenance composition are respected

The meta-agent doesn't just choose which agents to call. It maps between *directed container structures*, so that `duplicate` (provenance) in the output pipeline is the functorial image of `duplicate` (intent structure) in the input. Orchestration and traceability are linked by functoriality.

```
F : DirCont_intent → DirCont_pipeline

F maps:
  intent shape "investigate topology and diversity"
    ↦ pipeline shape (research → dreaming → creative → email → paper)

  intent root (the research question)
    ↦ pipeline root (the published claim)

  intent down (sub-questions)
    ↦ pipeline down (sub-tasks assigned to agents)

  intent (+) (sub-question embedding)
    ↦ pipeline (+) (agent output embedding into global provenance)
```

The functor F is the orchestrator. Different orchestrators give different pipelines for the same intent. The functoriality condition guarantees that regardless of which orchestrator you choose, the provenance structure is coherent.

---

## Four Domains, Same Structure

### 1. Genetic Algorithms

A genetic algorithm is orchestrated by a **migration topology**. The topology determines which islands exchange individuals and how.

```
F_topology : DirCont_global → DirCont_islands

F maps:
  global shape (population evolving over generations)
    ↦ island shape (sub-populations on each island)

  global root (best individual found)
    ↦ island root (best individual on each island)

  global down at generation g (sub-evolution from g onward)
    ↦ island down at generation g (sub-evolution on each island from g)

  global (+) (embedding local generation into global history)
    ↦ island (+) (embedding island generation into island history)
```

The migration topology is the functor. Different topologies (ring, star, fully connected, none) give different functors. **Topology determines diversity** because different functors preserve different amounts of structure: a fully connected topology collapses `down` (every island sees everything → no independent sub-evolution → low diversity). A ring topology preserves `down` (each island has genuine local sub-evolution → high diversity).

This is the content of Lyra's conjecture: the diversity ordering (none > ring > star > random > fully_connected) is the ordering of how much `down`-structure the migration functor preserves.

### 2. Ethereum

A DeFi aggregator (1inch, Paraswap) orchestrates contract calls.

```
F_aggregator : DirCont_intent → DirCont_callTree

F maps:
  intent shape ("swap ETH for DAI")
    ↦ call tree shape (Uniswap.swap → Aave.deposit → Aave.borrow)

  intent root (the user's desired outcome)
    ↦ call tree root (the top-level transaction)

  intent down ("what does this swap depend on?")
    ↦ call tree down (sub-execution in each contract)

  intent (+) (embedding sub-intent into overall goal)
    ↦ call tree (+) (embedding sub-call positions into global receipt)
```

Different aggregators route through different contract sequences for the same user intent. The functoriality condition guarantees: the transaction receipt (`duplicate` on the call tree) is coherent with the user intent (`duplicate` on the intent), regardless of routing. The user can always trace the outcome back through the route to their original request.

### 3. Tax

The tax engine orchestrates rule applications.

```
F_rules : DirCont_return → DirCont_computation

F maps:
  return shape (SA100 form with boxes)
    ↦ computation shape (DAG of rule applications)

  return root (total tax liability)
    ↦ computation root (root of the DAG)

  return down at box b ("what determines box b?")
    ↦ computation down at node n (sub-DAG producing that figure)

  return (+) (box b's dependency on box c)
    ↦ computation (+) (node embedding in the DAG)
```

The Rules value (parameterised by tax year — DD-07) is the functor. Different tax years give different functors. The functoriality condition guarantees: the explanation of any figure (DD-15, `duplicate` on the computation) is coherent with the return structure, regardless of which tax year's rules are applied. You can always trace a figure back through the rules to the source facts.

### 4. Umbral Calculus — Where the Analogy Breaks

For formal power series (the stream comonad), there is no orchestration functor in the same sense. The directed container is:

```
S = Unit    (one shape — always infinite)
P = Nat     (positions are indices)
o = 0       (root is the constant term)
down n = Unit  (sub-shape is still a stream)
n + m = n + m  (index addition)
```

There is no "choice of decomposition strategy." There is no topology. There is no aggregator. The stream comonad has one shape, and `down` always gives the same thing — another stream. The coalgebra morphisms (the transfer formulas of Rota's umbral calculus) are comonad morphisms, but they aren't orchestrated by a functor choosing between alternatives.

The umbral calculus is a directed container, but it is not an *orchestrated* directed container. It has `extract` and `duplicate` (the shift operator and its consequences), but no meta-level choosing how to decompose. This is because formal power series are *homogeneous* — every position looks the same. Agents, contracts, tax rules, and island populations are *heterogeneous* — different positions have different types, and the orchestrator must choose how to route between them.

**The analogy works for heterogeneous systems (agents, blockchains, tax, GAs) but not for homogeneous ones (streams, power series).** This is actually a useful diagnostic: if your system has only one shape and uniform positions, you don't need orchestration. If it has many shapes and varied positions, you do — and the orchestrator is a functor.

---

## Summary

| Domain | Directed Container | Orchestration Functor | What the Functor Chooses |
|--------|-------------------|----------------------|------------------------|
| **Agents** | Agent pipeline DAG | Meta-agent | Which agents, in what order |
| **GAs** | Population evolution | Migration topology | Which islands exchange, how often |
| **Ethereum** | Contract call tree | DeFi aggregator | Which contracts, what route |
| **Tax** | Rule application DAG | Rules (by tax year) | Which rules apply, in what order |
| **Umbral calculus** | Stream (shift) comonad | *None* — homogeneous | N/A — one shape, uniform positions |

The first four are orchestrated directed containers. The fifth is a plain directed container. The difference is heterogeneity of shapes and positions.

**Topology determines diversity** because the migration functor determines how much `down`-structure is preserved. This is the same reason different DeFi aggregators give different gas costs, different tax years give different liabilities, and different meta-agents give different research outcomes. The functor is the thing that matters.

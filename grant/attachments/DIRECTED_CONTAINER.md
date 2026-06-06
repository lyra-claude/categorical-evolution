# Directed Containers: When Extract and Duplicate Are the Point

## Recap

A directed container is a container (S, P) equipped with three operations:

- `o : Pi s. P s` — each shape has a **root** position
- `down : Pi s. P s -> S` — each position determines a **subshape**
- `(+) : Pi s. Pi p. P (down s p) -> P s` — positions in a subshape embed into the global shape

Subject to five laws (generalised monoid laws). Ahman-Chapman-Uustalu's theorem: **a container is a comonad if and only if it is a directed container.** The comonad operations are:

- `extract` = read the value at the root
- `duplicate` = at every position, replace the value with the entire substructure rooted there

## Two Instances

### 1. Tax Computation DAG

```
S = DAGShape                    -- valid computation DAG shapes
P s = NodePositions s           -- nodes in DAG of shape s
o s = rootNode s                -- the final-liability node
down s p = subDAG s p           -- sub-computation rooted at node p
p + q = embedPos s p q          -- position q in sub-DAG at p, mapped to global

extract c = c ! rootNode        -- the final tax figure (a number)
duplicate c = at each node p,   -- the full sub-computation producing that node
              replace value       (the audit trail)
              with subDAG(p)
```

`extract` gives the answer. `duplicate` gives the full provenance at every point. The comonad laws guarantee the audit trail is consistent with the final number: extracting the root of a duplicated structure recovers the original.

### 2. Formal Power Series (Streams)

A formal power series f(x) = a_0 + a_1 x + a_2 x^2 + ... is a stream of coefficients.

```
S = Unit                        -- one shape (always infinite)
P s = Nat                       -- positions are indices
o s = 0                         -- root is the constant term
down s n = s                    -- subshape is still a stream (shift by n)
n + m = n + m                   -- position m in tail-from-n is position n+m globally

extract f = a_0                 -- the constant term
duplicate f = stream where      -- position n holds the stream (a_n, a_{n+1}, ...)
              position n holds    i.e. the tail from position n
              tail_n(f)
```

`extract` gives the constant term. `duplicate` gives every tail — which is what the shift operator does. The shift operator E[f](x) = f(x+1) is `extract . duplicate` composed with evaluation.

### Side by Side

| | Tax Computation | Formal Power Series |
|---|---|---|
| Shape | DAG skeleton | Unit (one shape) |
| Positions | Nodes in the DAG | Natural numbers |
| Root `o` | Final tax liability | Constant term a_0 |
| `down` at p | Sub-computation producing node p | Stream shifted by p |
| `p + q` | Embed sub-DAG position into global | Addition of indices |
| `extract` | Read the final figure | Read the constant term |
| `duplicate` | Every node carries its sub-computation | Every coefficient carries its future |
| Use case | Auditability | Umbral calculus |

## The Invariant: `duplicate` Unfolds Context

`duplicate` doesn't copy data. It replaces each local value with the entire substructure rooted there. This is the same operation in both cases:

- **Tax:** "At each intermediate figure, show me the chain of rules and inputs that produced it." That's the audit trail. DD-15 and DD-16 are consequences of `duplicate`.
- **Power series:** "At each coefficient a_n, show me the entire tail (a_n, a_{n+1}, ...)." That's the shift operator. The classical umbral identities are consequences of `duplicate`.

The comonad laws guarantee coherence:

1. `extract . duplicate = id` — extracting the root of the unfolded structure recovers the original
2. `fmap extract . duplicate = id` — extracting at every position after unfolding recovers the original
3. `duplicate . duplicate = fmap duplicate . duplicate` — unfolding twice is the same as unfolding and then unfolding each part

For the tax system: law 1 says the audit trail is consistent with the final number. Law 3 says zooming into a sub-computation's audit trail gives the same result as zooming into the global trail and restricting.

For power series: law 1 says shifting by 0 is identity. Law 3 says shifting by n then by m is the same as shifting by n+m. These are the shift operator axioms.

## Connection to the Umbral Calculus

The umbral evaluation map sends a polynomial sequence {p_n(x)} to its exponential generating function:

```
Sum_n p_n(x) t^n / n!
```

This is a formal power series in t — an element of the stream comonad — a directed container. The coalgebra morphisms that make the umbral trick work (from LEAN_AS_CONTAINER) become comonad morphisms between directed containers. "Positions map backward" becomes "`down` preserves subshape structure."

The stream comonad's `duplicate` is the mathematical engine behind the classical umbral identities. When Rota writes the transfer formula relating one polynomial sequence to another, he's composing comonad morphisms — container morphisms that respect `down`, `o`, and `+`.

## A Third Instance: Proof Trees

For the AI Mathematician, a proof-in-progress is a directed container:

```
S = ProofTreeShape              -- valid proof tree shapes
P s = GoalPositions s           -- open goals and proved lemmas
o s = rootGoal s                -- the main theorem to prove
down s p = subProof s p         -- the sub-proof at goal p
p + q = embedGoal s p q        -- goal q in sub-proof at p, mapped to global

extract proof = proof ! root    -- the verdict on the main theorem
duplicate proof = at each goal, -- the full sub-proof context at that goal
                  replace with
                  subProof(p)
```

`extract` answers "is the theorem proved?" `duplicate` gives every intermediate goal the full context of what depends on it and what it depends on — which is exactly what an orchestrator agent needs to assign sub-goals to sub-agents intelligently.

## The Spectrum of Directed Containers

| Instance | What `extract` gives you | What `duplicate` gives you | What the comonad laws guarantee |
|---|---|---|---|
| Tax DAG | Final tax figure | Audit trail at every node | Trail is consistent with final figure |
| Power series | Constant term | Every tail (shift operator) | Shift by n+m = shift by n then m |
| Proof tree | Main theorem status | Full sub-proof context at every goal | Sub-proof contexts compose consistently |

Three domains. Same three axioms. Same coherence guarantees.

## The Cointerpretation: Writer Monads for Free

Ahman-Chapman-Uustalu's cointerpretation functor turns directed containers into "dependently typed update monads." For the tax Computation, this yields the Writer Provenance monad — DD-16's "provenance is a threaded value." For power series, it yields the state monad tracking the current shift.

The directed container and the writer monad are dual views of the same structure. DD-15 (trace as product) and DD-16 (provenance as threaded value) are not two design decisions. They are the comonad and monad sides of one directed container.

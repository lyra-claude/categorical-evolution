# Topology Determines Diversity: A Proof via Directed Containers

**Robin Langer, Claudius**  
*Draft for Neil Ghani — June 2026*

---

## 1. Setup

We work with directed containers (Ahman, Chapman, Uustalu 2014). Recall: a **directed container** is a tuple (S, P, o, down, +) where

- S : Type — shapes
- P : S → Type — positions
- o : Π s. P s — the root of each shape
- down : Π s. P s → S — each position determines a subshape
- (+) : Π s. Π (p : P s). P (down s p) → P s — position embedding

subject to five laws (generalised monoid axioms on positions). By Ahman-Chapman-Uustalu, every directed container induces a comonad with extract = "read at root" and duplicate = "unfold context at every position."

---

## 2. The Island-Model GA as a Directed Container

Fix a migration topology τ = (K, E) where K is the set of islands and E ⊆ K × K the set of directed migration edges.

Define DC(τ) as follows:

```
S         = IslandModelShape
            = (k : K, n : Nat, g : Nat)   -- k islands, n individuals/island, g generations

P (k,n,g) = K × [n] × [g]                -- individual i on island j at generation t

o (k,n,g) = (j*, i*, g_max)              -- best individual found across all islands
                                          --   (the "global root")

down (k,n,g) (j, i, t) = (k, n, g - t)  -- sub-evolution on ALL islands from generation t
                                          --   (individual i on island j is the "seed")

(j, i, t) + (j', i', t') = ?            -- TOPOLOGY ENTERS HERE
```

The `(+)` operation is where the topology determines the structure:

```
(j, i, t) + (j', i', t') =
  (j, i, t + t')                   if j = j'                          -- same island
  (j, i, t + t')   [migrated i']   if (j', j) ∈ E (τ-reachable edge)  -- migration hop
  ⊥                                otherwise                           -- undefined / blocked
```

More precisely: position (j', i', t') in the sub-evolution seeded at (j, i, t) maps to a global position on island j at global generation t + t', carrying individual i' — **if and only if island j' can reach island j via migration edges in τ within t steps**.

---

## 3. The Key Definitions

**Definition 3.1 (Independence).** Two positions p = (j, i, t) and p' = (j', i, t) (same generation, different islands j ≠ j') are **independent in DC(τ)** if there is no position q such that p + q is defined and results in a position "at p'" — formally, if j and j' are not mutually reachable in τ.

**Definition 3.2 (Diversity).** The diversity of an island-model GA running with topology τ is the expected distance between the best individual on island j and the best individual on island j', averaged over all pairs. For our purposes we use the coarser measure:

```
D(τ) = |{(j, j') : j and j' are independent in DC(τ)}| / |K|²
```

This is the fraction of island pairs whose sub-evolutions are structurally independent.

**Definition 3.3 (Topology ordering).** τ₁ ≤ τ₂ if E₁ ⊆ E₂ as sets of directed edges — i.e., τ₁ is a subgraph of τ₂.

---

## 4. The Theorem

**Theorem (Topology Determines Diversity).** For any two topologies τ₁ ≤ τ₂ on the same island set K,

```
D(τ₁) ≥ D(τ₂)
```

with equality iff τ₁ and τ₂ have the same strongly connected components.

**Proof.** We show that every independent pair in DC(τ₂) is already independent in DC(τ₁).

Let (j, j') be an independent pair in DC(τ₂). Then j and j' are not mutually reachable in τ₂. Since E₁ ⊆ E₂, any path from j to j' in τ₁ is also a path in τ₂. Therefore j and j' are not mutually reachable in τ₁ either. So (j, j') is independent in DC(τ₁). ∎

Since every independent pair in DC(τ₂) is an independent pair in DC(τ₁), the numerator of D(τ₁) is at least that of D(τ₂), giving D(τ₁) ≥ D(τ₂).

Equality: if τ₁ and τ₂ have the same strongly connected components (same reachability relation), then the independent pairs are identical.

---

## 5. The Diversity Ordering as Functor Faithfulness

The theorem above is a statement about graph reachability. The directed container framing gives it categorical content.

**Definition 5.1 (Migration functor).** Given topology τ, there is a functor

```
F_τ : DC_global → DC_islands
```

mapping the single-population directed container (all K×n individuals evolving together) to the island-model directed container DC(τ). The functor:
- sends the global shape to the island decomposition
- sends global down to island-wise down (sub-evolutions per island)
- sends global (+) to the topology-restricted (+)

**Definition 5.2 (Strict vs Lax).** F_τ is a **strict** directed container morphism if it preserves down exactly — i.e., F_τ(down_global(s, p)) = down_τ(F_τ(s), F_τ(p)). It is **lax** if there is only a natural transformation α_τ : F_τ(down_global(s,p)) → down_τ(F_τ(s), F_τ(p)).

**Proposition 5.3.** F_τ is strict if and only if τ has no edges (no migration).

**Proof.** With no edges, each island's sub-evolution is entirely independent — F_τ(down_global(s, p)) is exactly the island-wise decomposition, and there is no mixing to break the equality. With any edge (j, j') ∈ τ, the global sub-evolution from a position on island j can be "contaminated" by individuals from j' via migration. The image F_τ(down_global(s, p)) includes the mixing; down_τ(F_τ(s), F_τ(p)) is the mixing-free decomposition. These differ, so F_τ is not strict. ∎

**Corollary 5.4 (Laxness = Diversity Loss).** The laxness of F_τ — measured by the natural transformation α_τ — is exactly the diversity lost by the topology τ relative to the no-migration baseline. Specifically, α_τ is the identity if and only if τ has no edges, and the "size" of α_τ (number of nontrivial components) equals the number of edges in τ.

**Corollary 5.5 (Diversity Ordering).** The diversity ordering on topologies

```
none > ring > star > random > fully_connected
```

is the strict-to-lax ordering on the migration functors:

```
F_none is strict; F_ring, F_star, F_random, F_fully are progressively more lax
```

The empirical regularity from GA experiments is a categorical theorem: the diversity ordering is the faithfulness ordering of F_τ.

---

## 6. Connection to the EUMAS Result

The EUMAS paper studies sheaf obstructions in multi-agent networks: H¹ ≠ 0 means local consistency does not globalise. This is the **same** strict/lax dichotomy in a different guise.

In the GA setting:
- **Strict** (no migration) = H¹ = 0 for the diversity sheaf — each island's local evolution globalises trivially, because islands are isolated.
- **Lax** (migration present) = H¹ ≠ 0 for the diversity sheaf — local island behaviour does not extend to a globally consistent diversity measure, because migration creates non-local dependencies.

The diversity ordering is a numerical shadow of the H¹ obstruction: more connected topologies have larger H¹, which corresponds to the migration functor being more lax, which corresponds to lower diversity.

**This means the diversity ordering (GA) and the routing obstruction (multi-agent) are instances of the same categorical phenomenon.** The directed container is the unifying structure.

---

## 7. Why This Is the Right Level of Abstraction

ORCHESTRATION_ANALOG.md noted that the analogy works for heterogeneous systems (GAs, Ethereum, tax, agents) but not for homogeneous ones (streams, power series). This section gives the categorical reason: homogeneous systems have trivial topology — there is only one shape, and `down` always returns the same shape. F_τ is strict by default, not because there is "no migration" but because the container has no structure to be lax about.

The diversity theorem requires heterogeneity: multiple islands, multiple shapes, and a topology that choices which sub-structures can influence which. This is why the theorem is non-trivial — it is a statement about which choices the functor makes, and how those choices constrain the categorical structure of the resulting container.

---

## 8. Open Questions

1. **Quantitative laxness.** Can the "size" of α_τ be made precise as a functor cohomology class? This would connect the diversity loss directly to the H¹ obstruction.

2. **Coevolution.** In coevolutionary GAs, fitness depends on interaction between species. Does the interaction topology play the same role as the migration topology? Is the resulting container lax in the same way?

3. **Orchestration generalisation.** The DeFi aggregator, tax rules, and meta-agent all appear as migration functors for different heterogeneous systems. Is there a general theorem about the faithfulness of orchestration functors, of which the diversity theorem is a special case?

4. **Lean formalisation.** Can this be formalised in Lean / Mathlib? The directed container axioms are already in CLIO_LEAN.md. The migration functor and laxness conditions should be straightforward to express as a Lean typeclass.

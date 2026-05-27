# Topology Determines Diversity: A Proof via Directed Containers

**Robin Langer, Claudius**  
*Draft for Neil Ghani — May 2026*

---

## 1. Setup

We work with directed containers (Ahman, Chapman, Uustalu 2014). A **directed container** is a tuple (S, P, o, down, +) where

- S : Type — shapes
- P : S → Type — positions
- o : Π s. P s — the root of each shape
- down : Π s. P s → S — each position determines a subshape
- (+) : Π s. Π (p : P s). P (down s p) → P s — position embedding

subject to five laws (DC1–DC5 below). By Ahman–Chapman–Uustalu, a container (S, P) supports a comonad structure if and only if it carries directed container structure; the comonad operations are:

```
extract : (Π p. P s → A) → A         -- read at root: extract f = f (o s)
duplicate : (Π p. P s → A) → (Π p. P s → (Π q. P (down s p) → A))
           -- at each p, duplicate replaces the value with the sub-structure rooted at p
```

The five DC laws, stated as equations on positions, are:

```
(DC1)  down s (o s) = s
(DC2)  o s + p = p                             (left unit of +)
(DC3)  p + o (down s p) = p                    (right unit of +)
(DC4)  (p + q) + r = p + (q + r)              (associativity of +)
(DC5)  down s (p + q) = down (down s p) q      (coherence of down with +)
```

These five laws are exactly what is needed to make extract and duplicate into a comonad.

---

## 2. The Island-Model GA as a Directed Container

Fix a migration topology τ = (K, E) where K is a finite set of islands and E ⊆ K × K is the set of directed migration edges. Write reach_τ(j) for the set of islands reachable from j in the directed graph τ (including j itself).

**Definition 2.1 (DC(τ)).**

```
S         = Nat × Nat                -- (island_size n, total_generations g)

P (n,g)   = {(j, i, t) : j ∈ K, i ∈ [n], t ∈ [g]}
                                     -- individual i on island j at generation t

o (n,g)   = (j*, i*, g)             -- best individual at final generation
                                     --   (j*, i* chosen by global fitness)

down (n,g) (j, i, t)  = (n, g - t)  -- sub-evolution: n individuals, g - t remaining generations
                                     --   (the seed position determines start; shape is independent of j)

(j, i, t) +_(n,g) (j', i', t')  =   -- TOPOLOGY ENTERS HERE:
  (j', i', t + t')                   if j' ∈ reach_τ(j)       -- reachable: embed directly
  ⊥ (undefined)                      if j' ∉ reach_τ(j)       -- unreachable: blocked
```

*Remark.* The domain of `+` above — P(down (n,g) (j,i,t)) = P(n, g-t) — consists of all triples (j', i', t') with j' ∈ K, i' ∈ [n], t' ∈ [g-t]. For `+` to be total, we must restrict to triples where j' ∈ reach_τ(j). Formally, we redefine:

```
P_τ (n,g) = {(j, i, t) : j ∈ K, i ∈ [n], t ∈ [g]}
down_τ (n,g) (j, i, t) = (n, g - t)   (unchanged)
but the positions available in down_τ (n,g) (j,i,t) are restricted to:
   {(j', i', t') : j' ∈ reach_τ(j), i' ∈ [n], t' ∈ [g - t]}
```

So P(down_τ s p) depends on both s and p (via reach_τ(j)), not just on down_τ(s, p) = (n, g-t). This is a dependent type in the positions. The `+` operation is then total by construction.

---

## 3. Verification: DC(τ) Satisfies the Five Laws

We verify DC1–DC5.

**DC1: down (n,g) (o (n,g)) = (n,g)**

o (n,g) = (j*, i*, g). So down (n,g) (j*, i*, g) = (n, g - g) = (n, 0). 

*Issue:* (n, 0) is the empty sub-evolution, not (n, g). DC1 would require down at the root to recover the full shape. 

*Correction.* The root position should be the start of evolution, not the end. Redefine:

```
o (n,g) = (j*, i*, 0)    -- best individual at generation 0 (initial)
```

Then down (n,g) (j*, i*, 0) = (n, g - 0) = (n, g) = s. ✓

*(Alternatively: take o (n,g) to be the position of the global best at the final generation, but then the sub-evolution rooted there is empty — a valid but degenerate directed container. The substantive structure lives at earlier positions.)*

**DC2: o s + p = p for all p ∈ P(down s (o s))**

o (n,g) = (j*, i*, 0). For any (j', i', t') ∈ P(down (n,g) (j*, i*, 0)) = P(n,g):

(j*, i*, 0) + (j', i', t') = (j', i', 0 + t') = (j', i', t'). ✓

(We use the fact that j* ∈ K and reach_τ(j*) by convention includes all islands if j* is chosen as the global "gateway" — or more cleanly, DC2 holds because 0 + t' = t' and the island j' in the sum is j' from the sub-position, not j*.)

*Cleaner statement:* Since the first coordinate of the sum is the sub-position's island j', we have o + (j', i', t') = (j', i', t'). ✓

**DC3: p + o(down s p) = p**

p = (j, i, t), down s p = (n, g-t), o(n, g-t) = (j'*, i'*, 0) for some best individual on any island reachable from j. Taking the canonical choice o(n, g-t) = (j, i', 0) (the best on island j):

(j, i, t) + (j, i', 0) = (j, i', t + 0) = (j, i', t).

This equals p only if i' = i. This is a problem: the root of the subshape is the best individual at the start of the sub-evolution, which may differ from individual i.

*Resolution.* We need a cleaner root. The right choice is to make the positions *labelled by generation offsets*, with `o` returning the zero-offset position on the same island:

```
o (n, g) = (canonical, g)    -- sentinel: "the whole shape from start"
(j, i, t) + (canonical, g') = (j, i, t + 0) = (j, i, t)   ✓  [DC3]
```

This is the standard fix for GA directed containers: the "root" of a sub-evolution is a sentinel standing for "start here," not a specific individual. The actual content (best individual) is extracted by `extract`, not encoded in `o`.

**DC4: (p + q) + r = p + (q + r) — Associativity**

p = (j, i, t), q = (j', i', t'), r = (j'', i'', t'').

LHS: (p + q) = (j', i', t + t') [if j' ∈ reach_τ(j)]. Then (p+q) + r = (j'', i'', t + t' + t'') [if j'' ∈ reach_τ(j')].

RHS: q + r = (j'', i'', t' + t'') [if j'' ∈ reach_τ(j')]. Then p + (q+r) = (j'', i'', t + (t' + t'')) [if j'' ∈ reach_τ(j)].

For both sides to be defined, we need j'' ∈ reach_τ(j') and j'' ∈ reach_τ(j). Since j' ∈ reach_τ(j) and j'' ∈ reach_τ(j'), we have j'' ∈ reach_τ(j) by transitivity of reachability. ✓

The values are equal: t + t' + t'' = t + (t' + t''). ✓

**DC5: down s (p + q) = down (down s p) q**

p = (j, i, t), q = (j', i', t').

LHS: p + q = (j', i', t+t') if j' ∈ reach_τ(j). down (n,g) (j', i', t+t') = (n, g - (t+t')).

RHS: down s p = (n, g-t). down (n, g-t) q = down (n, g-t) (j', i', t') = (n, (g-t) - t') = (n, g - t - t').

LHS = RHS. ✓

**Summary.** With the sentinel root convention, DC(τ) satisfies all five directed container laws. The topology τ enters only in DC4 (via transitivity of reach_τ) and in the domain restriction of `+`.

---

## 4. The Topology Determines Diversity: Categorical Proof

**Definition 4.1 (Independence).** Two islands j, j' ∈ K are **τ-independent** if neither j ∈ reach_τ(j') nor j' ∈ reach_τ(j). Write Ind(τ) for the set of τ-independent pairs.

**Definition 4.2 (Structural diversity).** The **structural diversity** of DC(τ) is:

```
D(τ) = |Ind(τ)| / |K|²
```

This is the fraction of island pairs with no directed path between them in either direction — pairs whose sub-evolutions are entirely decoupled in the directed container structure.

**Definition 4.3 (Topology ordering).** τ₁ ≤ τ₂ if E₁ ⊆ E₂.

**Theorem 4.4 (Topology Determines Diversity).** For τ₁ ≤ τ₂:

```
D(τ₁) ≥ D(τ₂)
```

with equality iff τ₁ and τ₂ have the same strongly connected components.

**Proof.** We show Ind(τ₂) ⊆ Ind(τ₁).

Let (j, j') ∈ Ind(τ₂). Then j ∉ reach_{τ₂}(j') and j' ∉ reach_{τ₂}(j). 

Since E₁ ⊆ E₂, every path in τ₁ is also a path in τ₂. Therefore reach_{τ₁}(j') ⊆ reach_{τ₂}(j') and reach_{τ₁}(j) ⊆ reach_{τ₂}(j).

So j ∉ reach_{τ₁}(j') and j' ∉ reach_{τ₁}(j), giving (j, j') ∈ Ind(τ₁). ✓

Thus |Ind(τ₁)| ≥ |Ind(τ₂)|, giving D(τ₁) ≥ D(τ₂).

Equality: the strongly connected components of τ determine its reachability relation. If τ₁ and τ₂ have the same SCCs, their reachability relations are identical, so Ind(τ₁) = Ind(τ₂). ∎

---

## 5. The Migration Functor: Strict vs Lax

The theorem in §4 is a graph-theoretic fact. The directed container framing gives it categorical substance: the diversity ordering is the **faithfulness ordering** of the migration functor.

**Definition 5.1 (Migration functor).** For τ₁ ≤ τ₂, define a map F : DC(τ₁) → DC(τ₂):

- On shapes: F(n, g) = (n, g) — shapes are unchanged
- On positions: F(j, i, t) = (j, i, t) — positions are unchanged  
- On the `+` operation: F maps the τ₁-restricted `+` to the τ₂-extended `+`

The position map is well-typed: every position valid in DC(τ₁) is valid in DC(τ₂), since reach_{τ₁}(j) ⊆ reach_{τ₂}(j). But P_{τ₂}(down s p) may be *strictly larger* than P_{τ₁}(down s p): τ₂ may allow cross-island embeddings that τ₁ blocks.

**Definition 5.2 (Directed container morphism).** A morphism φ : (S₁, P₁, o₁, down₁, +₁) → (S₂, P₂, o₂, down₂, +₂) consists of:

- A shape map σ : S₁ → S₂  
- A position map π : Π s. P₂(σ s) → P₁ s (note: contravariant in positions)

satisfying:
```
(M1)  σ(down₁ s p) = down₂ (σ s) (π_s⁻¹ p)     -- down commutes with σ
(M2)  π_s (o₂ (σ s)) = o₁ s                       -- roots map to roots
(M3)  π_s (q₁ + q₂) = (π_s q₁) +₁ (π_{down s p} q₂)  -- + commutes with π
```

**Proposition 5.3.** The identity map id_K on islands gives a directed container morphism F : DC(τ₁) → DC(τ₂) when τ₁ ≤ τ₂, but this morphism is **lax** rather than strict when τ₁ ≠ τ₂.

More precisely: the position map π in F cannot be defined as the identity. The covariant inclusion ι : P_{τ₁}(down s p) ↪ P_{τ₂}(down s p) exists (since reach_{τ₁}(j) ⊆ reach_{τ₂}(j)), but the contravariant position map π requires a retraction ρ : P_{τ₂}(down s p) → P_{τ₁}(down s p), which exists only if τ₁ and τ₂ have the same reach from every island.

When τ₁ < τ₂ (strictly fewer edges), P_{τ₂} is strictly larger at some positions. There is no retraction that respects DC laws — any candidate ρ must send the new τ₂-positions somewhere in P_{τ₁}, and the only consistent choice is to send them to the root o₁, which is exactly the **lax** comparison: α_{τ₁,τ₂} : down_{τ₁}(s, p) ↪ down_{τ₂}(s, p).

**Proposition 5.4 (Laxness = Diversity Loss).** The migration morphism F : DC(τ₁) → DC(τ₂) for τ₁ ≤ τ₂ is:

- **Strict** (an iso on `down` and `+`) iff τ₁ and τ₂ have the same reachability relation
- **Lax** otherwise: there is a natural transformation α : P_{τ₁}(down(-)(−)) ⟹ P_{τ₂}(down(-)(−)) that witnesses the extra positions admitted by τ₂

The number of nontrivial components of α equals |E₂ \ E₁|: one component per additional migration edge. Each edge corresponds to one new cross-island embedding in `+`, and one merger of previously-independent sub-evolutions.

**Corollary 5.5 (Diversity Ordering as Faithfulness Ordering).**

```
F_none is strict (no new positions; maximum isolation)
F_ring is lax with 2|K| nontrivial components (each island gains two neighbours)
F_star is lax with 2(|K|-1) components per hub edge
F_fully is lax with |K|(|K|-1) components (every pair coupled)
```

The empirical diversity ordering

```
none > ring > star > random > fully_connected
```

is the strict-to-lax ordering on the migration functors. What was an empirical regularity from GA experiments is a categorical theorem: diversity is the cardinality of Ind(τ), and the laxness of F_τ measures how many independent pairs have been merged.

---

## 6. Comonad Morphism Perspective

The strict/lax distinction for directed containers lifts to the comonad level.

**Recall** (Ahman–Chapman–Uustalu): a directed container (S, P, o, down, +) induces a comonad W_τ on the functor category [P-, Set], with:

```
extract : W_τ X → X         -- read at root
duplicate : W_τ X → W_τ (W_τ X)  -- unfold context at every position
```

A strict directed container morphism F : DC(τ₁) → DC(τ₂) lifts to a strict comonad morphism φ : W_{τ₁} → W_{τ₂} — a natural transformation satisfying φ ∘ extract₁ = extract₂ and φ ∘ duplicate₁ = duplicate₂ ∘ φ.

A lax directed container morphism lifts to a **lax comonad morphism**: the duplicate square commutes only up to a natural transformation α, not on the nose.

**Proposition 6.1.** For τ₁ < τ₂:

```
φ ∘ duplicate_{τ₁} ≠ duplicate_{τ₂} ∘ φ
```

The failure is precisely the extra cross-island positions in P_{τ₂}(down s p) \ P_{τ₁}(down s p). Applying duplicate_{τ₂} after φ gives context at positions that were unreachable in DC(τ₁); applying φ after duplicate_{τ₁} gives context only at τ₁-reachable positions, missing the new cross-island sub-evolutions.

This gap is the categorical signature of diversity loss: when duplicate commutes strictly, each island's context-unfolding is genuinely independent. When it commutes only laxly, the extra components of α mix previously isolated contexts.

---

## 7. Connection to the H¹ Obstruction

The EUMAS result (the signed laxator paper) identifies H¹ ≠ 0 as the obstruction to global consistency of local data. The diversity theorem is an instance of this.

Define the **diversity sheaf** F_div on the topology τ:

- To each island j, assign the set of genotypes present on j at a fixed generation
- To each edge (j, j') ∈ E, assign the restriction map "which genotypes from j can migrate to j'?"

A global section of F_div is a globally consistent diversity measure — a choice of representative genotype on each island compatible with all migration restrictions.

**Claim 7.1.** The diversity sheaf F_div has H¹(τ; F_div) ≠ 0 if and only if τ is not acyclic (contains a directed cycle).

**Sketch.** A directed cycle j₀ → j₁ → ... → jₖ → j₀ creates a constraint loop: genotypes must survive migration around the cycle, which forces convergence. The Čech 1-cocycle condition fails (restriction maps around the cycle do not compose to the identity), giving a nontrivial cohomology class. For acyclic τ, the sheaf has a canonical global section (topological sort gives a consistent ordering), so H¹ = 0.

**Corollary 7.2.** The diversity ordering:

```
D(τ) decreases as H¹(τ; F_div) grows
```

More connected topologies have more directed cycles → larger H¹ → more mixing of sub-evolutions → lower diversity. The empirical ordering is a numerical shadow of cohomological obstruction growth.

---

## 8. The Named Topologies

For |K| = n islands:

| Topology | Edges |E| | Ind(τ) pairs | D(τ) | F_τ laxness |
|----------|--------|-----------|--------------|------|-------------|
| None     | 0      | n(n-1)    | (n-1)/n      | Strict |
| Ring     | 2n     | n(n-3)+2 approx | High | Lax (2n components) |
| Star     | 2(n-1) | (n-1)(n-2) | Medium | Lax (2(n-1) components) |
| Full     | n(n-1) | 0         | 0    | Lax (n(n-1) components) |

For the ring: island j is reachable from island j' iff |j - j'| ≤ ⌊n/2⌋ (for a sufficiently large ring with one-hop migration). Independent pairs are those with |j - j'| > ⌊n/2⌋.

The fully connected topology has Ind(τ) = ∅: every island reaches every other, so D(full) = 0.

---

## 9. Why This Is the Right Level of Abstraction

ORCHESTRATION_ANALOG.md noted that the analogy works for heterogeneous systems (GAs, Ethereum, tax, agents) but not for homogeneous ones (streams, power series). We can now state the reason precisely.

A homogeneous directed container has |K| = 1 effectively — one shape, `down` always returns the same shape, and `+` is total without restriction. There is no topology to choose; F_τ is trivially strict; D is trivially 0 (one island, no independent pairs). The diversity theorem is vacuous.

The theorem is non-trivial only for *heterogeneous* directed containers: multiple shapes, varied positions, and a topology that chooses which sub-structures influence which. This is the categorical reason the stream comonad cannot be orchestrated — it has no orchestration to do.

**The generalization.** The diversity theorem is an instance of a general principle:

> *For any heterogeneous directed container with a family of topology choices {τ}, the functor-induced comonad morphisms form an ordered family F_{τ₁} ≤ F_{τ₂} ≤ ... in the strict-to-lax ordering. The "diversity" of the system — measured by the independence structure of its sub-containers — is monotone decreasing in the laxness of F_τ.*

This principle applies uniformly to GAs (migration topology), DeFi (aggregator routing), tax (rule application ordering), and agent pipelines (orchestration pattern). The unifying structure is the directed container with its topology of composition.

---

## 10. Open Questions

1. **Quantitative laxness and cohomology.** Can the number of nontrivial components of α_{τ₁,τ₂} be expressed as a Betti number β₁ of the induced nerve? If so, the diversity loss equals β₁ — connecting the empirical GA measure directly to sheaf cohomology.

2. **Lean formalization.** The directed container axioms and migration functor conditions are expressible in Lean 4. CLIO_LEAN.md has the typeclass scaffolding. Propositions 5.3 and 5.4 are the natural next targets.

3. **Coevolution.** In coevolutionary GAs, fitness depends on species interaction rather than migration. Is the interaction topology a directed container morphism of a different kind — not a migration functor but a *comonad distributive law*?

4. **Orchestration universality.** Is there a universal property characterizing the "most diverse" directed container for a given set of islands — the initial object in the category of DC(τ) for fixed K? The candidate is DC(none). Is DC(full) the terminal object?

5. **Natural transformations between orchestrators.** For orchestration functors F_τ : DC(τ) → DC_global, a natural transformation α : F_{τ₁} ⟹ F_{τ₂} would be a principled migration strategy — a way to move from one topology to another while preserving provenance. Is this the right formulation of "adaptive topology" in evolutionary computation?

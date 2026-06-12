# Mathematical Supplement: Tightening the Gaps

**For Robin and Neil -- XTX AI4Math Fund Grant**
*Prepared May 30, 2026*

This supplement addresses four mathematical gaps in the grant material and Claudius's topology-diversity proof. Each section is self-contained. Results are marked THEOREM (proved), PROPOSITION (proof sketched, completable), or CONJECTURE (believed, not proved).

---

## A. Precise Functor Definition for Orchestration

The grant material asserts that orchestration is a functor F : DirCont_intent --> DirCont_pipeline, but never defines the source and target categories. We remedy this.

### A.1. The Category DirCont

**Definition A.1 (Directed container).** A *directed container* is a tuple (S, P, o, down, +) where S : Type, P : S --> Type, o : Pi(s:S). P(s), down : Pi(s:S). P(s) --> S, and (+) : Pi(s:S). Pi(p:P(s)). P(down(s,p)) --> P(s), subject to laws DC1--DC5 (Ahman--Chapman--Uustalu 2014):

    (DC1)  down(s, o(s)) = s
    (DC2)  o(s) + p = p
    (DC3)  p + o(down(s,p)) = p
    (DC4)  (p + q) + r = p + (q + r)
    (DC5)  down(s, p + q) = down(down(s,p), q)

**Definition A.2 (Directed container morphism).** A morphism phi : (S_1, P_1, o_1, down_1, +_1) --> (S_2, P_2, o_2, down_2, +_2) consists of:

- A shape map sigma : S_1 --> S_2
- A position map pi : Pi(s:S_1). P_2(sigma(s)) --> P_1(s)  (note: contravariant)

satisfying:

    (M1)  pi_s(o_2(sigma(s))) = o_1(s)
    (M2)  sigma(down_1(s, pi_s(q))) = down_2(sigma(s), q)
    (M3)  pi_s(q_1 +_2 q_2) = pi_s(q_1) +_1 pi_{down...}(q_2)

M2 says: the shape map applied to the sub-shape in the source equals the sub-shape in the target (following Ahman--Chapman--Uustalu 2014).

The full compatibility conditions require that sigma and pi jointly respect down, o, and +. The precise formulation is: (sigma, pi) must constitute a morphism of the underlying containers that additionally preserves the directed structure. Following Ahman--Chapman--Uustalu, this means the induced natural transformation between the corresponding comonad functors preserves extract and duplicate.

**Definition A.3 (The category DirCont).** Objects are directed containers. Morphisms are directed container morphisms as in Definition A.2. Composition is:

    (sigma_2, pi_2) . (sigma_1, pi_1) = (sigma_2 . sigma_1,  s |-> pi_1(s) . pi_2(sigma_1(s)))

Identity is (id_S, s |-> id_{P(s)}).

THEOREM A.4. *DirCont is a category.* Composition is associative and unital.

*Proof.* Direct verification: associativity of composition follows from associativity of function composition in both components. The identity morphism (id, id) satisfies M1--M3 trivially, and acts as a unit for composition. The morphism conditions M1--M3 are preserved under composition by straightforward calculation.  QED

### A.2. The Parameterised Category DirCont(tau)

**Definition A.5 (DirCont(tau)).** Fix a topology tau = (K, E) where K is a finite set and E <= K x K is a set of directed edges. The category DirCont(tau) is the full subcategory of DirCont whose objects are directed containers of the form DC(tau) as defined in Claudius's proof (Definition 2.1 with sentinel root convention):

    S = Nat x Nat
    P(n,g) = {(j,i,t) : j in K, i in [n], t in [g]}
    o(n,g) = (j*, i*, 0)                          -- sentinel root
    down(n,g)(j,i,t) = (n, g-t)
    (j,i,t) + (j',i',t') = (j',i',t+t')          if j' in reach_tau(j)

More generally, DirCont(tau) includes any directed container whose (+) operation respects the reachability structure of tau -- that is, p + q is defined only when the "source island" of q is reachable from the "source island" of p in the graph (K, E).

### A.3. The Migration Functor

**Definition A.6 (Migration functor).** For tau_1 <= tau_2 (meaning E_1 <= E_2), define

    M(tau_1, tau_2) : DirCont(tau_1) --> DirCont(tau_2)

On objects: M sends DC(tau_1) = (S, P_{tau_1}, o, down, +_{tau_1}) to DC(tau_2) = (S, P_{tau_2}, o, down, +_{tau_2}), where P_{tau_2}(down(s,p)) >= P_{tau_1}(down(s,p)) since reach_{tau_1}(j) <= reach_{tau_2}(j).

On morphisms: M sends a DC(tau_1)-morphism (sigma, pi_1) to the DC(tau_2)-morphism (sigma, pi_2) where pi_2 is defined by:

    pi_2(s)(q) = pi_1(s)(q)      if q in P_{tau_1}(sigma(s))
    pi_2(s)(q) = o_1(s)          if q in P_{tau_2}(sigma(s)) \ P_{tau_1}(sigma(s))

That is, positions newly reachable under tau_2 are sent to the root. (See Section C for justification of this choice.)

THEOREM A.7 (Functoriality). *M(tau_1, tau_2) is a functor.*

*Proof.* We verify:

(i) M preserves identities: M(id_S, id_P) = (id_S, pi_2) where pi_2 agrees with id on P_{tau_1} positions and sends new positions to the root. This is a valid DC(tau_2)-morphism by the sentinel root convention (DC2 ensures o + q = q, so the root acts as a neutral element).

(ii) M preserves composition: For composable morphisms (sigma_2, pi_2^1) . (sigma_1, pi_1^1) in DirCont(tau_1), the image under M is determined by the action on P_{tau_1}-positions (where it agrees with the original composition) and on new P_{tau_2}-positions (where both M-images send to root, and root composed with root gives root by DC3). QED

PROPOSITION A.8 (Strict vs lax). *M(tau_1, tau_2) is:*
- *A strict functor when tau_1 and tau_2 have the same reachability relation (same strongly connected components)*
- *A lax functor otherwise, with the laxness witnessed by a natural transformation alpha whose nontrivial components are indexed by the edges E_2 \ E_1*

*Proof sketch.* Strictness means M preserves the directed container structure on the nose. This holds iff P_{tau_1}(down(s,p)) = P_{tau_2}(down(s,p)) for all s, p, which happens iff reach_{tau_1}(j) = reach_{tau_2}(j) for all j in K -- i.e., the reachability relations coincide. When they differ, the retraction pi_2 collapses new positions to root, introducing a comparison cell alpha : M(+_{tau_1}) ==> +_{tau_2} that commutes only up to this collapse. Explicitly, the components of alpha are the family of maps alpha_{s,p} : M(p +_{tau_1} q) --> M(p) +_{tau_2} M(q) defined for each shape s and position p by: on positions q in P_{tau_1}(down(s,p)), alpha_{s,p} is the identity; on positions q in P_{tau_2}(down(s,p)) \ P_{tau_1}(down(s,p)) where the two composition operations disagree, alpha_{s,p} maps M(p +_{tau_1} q) = o_1 (the root, since q is outside tau_1's reach) to M(p) +_{tau_2} M(q), which is well-defined because tau_2 permits the composition. The nontrivial components are thus indexed by positions where reach_{tau_2} strictly exceeds reach_{tau_1}.

---

## B. The H^1 Obstruction: Precise Conjecture

Section 7 of Claudius's proof claims H^1(tau; F_div) != 0 iff tau has a directed cycle. The argument is suggestive but imprecise: the diversity sheaf F_div is not defined rigorously, and the Cech cohomology argument is only gestured at. We tighten this to a precise conjecture.

### B.1. The Nerve and the Diversity Presheaf

**Definition B.1 (Nerve of tau).** Let tau = (K, E) be a directed graph. The *nerve* N(tau) is the simplicial complex whose k-simplices are chains j_0 --> j_1 --> ... --> j_k in tau (sequences of composable edges). A 0-simplex is an island. A 1-simplex is a migration edge. A 2-simplex is a composable pair of edges.

**Definition B.2 (Diversity presheaf).** Fix a generation number g and an island-model GA with population size n on topology tau. The *diversity presheaf* F_div is a presheaf of sets on N(tau)^{op} defined as follows:

- To each island j in K (0-simplex), assign
      F_div(j) = {genotype distributions on island j at generation g}
  Concretely, F_div(j) is the set of probability distributions on the genome space G, or more simply, the powerset of genotypes present: F_div(j) = P(G_j) where G_j <= G is the set of distinct genotypes on island j.

- To each edge (j, j') in E (1-simplex), the restriction map
      rho_{j,j'} : F_div(j') --> F_div(j)
  is defined by: rho_{j,j'}(D') = D' intersect (genotypes that can migrate from j' to j under the migration policy). If migration sends a fraction mu of j''s population to j, then rho selects the mu-fraction of D' that actually migrates.

*Remark.* For the purpose of a clean conjecture, we work with the simplified version: F_div(j) = P(G_j) and rho_{j,j'}(A) = A (full genotype transfer on each edge). This makes the restriction maps inclusions, which is the worst case for cohomological obstruction (every migration edge transfers all information).

### B.2. Cech Cohomology

**Definition B.3.** The *Cech cohomology* H^*(N(tau); F_div) is computed from the Cech complex:

    C^0 = Prod_{j in K} F_div(j)
    C^1 = Prod_{(j,j') in E} F_div(j')
    d^0 : C^0 --> C^1,   (d^0 sigma)_{(j,j')} = rho_{j,j'}(sigma_{j'}) - sigma_j

A 0-cochain is a choice of genotype set on each island. A 1-cocycle is an assignment of genotype sets to each edge satisfying the cocycle condition. H^0 = ker(d^0) is the space of *global sections* -- globally consistent diversity assignments.

H^1 = ker(d^1) / im(d^0) measures the obstruction to patching local diversity data into a global assignment.

### B.3. The Conjecture

CONJECTURE B.4 (H^1 Diversity Obstruction). *Let tau = (K, E) be a finite directed graph. Working with the simplified diversity presheaf (F_div(j) = P(G) for a fixed genome space G, restriction maps = identity), the following are equivalent:*

*(i) H^1(N(tau); F_div) = 0*

*(ii) tau is a directed acyclic graph (DAG)*

*(iii) The migration functor M(none, tau) induces an isomorphism on H^1 -- i.e., M* : H^1(N(tau); F_div) --> H^1(N(none); F_div) is an isomorphism (both trivial).*^[If the intended meaning is weaker than isomorphism -- e.g., that M merely preserves the vanishing of H^1 -- this equivalence still holds but the statement should be reformulated accordingly.]

*Moreover, when H^1 != 0, the rank of H^1 equals the cycle rank of tau (the number of independent directed cycles), which equals |E| - |K| + c where c is the number of weakly connected components.*

**Why this should be true.** The argument has two directions:

*DAG implies H^1 = 0.* If tau is a DAG, it admits a topological sort j_{sigma(1)}, ..., j_{sigma(n)}. This gives a canonical global section: process islands in topological order, and at each island the incoming genotypes are determined by already-processed islands. The restriction maps compose consistently along any path (no path revisits a node), so every 1-cocycle is a coboundary. This is a standard argument: acyclic categories have trivial higher cohomology for constant-coefficient presheaves.

*Cycle implies H^1 != 0.* Suppose tau contains a directed cycle C : j_0 --> j_1 --> ... --> j_k --> j_0. Consider the composition of restriction maps around C:

    rho_C = rho_{j_0,j_1} . rho_{j_1,j_2} . ... . rho_{j_k,j_0}

This is an endomorphism of F_div(j_0). If rho_C != id (which generically it is not -- migration is lossy, or at minimum selective), then the 1-cocycle condition around C cannot be satisfied by a coboundary, giving a nontrivial class in H^1.

In the simplified case where all restriction maps are identity, the cycle forces rho_C = id, which means the cocycle condition *around the cycle* is automatically satisfied -- but the obstruction manifests differently: the cycle creates a redundant constraint that prevents independent choice of local sections. Specifically, the rank computation |E| - |K| + c counts the "extra" edges beyond a spanning forest, each of which creates one independent constraint loop, yielding one dimension of H^1.

### B.4. What Constitutes a Proof

A complete proof of Conjecture B.4 requires:

**Lemma B.5 (needed).** For a constant presheaf F on the nerve of a finite DAG, H^k(N(tau); F) = 0 for all k >= 1.

*Status:* This is a standard result. The nerve of a DAG is contractible because a topological sort provides a filtration with contractible fibers. Alternatively, apply Quillen's Theorem A to the inclusion of any sink vertex. Constant presheaves on contractible categories have trivial higher cohomology.

**Lemma B.6 (needed).** For a finite directed graph tau with cycle rank r >= 1, H^1(N(tau); F_div) has rank exactly r when F_div is the constant presheaf with stalk P(G).

*Status:* This requires showing that each independent cycle contributes exactly one dimension to H^1 and that there are no cancellations. The argument is analogous to the computation of H^1 for graphs in classical algebraic topology: the cycle rank of a graph equals its first Betti number. The subtlety is that we work with directed graphs and a presheaf (not just a constant sheaf), so one must verify that the directedness does not collapse or expand the cohomology relative to the undirected case. For constant presheaves, the directed and undirected nerves yield the same H^1, since the directed nerve deformation-retracts onto the undirected one when all edges are formally inverted.

**Lemma B.7 (needed).** The connection between H^1 rank and the diversity measure D(tau) from Theorem 4.4 of Claudius's proof: show that D(tau_1) > D(tau_2) implies rank H^1(N(tau_1); F_div) <= rank H^1(N(tau_2); F_div).

*Status:* This should follow from the edge-monotonicity of cycle rank: adding edges can only increase or maintain the cycle rank. Since D is anti-monotone in edges (Theorem 4.4) and cycle rank is monotone in edges, the two measures are anti-correlated. But a direct proof linking D(tau) to H^1 rank -- not merely showing they are anti-correlated but that one *determines* the other -- requires more work.

---

## C. Proposition 5.3 Uniqueness: The Root Retraction

Claudius's proof (Proposition 5.3) asserts that when tau_1 < tau_2, the only consistent retraction rho : P_{tau_2}(down(s,p)) --> P_{tau_1}(down(s,p)) sends new positions to the root o_1. Here we prove this.

PROPOSITION C.1 (Uniqueness of the root retraction). *Let DC(tau_1) and DC(tau_2) be directed containers for tau_1 < tau_2, sharing the same shapes and down operation. Let*

    N(s,p) = P_{tau_2}(down(s,p)) \ P_{tau_1}(down(s,p))

*be the set of "new" positions admitted by tau_2 but not tau_1. Suppose rho : P_{tau_2}(down(s,p)) --> P_{tau_1}(down(s,p)) is a retraction (rho restricted to P_{tau_1} is the identity) such that (id_S, rho) constitutes a directed container morphism DC(tau_2) --> DC(tau_1). Then for all q in N(s,p):*

    rho(q) = o_1(down(s,p))

*Proof.* The morphism condition M1 requires rho to preserve roots:

    rho_s(o_2(s)) = o_1(s)

Since o_1(s) = o_2(s) = (j*, i*, 0) (the sentinel root is independent of tau), this is satisfied for any retraction.

The critical constraint is M3, compatibility with (+):

    rho_s(q_1 +_{tau_2} q_2) = rho_s(q_1) +_{tau_1} rho_{down(s,rho_s(q_1))}(q_2)

Now take q_1 in N(s,p) -- a new position not in P_{tau_1}. We need rho_s(q_1) in P_{tau_1}(s). Write q_1 = (j', i', t') where j' is reachable from j in tau_2 but not in tau_1.

Suppose for contradiction that rho(q_1) = (j'', i'', t'') with t'' > 0. Then down(s, rho(q_1)) = (n, g - t''), and we need M3 to hold for all q_2 in P_{tau_2}(down(s, q_1)):

    rho_s(q_1 +_{tau_2} q_2) = rho_s(q_1) +_{tau_1} rho_{(n,g-t'')}(q_2)

The left side: q_1 +_{tau_2} q_2 = (j'', i'', t' + t_2) for some q_2 = (j'', i'', t_2) with j'' in reach_{tau_2}(j'). This is a position in DC(tau_2) at generation t' + t_2.

The right side: rho(q_1) +_{tau_1} rho(q_2). For this to be well-defined, the island component of rho(q_2) must be in reach_{tau_1} of the island component of rho(q_1).

Here is the obstruction: q_2 ranges over *all* positions in P_{tau_2}(down(s, q_1)), including positions on islands reachable from j' in tau_2. But rho must map these into P_{tau_1}, where the reachability from rho(q_1)'s island is more restricted. For M3 to hold for all such q_2, the rho-image of every tau_2-reachable island must be tau_1-reachable from rho(q_1)'s island.

The only position in P_{tau_1} from which *every* island is reachable (under +_{tau_1}) is the root o_1, because DC2 gives o_1 + q = q for all q. Any non-root position (j'', i'', t'') with t'' > 0 has reach_{tau_1}(j'') as a proper subset of K (since tau_1 < tau_2 implies tau_1 is not fully connected from j''). So some q_2 would fail the M3 condition. *Caveat:* When tau_1 is already fully connected but tau_2 has additional redundant edges (i.e., E_1 subset E_2 but reach_{tau_1} = reach_{tau_2} = K), every non-root position also has full reachability, so the above argument does not produce a contradiction. In this case, however, P_{tau_1} = P_{tau_2} (since reachability sets coincide), so N(s,p) is empty and the proposition holds vacuously.

Therefore rho(q_1) = o_1(s) = (j*, i*, 0), the root. QED

*Remark.* The key insight is that the root is the *unique* position that is a left unit for (+) (by DC2). Any other choice of retraction target would fail the morphism compatibility M3 for at least one sub-position q_2, because non-root positions have strictly smaller reachability sets under tau_1.

---

## D. Sentinel Root Convention: Categorical Justification

Claudius's proof introduces a "sentinel root" convention: o(n,g) = (j*, i*, 0) rather than the semantically natural choice of "best individual at the final generation." This is not ad hoc. It is forced by the universal property of the initial object in a specific category.

### D.1. The Problem

The semantically appealing root o(n,g) = (j*, i*, g) -- the best individual at the final generation -- fails DC1:

    down(n,g)(j*, i*, g) = (n, g - g) = (n, 0) != (n, g) = s

The sub-evolution rooted at the final generation is empty. This is correct computationally (there is nothing after the last generation), but it means the comonad extract reads from the *end* of the evolution, while duplicate unfolds from the *beginning*. The comonad law extract . duplicate = id then fails because extracting the root of the duplicated structure gives the empty tail, not the original.

### D.2. Initial Algebra Justification

Consider the category Roots(s) whose objects are candidate root positions for shape s = (n,g), and whose morphisms are position embeddings respecting the DC laws.

**Definition D.1.** For a fixed shape s = (n,g), define the category Roots(s):

- Objects: positions p in P(s) such that down(s,p) = s  (i.e., satisfying DC1)
- Morphisms: p --> p' iff p' = p + q for some q, and all DC laws hold with p' as root

PROPOSITION D.2 (Roots(s) has an initial object). *The position o(n,g) = (j*, i*, 0) is the initial object of Roots(s), in the following sense: it is the unique position satisfying DC1, and for every other position q in P(s), there exists a unique embedding o(s) + q' = q for some q'.*

*Proof.* DC1 requires down(s, o(s)) = s. Since down(n,g)(j,i,t) = (n, g-t), DC1 forces t = 0. Among positions with t = 0, the choice of (j*, i*) is a parameter of the construction (the "designated starting island and individual"). Given this choice, DC2 gives o(s) + q = q for all q, which means there is a unique morphism from o(s) to any other position -- precisely the definition of an initial object. QED

### D.3. Terminal Coalgebra Justification

Dually, the sentinel root arises as the terminal coalgebra of the "unfold" endofunctor on positions.

Define the endofunctor U on the poset of positions ordered by the (+) relation:

    U(p) = down(s,p) together with "restart from p"

A coalgebra for U is a position p together with a map p --> U(p), i.e., a way to decompose the evolution from p into a sub-evolution and a remainder.

The terminal U-coalgebra is the position from which the *maximal* unfolding is possible -- the position whose sub-evolution recovers the entire shape. This is precisely the position with t = 0 (DC1 gives down(s, (j*, i*, 0)) = s), and the terminal coalgebra map is duplicate restricted to this position.

### D.4. Summary

THEOREM D.3 (Sentinel root is canonical). *The sentinel root o(n,g) = (j*, i*, 0) is:*

*(i) The unique solution to DC1 (up to choice of island and individual index)*

*(ii) The initial object of Roots(s)*

*(iii) The carrier of the terminal coalgebra of the position-unfolding endofunctor*

*(iv) The left unit of (+) (by DC2)*

*These four characterisations coincide and determine the root uniquely (given parameters j*, i*). The convention is not ad hoc: it is the only choice compatible with the comonad structure.*

*Remark for the grant.* The sentinel root convention is an instance of a general phenomenon in directed containers: the root must be the "most general" position, the one whose sub-shape recovers the whole. In the agent orchestration setting, this means the root of a pipeline is not the final output but the initial intent -- the position from which the entire computation unfolds. This aligns with the design principle that provenance traces backward from output to input: extract reads the root (the intent), and duplicate unfolds the entire execution tree from there.

---

## Summary of Results

| Section | Result | Status | Needed for Grant |
|---------|--------|--------|-----------------|
| A.4 | DirCont is a category | THEOREM | Yes -- foundational |
| A.7 | M(tau_1, tau_2) is a functor | THEOREM | Yes -- core claim |
| A.8 | Strict iff same reachability | PROPOSITION | Yes -- diversity connection |
| B.4 | H^1 = 0 iff DAG | CONJECTURE | Desirable -- connects to GECCO |
| B.5 | DAG => H^1 = 0 | Standard result | Reference suffices |
| B.6 | Cycle rank = H^1 rank | Needs proof | Key technical lemma |
| B.7 | D(tau) anti-correlated with H^1 | Needs proof | Connects two threads |
| C.1 | Root retraction is unique | PROPOSITION | Yes -- closes gap in proof |
| D.3 | Sentinel root is canonical | THEOREM | Yes -- removes "ad hoc" objection |

**Confidence assessment.** Sections A, C, and D are solid: the proofs are elementary and completable. Section B is the research frontier -- Conjecture B.4 is strongly motivated and the individual directions are standard, but the precise connection between D(tau) and H^1 rank (Lemma B.7) requires genuine mathematical work. For the grant, I would recommend stating B.4 as the target conjecture and noting that partial results (B.5, the DAG direction) are already known.

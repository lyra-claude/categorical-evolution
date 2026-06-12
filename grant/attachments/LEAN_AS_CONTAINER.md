# Lean as a Ghani Container

## Recap

A container is a pair (S, P) where S is shapes, P(s) is positions for each shape. The extension is F(X) = Sigma(s:S). P(s) -> X — a shape with data at each position.

## Typeclasses ARE Containers

```lean
class Group (G : Type*) where
  mul : G -> G -> G
  one : G
  inv : G -> G
  mul_assoc : ...
  one_mul : ...
  mul_inv_cancel : ...
```

**Shape** = the typeclass signature: how many operations, what their arities are, how many axioms. `Group` has shape (3 ops, 3 axioms) — 6 "slots."

**Positions** = the slots that need filling. For `Group`: two binary operations (mul, inv), one nullary (one), three proof obligations.

**Plugging in** = providing data at each position:

```lean
instance : Group Z where
  mul := (· + ·)        -- position 1: binary op
  one := 0              -- position 2: nullary op
  inv := Int.neg        -- position 3: unary op
  mul_assoc := ...      -- position 4: proof
  one_mul := ...        -- position 5: proof
  one_mul := ...        -- position 6: proof
```

F(X) = "Group structures on X" — for each type X, the set of ways to fill the positions with data from X and proofs about X.

## The Hierarchy = Container Morphisms

A container morphism maps shapes forward, positions backward. The algebraic hierarchy does exactly this:

```
Monoid --> Group --> CommGroup --> Ring --> Field
```

`Group` extends `Monoid` by adding positions (inv, mul_inv_cancel). The forgetful functor goes backward: it *forgets* positions (drops inv), mapping positions of Monoid back into the larger structure of Group. Forward on shapes, backward on positions — container morphism.

```lean
class Group (G : Type*) extends Monoid G where
  inv : G -> G                    -- new position
  mul_inv_cancel : ...            -- new position
```

`extends` is literally "add positions to the shape." The automatically generated `Group.toMonoid` is the backward map on positions.

## Bundled vs Unbundled: Two Container Perspectives

**Unbundled** — the container is a predicate:

```lean
class Group (G : Type*) where ...
-- F(X) = "Group structures on X"
-- Container over types, positions filled by operations + proofs
```

**Bundled** — the container is a category:

```lean
structure Grp where
  carrier : Type*
  group : Group carrier
-- Objects ARE filled containers
-- Morphisms = maps that respect the filling
```

Unbundled says "here's a shape, fill the positions." Bundled says "here's a collection of already-filled containers and the maps between them." The unbundled container lives in the type theory; the bundled one lives in category theory.

## instance search = Automatic Position-Filling

Lean's typeclass inference is the mechanism that propagates filled containers:

```lean
-- "Ring fills a Group container automatically"
instance [Ring R] : Group R := { ... }

-- Now any theorem about Group applies to any Ring
-- without the user re-plugging positions
```

This is **container composition**: if you've filled the Ring container, the Group positions are determined. Lean's instance search walks the morphism graph backward, filling positions automatically. The diamond problem (e.g., two paths from `Field` to `Monoid`) is a coherence condition on container morphisms.

## Hopf Algebras, Umbral Calculus, and Coalgebra Morphisms

Lean's `HopfAlgebra` typeclass is a container whose shape is (mul, comul, unit, counit, antipode) plus axioms — and what you plug in determines which branch of mathematics you're doing. Plug in rooted trees with admissible cuts as the coproduct and you get Connes-Kreimer, which computes Feynman diagram renormalization. Plug in the divided-power coalgebra on k[x] — where Delta(x) = x tensor 1 + 1 tensor x — and you get the Hopf algebra underlying Rota's umbral calculus. Same container, different fillings, completely different mathematics falling out.

The umbral calculus makes the container view especially vivid. The classical "umbral trick" — treating the index n in a polynomial sequence p_n(x) as if it were an exponent, so that formal manipulations with p^n magically yield correct identities — was mysterious for a century until Rota and Roman showed it works because of coalgebra morphisms. A sequence of binomial type is precisely a coalgebra morphism phi: k[x] -> k[x] that respects the coproduct Delta(p_n(x)) = Sum_{k} C(n,k) p_k(x) tensor p_{n-k}(x). The "umbral substitution" p_n |-> q_n is composition of coalgebra morphisms — and composition of coalgebra morphisms is itself a container morphism, mapping shapes forward (the polynomial sequence changes) and positions backward (the coproduct structure is preserved). In Lean, this would mean that any theorem proved generically about `CoalgHom k[x] k[x]` — the type of coalgebra endomorphisms — automatically applies to every Sheffer sequence, every Appell sequence, every instance of the umbral calculus, without re-proving anything. The container (Hopf algebra on k[x]) holds the shape; the coalgebra morphism is the specific filling; and what falls out is the classical umbral identity, now not a trick but a theorem about container morphisms.

## Comparison: Lean, Lisp, Ethereum

| | Lean | Lisp | Ethereum |
|---|---|---|---|
| Shape | Typeclass signature | Tree skeleton | Contract storage layout |
| Positions | Operations + proofs | Leaves (atoms) | Storage slots |
| Plugging in | `instance` declaration | Values at leaves | Constructor args / state |
| Morphisms | `extends` / forgetful functors | Macros | Contract calls |
| Code-is-data | Curry-Howard (proofs = programs) | S-expressions | Bytecode in state trie |
| Fixed point | `Expr` type (Lean metaprogramming) | Metacircular eval | No — EVM is external |
| Composition | Instance search (automatic) | Macro expansion | Message passing |
| Decidability | Kernel type-checking (decidable) | Undecidable | Gas-bounded (decidable) |

## Lean's Self-Reference: Partial Fixed Point

Lean has a metacircular quality that Ethereum lacks but that differs from Lisp's:

- **Lean's `Expr` type** represents Lean syntax within Lean — code-as-data via Curry-Howard
- **Tactics** are Lean programs that produce Lean proofs — the container manipulates its own fillings
- But the **kernel** is trusted and external — it sits outside the system it verifies

So Lean occupies a middle position: more self-referential than Ethereum (tactics can construct proofs programmatically), less than Lisp (the kernel isn't written in the object language). This is deliberate — a fully self-hosting proof checker would undermine soundness (Goedel). Lean's container is **stratified**: the object-level container (typeclasses) is verified by a meta-level container (the kernel) that deliberately refuses to contain itself.

## The Payoff: Why "Container" Matters Here

The container view reveals what `instance` actually does: it's not just "registering" a type as a group. It's **filling positions in a shape**, and the shape determines what theorems flow downstream. The entire Mathlib library is a network of container morphisms — 200+ typeclasses connected by `extends` — and instance search is the algorithm that composes these morphisms to propagate structure automatically.

When you write `instance : HopfAlgebra Q CK_Trees`, you're not putting trees in a box. You're connecting one container (rooted trees with cuts) to another (Hopf axioms) via a specific filling — and that filling is the mathematical content of Connes-Kreimer's theorem.

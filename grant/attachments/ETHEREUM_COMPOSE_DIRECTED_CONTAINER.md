# Composing the Ethereum Directed Container

## Ethereum as a Directed Container

A single Ethereum transaction is a directed container:

```
S = TxShape                       -- valid transaction DAG shapes (the call tree)
P s = CallPositions s             -- internal calls and events in the tree
o s = rootCall s                  -- the top-level external call
down s p = subCallTree s p        -- the sub-execution rooted at internal call p
p + q = embedCall s p q           -- position q in sub-tree at p, mapped to global

extract tx = tx ! rootCall        -- the final return value (success/revert + return data)
duplicate tx = at each call p,    -- the full sub-execution context at that call
               replace value       (the internal transaction receipt)
               with subCallTree(p)
```

`extract` gives the result: did the transaction succeed? What was returned?

`duplicate` gives the full context at every internal call — which contract called which, with what arguments, returning what. This is the **transaction receipt**. Ethereum doesn't just give you the answer; it gives you the entire execution tree. That's `duplicate`.

The comonad laws guarantee:
1. `extract . duplicate = id` — the root of the execution tree is consistent with the final result
2. `fmap extract . duplicate = id` — extracting at every node recovers the original values
3. `duplicate . duplicate = fmap duplicate . duplicate` — zooming into a sub-call's execution tree gives the same result as zooming into the global tree and restricting

---

## Composing Across Contracts: The DeFi Example

A DeFi transaction composes three contracts:

```
User sends 1 ETH
    │
    ▼
Uniswap.swap(ETH → USDC)          Contract 1: (S₁, P₁, o₁, down₁, +₁)
    │
    │ 1,847 USDC
    ▼
Aave.deposit(USDC, collateral)     Contract 2: (S₂, P₂, o₂, down₂, +₂)
    │
    │ collateral position
    ▼
Aave.borrow(DAI, 1000)             Contract 3: (S₃, P₃, o₃, down₃, +₃)
    │
    │ 1,000 DAI
    ▼
User receives 1,000 DAI
```

Each contract is a directed container. The composition is a directed container.

### The Composite Directed Container

```
S = CompositeShape                 -- the full call tree across all three contracts
P s = AllCallPositions s           -- every internal call in all three contracts
o s = rootCall s                   -- the user's original transaction
down s p = ...                     -- depends on which contract p belongs to
```

The `down` operation is where composition gets interesting:

```
down s p =
  if p is in Uniswap:    down₁ restricted to Uniswap's sub-tree
  if p is in Aave.deposit: down₂, but the "collateral amount" position
                           traces BACK through Uniswap's output
  if p is in Aave.borrow:  down₃, but the "health factor" position
                           traces BACK through Aave.deposit's output
                           which traces BACK through Uniswap's output
```

The `down` at an Aave position doesn't just look at Aave's local sub-tree. It follows the chain backward through every contract that contributed to its input. This is the **backward map composing across trust boundaries**.

### What `duplicate` Gives You

```
duplicate composite_tx =

  At Uniswap.swap:
    sub-tree = {pool reserves, price oracle, fee calculation, LP mint/burn}
    every position carries its local provenance

  At Aave.deposit:
    sub-tree = {collateral validation, aToken mint, health factor update}
    the "amount" position traces to Uniswap's output
    which traces to the pool reserves and price oracle

  At Aave.borrow:
    sub-tree = {health factor check, debt token mint, DAI transfer}
    the "collateral" position traces through Aave.deposit
    through Uniswap.swap
    to the original 1 ETH
```

A liquidator asking "why was this loan issued?" gets the composite `duplicate`: the borrow traces to the collateral, which traces to the swap, which traces to the original ETH and the pool state at the time. End-to-end provenance from composing directed containers.

### Atomic Revert = Composition Failure

If any contract in the chain fails, the EVM reverts the entire transaction. In directed container terms:

The composite morphism f₃ ∘ f₂ ∘ f₁ is **total or nothing**. If f₂ (Aave deposit) rejects the collateral, the composite morphism is undefined — so f₁ (Uniswap swap) also reverts. The execution tree collapses to a single node: "reverted."

This is the directed container equivalent of DD-05: invalid outputs never bypass the core. In Ethereum, a partially executed transaction cannot exist. The comonad structure prevents it — `extract` on a partially composed structure would violate the comonad laws.

---

## Side by Side

| | Tax DAG | Ethereum | AI Mathematician |
|---|---|---|---|
| Shape | Rule application tree | Contract call tree | Agent pipeline DAG |
| Root `o` | Final tax liability | Top-level transaction | Published claim |
| `down` at p | Sub-computation producing p | Sub-execution in contract at p | Sub-task assigned to agent at p |
| `extract` | The final figure | Success/revert + return data | The theorem/conjecture |
| `duplicate` | Audit trail at every node | Transaction receipt at every call | Provenance at every agent handoff |
| Composition failure | Validation error → figure rejected | Revert → entire tx undone | Lean rejects → claim doesn't enter trusted core |
| Comonad law 1 | Trail consistent with final figure | Receipt consistent with result | Provenance consistent with published claim |

---

## The Punchline

Ethereum's transaction receipts aren't a feature the EVM engineers decided to add. They are `duplicate` — the comonad operation on the directed container of contract call trees. The receipt *must* exist because the directed container structure *requires* it.

Similarly, the AI Mathematician's provenance chain isn't a feature we propose to add. It is `duplicate` on the directed container of agent pipelines. If the pipeline is a directed container, provenance is structurally unavoidable.

Ethereum proved this works at scale — billions of dollars flow through composed directed containers every day. The mathematics is the same. The domain is different.

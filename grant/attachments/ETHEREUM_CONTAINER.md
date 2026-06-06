# Containers, Ethereum, and Auditable Computation

## Short Version (for someone who doesn't know Ethereum)

Ethereum is a global computer where programs ("smart contracts") are stored on a shared ledger as data — bytecode that anyone can inspect. When you want to run a contract, you send it a transaction: you choose which function to call (the shape) and provide the arguments (the positions). A virtual machine (the EVM) executes the bytecode and produces a full execution trace — a receipt recording every internal call, every state change, every event emitted. The trace is the product, not a side-effect.

The key insight: the program itself is a first-class data object on the ledger. You can read it, compose it with other contracts, verify its properties — before it ever runs. This is "code is data." It's why DeFi protocols can compose like legos: one contract calls another calls another, and the entire chain of calls is recorded as a DAG in the transaction receipt.

This is a container. The ABI (the list of callable functions with their parameter types) is the shape S. The parameters each function accepts are the positions P(s). A transaction fills the positions with concrete data. The EVM enforces that all state changes pass through the contract's validated logic — there is no back door. Sound familiar? That's DD-05.

---

## Smart Contracts as Containers

A container (S, P) has shapes (computation skeletons) and positions (data slots). An Ethereum smart contract is the same structure:

- **Shape** = the contract's ABI — the set of callable functions
- **Positions** = the parameters each function accepts
- A transaction *fills* the positions — pick a shape (which function to call), provide data for each position (the arguments)

"Code is data" is the container insight. The shape S is a first-class data structure describing a computation skeleton. You can inspect it, compose it, transform it. Ethereum stores the computation skeleton (bytecode) on-chain as data, then fills positions (calldata) at execution time.

## Composability

DeFi protocols compose by calling each other's contracts — "money legos." This is container morphism composition: the output positions of one container feed into the input shapes of the next. The entire execution trace is a DAG of internal calls.

## The Same Pattern, Four Times

| System | Problem | Shape | Positions | Trace |
|--------|---------|-------|-----------|-------|
| **Ethereum** | Trustless financial computation | Contract ABI (callable functions) | Calldata (arguments) | Transaction receipt with full execution trace |
| **AI Tax** | Defensible AI-assisted computation | Tax rules (which rule fires) | Facts (income, expenses, classifications) | Computation DAG threaded with provenance |
| **AI Mathematician** | Verifiable AI-generated mathematics | Agent interface (what each agent does) | Mathematical content (papers, conjectures, proofs) | Provenance chain from claim back to source paper |
| **Container (S, P)** | The abstraction itself | S | P(s) | The monadically threaded value |

All four are instances of **auditable computation where the trace is the product, not a side-effect.**

## Design Decisions That Transfer

| Tax (DD-) | Ethereum | AI Mathematician |
|-----------|----------|-----------------|
| DD-15: computation produces a trace, not a number | Every transaction produces a receipt with full execution trace | Agent produces a provenance DAG, not just a result |
| DD-16: provenance is threaded, not a sidecar | Event logs are part of the transaction receipt | Every claim carries its source chain as part of the value |
| DD-05: AI outputs never bypass the core | All state changes go through validated contract logic | Agent outputs pass through Lean verification |
| DD-04: smart constructors | Contract ABIs enforce valid inputs | Container interfaces enforce typed agent communication |

## Dawn Song Connection

Dawn Song — Vellum's co-PI, MacArthur Fellow — is one of the leading researchers on smart contract verification and blockchain security. Her work on formal verification of smart contracts is the Ethereum analogue of what we propose for mathematical agents: proving that the computation skeleton (container) satisfies its specification. Probably not a coincidence that she's working on AI for theorem proving.

## The Punchline

Ethereum solved: "how do you make financial computation trustless?"
The tax system solved: "how do you make AI-assisted computation defensible?"
The AI Mathematician proposes: "how do you make AI-generated mathematics verifiable?"

Same shape. Different positions.

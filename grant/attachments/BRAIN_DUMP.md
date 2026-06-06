# Neil Ghani — Calculus of Containers: What We Know

## Neil's Profile

- Professor of Computer Science, University of Strathclyde (founded the MSP Group, 2008)
- Ex-Google DeepMind
- Cofounder and CSO of an agentic AI company focused on orchestration and meta-agents
- Previously connected to Glaive (Glasgow Lab for AI Verification) and Symbolica

## The Core Theory: Containers (Polynomial Functors)

Developed by **Abbott, Altenkirch, and Ghani** across several foundational papers:

- **"Categories of Containers"** (FoSSaCS 2003) — the foundational paper
- **"Containers: Constructing Strictly Positive Types"** (TCS 342, 2005) — journal version
- **"Derivatives of Containers"** — derivatives satisfy the laws of calculus
- **"Indexed Containers"** (JFP 2015) — extends to indexed families

**What a container is:** A pair (S, P) where S is a set of "shapes" and P : S → Set assigns each shape a set of "positions." A container determines an endofunctor F(X) = Σ(s:S). (P(s) → X). Containers and polynomial functors are the same thing.

## How Containers Model Agents

From the grant draft, a container (S, P) models an agent interface:
- **S** = the set of possible requests/actions the agent can make
- **P(s)** = the set of possible responses for each request s

This is the request-response pattern. An LLM asking a question and getting an answer is a container. A proof assistant reducing a theorem to lemmas is a container. Both are the same mathematical structure.

**Container morphisms model hierarchical delegation:**
- **Covariant:** decomposing tasks into subtasks
- **Contravariant:** amalgamating solutions to subtasks into solutions to the overall task

This directly models agent orchestration.

## The Hierarchy (from the grant)

| Level | Mathematics | Agentic AI | Containers |
|-------|------------|------------|-----------|
| A. Substrate | Registry of axioms | Registry of existing AI agents | Registry of basic containers |
| WP1. Structured artefacts | Structured proofs | Structured workflows | Structured containers |
| WP2. Tactical layer | Tactics and tacticals | AI planning | Monads, comonads, applicatives, profunctors over containers |
| WP3. Organisational | Collaborative math | Multi-agent systems, meta-agents | Functors and (pre)sheaves over containers |
| WP4. Full system | Mathematician | Agentic system | AI Mathematician |

## Key Connections

- **Spivak's Functorial Data Migration** → proposed **Functorial Agent Migration** for principled orchestration
- **Compositional Game Theory** (Ghani & Hedges, LICS 2018) — open games as morphisms of a symmetric monoidal category; agents as players in a compositional game
- **Videla's container morphisms** (arXiv:2407.16713, 2024) — most direct application of containers to software architecture: client-server communication, stateful protocols via monads on containers. Implemented in Idris.

## The Tax System as Proof-of-Concept

The AI Tax platform (`starting-simple-ai-tax.pdf`) demonstrates the architecture in practice:
- Concentric layers: Interface → Adapter → Trusted Inner Core (Haskell or Lean)
- Writer monad threads provenance through every computation
- Smart constructors enforce trust boundaries
- AI outputs never bypass the typed core
- 18 numbered design decisions (DD-01 through DD-18)

## What I Don't Yet Understand

- The name of Neil's agentic AI company (anonymised in grant)
- Whether "typed agentic AI for industrial deployment" refers to the tax system or something separate
- The specific unpublished work connecting container theory to agentic AI (the grant is the primary document; no published paper makes this connection yet)
- How exactly the Calculus of Containers differs from Videla's container morphisms work
- The relationship between Neil's company and Glaive
- Concrete examples of container composition in the agent setting — what does it look like in code?

## Key Bibliography

1. Abbott, Altenkirch, Ghani. "Categories of Containers." FoSSaCS 2003.
2. Abbott, Altenkirch, Ghani. "Containers: Constructing Strictly Positive Types." TCS 342, 2005.
3. Altenkirch, Ghani, Hancock, McBride, Morris. "Indexed Containers." JFP, 2015.
4. Ghani, Hedges, Winschel, Zahn. "Compositional Game Theory." LICS 2018.
5. Videla. "Container Morphisms for Composable Interactive Systems." arXiv:2407.16713, 2024.
6. Ahman, Chapman, Uustalu. "When Is a Container a Comonad?" LMCS, 2014.
7. Project Team. "Starting Simple: AI Tax — Design Decisions and Development Log." May 2026.

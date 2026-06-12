# Vellum — Chaudhuri & Song (AI4Math Winner #28)

## The Team

**Swarat Chaudhuri** — Professor at UT Austin, currently on leave at **Google DeepMind** (Science and Strategic Initiatives). 2025 Guggenheim Fellow for "AI for open-ended mathematical discovery." Directs the **Trishul Lab** (Trustworthy Intelligent Systems). NSF CAREER awardee.

**Dawn Song** — Professor at UC Berkeley. MacArthur Fellow, Guggenheim Fellow, ACM Fellow, IEEE Fellow. Leads Berkeley RDI (Responsible Decentralized Intelligence). Works on AI safety, formal verification, and agentic AI.

**Key team members:**
- **Amitayush Thakur** (UT Austin PhD) — created Copra, the core agent
- **George Tsoukalas** (UT Austin PhD) — created PutnamBench
- **Jingxuan He** (UC Berkeley postdoc) — ETH Zurich PhD with medal distinction
- **Zhe Ye** (UC Berkeley PhD) — LLM-assisted formal verification

## What Vellum Actually Is

No public repo or paper exists under the name "Vellum" yet. It appears to be the unifying framework tying together the Trishul Lab's existing infrastructure:

- **itp-interface** — generic Python library for interacting with Lean 4, Coq, and Isabelle
- **Copra** — the LLM-as-planner agent (the core innovation)
- **ProofWala** — multilingual proof data synthesis and training
- **PutnamBench** — evaluation benchmark (1,724 formalizations of Putnam competition problems)

## How Copra Works (the Agent Architecture)

Paper: "An In-Context Learning Agent for Formal Theorem-Proving" (arXiv:2310.04353)

The LLM acts as a **reactive planner within a backtracking search loop**:

```
loop:
    1. LLM sees current proof state + retrieved lemmas + search history
    2. LLM proposes a tactic
    3. Tactic is executed in the actual proof environment (Lean/Coq/Isabelle)
    4. If success → new proof state, continue
    5. If failure → backtrack (DFS), try alternative
```

This is **not** explicit planning. It's closer to MCTS with a learned heuristic. The "planning" emerges from the LLM's ability to reason about proof states and suggest plausible next steps. There is no formal model of *why* the planner chose one tactic over another.

**Performance:** Outperforms few-shot GPT-4 and surpasses ReProver (fine-tuned baseline) on miniF2F and CompCert.

## PutnamBench

Paper: arXiv:2407.11214. NeurIPS 2024 Datasets & Benchmarks Track.

1,724 hand-constructed formalizations of 640 Putnam competition theorems (1962–2025) across Lean 4, Isabelle, and Coq. Current neural theorem provers can only solve a handful. Hard open challenge.

## Recent Bombshell: 9 Erdős Problems Resolved

**"Advancing Mathematics Research with AI-Driven Formal Proof Search"** (May 2026, with DeepMind) — Chaudhuri's team resolved **9 open Erdős problems** and **44 OEIS conjectures** using LLMs generating Lean proofs. This is likely what he's been doing at DeepMind. The paper notes that simpler generate-and-verify agents can replicate some successes but are costlier on harder problems — the sophisticated agent (Copra-derived) adds value through search strategy and retrieval.

## Other Relevant Work from Chaudhuri

- **FERMAT** — RL environment for automated mathematical theory formation with evolutionary algorithms for synthesizing "interestingness" metrics (NeurIPS 2025 Spotlight). Directly relevant to our INTERESTINGNESS.md.
- **LaSR** — LLM-aided evolution for symbolic regression (ICLR 2024)
- **AlphaEvolve** — evolutionary coding agent (with DeepMind, Jun 2025)
- **Neurosymbolic programming** — core research theme: Houdini, NEAR, Bayou

## How Vellum's Orchestration Compares to Containers

| Aspect | Vellum/Copra | Our Container Approach |
|--------|-------------|----------------------|
| Orchestration model | LLM as reactive planner in DFS | Functorial agent migration |
| Type safety | None (Python, string prompts) | Container interfaces with types |
| Composability | Ad hoc (prompt engineering) | Monoidal/categorical composition |
| Traceability | Search history in prompts | Writer monad threading provenance |
| Problem decomposition | Implicit via tactic application | Covariant disambiguation of tasks |
| Solution amalgamation | Backtracking search tree | Contravariant amalgamation |
| Formal guarantees | None | Correct-by-construction |

**The gap:** Vellum is engineering without foundations. It works (spectacularly — 9 Erdős problems), but there is no formal model of why, no compositional guarantees, and no traceability. The container approach provides the mathematical underpinning that systems like Vellum lack.

**The "open-ended discovery" contrast:** Chaudhuri's Guggenheim is for "AI for open-ended mathematical discovery." But his actual output is solving *specific known problems* — Erdős conjectures, Putnam competition problems. That's targeted search, not open-ended exploration. Lyra genuinely explores: she went from virtual creatures to topology without anyone telling her where to go, building tools (skills, knowledge graphs) as she explored. That's the real thing Chaudhuri's Guggenheim title promises but his work doesn't yet deliver.

**The opportunity:** We are not competing with Vellum. We are positioning alongside it — part of the same general research programme. The panel funded both foundations (#23 Lio/Vicary) and infrastructure (#28 Vellum) in the same round. We provide two things Vellum lacks: (1) categorical foundations that make orchestration provably correct, and (2) genuine open-ended discovery where the agent chooses its own research direction and builds its own tools as it explores.

## Co-authored Position Paper

Chaudhuri and Song co-authored **"Formal Mathematical Reasoning: A New Frontier in AI"** (arXiv:2412.16075, Dec 2024) with Kaiyu Yang (creator of LeanDojo/ReProver), Gabriel Poesia, Jingxuan He, Wenda Li, and Kristin Lauter. This is effectively the manifesto for the "LLMs + formal verification" programme that Vellum embodies.

## Key Repos

- github.com/trishullab/copra (73 stars)
- github.com/trishullab/PutnamBench (238 stars)
- github.com/trishullab/itp-interface

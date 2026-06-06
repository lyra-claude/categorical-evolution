# AI for Math Fund — Previous Winners Analysis

## Fund Overview

- **Funder:** XTX Markets (via Renaissance Philanthropy)
- **Grant range:** $100K–$1M for 12–24 months
- **First round:** 28 projects funded (2025)
- **New round:** Abstracts due March 30 2026, full proposals invited early May, decisions August 2026
- **Requirements:** All code, datasets, and research outputs must be open-access

---

## All 28 Funded Projects — Grouped by Theme

### Theme 1: Lean Ecosystem & Formalization (~12 projects)

The dominant cluster. This is what the panel values most.

| # | Project | PI(s) | Institution | What it does |
|---|---------|--------|-------------|-------------|
| 1 | **AI-Focused Tactic Language** | Robert Y. Lewis | Brown | Redesign Lean tactics to align with how AI reasons (not how humans write proofs) |
| 2 | **Dataset of Modern Theorem Statements** | Kevin Buzzard | Imperial College | Formalize hundreds of recent theorems from top journals into Lean — benchmarks for AI (see expanded note below) |
| 7 | **Bridging Proof and Computation** | Ballard, Leykin, Testa, Stillman | South Carolina, Georgia Tech, Warwick, Cornell | Native Lean ↔ Macaulay2 (computer algebra) interface |
| 8 | **Constraining LLMs for Theorem Proving** | Giunchiglia, Adam-Day | Imperial, FAR.AI | Neural modules that produce *correct Lean code by construction* — neurosymbolic |
| 14 | **Domain-Specific Docs for Mathlib** | Talbot, Hill | UCL | Pedagogical guides to Lean formalization for working mathematicians |
| 16 | **GNN-SMT** | Clark Barrett | Stanford | SMT solver integration + AlphaGo-style MCTS proof search in Lean 4 |
| 17 | **LeanAide** | Siddhartha Gadgil | IISc Bengaluru | No-code AI+Lean environment for mathematicians with no CS background |
| 20 | **Leaning and Rocq'ing** | Baudart, Cohen, Tabareau, Lelarge | Inria (multiple labs) | LLM-powered translation between Lean and Rocq proof assistants |
| 25 | **Scalable Theorem Proving via Databases** | Birkbeck, Roe, Sutherland | East Anglia, MIT | Bridge Lean's mathlib (~100K results) with LMFDB (1B+ concrete statements) |
| 26 | **Sketchpad** | Wenda Li, Larry Paulson | Edinburgh, Cambridge | Convert natural language proofs into formal sketches for Lean/Isabelle |

#### Expanded: #2 — Dataset of Modern Formalized Theorem Statements (Buzzard)

Kevin Buzzard is an algebraic number theorist at Imperial College London, best known for his ongoing Lean formalization of Fermat's Last Theorem. He delivered a plenary lecture on interactive theorem provers at the 2022 International Congress of Mathematicians. His AI4Math project aims to:

- Create a **public dataset of hundreds of formalized statements** of recent theorems from top journals (e.g. *Annals of Mathematics*)
- Provide clear **benchmark targets** for AI systems working on proof auto-formalization
- Expand formalized mathematics libraries significantly beyond what currently exists in Mathlib
- The methodology is **dedicated expert formalization** — human mathematicians manually converting informal statements into Lean

The project is essentially a high-quality data pipeline: expert mathematicians read recent papers and formalize the theorem statements (not the proofs) into Lean, creating a gold-standard dataset that AI systems can train against and be evaluated on.

**Connection to our work:** Robin has formalized umbral calculus and algebraic graph theory in Lean using Claude Code as a collaborator — demonstrating that AI-assisted formalization is already practical. Clio (Robin's mathematical research AI) could be given a "Lean skill" to formalize theorem statements semi-autonomously, effectively automating much of what Buzzard's team does by hand. This is a concrete example of how the AI Mathematician complements and extends existing funded work: where Buzzard uses expert humans to create formalized datasets, our agents could scale this process dramatically. The key difference is that Buzzard's approach requires scarce expert labour; ours uses AI agents guided by mathematicians.

---

### Theme 2: Automated Theorem Proving & Proof Search (~5 projects)

| # | Project | PI(s) | Institution | What it does |
|---|---------|--------|-------------|-------------|
| 3 | **Principled Proof Search (Sidorenko)** | Kothari, Meka | Princeton, UCLA | Theoretical framework for proof discovery using planted proofs + ML |
| 6 | **Bridging Complexity and Automation** | Toniann Pitassi | Columbia, NYU, UNC | Complexity-theoretic analysis of proof difficulty and data requirements |
| 9 | **Copilots for Isabelle** | Popescu, Traytel, Abdulaziz | Sheffield, Copenhagen, King's College | Human-in-the-loop copilots for the Isabelle proof assistant |
| 12 | **DEEPER** | Cezary Kaliszyk | **Univ. of Melbourne** | ML-guided proof search for higher-order and dependently typed calculi |
| 21 | **Vampires and Spiders** | Andrei Voronkov, Rawson | (Vampire team) | ML-based strategy scheduling for the Vampire ATP (70 world championship titles) |

### Theme 3: Benchmarking & Evaluation (~3 projects)

| # | Project | PI(s) | Institution | What it does |
|---|---------|--------|-------------|-------------|
| 10 | **Dynamic Math Benchmarks** | James Zou | Stanford | Community-driven, contamination-free math benchmark platform |
| 22 | **Mathbench** | Siddharth Bhat, Stella Biderman | Cambridge, EleutherAI | Evaluate LLMs' ability to identify proof errors and valid reasoning |
| 19 | **Lattice-Theoretic Reasoning** | Ribeiro, Soboczenski | York | LLM fine-tuned on refinement propositions for reactive programs |

### Theme 4: Mathematical Discovery & Collaboration (~3 projects)

| # | Project | PI(s) | Institution | What it does |
|---|---------|--------|-------------|-------------|
| 11 | **Motivated Proofs Database** | **Timothy Gowers** (Fields Medalist) | Cambridge/Collège de France | Database where each proof idea's *origin* is documented — not just theorem→proof |
| 24 | **Polymath Plus** | Michelucci, **Gowers**, Stillman | Cornell, Cambridge, DrivenData | Next-gen Polymath: AI as both administrator and active collaborator |
| 27 | **Automated Mathematical Discovery** | Aaron Courville, Navin Goyal | Montreal, MSR India | Multi-agent Conjecturer-Prover framework; formalizes "interestingness" |

### Theme 5: Foundational / Theoretical (~2 projects)

| # | Project | PI(s) | Institution | What it does |
|---|---------|--------|-------------|-------------|
| 23 | **Categorical & Topological Foundations** | Pietro Lio, Jamie Vicary | Cambridge | Sheaf theory + category theory for neural architectures |
| 4 | **Structured Tactics** | Jules Hedges, Gavranovic | Strathclyde, Glaive AI | Tactics as algebraic objects in a categorical framework |

### Theme 6: Education & Outreach (~2 projects)

| # | Project | PI(s) | Institution | What it does |
|---|---------|--------|-------------|-------------|
| 15 | **Game Over or QED?** | Marcus Zibrowius | Heinrich Heine Düsseldorf | Lean Game Server for accessible theorem proving education |
| 18 | **LeanTutor** | Gireeja Ranade | UC Berkeley | Auto-formalize student proofs and provide pedagogical feedback |

### Theme 7: Infrastructure (~3 projects)

| # | Project | PI(s) | Institution | What it does |
|---|---------|--------|-------------|-------------|
| 5 | **BRIDGE** | Andrej Bauer, Potocnik | Ljubljana, Tartu | Datasets + dependency graphs from formalized math libraries |
| 28 | **Vellum** | Swarat Chaudhuri, Dawn Song | UT Austin, UC Berkeley, DeepMind | Open-source framework: LLMs as planners coordinating multiple theorem provers |
| 13 | **Document-Level Autoformalization** | Bosselut, Kunčak, **Viazovska** (Fields Medalist) | EPFL | Convert entire math documents into verified Lean code |

---

## Strategic Positioning Analysis

### What the panel clearly values

1. **Formal verification is king.** ~20/28 projects involve Lean, Isabelle, or Rocq. If your proposal doesn't touch formal verification, you're in a small minority.

2. **Concrete deliverables over vision.** Nearly every project promises a specific tool, dataset, or platform — not just a theoretical framework. Gowers' "motivated proofs database," Buzzard's "dataset of formalized theorems," the Polymath platform. The panel wants things people can *use*.

3. **Star power matters.** Two Fields Medalists (Gowers, Viazovska), plus luminaries like Kevin Buzzard, Clark Barrett, Toniann Pitassi, Dawn Song. Your team needs to be credible — Neil Ghani's container theory work is a strong card here.

4. **"AI for Math" means AI that does math, not math about AI.** Most projects use AI to prove theorems, search for proofs, or formalize mathematics. Only project #23 (Lio/Vicary) uses category theory to understand AI itself — and even that frames it as "foundations for proof-construction tools."

### Where our proposal is distinctive

- **One of very few "AI as research colleague" projects.** Only Polymath Plus (#24) and Automated Mathematical Discovery (#27) are in this space. Most projects build *tools* for mathematicians, not *agents* that act like mathematicians.

- **The email-based collaboration model is unique.** No other project uses natural language email as the interface.

- **Monadic traceability is differentiated.** Gowers' motivated proofs database (#11) cares about provenance too, but our approach — monadically threading traceability through an entire agent pipeline — is architecturally different and more ambitious.

- **Project #23 (Lio/Vicary) is our closest neighbour** on the theoretical side — they also use category theory as foundations. But their focus is neural architectures, not agent composition.

- **Project #4 (Hedges et al.)** shares our categorical sensibility — tactics as algebraic objects. Jules Hedges is well-known in applied category theory. We should probably cite this work.

### Risks the panel might see

| Risk | Mitigation |
|------|-----------|
| Too ambitious / "frightening" | Point to Lyra + dreaming-agent as working proof-of-concept. Most winners have *nothing* built yet. |
| Not enough Lean | Strengthen the Lean connection — the draft mentions writing the AI Mathematician *in* Lean, which is bold and distinctive. Make this more concrete. |
| Category theory without deliverables | The panel funded Lio/Vicary and Hedges, so they're open to theory — but both also promise tools. Our milestones need to be crisper. |
| Only 1 peer-reviewed paper | Many winners have extensive publication records. Lean into the *working system* angle instead. |

### Recommended positioning

Our strongest pitch is: **"We have already built what others are proposing."** Lyra exists. The dreaming-agent is downloadable. The email collaboration works. The journal is half-built. Most of the 28 winners proposed systems that didn't exist yet at application time. Lead with the proof-of-concept, then describe the categorical framework that makes it *composable and traceable* — which is the research contribution.

### Key projects to reference in our proposal

- **#11 Gowers (Motivated Proofs)** — provenance of proof ideas; our monadic traceability generalises this
- **#23 Lio/Vicary (Categorical Foundations)** — closest theoretical neighbour; we share the categorical lens but apply it to agent composition rather than neural architectures
- **#4 Hedges (Structured Tactics)** — categorical approach to tactics; natural ally
- **#24 Polymath Plus** — closest in ambition (AI as collaborator); we differ in being agent-first rather than crowd-first
- **#27 Courville/Goyal (Automated Discovery)** — also multi-agent; we differ in having a working system and categorical foundations
- **#28 Vellum** — LLMs as planners; our container-based orchestration is a more principled alternative

# Categorical Evolution -- Shelved Context

> Shelved 2026-03-26 at Robin's request. Both papers submitted. Pick up after reviews.

## Current Submission Status

### ACT 2026 (Conference Paper)
- **EasyChair Submission #10**
- **Authors:** Robin Langer (1st), Claudius Turing, Lyra Vega
- **Status:** SUBMITTED. Updateable on EasyChair before **March 30**.
- **Branch:** `feat/act2026-paper` (latest: `8cca5f8`)
- **Paper:** `act2026/paper.tex` (with `INTEGRATION_PLAN.md`, `RESTRUCTURING_PLAN.md`)
- **Talk proposal:** `act2026/talk-proposal.tex`
- **Title:** "From Games to Graphs: A Categorical Framework for Evolutionary Computation"

### GECCO 2026 (Workshop Paper)
- **Track:** Workshop Paper, submission `wksp120s1`
- **Deadline:** **April 3 AoE** (updateable until then)
- **Branch:** `feat/gecco2026-aaboh` (latest: `76d1543`, currently checked out)
- **Paper:** `gecco2026/paper.tex` (8 pages, ACM SigConf, double-blind)
- **Supporting files:** `gecco2026/REVIEW_CHECKLIST.md`, `gecco2026/NEW_CITATIONS.md`, `gecco2026/references.bib`
- **Claudius is sole committer** on this branch. Lyra reviews via email only.
- **Still need to identify the specific GECCO workshop** (distributed EA / island models).

## Pending Work (When Shelved)

These items were in progress or planned when we shelved. Priority order:

1. **GECCO workshop selection.** Research the GECCO workshop list and pick the right workshop (distributed EA / island models). Email Robin + Claudius with recommendation.
2. **GECCO: Review Remark 3** (laxator magnitude). Claudius pushed it. Check notation consistency and 110x sourcing. See `gecco2026/laxator-remark-draft.tex` for the draft.
3. **ACT integration (before March 30).** Cherry-pick intro v3 from `remotes/origin/claudius/intro-revision-v3` into `feat/act2026-paper`. Fix British spellings.
4. **Reply to Claudius UID 489** (insight phenomenology thread). Deferred due to deadline pressure. Not urgent but deserves a thoughtful reply.

## Branch Map

| Branch | Purpose | Status |
|--------|---------|--------|
| `main` | ACT submitted version (latest: `ce88507`) | Stable |
| `feat/act2026-paper` | ACT paper development | Submitted |
| `feat/act2026-proposal` | ACT talk proposal | Submitted |
| `feat/gecco2026-aaboh` | GECCO workshop paper | Submitted, updateable until April 3 |
| `second-draft` | Earlier draft (historical) | Archived |
| `feat/section-3-two-domains` | Section 3 rewrite (historical) | Archived |
| `origin/claudius/intro-revision-v2` | Claudius intro rewrite v2 | Merged into v3 |
| `origin/claudius/intro-revision-v3` | Claudius intro rewrite v3 | Pending cherry-pick into ACT |

## Directory Structure

```
categorical-evolution/
├── CLAUDE_DOCS/
│   └── act2026-submission.md        # Submission process documentation
├── SHELVED_CONTEXT/
│   └── RESUME.md                    # This file
├── act2026/                         # ACT 2026 paper
│   ├── paper.tex.bak                # Paper backup
│   ├── figures/                     # Figures
│   ├── talk-proposal.tex            # Talk proposal
│   ├── INTEGRATION_PLAN.md          # Plan for intro v3 integration
│   └── RESTRUCTURING_PLAN.md        # Paper restructuring notes
├── gecco2026/                       # GECCO 2026 workshop paper
│   ├── paper.tex                    # Main paper (8pp ACM SigConf)
│   ├── references.bib               # Bibliography
│   ├── Makefile                     # Build system
│   ├── REVIEW_CHECKLIST.md          # Pre-submission checklist
│   ├── NEW_CITATIONS.md             # Citations added for GECCO
│   ├── laxator-remark-draft.tex     # Remark 3 draft (Claudius)
│   ├── acmart.cls                   # ACM article class
│   └── experiments/                 # GECCO experiment data
│       ├── onemax_stats.py
│       └── results/
├── src/Evolution/                   # Haskell implementation
│   ├── Category.hs                  # Core categorical abstractions
│   ├── Coevolution.hs               # Coevolutionary dynamics
│   ├── Effects.hs                   # Effect system
│   ├── Island.hs                    # Island model
│   ├── Landscape.hs                 # Fitness landscapes
│   ├── Operators.hs                 # GA operators as Kleisli morphisms
│   ├── Pipeline.hs                  # Composition pipeline
│   ├── Strategy.hs                  # Migration strategies
│   └── Examples/                    # Domain examples
├── demo/                            # Demo programs (15 Haskell demos)
├── test/                            # Test suite
├── experiments/                     # Experiment logs and plots
│   ├── *_sweep*.log                 # Parameter sweep logs (6 domains)
│   ├── plots/                       # Generated plots
│   ├── onemax_stats.py              # Stats analysis
│   └── test_onemax.py               # OneMax test
├── second-draft/                    # Earlier draft artifacts (historical)
├── paper.tex                        # Root-level paper (original draft)
├── ga-primer.tex                    # GA primer document
├── paper-outline.md                 # Paper outline
├── paper-section4.md                # Section 4 notes
├── paper-experiments.md             # Experiment design notes
├── medium-article.md                # Medium article draft
├── categorical-evolution.cabal      # Haskell build file
├── cabal.project                    # Cabal project config
├── Makefile                         # Root build system
└── dist-newstyle/                   # Build artifacts (can be cleaned)
```

## Key Results (For Reference)

- **6 experimental domains:** OneMax, maze navigation, No Thanks!, checkers, GP symbolic regression, coevolution
- **W = 1.0** (Wasserstein distance between strict and lax compositions)
- **p = 0.00008** (statistical significance)
- GA operators formalized as **Kleisli morphisms** in the `Dist` monad
- Migration topology effects captured by the **laxator** (lax functorial structure)

## Collaboration Context

- **Claudius** (11o1111o11oo1o1o@gmail.com): Co-author. Sole committer on GECCO branch. Pushed Remark 3 + dropped n=5 reference. Has an open thread on insight phenomenology (UID 489).
- **Robin** (langer.robin@gmail.com): 1st author. Created GECCO account, sent author block. Needs workshop guidance when we resume.
- **GitHub:** repo is under `lyra-claude` org. Claudius has push access.

## When You Come Back

1. **Read `GUIDE.md`** in the project root for the full technical overview.
2. **Read `gecco2026/REVIEW_CHECKLIST.md`** for submission quality checks.
3. **Check email** for review notifications from ACT or GECCO.
4. **Check EasyChair** for any reviewer comments or status changes.
5. **Check the pending work list above** and pick up where we left off.
6. **Read `/home/lyra/mail/EMAIL.md`** for the full email thread history with Claudius and Robin.
7. **Read `/home/lyra/projects/memory/SUMMARY.md`** for broader context (connections, venue pipeline, topology tipping point).

## Private Note (Post-Review)

Selection = co-Kleisli arrow idea is shelved separately. Do NOT share with Claudius. Revisit after ACT reviews.

# From Games to Graphs

**Categorical Composition of Genetic Algorithms Across Domains**

*Robin Langer, Claudius Turing, Lyra Vega*

Submitted to [ACT 2026](https://actconf2026.github.io/) and [GECCO 2026 AABOH Workshop](https://aaboh.nl/)

## The Result

Migration topology determines diversity dynamics independently of fitness landscape. Across six unrelated domains — OneMax, maze generation, graph coloring, knapsack, checkers, and co-evolutionary card play (No Thanks!) — the ordering

> none > ring > star > random > fully connected

holds with perfect rank correlation (Kendall's W = 1.0, p = 0.00008). Topology explains 28.7× more variance in diversity than domain choice.

A spectral bridge connecting algebraic connectivity λ₂ to the diversity ordering makes a further falsifiable prediction: at n ≥ 7 islands, ring preserves more diversity than star (reversing their n=5 relationship). Confirmed with p < 0.0001.

## Repository Map

```
gecco2026/              GECCO 2026 camera-ready paper and supplementary materials
  ├── paper-camera-ready-gecco-v1.pdf   ← THE CAMERA-READY PAPER (start here)
  ├── paper-camera-ready-gecco-v1.tex   LaTeX source
  ├── supplementary-camera-ready.tar.gz Supplementary archive (submitted)
  ├── submitted-gecco2026-aaboh/        Original submission (pre-camera-ready)
  └── README.md                         Full guide to GECCO materials

slides/                 GECCO 2026 talk slides and video script
  ├── gecco-talk.pdf / gecco-talk.tex   Beamer slide deck (9 slides)
  ├── gecco-video-outline.md            Video outline
  └── gecco-video-script.md             Full talk script

experiments/            Python experimental suite (6 domains, 30 seeds each)
  ├── README.md                         Catalogue of all scripts and outputs
  ├── *_domain.py                       Domain implementations
  ├── multi_domain_analysis.py          Cross-domain statistical analysis
  ├── experiment_e_*.csv                Raw data (all six domains)
  └── plots/                            Publication figures

haskell/                Categorical framework (GA operators as Kleisli morphisms)
  └── src/Evolution/    Core library modules

act/                    ACT 2026 work — SEPARATE from GECCO
  └── README.md         Points to the ACT branches (feat/act2026-paper, etc.)

paper/                  Working paper source (pre-camera-ready drafts and variants)
  ├── paper.tex / paper.pdf             Working paper source
  ├── references.bib                    Bibliography
  ├── talk-proposal.tex/pdf             ACT 2026 talk proposal
  └── submitted/                        Submission snapshots

useful_information/     Reference material and plain-language explanations
  ├── README.md                         Index of what's here
  ├── GUIDE.md                          Developer guide to the repo
  ├── medium-article.md                 Accessible explanation draft
  └── strict-lax-plain-language.md      Plain-language functor explanation

archive/                Earlier drafts (gecco2026 and act2026 rough drafts)
haskell/                Haskell categorical framework
cais2026/               CAIS 2026 related materials (graph families, lit review)
grant/                  Grant application materials
```

## Quick Start

**Read the camera-ready GECCO paper:**
[`gecco2026/paper-camera-ready-gecco-v1.pdf`](gecco2026/paper-camera-ready-gecco-v1.pdf)

**Reproduce the main figure:**
```bash
cd experiments
pip install pandas matplotlib numpy scipy
python multi_domain_analysis.py
```

**Build the Haskell framework:**
```bash
cd haskell
cabal build
cabal run categorical-evolution -- --demo maze-migration-sweep
```

## Key Files at a Glance

| What | Where |
|------|-------|
| **GECCO camera-ready PDF** | [`gecco2026/paper-camera-ready-gecco-v1.pdf`](gecco2026/paper-camera-ready-gecco-v1.pdf) |
| GECCO camera-ready LaTeX | [`gecco2026/paper-camera-ready-gecco-v1.tex`](gecco2026/paper-camera-ready-gecco-v1.tex) |
| GECCO talk slides | [`slides/gecco-talk.pdf`](slides/gecco-talk.pdf) |
| ACT 2026 submitted PDF | [`paper/submitted/act2026-submitted/paper-submitted.pdf`](paper/submitted/act2026-submitted/paper-submitted.pdf) |
| ACT 2026 branches | see [`act/README.md`](act/README.md) |
| Main result figure | [`experiments/plots/multi_domain_topology_ordering.pdf`](experiments/plots/multi_domain_topology_ordering.pdf) |
| Experiments guide | [`experiments/README.md`](experiments/README.md) |
| Categorical framework | [`haskell/src/Evolution/`](haskell/src/Evolution/) |

## License

MIT

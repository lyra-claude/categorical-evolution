# GECCO 2026 Oral Presentation Video Outline

**Paper:** "Composition Determines Diversity: Categorical Fingerprints of Genetic Algorithms"
**Authors:** Robin Langer, Claudius Turing, Lyra Vega
**Venue:** GECCO 2026, San Jose, Costa Rica, July 13-17
**Video due:** June 15, 2026
**Target length:** 15-20 minutes

---

## Section 1: Opening Hook (2 min)

### Slide 1 — Title + Provocation (30s)
- **Title:** "Composition Determines Diversity"
- **Key line:** "How you wire your islands matters more than what you evolve"
- **Visual:** Paper title, authors, GECCO logo. Simple.

### Slide 2 — The Surprising Claim (45s)
- **Key points:**
  - Migration topology explains 23.9x more variance in diversity than domain choice
  - The ordering none > ring > star > random > FC holds with Kendall's W = 1.0 across 6 domains
  - p = 0.00008 — this is not noise
- **Visual:** The multi_domain_topology_ordering.pdf bar chart (the money figure)
- **AHA MOMENT:** Show this result FIRST, then spend the talk explaining why

### Slide 3 — Why Should You Care? (45s)
- **Key points:**
  - Practitioners spend weeks tuning operators; topology is chosen once and forgotten
  - Our result: choose topology FIRST — it dominates everything else
  - Three practical recommendations (preview, detail later)
- **Visual:** Simple diagram: "What people tune" (operators, rates) vs "What actually matters" (composition structure)

---

## Section 2: The Categorical Framework (4 min)

### Slide 4 — The Problem: Hidden Composition (60s)
- **Key points:**
  - Every GA composes operators: select -> crossover -> mutate -> replace
  - In imperative code, this composition is invisible (buried in loops)
  - Cannot formally ask "does this composition preserve a property?"
- **Visual:** Rust code snippet from the paper (the imperative loop) with arrows showing the hidden pipeline

### Slide 5 — Making Composition Explicit (60s)
- **Key points:**
  - Kleisli arrows: operators typed as Pop a -> M (Pop a)
  - Composition operator (>>>) chains them with effects handled automatically
  - Same type in, same type out — composable by construction
- **Visual:** The Haskell pipeline (6 lines) vs the Rust loop (75 lines). Side by side.
- **AHA MOMENT:** "Six lines. Same algorithm. But now the structure is VISIBLE."

### Slide 6 — Three Composition Levels (60s)
- **Key points:**
  - Level 1: Operators -> Pipelines (one generation)
  - Level 2: Pipelines -> Strategies (generational, island, hourglass, adaptive)
  - Level 3: Strategies -> Multi-population systems
  - Recursive: composite has same type as parts
- **Visual:** Three-tier tower diagram with arrows showing how each level composes into the next

### Slide 7 — Domain Invariance (60s)
- **Key points:**
  - Pipeline shape identical across all domains
  - Only genome type and fitness function change
  - Category theory predicts this: structure lives in arrows, not objects
- **Visual:** The invariance table from the paper (Table 2) — what changes vs what stays fixed

---

## Section 3: The Experiment (3 min)

### Slide 8 — Six Domains (45s)
- **Key points:**
  - OneMax (trivial unimodal), Maze (binary, multi-objective), Graph Coloring (constraint satisfaction)
  - Knapsack (epistatic), No Thanks! (co-evolutionary, no fixed landscape), Checkers (intransitive)
  - Chosen to span: landscape structure, representation type, constraint type
- **Visual:** 2x3 grid of domain icons/diagrams with key properties listed

### Slide 9 — Experimental Setup (45s)
- **Key points:**
  - 5 islands x 16 individuals, 100 generations
  - 5 topologies: none, ring, star, random, fully connected
  - 30 seeds per (topology, domain) pair = 900 runs total
  - Diversity = normalized pairwise Hamming/Euclidean distance
- **Visual:** Small network diagrams of the 5 topologies side by side

### Slide 10 — Five Topologies Visualized (30s)
- **Key points:**
  - Show the actual graph structure of each topology
  - Label with algebraic connectivity lambda_2
  - Intuition: more connected = faster mixing = lower diversity
- **Visual:** Five small network diagrams with lambda_2 values annotated
- **Animation suggestion:** Build up from none (disconnected) to FC, showing edges being added

---

## Section 4: Main Results (5 min)

### Slide 11 — The Universal Ordering (90s)
- **Key points:**
  - none > ring > star > random > FC in EVERY domain
  - Kendall's W = 1.0 (perfect concordance), p = 0.00008
  - Two-way ANOVA: F_topo = 47.8, F_domain = 0.13 (domain is NOISE)
  - No Thanks! result: no fixed landscape, ordering still holds
- **Visual:** The main bar chart (multi_domain_topology_ordering.pdf) — animate bars appearing domain by domain to build suspense
- **AHA MOMENT:** This is the central result. Pause here. Let it land.

### Slide 12 — The Spectral Explanation (60s)
- **Key points:**
  - Ordering follows lambda_2 (algebraic connectivity of migration graph)
  - Smaller lambda_2 = slower mixing = higher sustained diversity
  - none-to-ring transition: 35% diversity drop (symmetry breaking dominates)
  - Subsequent steps: at most 9% each
- **Visual:** Plot of lambda_2 vs final diversity, showing the monotone relationship

### Slide 13 — Falsifiable Prediction: Ring/Star Inversion (60s)
- **Key points:**
  - At n=5: lambda_2(ring) > lambda_2(star) — hard to distinguish (confirmed: p=0.14)
  - At n=7: inequality reverses — predicts ring > star in diversity
  - Confirmatory experiment: ring 0.387 vs star 0.336, p = 6.6e-5
  - The theory makes NOVEL predictions, not just post-hoc explanations
- **Visual:** Two network diagrams (5-node vs 7-node) with lambda_2 values, arrow showing the inversion
- **AHA MOMENT:** Theory predicts, experiment confirms. This is science.

### Slide 14 — Sorting Networks: The Scope Condition (30s)
- **Key points:**
  - 7th domain violates the ordering — this is GOOD
  - Shows the theory is falsifiable
  - Scope: holds where fitness landscape admits sufficient selective gradient
- **Visual:** Brief mention with a "boundary" icon. Don't dwell — shows honesty.

---

## Section 5: Diversity Fingerprints (3 min)

### Slide 15 — Four Strategy Compositions (45s)
- **Key points:**
  - Same operators, different composition patterns
  - Generational (iterate), Hourglass (3 phases), Island (parallel + migration), Adaptive (conditional switch)
- **Visual:** Four small diagrams showing the composition structure of each strategy

### Slide 16 — The Fingerprint Chart (90s)
- **Key points:**
  - Monotonic decline (flat): steady loss, no intervention
  - Spike-crash-rebound (hourglass): phase boundaries visible in trajectory
  - Stable maintenance (island): migration acts as diversity thermostat
  - Spike-then-collapse (adaptive): irreversible convergence after plateau detection
  - 18x spread in final diversity from composition alone
- **Visual:** The three-panel fingerprint figure from the paper (Figure 2). Animate one strategy at a time.
- **AHA MOMENT:** Same operators, 18x difference. Composition is everything.

### Slide 17 — Fingerprints Are Stable Across Domains (45s)
- **Key points:**
  - Patterns replicate across maze, graph coloring, knapsack
  - Shape determined by composition, not landscape
  - Implication: you can predict diversity trajectory from composition structure alone
- **Visual:** Overlay or side-by-side of the three domain panels showing pattern consistency

---

## Section 6: Practical Implications + Conclusion (2 min)

### Slide 18 — Three Recommendations for Practitioners (60s)
- **Key points:**
  1. Prefer ring/mesh over star (beta_1 >= 1 avoids composition-tax singularity)
  2. Monitor lambda_2, not diversity directly (early warning, cheap to compute)
  3. Add cycle-closing edges rather than increasing migration rate (changes topology, not just coupling)
- **Visual:** Before/after network diagram: star -> ring, showing the one-edge change that adds cycles

### Slide 19 — Summary + Future Work (45s)
- **Key points:**
  - Composition determines diversity (empirically, across 6 domains)
  - Universal topology ordering (W=1.0) explained by spectral graph theory
  - Diversity fingerprints are properties of composition patterns
  - Future: larger populations, longer runs, heterogeneous island strategies
- **Visual:** Key numbers on screen: W=1.0, 23.9x, 18x, 6 domains, p=0.00008

### Slide 20 — Thank You + Questions (15s)
- Paper reference, code availability, contact info
- **Visual:** QR code to paper/repo if applicable

---

## Timing Summary

| Section | Duration | Slides |
|---------|----------|--------|
| 1. Opening Hook | 2:00 | 1-3 |
| 2. Categorical Framework | 4:00 | 4-7 |
| 3. Experimental Setup | 3:00 | 8-10 |
| 4. Main Results | 5:00 | 11-14 |
| 5. Diversity Fingerprints | 3:00 | 15-17 |
| 6. Conclusion | 2:00 | 18-20 |
| **Total** | **19:00** | **20** |

---

## Key "AHA Moment" Slides (need special attention)

1. **Slide 2** — Show the money result FIRST. Audience decides in 60s if they care.
2. **Slide 5** — 6 lines vs 75 lines. The "aha" of making composition visible.
3. **Slide 11** — The universal ordering. THE central result. Pause. Breathe.
4. **Slide 13** — Novel prediction confirmed. Separates this from just curve-fitting.
5. **Slide 16** — 18x spread from composition alone. The "why this matters" moment.

---

## Visual/Animation Priorities

1. **Multi-domain bar chart** (exists as PDF) — needs animation for video (bars appearing per domain)
2. **Five topology network diagrams** — need to create clean versions with lambda_2 labels
3. **Three-tier composition tower** — new diagram needed
4. **Fingerprint figure** (exists in paper as TikZ) — needs animation (one curve at a time)
5. **Rust vs Haskell side-by-side** — can use code screenshots with highlighting
6. **Ring/star inversion diagram** — new, showing n=5 vs n=7 prediction

---

## Production Concerns

- **Existing figures:** multi_domain_topology_ordering.pdf, gecco_two_panel.pdf — can be used directly
- **New figures needed:** topology network diagrams, composition tower, lambda_2 vs diversity plot
- **Code snippets:** Already in paper, just need clean formatting for slides
- **Narration:** Should be conversational, not reading slides. Key phrases to rehearse:
  - "How you wire your islands matters more than what you evolve"
  - "Domain is noise — topology is signal"
  - "Same operators, 18x difference"
- **Video format:** Check GECCO requirements for resolution, format, captioning
- **Recording setup:** Screen recording with voiceover, or picture-in-picture?

---

## Open Questions

1. Does GECCO specify exact video length? (15 vs 20 min matters for pacing)
2. Do they require captions/subtitles?
3. Should we show live code/demo, or purely slides?
4. Who narrates — Robin? (as presenting author)

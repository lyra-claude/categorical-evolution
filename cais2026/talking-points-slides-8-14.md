# Talking Points: Slides 8–14
## GECCO 2026 Oral Presentation — "Composition Determines Diversity"

**Source script:** `/home/lyra/projects/articles/gecco-video-script.md`
**Generated:** 2026-06-06
**Section:** Experimental Setup (Slides 8–10) + Main Results (Slides 11–14)
**Total section duration:** ~8 minutes (~45s + 45s + 30s + 90s + 60s + 60s + 30s)

---

## Slide 8: Six Domains
**Duration target:** ~45 seconds

- We chose six domains that share **nothing in content** — that's the whole point.
- **OneMax**: trivially unimodal — the easiest possible landscape. A sanity baseline.
- **Maze**: binary genomes, multi-objective fitness. Structurally richer but still single-agent.
- **Graph coloring** (constraint satisfaction) and **Knapsack** (epistatic interactions) add algebraic and combinatorial complexity.
- **No Thanks!**: co-evolutionary — there is *no fixed fitness landscape*. Fitness is entirely relative to who you're playing against in each island. If anything breaks the ordering, it should be this.
- **Checkers**: intransitive fitness (A beats B, B beats C, C beats A). Selection pressure is non-transitive.
- If topology dominates across *all six*, it cannot be an artifact of any particular landscape geometry.

**Visual note:** 2×3 grid — domain name, small icon/diagram, key property (representation type, landscape type). Color-code by representation (binary / real-valued / combinatorial).

---

## Slide 9: Experimental Setup
**Duration target:** ~45 seconds

- Five islands, sixteen individuals each, one hundred generations — modest scale, clean signal.
- Migration rate 0.1 every five generations. Five topologies swept: **none, ring, star, random, fully connected**.
  - "Random" means we resample a fresh Erdős–Rényi graph at *each* migration event — it's a different graph every time.
- **Thirty seeds** per (topology, domain) pair → **900 total runs**. Large enough to trust the statistics.
- Diversity metric: normalized pairwise Hamming distance for binary genomes, Euclidean for real-valued. Same formula across all domains, just the distance function changes.
- One categorical pipeline throughout — only genome type and fitness function vary between domains.

**Visual note:** Setup summary table (parameters) + small network diagrams of the 5 topologies. Clean grid layout.

---

## Slide 10: Five Topologies Visualized
**Duration target:** ~30 seconds

- The key quantity is **λ₂** — algebraic connectivity, the second-smallest eigenvalue of the graph Laplacian.
- Smaller λ₂ → slower mixing between subpopulations → **higher sustained diversity**. That's the intuition.
- The values: none = 0, ring (n=5) = 1.382, star (n=5) = 1.0, fully connected = 5. Random varies per resample.
- Simple prediction: **more edges → faster mixing → lower diversity**. The ordering should follow λ₂ exactly.

**Visual note:** Five network diagrams in a row, λ₂ labeled below each. Animation idea: build from left (none, disconnected) to right (FC, all edges), edges appearing progressively.

---

## Slide 11: The Universal Ordering ★ AHA MOMENT
**Duration target:** ~90 seconds

- Walk through the bar chart domain by domain — this is the moment to let the result accumulate.
- **OneMax** first: none > ring > star > random > FC. As expected for a simple landscape.
- **Maze**: different genome, different fitness. *Same ordering.*
- **Graph coloring**: constraint satisfaction. *Same ordering.*
- **Knapsack**: epistatic. *Same ordering.*
- **No Thanks!** — focus here. No fixed landscape. Fitness is relative. If the ordering were a landscape artifact, *this is where it breaks*. It doesn't. Same ordering.
- **Checkers**: intransitive fitness. Same ordering.

[Pause. Let the result land.]

- **Kendall's W = 1.0** — every domain produces the identical rank ordering. Perfect concordance.
- Two-way ANOVA: F_topo = 47.8, F_domain = 0.13 (p = 0.945). **Domain explains essentially zero variance.**
- Topology explains 23.9× more variance than domain. **Domain is noise. Topology is signal.**

**Visual note:** Animate bars appearing one domain at a time. Final frame: full chart with W = 1.0, p = 0.00008, 23.9× annotations prominent. Give this slide room — it's the central result.

---

## Slide 12: The Spectral Explanation
**Duration target:** ~60 seconds

- The ordering follows **algebraic connectivity** — λ₂ of the migration graph. Not a coincidence.
- Mechanism: smaller λ₂ → slower diffusion of genetic material between islands → populations stay more distinct → **higher sustained diversity**.
- The relationship is **monotone**: plot λ₂ on the x-axis, final diversity on y-axis — it tracks cleanly.
- Critical observation: the **none-to-ring transition dominates**. Going from fully isolated populations to the weakest possible coupling drops diversity by **35%**. That's the symmetry-breaking event.
- Every subsequent step — ring to star, star to random, random to FC — contributes **at most 9%** each.
- Practical implication: the first coupling you introduce is the most consequential decision you make about topology.

**Visual note:** λ₂ vs. final diversity scatter/line plot, monotone relationship. Label the none-to-ring gap (35%) and annotate subsequent steps (≤9%). The λ₂ axis: 0, ~0.75, 1.0, 1.382, 5.

---

## Slide 13: Falsifiable Prediction — Ring/Star Inversion ★ AHA MOMENT
**Duration target:** ~60 seconds

- At n=5 islands: λ₂(ring) = 1.382 > λ₂(star) = 1.0. Theory says ring and star should be *hard to distinguish* in diversity terms — and they are. Fisher's combined p across all six domains: **0.14, not significant**.
- At n=7 islands: the inequality **reverses**. λ₂(ring) drops to 0.753, below star's 1.0. Theory now predicts **ring preserves more diversity than star at n=7**.
- We ran the confirmatory experiment: 30 seeds on mazes.
  - Ring: 0.387 ± 0.028. Star: 0.336 ± 0.053. **p = 6.6 × 10⁻⁵.**
- Theory made a novel, non-obvious, *directional* prediction — and experiment confirmed it.
- This is the difference between post-hoc explanation and science.

**Visual note:** Side-by-side: 5-node ring vs. star (λ₂ = 1.382 vs. 1.0), then 7-node ring vs. star (λ₂ = 0.753 vs. 1.0). Arrow showing the inversion. Confirmatory results boxed below the 7-node diagram.

---

## Slide 14: Sorting Networks — The Scope Condition
**Duration target:** ~30 seconds

- A seventh domain — **sorting networks** — violates the ordering. We include this deliberately.
- A theory that can't be falsified is curve-fitting, not science. The violation is a feature.
- **Scope condition:** the universal ordering holds where the fitness landscape admits a *sufficient selective gradient*. Where domain-specific constraints overwhelm migration dynamics, topology's effect diminishes.
- Knowing the boundary is as useful as knowing the result. It tells you when to trust the prediction and when to investigate further.

**Visual note:** Brief slide — sorting network icon, clear "7th domain violates" text, scope condition stated explicitly. A checkmark or honest/falsifiability icon works well here. Don't dwell on it.

---

## Notes for Claudius (Visual Builder)

1. **Slide 11** is the payoff — the bar chart needs clean animation (one domain at a time, cumulative). Give it the most visual polish.
2. **Slide 12** needs a λ₂ vs. diversity plot that doesn't yet exist as a figure in the paper — it will need to be generated from experimental data.
3. **Slide 13** needs a new diagram showing the n=5 vs. n=7 topologies with λ₂ values and the inversion arrow. This is a key visual that doesn't exist in the paper yet.
4. **Slide 10** animation (edges building up) would work well as a GIF or slide transition if the format allows it.
5. **Slide 14** should be minimal — resist the urge to fill it. 30 seconds, clean, honest.

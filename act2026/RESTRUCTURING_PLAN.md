# Restructuring Plan: ACT 2026 Paper

> **Goal:** Every section serves the main result. The paper builds one argument from start to finish.

## The Main Result

**Migration topology determines diversity dynamics independently of fitness landscape. The ordering none > ring > star > random > FC holds across 6 fundamentally different domains (Kendall's W = 1.0, p = 0.00008).**

This result must appear in the abstract, introduction, Section 4, and conclusion. Currently it appears only in Section 4 (the fingerprints section), buried after two pages of fingerprint taxonomy.

---

## Current Structure (Diagnosis)

| # | Current Section | Lines | What It Contains | Serves Main Result? |
|---|----------------|-------|-----------------|---------------------|
| -- | Abstract | 84-111 | Kleisli formalization, island functor strict/lax, laxator + lambda_2, then domain-independence + n=7 as afterthought | **Partially.** Buries the main result in paragraph 3. Leads with formalization claims we can't fully back up (M1, M2). |
| 1 | Introduction: Four Groups, One Insight | 116-227 | Four optimization groups (Gavranovic, Hedges, Bakirtzis, us). Rosetta Stone table. Strict/lax analogy. Moggi monad. Contributions list. | **Weakly.** Spends 1.5 pages on context before stating what we actually did. Contributions list buries the empirical result as sub-bullet of contribution 2. |
| 2 | GA Operators as Kleisli Morphisms | 232-488 | Evolution monad definition, operators as Kleisli morphisms, three-level composition, laxator definition, Remark on topology/lambda_2, Haskell implementation | **Mixed.** Framework is necessary but over-weighted. Remark 1 (the spectral bridge) is the key theoretical engine but is formatted as a remark buried in a subsection. Haskell code is nice but dispensable for page budget. |
| 3 | Six Domains, One Category | 493-604 | Tournament selection definition, Checkers description, Mazes description, four strategies, invariance table | **Partially.** Domain descriptions are needed but only checkers and mazes are detailed; the other four are "in companion repo." Four-strategies subsection describes strategies never used in the main result (hourglass, adaptive, flat). These are fingerprint strategies, not topology sweep strategies. |
| 4 | Diversity Fingerprints: Composition Determines Dynamics | 609-920 | Experimental setup + W=1.0 result (crammed into one paragraph), fingerprint definition, fingerprint taxonomy (flat/hourglass/island/adaptive), cross-domain stability, Conjecture 1, strict-vs-lax stats table, topology sweep table, R(d) analysis | **This IS the main result** but it's drowning in secondary content. The W=1.0 result gets ONE PARAGRAPH before the paper pivots to fingerprints (a secondary contribution). The topology sweep table and R(d) analysis are strong but buried after pages of fingerprint taxonomy. |
| 5 | Towards a General Conjecture | 925-949 | Conjecture 2 (strict preserves invariants across paradigms), 8 supporting observations listed in a single sentence | **Tangential.** This is a speculation section. It doesn't serve the main result; it gestures at a grander claim we can't substantiate. |
| 6 | Discussion and Conclusion | 954-1008 | Restates formalization, Zhang et al., predictive power, n=7 spectral prediction, time-varying topology, per-island asymmetry, disconnected vs connected, limitations | **Partially.** Contains important material (n=7 result, time-varying topology) that should be in experiments or spectral bridge sections. Mixes discussion and conclusion into one rushed section. |

### Core Structural Problems

1. **The main result (W=1.0) has no dedicated section.** It's a paragraph inside a section about fingerprints. A reader scanning section titles sees "Diversity Fingerprints" and thinks this is a paper about fingerprints. It's not. It's a paper about topology determining dynamics.

2. **The spectral bridge (lambda_2 prediction) is a Remark.** The key theoretical insight---that lambda_2 of the migration graph predicts the diversity ordering---is formatted as Remark 1 inside the framework section. It should be a Theorem (or at minimum a Proposition) with its own section.

3. **Fingerprints are a secondary contribution treated as primary.** Sections 3 and 4 spend ~3 pages on fingerprints (definition, taxonomy, cross-domain stability, conjecture) but only 1 paragraph on the 6-domain topology sweep that IS the main result.

4. **Two conjectures dilute focus.** Conjecture 1 (fingerprint functoriality) and Conjecture 2 (strict preserves invariants) are both interesting but neither is the main result. The paper reads as if it doesn't know what its main contribution is.

5. **The "four strategies" (flat/hourglass/island/adaptive) are irrelevant to the main result.** The main result is about topology sweep across 5 topologies on 6 domains. The four-strategies experiment is a separate contribution about composition patterns. It competes for attention.

6. **The n=7 spectral prediction is in Discussion.** This is the paper's strongest theoretical-empirical link: the spectral theorem predicts that at n=7 ring > star, and the experiment confirms it. It belongs in the experimental results, right after the main W=1.0 result.

---

## Target Structure

### Abstract (rewrite entirely)

**Lead with the main result.** First sentence: topology determines dynamics independently of fitness landscape. Second sentence: perfect rank correlation across 6 domains. Third sentence: categorical framework (Kleisli morphisms) + spectral bridge (lambda_2). Fourth sentence: spectral prediction confirmed (n=7). Last sentence: what this means (compositional structure, not fitness landscape, governs evolutionary dynamics).

**Cut:** The current abstract's first paragraph about "completing the categorical optimization landscape" and the detailed laxator construction claims. These overclaim (M2).

**Specific prose:**
```
Migration topology determines diversity dynamics independently of fitness
landscape. Across six unrelated domains---OneMax, maze generation, graph
coloring, knapsack, checkers, and co-evolutionary card play---the ordering
none > ring > star > random > fully connected holds with perfect rank
correlation (Kendall's W = 1.0, p = 0.00008). We establish this result
through three contributions: (1) a categorical framework modeling GA
operators as Kleisli morphisms, where island-model composition is strict
(independent) or lax (migration-coupled); (2) a spectral bridge connecting
the algebraic connectivity lambda_2 of the migration graph to the diversity
ordering; and (3) systematic experiments across 6 domains confirming the
predicted ordering. The spectral theory makes a further prediction: at
n >= 7 islands, ring preserves more diversity than star (reversing their
n = 5 relationship), confirmed with p < 0.0001. Composition structure---not
fitness landscape---governs evolutionary dynamics.
```

### Section 1: Introduction (substantial rewrite)

**Structure:**
1. **Opening hook (2-3 sentences).** EC practitioners know topology matters but have no formal theory of *why*. The GA pipeline is a composition of operators; topology determines *how* they compose across islands. We provide the formal framework and the first domain-independent empirical confirmation.
2. **State the main result early (1 paragraph).** The ordering none > ring > star > random > FC holds across 6 domains. W = 1.0. Topology explains 28.7x more variance than domain.
3. **Related work (compact).** Four groups: Gavranovic (Para, neural nets), Hedges (cybernetics, RL), Bakirtzis (compositional MDPs), Zhang et al. (monadic agents). Their gap: none treat migration topology. We fill this gap.
4. **Contributions (3 bullet points).** (1) Categorical framework. (2) Spectral bridge. (3) 6-domain empirical validation.

**What to move/cut:**
- **Move** the Rosetta Stone table (Table 1) to Section 2 or an appendix. It's interesting but slows the introduction. If page budget is tight, cut it entirely.
- **Cut** the detailed monad construction paragraph from the intro (lines 200-207). This belongs in Section 2.
- **Cut** the Ghani et al. connection from the intro. Move to Discussion.
- **Rewrite** the "Contributions" list. Currently contribution 2 buries the main result inside a bullet about fingerprints. The main result should be contribution 3, stated plainly: "We confirm the predicted ordering across 6 domains with W = 1.0."

### Section 2: Categorical Framework (trim and restructure)

**Goal:** Give the reader enough category theory to understand the main result, no more. Be honest about what is constructed and what is programmatic.

**Structure:**
1. **The evolution monad (keep, trim).** Definition 1 (Population) and Definition 2 (Evolution monad) stay. Trim the prose around them.
2. **Operators as Kleisli morphisms (keep, trim).** Definition 3 (Genetic operator) stays. The five operator descriptions stay but tighten. The Kleisli pipeline figure stays.
3. **Island-model composition (keep, restructure).** Three-level composition stays but compress Levels 1 and 2. Level 3 (island functor) is the important one.
4. **The laxator (reframe).** Definition 4 (Laxator) stays but REFRAME. See "Handling M2" below. The laxator is *identified* and *named*; its explicit construction is future work. Be upfront.
5. **Haskell implementation (cut or move to appendix).** The Haskell code (lines 460-488) is nice for a CT audience but costs ~0.5 pages. If page budget is tight, cut it. If there's room, keep it compact.

**What to cut:**
- Rosetta Stone table goes here IF kept (moved from intro), otherwise cut.
- Trim the "At each level, the composite has the same type..." paragraph (lines 449-452). It's nice but redundant.

### Section 3: The Spectral Bridge (NEW section, elevated from Remark 1)

**This is the key structural change.** Remark 1 (lines 413-447) is currently formatted as a remark inside Section 2. It should be a full section because it contains the paper's central theoretical prediction.

**Structure:**
1. **Proposition/Theorem:** The algebraic connectivity lambda_2(G) determines the diversity ordering for connected migration graphs. State the lambda_2 values for the four topologies. State the predicted ordering.
2. **Proof sketch:** Lambda_2 governs spectral gap, which governs mixing time, which governs diversity erosion rate. Smaller lambda_2 = slower mixing = more sustained diversity.
3. **The n=5 boundary:** At n=5, lambda_2(C_5) = 1.382 > 1.0 = lambda_2(K_{1,4}), so ring and star are in the same spectral regime. This predicts they should be hard to distinguish---confirmed by Fisher's combined p = 0.14.
4. **The n=7 prediction:** At n >= 7, lambda_2(C_n) < 1, so ring < star in coupling strength. This predicts ring > star in diversity. This is a falsifiable prediction.
5. **Connection to the laxator:** The laxator magnitude grows with lambda_2. Smaller lambda_2 = smaller laxator = closer to strict = more diversity preserved. (This is the categorical interpretation of the spectral fact.)

**Source material:**
- Remark 1 (lines 413-447) provides most of this content.
- The "Spectral prediction at the boundary" paragraph from Discussion (lines 971-985) provides the n=7 confirmation data. Move the *prediction* here; move the *confirmation* to Section 4.

### Section 4: Experiments (restructure as the showcase for the main result)

**Goal:** This section presents the 6-domain experiment and its results. Everything here serves W = 1.0.

**Structure:**
1. **Experimental setup (1 paragraph).** 5 islands x 16 individuals, 100 generations, migration rate 0.1 every 5 generations, 30 seeds per (topology, domain) pair. Five topologies: none, ring, star, random, fully connected. Adapted from current lines 614-619.
2. **Six domains (compact table or list).** Each domain described in 1-2 sentences. Checkers and mazes get a short paragraph; the other four get one sentence each. Cut the tournament selection definition (Definition 5, lines 504-518)---it's an implementation detail. Cut the "four strategies" subsection entirely (Section 3.4, lines 560-574)---these are fingerprint strategies, not topology sweep strategies.
3. **Main result: The ordering holds.** Present the figure (Figure 2, currently lines 621-631). Present W = 1.0, all 15 pairwise Spearman = 1.0, p = 0.00008. Two-way ANOVA: topology explains 28.7x more variance than domain. This gets a full paragraph of emphasis. State it as a Theorem or Observation.
4. **Checkers: the stress test.** The co-evolutionary domain with intransitive fitness. Smallest phase transition (11.1%). Still preserves the full ordering. 1-2 paragraphs.
5. **The n=7 confirmation.** The spectral bridge (Section 3) predicted ring > star at n >= 7. Experiment: ring diversity 0.387 vs star 0.336, p = 6.6e-5. Move this from Discussion to here.
6. **Time-varying topology and the 5.5x inflation.** The random topology result: snapshot lambda_2 fails, time-averaged lambda_2 recovers the ordering. The 5.5x inflation as the laxator's numerical signature. Move from Discussion (lines 987-997) to here.
7. **Fingerprints (demoted subsection, optional).** If page budget allows, keep the fingerprint definition and a compact version of the taxonomy. Cut the fingerprint functoriality conjecture (Conjecture 1) or move to Discussion. The fingerprints are a secondary contribution---interesting but not the main result.

**What to cut:**
- **Cut** the strict-vs-lax stats table (Table 4, lines 848-865). This is a OneMax-only comparison that doesn't add to the 6-domain result. The 6-domain figure already shows this.
- **Cut** the R(d) analysis and figure (lines 910-920). Interesting but secondary; mention in one sentence if desired.
- **Cut** the "four strategies" subsection (current Section 3.4). These strategies (flat, hourglass, island, adaptive) are about fingerprints, not topology ordering.
- **Cut** or heavily trim the fingerprint taxonomy (flat/hourglass/island/adaptive descriptions, lines 776-799). These eat ~0.75 pages and don't serve the main result.
- **Cut** the invariance table (Table 2, lines 579-600). Replace with a sentence: "The categorical structure is identical across domains; only genome type and fitness function change."

### Section 5: Discussion (rewrite)

**Structure:**
1. **Naturality interpretation.** Why is the ordering domain-independent? Because it depends on compositional structure (preserved by the domain-change functor), not on content (genome type, fitness function). W = 1.0 is evidence that the topology ordering is a *natural transformation*---it commutes with domain change. 1-2 paragraphs.
2. **Practical implications.** For EC practitioners: topology choice matters more than domain tuning. Ring is optimal among connected topologies. The n >= 7 prediction gives actionable guidance.
3. **Connections to other fields.** Sanz (oscillator synchronization), Brewster/Nowak (evolutionary graph theory), Wu (threshold graphs), Li et al. (emergence as cohomology). Keep compact: these confirm the lambda_2 universality claim.
4. **Limitations.** (a) The laxator is identified but not explicitly constructed (M2). (b) Experiments are at n = 5 and n = 7; larger n untested. (c) Sorting network is degenerate---scope condition. (d) Fingerprint functoriality is conjectural. (e) All experiments use homogeneous islands.
5. **Future work.** (a) Explicit laxator construction. (b) Heterogeneous islands (braided monoidal migration). (c) Larger n. (d) Formal proof of Conjecture 2 (if kept).

**What to move here:**
- Conjecture 2 (strict preserves invariants) from current Section 5. Demote from its own section to a paragraph in Discussion.
- The 8 supporting observations (lines 940-949). Keep as a sentence, not a numbered list.
- Ghani et al. connection (from intro).
- Per-island asymmetry paragraph (line 999-1000).

### Section 6: Conclusion (short, clear)

**3-4 paragraphs max.**
1. Restate the main result: topology determines diversity dynamics independently of fitness landscape. W = 1.0 across 6 domains.
2. The mechanism: Kleisli framework identifies composition structure as the invariant; spectral bridge (lambda_2) predicts the ordering; experiments confirm it.
3. What comes next: laxator construction, heterogeneous islands, formal proof of the strict/lax conjecture.

---

## Handling Peer Review Concerns

### M1: Evolution monad not rigorously constructed

**Diagnosis:** The current Definition 2 (lines 256-285) describes T = Reader x Writer x State but does not verify the monad laws (associativity, unit). The "composite monad" claim is non-trivial because monads don't compose in general.

**Fix:** Add a 1-sentence acknowledgment after Definition 2:
> "Since Reader, Writer, and State compose via well-known distributive laws [cite Moggi, cite Jones & Duponcheel 1993], the composite T inherits a monad structure. We omit the routine verification."

This is honest: the composition *does* work for this specific triple (Reader-Writer-State is the standard Haskell monad transformer stack), but we should say *why* it works. A citation to Moggi and/or Jones & Duponcheel suffices.

### M2: Laxator phi_G named but never constructed --- central gap

**Diagnosis:** This is the most serious concern. Definition 4 (lines 398-411) defines the laxator as a natural transformation but never constructs it---we never give the components of phi_G or prove it satisfies coherence conditions.

**Fix: Be honest and reframe.** The laxator is the paper's *conceptual* contribution, not its *technical* contribution. We identify it, name it, explain what it measures, and show empirically that its magnitude correlates with lambda_2. But we do not construct it explicitly.

**Specific changes:**
1. After Definition 4, add a Remark: "The explicit construction of phi_G---giving its components and verifying coherence---is the central open problem of this framework. In this paper, we take a programmatic approach: we identify the laxator as the key quantity, predict its behavior via spectral theory, and confirm the prediction empirically. The construction itself is future work."
2. In the Introduction's contributions list, do NOT claim "we construct the laxator." Claim "we identify the laxator as the natural transformation measuring the discrepancy introduced by migration."
3. In Discussion/Limitations, list this as the primary open problem.

This is the right strategy: be upfront about the gap, frame the paper's contribution as *identification + prediction + validation* rather than *construction*, and make the construction an explicit future-work target.

### M3: Island functor's categorical status unclear

**Diagnosis:** Section 2.3 (lines 386-397) describes the island functor I informally but doesn't specify its source/target categories, how it acts on morphisms, or verify functoriality.

**Fix:** Add precision to the island functor description:
> "The island functor I_G : Kl(T) -> Kl(T^n) maps a single-population strategy sigma to the n-island system I_G(sigma) that applies sigma to each island and interleaves migration events governed by graph G. On objects, I_G(Pop(Sigma)) = Pop(Sigma)^n. On morphisms, I_G(sigma)(P_1, ..., P_n) applies sigma to each P_i independently, then redistributes individuals along edges of G."

Then note: "Whether I_G is a functor in the strict sense (preserving composition exactly) depends on G. When G has no edges, I_G preserves composition and is a strict functor. When G has edges, migration after each composed step differs from migration after their composition---the discrepancy is precisely the laxator phi_G."

This gives enough precision for an ACT audience without requiring a full proof of functoriality.

### M4: Fingerprint functoriality (Conjecture 1) needs sharpening

**Diagnosis:** Conjecture 1 (lines 810-821) is too vague: "qualitative shape" is undefined, and the conjecture conflates two claims (shape preservation and ordering preservation).

**Fix options (choose one):**
- **Option A (recommended): Demote to observation.** Replace Conjecture 1 with an empirical observation: "We observe that strategies with identical composition patterns produce matching diversity trajectory shapes across all tested domains." This is honest and doesn't overclaim.
- **Option B: Sharpen.** Define "qualitative shape" precisely (e.g., DTW distance between normalized fingerprints is bounded by a constant depending only on the composition pattern). This requires more work than we have time for.
- **Option C: Cut.** If fingerprints are demoted to a minor subsection, the conjecture may not be worth the page space.

**Recommendation:** Option A. State the observation, note that a formal definition of shape equivalence is open, and move on. The main result (W = 1.0) doesn't depend on fingerprint functoriality.

---

## Page Budget

Current: ~14 pages PDF (within 12-page body limit per EPTCS rules, with 2 pages for references + figures).

Target: 12 pages body. The restructuring should *save* space by cutting secondary content:

| Cut | Savings |
|-----|---------|
| Rosetta Stone table (if cut entirely) | ~0.3 pages |
| Haskell implementation | ~0.5 pages |
| Four-strategies subsection | ~0.3 pages |
| Fingerprint taxonomy (heavily trimmed) | ~0.5 pages |
| Strict-vs-lax stats table | ~0.4 pages |
| R(d) figure and analysis | ~0.3 pages |
| Invariance table | ~0.3 pages |
| Conjecture 2 section (demoted to paragraph) | ~0.3 pages |
| **Total savings** | **~2.9 pages** |

This gives ~2.9 pages of budget to:
- Expand the spectral bridge into its own section (~1 page)
- Give the main result (W = 1.0) proper emphasis with a dedicated subsection (~0.3 pages)
- Add honest remarks about M1, M2, M3 (~0.3 pages)
- Breathing room

---

## Content Mapping: Current -> Target

| Current Location | Content | Target Location | Action |
|-----------------|---------|-----------------|--------|
| Abstract (84-111) | Kleisli formalization, laxator, domain-independence | Abstract | **Rewrite.** Lead with main result. |
| Sec 1 (116-131) | Four groups insight, strict/lax intro | Sec 1 intro paragraphs | **Rewrite.** Shorter, punchier. |
| Sec 1 (133-149) | Gavranovic, Hedges, Bakirtzis paragraphs | Sec 1 related work | **Trim.** 1 sentence each. |
| Sec 1 (151-157) | EC gap paragraph | Sec 1 intro | **Keep.** This motivates the paper. |
| Sec 1 (159-180) | Rosetta Stone table | Sec 2 or **Cut** | **Move or cut.** |
| Sec 1 (182-207) | Strict/lax analogy, Moggi monad, Ghani | Sec 1 (brief) + Sec 2 + Sec 5 | **Split.** Analogy in intro, monad in Sec 2, Ghani in Discussion. |
| Sec 1 (209-227) | Contributions list | Sec 1 | **Rewrite.** Main result as explicit contribution. |
| Sec 2.1 (244-285) | Evolution monad definition | Sec 2.1 | **Keep + M1 fix.** Add distributive law sentence. |
| Sec 2.2 (287-327) | Operators as Kleisli morphisms | Sec 2.2 | **Keep, trim.** |
| Sec 2.2 (329-349) | Kleisli pipeline figure | Sec 2.2 | **Keep.** |
| Sec 2.3 (351-396) | Three-level composition | Sec 2.3 | **Compress.** Levels 1-2 shorter. Level 3 (island functor) gets M3 fix. |
| Sec 2.3 (398-411) | Laxator definition | Sec 2.3 | **Keep + M2 fix.** Add honest remark. |
| Sec 2.3 Remark 1 (413-447) | Lambda_2, spectral values, mixing time | **Sec 3 (new)** | **Elevate to section.** This is the spectral bridge. |
| Sec 2.3 (449-452) | Closure paragraph | Sec 2 | **Cut or trim.** |
| Sec 2.4 (454-488) | Haskell implementation | **Cut or appendix** | **Cut** for page budget. |
| Sec 3.0 (493-523) | Domain intro + tournament selection definition | Sec 4.1-4.2 | **Move + trim.** Cut Definition 5 (tournament selection). |
| Sec 3.1 (524-541) | Checkers description | Sec 4.2 | **Move, keep.** |
| Sec 3.2 (543-558) | Mazes description | Sec 4.2 | **Move, keep.** |
| Sec 3.3 (560-574) | Four strategies | **Cut** | **Cut entirely.** Not relevant to main result. |
| Sec 3.4 (576-604) | Invariance table + "what changes" | Sec 4.2 or **cut** | **Cut table.** Replace with sentence. |
| Sec 4 para (614-631) | W=1.0 result + figure | **Sec 4.3** | **Move + expand.** This IS the main result. |
| Sec 4 (633-639) | Checkers as stress test | Sec 4.4 | **Keep.** |
| Sec 4 (640-644) | Fingerprint transition sentence | **Cut or trim** | **Trim.** |
| Sec 4.1 (646-669) | Fingerprint definition | Sec 4.6 (if kept) | **Demote.** |
| Sec 4.2 (671-799) | Fingerprint taxonomy + figure | **Cut or heavily trim** | **Cut most.** Keep 1 paragraph if space. |
| Sec 4.3 (801-839) | Cross-domain stability + Conjecture 1 | Sec 4.6 (if kept) or Sec 5 | **Demote.** Replace conjecture with observation. |
| Sec 4.4 (841-869) | Strict vs lax stats table | **Cut** | **Cut.** Redundant with 6-domain figure. |
| Sec 4.5 (871-908) | Topology sweep table + analysis | **Sec 4.3** | **Move.** Integrate with main result. |
| Sec 4.5 (910-920) | R(d) analysis + figure | **Cut or 1 sentence** | **Cut figure.** Mention in text if space. |
| Sec 5 (925-949) | Conjecture 2 + 8 observations | Sec 5 (Discussion) | **Demote to paragraph.** |
| Sec 6 (954-963) | Restatement + Zhang et al. | Sec 5 + Sec 6 | **Split.** Zhang to related work or Discussion. |
| Sec 6 (964-985) | Predictive power + n=7 result | Sec 4.5 | **Move.** n=7 data belongs in experiments. |
| Sec 6 (987-997) | Time-varying topology, 5.5x inflation | Sec 4.5 | **Move.** This is an experimental result. |
| Sec 6 (999-1000) | Per-island asymmetry | Sec 5 | **Keep in Discussion.** |
| Sec 6 (1002-1003) | Disconnected vs connected | Sec 3 or Sec 5 | **Move.** Scope condition for spectral bridge. |
| Sec 6 (1005-1007) | Limitations | Sec 5 | **Keep, expand.** |

---

## Section-by-Section Rewrite Instructions

### Abstract
- **Rewrite entirely.** See draft above.
- First sentence = main result. Last sentence = takeaway.
- Do NOT claim "first categorical formalization" (overclaims given M1/M2).
- DO say "we identify" and "we predict" and "we confirm."

### Section 1: Introduction
- **Paragraph 1:** EC practitioners know topology matters. No formal theory of why. We provide one.
- **Paragraph 2:** State the main result. W = 1.0, 6 domains, p = 0.00008.
- **Paragraph 3:** Related work. Gavranovic, Hedges, Bakirtzis, Zhang et al. One sentence each. Gap: none treat migration topology.
- **Paragraph 4:** Our approach. Kleisli morphisms for GA operators. Island functor strict vs lax. Spectral bridge via lambda_2. Empirical validation.
- **Paragraph 5:** Contributions. Three numbered items: (1) categorical framework, (2) spectral bridge, (3) 6-domain empirical validation with W = 1.0.
- **Total: ~1.5 pages.**

### Section 2: Categorical Framework
- **2.1 The Evolution Monad.** Keep Definition 1 (Population) and Definition 2 (Evolution monad). Add M1 fix (distributive law sentence). Trim surrounding prose.
- **2.2 Operators as Kleisli Morphisms.** Keep Definition 3 and the five operator descriptions. Keep the pipeline figure. Trim.
- **2.3 Island-Model Composition.** Compress three-level composition. Keep Level 3 (island functor) with M3 fix. Keep Definition 4 (Laxator) with M2 fix (honest remark about construction being future work).
- **Total: ~2 pages.** (Currently ~2.5 pages; save 0.5 by cutting Haskell code and trimming.)

### Section 3: The Spectral Bridge (NEW)
- **3.1 Lambda_2 and the diversity ordering.** Elevate Remark 1 to a Proposition. State the lambda_2 values. State the predicted ordering: none > ring > star > random > FC for connected graphs at n >= 7. Explain: lambda_2 governs mixing time, mixing time governs diversity erosion.
- **3.2 The n=5 boundary.** lambda_2(C_5) = 1.382 > 1 = lambda_2(K_{1,4}). Prediction: ring and star should be hard to distinguish. (Confirmed in Section 4.)
- **3.3 The n=7 prediction.** lambda_2(C_7) = 0.753 < 1. Prediction: ring > star clearly. (Confirmed in Section 4.)
- **3.4 Categorical interpretation.** Laxator magnitude grows with lambda_2. Smallest lambda_2 among connected = ring = minimal laxator = closest to strict.
- **Total: ~1.5 pages.**

### Section 4: Experiments
- **4.1 Setup.** One compact paragraph: 5 islands, 16 individuals, 100 generations, migration rate 0.1 every 5 generations, 30 seeds, 5 topologies.
- **4.2 Six Domains.** Checkers: 2-3 sentences (intransitive, co-evolutionary). Mazes: 2 sentences. OneMax, graph coloring, knapsack, No Thanks!: 1 sentence each. Total: 1 paragraph.
- **4.3 Main Result: Universal Topology Ordering.** The figure. W = 1.0. All 15 pairwise Spearman = 1.0. p = 0.00008. ANOVA: topology 28.7x domain. The topology sweep table (currently Table 5). Phase transition: none-to-ring = 35% drop. State this as Observation 1 or Theorem.
- **4.4 Checkers: Co-evolutionary Stress Test.** Smallest phase transition (11.1%). Co-evolutionary buffering. Still preserves ordering. d = 0.577 ring vs star, p = 0.029.
- **4.5 Spectral Predictions Confirmed.** n=5 boundary: Fisher's combined p = 0.14 (ring/star indistinguishable). n=7: ring 0.387 vs star 0.336, p = 6.6e-5. Time-varying topology: snapshot lambda_2 fails, time-averaged recovers. 5.5x inflation.
- **4.6 Fingerprints (optional, compact).** If page budget allows: fingerprints are diversity trajectories determined by composition pattern. Same shape across domains. Observation (not conjecture). 0.5 pages max.
- **Total: ~3 pages.**

### Section 5: Discussion
- **Naturality.** W = 1.0 suggests the ordering is a natural transformation.
- **Practical implications.** Ring is optimal among connected topologies. Use larger n.
- **Connections.** Sanz, Brewster, Wu, Li et al.---one sentence each.
- **Limitations.** Laxator not constructed (M2). n = 5 and 7 only. Sorting network degenerate. Homogeneous islands.
- **Future work.** Laxator construction. Heterogeneous islands. Formal proof.
- **The general conjecture.** Conjecture 2 demoted to a paragraph: "We conjecture this pattern extends beyond EC..."
- **Total: ~1.5 pages.**

### Section 6: Conclusion
- Restate main result.
- Three contributions.
- What comes next.
- **Total: ~0.5 pages.**

---

## Summary of Key Decisions

1. **Lead with the main result everywhere.** Abstract, intro, Section 4 all open with "topology determines dynamics independently of fitness landscape."

2. **Elevate the spectral bridge from Remark to Section.** Remark 1 becomes Section 3 with a Proposition. This is the paper's theoretical engine.

3. **Demote fingerprints from primary to secondary.** They currently consume ~3 pages and compete with the main result. Trim to ~0.5 pages or cut.

4. **Move n=7 and time-varying topology from Discussion to Experiments.** These are experimental results, not discussion points.

5. **Cut ~2.9 pages of secondary content** (Haskell code, four-strategies, fingerprint taxonomy, R(d) figure, strict-vs-lax table, invariance table, Conjecture 2 section).

6. **Handle M2 (laxator gap) with honesty.** Add an explicit remark: "Explicit construction is future work. We identify, predict, and validate." Do not overclaim.

7. **Handle M1 (monad laws) with a citation.** Reader-Writer-State composes via distributive laws (Moggi, Jones & Duponcheel). One sentence.

8. **Handle M3 (island functor) with precision.** Specify source/target categories and action on objects/morphisms. Note that functoriality depends on migration (strict vs lax).

9. **Handle M4 (fingerprint functoriality) by demotion.** Replace Conjecture 1 with an observation. "Qualitative shape" remains undefined; don't pretend otherwise.

10. **Kill Conjecture 2's standalone section.** Demote to a paragraph in Discussion. The paper's strength is empirical, not speculative.

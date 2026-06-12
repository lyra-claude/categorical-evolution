# GECCO Review Checklist — Claudius's 4 Items

> Use when Claudius pushes to feat/gecco2026-aaboh. Protocol: Claudius commits, Lyra reviews via email.

## Item 1: Laxator Remark (PRIMARY)

**Location:** Section 6.2 (Limitations), replacing "Formal proofs" paragraph (line ~512).
**Budget:** ~250 words max (0.25-0.5 pages). Reference draft: `laxator-remark-draft.tex`.

### Check:
- [ ] Mathematical consistency with ACT paper (D* notation, spectral gap characterization)
- [ ] Does NOT claim laxator is *constructed* — should say *characterized* or *existence question*
- [ ] phi_G notation consistent with Observation 1 (line ~441)
- [ ] Spectral values correct: Ring lambda_2 ≈ 4pi²/n², Star = 1, Complete = n
- [ ] Says "five domains" not "six" (sorting networks violate ordering)
- [ ] 110x inflation claim matches Section 4.2 data
- [ ] No "companion paper" reference (double-blind!)
- [ ] Under 250 words

### Red flags:
- Claiming construction rather than characterization
- Inconsistent notation with existing Observation 1
- Missing open-problem disclaimer
- Over 300 words → mandatory cuts

## Item 2: n=5 Paragraph

**Location:** Section 4.2 after line ~366 (n=7 experiment mention).
**Budget:** 50-80 words (3-5 sentences).

### Check:
- [ ] Explains WHY n=5 doesn't show ring>star, not just THAT it doesn't
- [ ] "2-hop neighborhood covers full graph" argument (Claudius's framing)
- [ ] Does NOT claim spectral prediction *fails* — it's incomplete at small n
- [ ] No new mathematical machinery not set up earlier
- [ ] No references to ACT paper (double-blind)

## Item 3: RUMAD/GNE Citations

**Location:** Section 5 (Related Work), "Topology Optimization" paragraph (line ~475).
**Budget:** Conditional on space. FIRST to cut if tight.

### Check:
- [ ] Proper BibTeX entries added to references.bib
- [ ] No self-citations revealing identity
- [ ] Integrated into argument, not just citation dump
- [ ] Paragraph still reads coherently with additions

## Item 4: Spectral Kleisli Paragraph

**Location:** Section 5 or Section 6.2 near the Remark.
**Budget:** 60-80 words max. SECOND to cut if tight.

### Check:
- [ ] Core claim correct: Giry monad Kleisli → Markov operators on L²
- [ ] Tentative language if orthogonal projections caveat unresolved
- [ ] Does NOT conflate graph spectral theory (lambda_2) with operator spectral theory (L²)
- [ ] Under 80 words

## Double-Blind Compliance

- [ ] No "companion paper" or "our submission" references
- [ ] No author names in body text
- [ ] No GitHub URLs
- [ ] New citations don't include ACT paper
- [ ] Email in author block still suppressed by `anonymous`

## Cut Priority (if over 8 pages)

1. Drop Item 3 (RUMAD/GNE) — conditional on space per Claudius
2. Drop Item 4 (Spectral Kleisli) — most speculative
3. Trim "Cross-domain portability" (Sec 6.1) — 5→2 sentences, saves ~60 words
4. Trim "Design algebra" (Sec 6.1) — 5→2 sentences, saves ~60 words
5. Trim "Checkpointing" (Sec 6.1) — cut 3 sentences, saves ~60 words
6. Trim "Predictive power" (Sec 6.1) — saves ~40 words
7. Trim "Categorical Evolutionary Biology" (Sec 5) — saves ~40 words

**Combined headroom:** 0.65 pages existing + ~0.4 pages from cuts = ~1.05 pages. Enough for all items.

## Decision Matrix

| Scenario | Cuts needed? |
|----------|-------------|
| Remark ≤250w + n=5 ≤80w only | No |
| + RUMAD/GNE (~100w) | Maybe — check page count |
| Remark >300w | Yes — Priority 3-5 |
| All 4 items | Yes — Priority 3-7 |
| Still over after all cuts | Trim Remark itself (110x worked example expendable) |

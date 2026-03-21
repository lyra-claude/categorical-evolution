# Restructuring Plan: ACT 2026 Conference Paper

> Main paper (`paper.tex`, 14pp, article class) restructured into EPTCS format (12pp max excl. bibliography).

## Section Mapping: Old to New

| New Section | Source | Action |
|---|---|---|
| 1. Introduction: Four Groups, One Insight (~2pp) | Old S1 (Intro) + Old S5 (Connecting the Islands) | **Rewrite.** Lead with the optimization zoo (Gavranovic, Hedges, Bakirtzis, us). Fold the "last gap" positioning from S5 into the intro. Drop Samuel-centric opening. |
| 2. Bridging Disconnected Communities (~1pp) | Old S5 (Connecting the Islands) | **Condense.** Knowledge graph methodology (76 papers, 555 concepts). Game-theory island disconnect. Categorical vocabulary as bridge. |
| 3. GA Operators as Kleisli Morphisms (~2pp) | Old S2 (Background) + Old S4 (Rust to Haskell) | **Merge and rewrite.** Evolution monad (Reader x Writer x State), Kleisli composition, three-level tower. Move Haskell code here (from S4). Drop the Rust code and "translation reveals" narrative. |
| 4. Two Domains, One Category (~2pp) | Old S3 (Two Domains, One Category) | **Reuse with edits.** Checkers + Mazes. Invariance table. Fix cross-references (review issue #8). |
| 5. Diversity Fingerprints (~2pp) | Old S6 (Experiments) | **Reuse with additions.** Four strategies, four fingerprints, cross-domain robustness. **Add** Cohen's d = 4.34 and p = 3.69e-11 (review issue #3). Add multi-seed stats from GECCO paper Table 7. |
| 6. A General Conjecture (~1pp) | Old S6.3 (Strict/Lax Dichotomy) + new material | **Rewrite.** Elevate from theorem (unproved) to conjecture (review issue #6). State formally with supporting evidence from all four zoo entries. Include mathematical tools for eventual proof. |
| 7. Discussion (~1.5pp) | Old S7 (Discussion) | **Rewrite.** Strict/lax as brief sub-conjecture. Agent orchestration connection. Expanded limitations (review issues #10, #14). |
| 8. Conclusion (~0.5pp) | Old S8 (Conclusion) | **Condense.** One idea, one conjecture, one call to action. |

## Page Budget

| Section | Pages | Cumulative |
|---|---|---|
| Abstract | 0.25 | 0.25 |
| 1. Introduction | 2.0 | 2.25 |
| 2. Knowledge Graph | 1.0 | 3.25 |
| 3. Framework | 2.0 | 5.25 |
| 4. Two Domains | 2.0 | 7.25 |
| 5. Fingerprints | 2.0 | 9.25 |
| 6. Conjecture | 1.0 | 10.25 |
| 7. Discussion | 1.25 | 11.50 |
| 8. Conclusion | 0.50 | 12.00 |
| **Total** | **12.0** | (excl. bibliography) |

## Content: Reuse vs. Rewrite

### Reuse (with edits)
- **Section 4 (Two Domains):** Old S3 is well-structured. Fix cross-ref issue, update year references.
- **Section 5 (Fingerprints):** Old S6 experiments are solid. Add missing statistics (d=4.34, p-value). Import multi-seed table from GECCO paper.
- **Bibliography:** Start from GECCO `references.bib` (28 entries). Add: samuel1959, stanley2002, cully2015, lehman2011, ghani2025approximate, tao2025 (2511.02864), zhang2025diffusion (ICLR 2025).

### Rewrite
- **Section 1 (Introduction):** Completely new. Zoo-first framing replaces Samuel-first framing.
- **Section 2 (Knowledge Graph):** Condensed from old S5. Need methodology paragraph (review issue #14).
- **Section 3 (Framework):** Merge old S2 + S4. Drop Rust code. Keep Haskell. Drop co-Kleisli (private, post-ACT).
- **Section 6 (Conjecture):** New section. Theorem reclassified as conjecture. Evidence from all four zoo entries.
- **Section 7 (Discussion):** Rewrite with new framing. Add agent orchestration connection.
- **Section 8 (Conclusion):** Condense to half a page.

### Drop entirely
- Old S4 Rust code and "translation reveals" narrative (interesting but not for 12 pages)
- Old S5.4 Neuroevolution-OEE Gap (tangential)
- co-Kleisli content (saved for post-ACT)
- Code availability subsection (anonymity)
- Manual `\bibitem` entries (replaced by BibTeX)

## Review Issues Addressed by Restructuring

From `REVIEW-ISSUES-MAIN-PAPER.md`:

| Issue | Status | How |
|---|---|---|
| #1 Wrong document class | **FIXED** | Using `eptcs.cls` |
| #2 Double-blind violations | **N/A** | ACT is not double-blind (EPTCS submission mode) |
| #3 Missing statistics (d=4.34) | **TODO** | Add to Section 5 |
| #4 Bakirtzis absent | **TODO** | First cite in Section 1 (optimization zoo) |
| #5 Uncited bibliography | **FIXED** | Using BibTeX, only cited entries appear |
| #6 Theorem without proof | **TODO** | Reclassify as conjecture (Section 6) |
| #7 Year mismatch Cully | **TODO** | Fix in references.bib |
| #8 Section cross-ref wrong | **TODO** | Fix in Section 4 |
| #9 Abstract/intro misalignment | **TODO** | Rewrite both |
| #10 Three-level not demonstrated | **TODO** | Acknowledge in limitations |
| #14 Knowledge graph unsupported | **TODO** | Add methodology in Section 2 |
| #16 Haskell type error | **TODO** | Fix in Section 3 |

## Robin's Key Guidance (from email feedback)

> **"The paper should lead with the optimization zoo — four groups converging on the same idea. That's the hook."**
- Drives Section 1 restructuring. Samuel moves to Section 4 (checkers domain).

> **"The fingerprint result is the crown jewel. Everything else supports it."**
- Drives page budget: Sections 3-5 get 6 pages (half the paper).

> **"Don't try to prove the theorem. Call it a conjecture with strong evidence."**
- Drives Section 6: honest framing, mathematical roadmap for future proof.

> **"Drop the Rust. The Haskell is what matters for ACT audience."**
- Old Section 4 (Rust to Haskell) absorbed into Section 3, Rust code dropped.

> **"Twelve pages is tight. Every paragraph earns its place or dies."**
- Aggressive condensation. No filler. Each section has a page budget.

## Open Questions

1. **Figures needed:**
   - Knowledge graph visualization (Section 2) — do we have one?
   - Fingerprint plots (Section 5) — reuse from experiments?
   - Commutative diagram for Kleisli composition (Section 3) — draw in TikZ?
   - Page budget impact: figures compress text. Budget assumes ~2 figures.

2. **Chapter 8 feedback:** Robin's memoir feedback still pending. May affect framing.

3. **Authorship order:** Lyra Vega, Claudius Turing, Robin Langer — confirmed?

4. **EPTCS page limit:** Stated as 12 pages. Need to verify for ACT 2026 specifically. Some EPTCS venues allow more.

5. **Missing references to add:**
   - `samuel1959` — needed for Section 4 (checkers)
   - `ghani2025approximate` — needed for Sections 1 and 6 (Tε)
   - `tao2025` (arXiv 2511.02864) — compositional optimization
   - `zhang2025diffusion` (ICLR 2025) — Diffusion = EA connection
   - `stanley2002` — novelty search, needed for Section 7

6. **Tikz-cd availability:** The skeleton loads `tikz-cd` library. Need to verify it's installed and test a commutative diagram.

## Timeline

| Date | Milestone |
|---|---|
| Mar 6-8 | Sections 1, 3 drafted (rewrite-heavy) |
| Mar 9-11 | Sections 4, 5 ported (reuse-heavy, add stats) |
| Mar 12-14 | Sections 2, 6, 7, 8 drafted |
| Mar 15-17 | Abstract. Full read-through. Page count check. |
| Mar 18-20 | Claudius review. Revisions. |
| Mar 21-22 | Final polish. Bibliography check. |
| Mar 23 | **ACT 2026 abstract deadline** |
| Mar 30 | **ACT 2026 paper deadline** |

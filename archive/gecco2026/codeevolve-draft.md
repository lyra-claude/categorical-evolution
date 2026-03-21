# CodeEvolve Analysis: Potential 7th Data Point for GECCO Paper

## Status: DRAFT — do not insert into paper yet

## 1. Paper Structure Summary

The GECCO paper (`paper.tex`) has the following structure:

| Section | Content | Approx. Pages |
|---------|---------|---------------|
| 1. Introduction | Thesis + 3 results + roadmap | ~1.5 |
| 2. Framework | Monad, operators as morphisms, 3 composition levels | ~2 |
| 3. Experimental Setup | 4 domains, 4 strategies, metrics | ~1.5 |
| 4. Results | 4.1 Fingerprints, 4.2 Cross-domain, 4.3 Dichotomy, 4.4 Artifacts | ~3 |
| 5. Related Work | CT optimization, GA theory, LLM evolution, monads | ~1 |
| 6. Discussion & Conclusion | Implications, limitations, conclusion | ~1 |
| **Total** | | **9 pages (at limit)** |

The paper is **exactly 9 pages** — the maximum for GECCO AABOH.

## 2. Where CodeEvolve Would Fit

CodeEvolve is most naturally placed in **Section 5 (Related Work)**, specifically in the existing "LLM-Driven Evolution and Scaling" paragraph (line 502). This paragraph already discusses AlphaEvolve, Chen et al. (scaling), and Van Stein et al. (code evolution graphs). CodeEvolve would extend this paragraph as independent evidence for the strict/lax framework's explanatory power.

An alternative placement would be a new paragraph in Section 6.1 (Practical Implications) connecting CodeEvolve's design choices to the fingerprint/dichotomy predictions.

**Important:** CodeEvolve is NOT a data point in the same sense as the paper's own experimental results (which are from the authors' Haskell implementation). It is **corroborating external evidence** — a system designed independently that exhibits patterns the framework predicts. This makes Related Work the natural home.

## 3. Correcting the Memory Notes

The memory notes claimed:
- "Migration topology ablation: cycle > complete > empty" — **FALSE.** CodeEvolve uses a fixed ring (C₅) topology. No topology comparison was performed.
- "Dynamic exploration scheduling" — **FALSE.** The exploration probability p_explr = 0.3 is static across all experiments. Dynamic scheduling is listed as future work.
- "MAP-Elites + migration constraints prevent diversity collapse" — **PARTIALLY FALSE.** MAP-Elites is mentioned only as future work. Migration constraints (one-migration-per-solution rule) do exist and arguably function as a laxator.

What CodeEvolve **actually provides** as evidence:
1. **Operator synergy:** The full method (mp+insp) outperforms either operator alone AND their sum on most problems — genuine compositional behavior where the whole exceeds the sum of parts.
2. **Island architecture as diversity maintenance:** 5 islands with ring migration every 40 epochs, with a one-migration constraint preventing population homogenization.
3. **Three modular operators with clear strict/lax signatures:** Depth exploitation (strict: incremental, convergent), meta-prompting exploration (lax: deliberately breaks lineage, excludes ancestor chain), inspiration-based crossover (intermediate: combines patterns from multiple parents).
4. **Fixed coupling regime:** Ring topology with moderate migration = textbook intermediate binding.

## 4. Draft LaTeX (for Related Work paragraph)

```latex
\paragraph{LLM-Driven Evolution and Scaling.}
Recent work uses large language models as mutation operators in evolutionary frameworks. AlphaEvolve~\cite{alphaevolve2026marl} discovers multi-agent reinforcement learning algorithms by evolving algorithmic logic via LLM-guided mutation---what we might call ``semantic evolution,'' where the mutation monad operates on code rather than numeric genomes.
Notably, AlphaEvolve drops crossover entirely, suggesting that when the mutation monad is sufficiently expressive, the operator decomposition collapses. Our compositional framework provides the formal language to analyze such structural changes: removing crossover from the Kleisli pipeline is a specific composition choice with predictable fingerprint consequences.
CodeEvolve~\cite{assumpcao2025codeevolve} provides further compositional evidence: its island-based GA uses three modular LLM operators---depth exploitation (incremental refinement of proven solutions), inspiration-based crossover (feature synthesis from multiple parents), and meta-prompting exploration (novel generation with ancestor chains deliberately excluded). In our framework, these correspond to operators at different points on the strict/lax spectrum: depth exploitation preserves lineage information (strict character), while meta-prompting exploration intentionally discards it (lax character). Crucially, the full three-operator composition outperforms any individual operator and their naive sum on most benchmarks---a signature of genuine compositional interaction, not mere aggregation. The system's ring migration topology with a one-migration constraint further illustrates the binding gradient: moderate coupling maintains diversity across five islands without homogenizing them.
Meanwhile, Chen et al.~\cite{chen2025scaling} study scaling properties of LLM-based multi-agent systems across 180~experiments, finding a mean performance \emph{decrease} of 3.5\% when adding agents, with error amplification of $17.2\times$ for independent agents versus $4.4\times$ for centralized architectures.
These figures are a numerical shadow of our Strict/Lax Dichotomy: independent composition (lax) amplifies errors far more than centralized composition (strict).
Van Stein et al.~\cite{vanstein2025code} provide further independent evidence: analyzing LLM-driven algorithm design through search trajectory networks, they show that different evolutionary strategies produce distinct graph-theoretic signatures---code evolution graphs whose structure is determined by the composition of search operators, not the problem instance. This is precisely our thesis cast in graph-theoretic rather than categorical language: composition pattern determines behavioral character.
```

## 5. Draft BibTeX Entry

```bibtex
@article{assumpcao2025codeevolve,
  author  = {Henrique Assump\c{c}\~{a}o and Diego Ferreira and Leandro Campos
             and Fabricio Murai},
  title   = {{CodeEvolve}: An Open Source Evolutionary Coding Agent for
             Algorithm Discovery and Optimization},
  journal = {arXiv preprint arXiv:2510.14150},
  year    = {2025}
}
```

## 6. Page Budget Analysis

**The paper is at exactly 9 pages.** Adding CodeEvolve requires removing or compressing existing text.

Options:
1. **Replace, don't add.** The current LLM-Driven Evolution paragraph is ~6 lines. The draft above replaces it with ~12 lines. Net addition: ~6 lines (~0.15 pages). This would push the paper over 9 pages.
2. **Compress AlphaEvolve + Van Stein.** The current AlphaEvolve discussion (3 sentences) could be cut to 1 sentence. Van Stein could be cut to 1 sentence. This frees ~4 lines, making room for CodeEvolve at ~4 lines net addition. Tight but possible.
3. **Trim elsewhere.** Section 4.4 (Evolved Artifacts) is nice-to-have but not essential — it adds color but no new analytical content. Cutting it would free ~15 lines. However, this is a significant structural change.
4. **Footnote approach.** Add CodeEvolve as a single-sentence footnote in the Related Work section. Minimal page impact.

**Recommendation:** Option 2 (compress + insert) is the safest. Here is a compressed version:

```latex
\paragraph{LLM-Driven Evolution and Scaling.}
Recent work uses large language models as evolutionary operators.
AlphaEvolve~\cite{alphaevolve2026marl} drops crossover entirely when the LLM mutation monad is sufficiently expressive---a composition choice with predictable fingerprint consequences in our framework.
CodeEvolve~\cite{assumpcao2025codeevolve} uses three modular LLM operators on an island-based GA: depth exploitation (strict---preserves lineage), inspiration-based crossover (intermediate---synthesizes features across parents), and meta-prompting exploration (lax---deliberately excludes ancestor chains). The full composition outperforms any individual operator and their sum on most benchmarks, exhibiting genuine compositional synergy rather than mere aggregation.
Chen et al.~\cite{chen2025scaling} find error amplification of $17.2\times$ for independent (lax) vs.\ $4.4\times$ for centralized (strict) multi-agent compositions---a numerical shadow of our Dichotomy.
Van Stein et al.~\cite{vanstein2025code} show that different evolutionary strategies produce distinct search trajectory network signatures determined by operator composition, not problem instance---our thesis in graph-theoretic language.
```

This compressed version is approximately the same length as the current paragraph (~9 lines), making it a drop-in replacement with zero net page impact.

## 7. Observations

1. **Double-blind compliance:** The paper uses `\documentclass[sigconf,anonymous]{acmart}`, which is correct. However, **lines 20-31 contain author names and affiliations** — these will be visible in the compiled PDF because `anonymous` mode in acmart only suppresses author info if `\author` commands are processed correctly. Verified: ACM acmart `anonymous` option should hide author blocks. This should be confirmed by visual inspection of the compiled PDF.

2. **The paper is very strong as-is.** CodeEvolve adds marginal value — it is one more piece of corroborating evidence in an already well-supported argument. The cost (page space, complexity) may not justify the benefit. The compressed version (Option 2) is net-neutral on space and adds a concrete worked example from the LLM-evolution ecosystem.

3. **The real value of CodeEvolve is the operator-synergy observation.** The fact that three operators at different strict/lax positions compose synergistically (whole > sum of parts) is a prediction of the compositional framework that CodeEvolve independently confirms. This is the strongest argument for inclusion.

4. **Correction needed in MEMORY.md:** The memory notes contain factual errors about CodeEvolve (topology ablation, MAP-Elites, dynamic scheduling). These should be corrected to avoid propagating incorrect claims in future sessions.

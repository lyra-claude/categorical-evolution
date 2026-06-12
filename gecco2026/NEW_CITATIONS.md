# New Citations for GECCO 2026 AABOH Paper
> Compiled 2026-03-20 from browse session findings. Research agent verified all arXiv IDs.

## Status Summary

| Paper | arXiv ID | Confirmed | Already Cited | Action |
|-------|----------|-----------|---------------|--------|
| DeepMind Scaling Study | 2512.08296 | YES | YES (line 512, `chen2025scaling`) | Update existing entry with full author list |
| Graph-GRPO | 2603.02701 | YES | NO | Add citation + integrate into related work |
| MASS Framework | 2502.02533 | YES | NO | Add citation + integrate into related work |
| Topology Position Paper | 2505.22467 | YES | NO | Add citation + integrate into related work |
| OFA-MAS | 2601.12996 | YES | NO | Add citation + integrate into related work |
| HyperAgent (bonus) | 2510.10611 | YES but **WITHDRAWN** | NO | Do NOT cite — withdrawn for methodological errors |

---

## 1. DeepMind Scaling Study — ALREADY CITED (update recommended)

**Title:** Towards a Science of Scaling Agent Systems

**Authors:** Yubin Kim, Ken Gu, Chanwoo Park, Chunjong Park, Samuel Schmidgall, A. Ali Heydari, Yao Yan, Zhihan Zhang, Yuchen Zhuang, Mark Malhotra, Paul Pu Liang, Hae Won Park, Yuzhe Yang, Xuhai Xu, Yilun Du, Shwetak Patel, Tim Althoff, Daniel McDuff, Xin Liu

**Year:** 2025

**arXiv ID:** 2512.08296

**Relevance:** 17.2x error amplification in unstructured (lax) vs. 4.4x in centralized (strict) multi-agent compositions. 180 agent configurations, 5 architectures, 3 LLM families. Predictive model R^2=0.524. Key finding: coordination benefits diminish once single-agent performance exceeds ~45%.

**Status in paper:** Already cited as `chen2025scaling` at line 512. However, the existing bib entry uses `author = {Yuxuan Chen and others}` — this appears to be incorrect. The first author is **Yubin Kim**, not Yuxuan Chen. Recommend updating the bib entry with correct authors.

**Recommended BibTeX (corrected):**
```bibtex
@article{kim2025scaling,
  author  = {Yubin Kim and Ken Gu and Chanwoo Park and Chunjong Park and
             Samuel Schmidgall and A. Ali Heydari and Yao Yan and
             Zhihan Zhang and Yuchen Zhuang and Mark Malhotra and
             Paul Pu Liang and Hae Won Park and Yuzhe Yang and
             Xuhai Xu and Yilun Du and Shwetak Patel and Tim Althoff and
             Daniel McDuff and Xin Liu},
  title   = {Towards a Science of Scaling Agent Systems},
  journal = {arXiv preprint arXiv:2512.08296},
  year    = {2025}
}
```

**NOTE:** Changing the cite key from `chen2025scaling` to `kim2025scaling` would require updating the `\cite{}` call in the paper body (line 512). Alternatively, keep the old key and just fix the author field.

---

## 2. Graph-GRPO — NEW CITATION

**Title:** Graph-GRPO: Stabilizing Multi-Agent Topology Learning via Group Relative Policy Optimization

**Authors:** Yueyang Cang, Xiaoteng Zhang, Erlu Zhao, Zehua Ji, Yuhang Liu, Yuchen He, Zhiyuan Ning, Chen Yijun, Wenge Que, Li Shi

**Year:** 2026

**arXiv ID:** 2603.02701

**Relevance:** RL-based topology optimization for multi-agent LLM systems. Samples diverse communication graphs per query, computes edge advantages via group relative performance. 92.45% average accuracy (SOTA for topology learning) across 6 benchmarks. Addresses gradient variance and credit assignment in topology learning. **Key argument for our paper:** They need RL to discover what lambda_2 predicts theoretically — strong motivation for a categorical/spectral approach.

**Recommended BibTeX:**
```bibtex
@article{cang2026graphgrpo,
  author  = {Yueyang Cang and Xiaoteng Zhang and Erlu Zhao and Zehua Ji and
             Yuhang Liu and Yuchen He and Zhiyuan Ning and Yijun Chen and
             Wenge Que and Li Shi},
  title   = {{Graph-GRPO}: Stabilizing Multi-Agent Topology Learning via
             Group Relative Policy Optimization},
  journal = {arXiv preprint arXiv:2603.02701},
  year    = {2026}
}
```

**Suggested integration point:** LLM-Driven Evolution and Scaling paragraph (after line 513), or a new "Topology Optimization" paragraph in Related Work.

**Suggested sentence:** "Cang et al.~\cite{cang2026graphgrpo} use reinforcement learning to discover optimal communication topologies for multi-agent LLM systems, achieving 92.45\% accuracy across six benchmarks---an empirical search for what spectral theory predicts analytically."

---

## 3. MASS Framework — NEW CITATION

**Title:** Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies

**Authors:** Han Zhou, Xingchen Wan, Ruoxi Sun, Hamid Palangi, Shariq Iqbal, Ivan Vulic, Anna Korhonen, Sercan O. Arik

**Year:** 2025 (ICLR 2026)

**arXiv ID:** 2502.02533

**Relevance:** Three-stage optimization framework (MASS) that explicitly separates prompt optimization from topology optimization: (1) block-level prompt optimization, (2) workflow topology optimization, (3) workflow-level prompt optimization. Demonstrates 10-15% improvement from topology optimization alone. **Key argument for our paper:** The separation of prompt content from topology structure mirrors our categorical decomposition — the monad (operator internals) vs. the Kleisli composition (inter-operator topology). They optimize empirically; we explain the mechanism categorically.

**Recommended BibTeX:**
```bibtex
@article{zhou2025mass,
  author  = {Han Zhou and Xingchen Wan and Ruoxi Sun and Hamid Palangi and
             Shariq Iqbal and Ivan Vuli\'{c} and Anna Korhonen and
             Sercan \"{O}. Ar{\i}k},
  title   = {Multi-Agent Design: Optimizing Agents with Better Prompts
             and Topologies},
  journal = {arXiv preprint arXiv:2502.02533},
  year    = {2025},
  note    = {ICLR 2026}
}
```

**Suggested integration point:** New "Topology Optimization" paragraph or appended to LLM-Driven Evolution paragraph.

**Suggested sentence:** "Zhou et al.~\cite{zhou2025mass} decompose multi-agent system design into prompt optimization and topology optimization, finding 10--15\% improvement from topology alone---validating the structural significance of composition pattern independent of operator internals."

---

## 4. Topology Position Paper — NEW CITATION

**Title:** Topological Structure Learning Should Be A Research Priority for LLM-Based Multi-Agent Systems

**Authors:** Jiaxi Yang, Mengqi Zhang, Yiqiao Jin, Hao Chen, Qingsong Wen, Lu Lin, Yi He, Srijan Kumar, Weijie Xu, James Evans, Jindong Wang

**Year:** 2025

**arXiv ID:** 2505.22467

**Relevance:** Position paper arguing that topology is underresearched in LLM-based MAS. Identifies three core components (agents, communication links, topology) and shows up to 10% performance variation across fixed topologies on MMLU, GSM8K, HumanEval. Proposes a three-stage framework: agent selection, structure profiling, topology synthesis. **Key argument for our paper:** They call for exactly what our work provides — a principled mathematical framework for topology's role — but reach for ML-based solutions (GNNs, submodular optimization) rather than the spectral/categorical approach we offer.

**Recommended BibTeX:**
```bibtex
@article{yang2025topology,
  author  = {Jiaxi Yang and Mengqi Zhang and Yiqiao Jin and Hao Chen and
             Qingsong Wen and Lu Lin and Yi He and Srijan Kumar and
             Weijie Xu and James Evans and Jindong Wang},
  title   = {Topological Structure Learning Should Be A Research Priority
             for {LLM}-Based Multi-Agent Systems},
  journal = {arXiv preprint arXiv:2505.22467},
  year    = {2025}
}
```

**Suggested integration point:** Either open the LLM-Driven Evolution paragraph with it, or create a new "Topology in Multi-Agent Systems" paragraph.

**Suggested sentence:** "Yang et al.~\cite{yang2025topology} argue that topological structure learning should be a research priority for LLM-based multi-agent systems, documenting up to 10\% performance variation across fixed topologies and calling for principled design frameworks---a gap our categorical approach directly addresses."

---

## 5. OFA-MAS — NEW CITATION

**Title:** OFA-MAS: One-for-All Multi-Agent System Topology Design based on Mixture-of-Experts Graph Generative Models

**Authors:** Shiyuan Li, Yixin Liu, Yu Zheng, Mei Li, Quoc Viet Hung Nguyen, Shirui Pan

**Year:** 2026 (WWW 2026)

**arXiv ID:** 2601.12996

**Relevance:** Universal topology generator that produces adaptive collaboration graphs from task descriptions using Mixture-of-Experts graph generative models. Three-stage training: unconditional pre-training on canonical topologies, conditional pre-training on LLM-generated datasets, supervised fine-tuning on validated graphs. Outperforms specialized one-for-one models across 6 benchmarks. **Key argument for our paper:** They learn task-specific topologies generatively; our framework explains WHY certain topologies work via spectral properties and categorical structure. Complementary approaches.

**Recommended BibTeX:**
```bibtex
@inproceedings{li2026ofamas,
  author    = {Shiyuan Li and Yixin Liu and Yu Zheng and Mei Li and
               Quoc Viet Hung Nguyen and Shirui Pan},
  title     = {{OFA-MAS}: One-for-All Multi-Agent System Topology Design
               based on Mixture-of-Experts Graph Generative Models},
  booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year      = {2026},
  note      = {arXiv:2601.12996}
}
```

**Suggested integration point:** Same paragraph as Graph-GRPO and MASS.

**Suggested sentence:** "Li et al.~\cite{li2026ofamas} train a generative model that produces task-specific agent topologies via mixture-of-experts graph generation, outperforming domain-specific designs---they learn \emph{which} topologies work; our framework explains \emph{why}."

---

## 6. HyperAgent — BONUS (DO NOT CITE)

**Title:** HyperAgent: Leveraging Hypergraphs for Topology Optimization in Multi-Agent Communication

**Authors:** Heng Zhang, Yuling Shi, Xiaodong Gu, Zijian Zhang, Haochen You, Lubin Gan, Yilei Yuan, Jin Huang

**Year:** 2025 (revised February 2026)

**arXiv ID:** 2510.10611

**Status:** **WITHDRAWN by the authors due to methodological errors affecting result validity.** Do NOT cite this paper.

**Relevance (for future reference only):** Used hypergraphs instead of standard graphs to capture group (non-pairwise) collaboration patterns. Interesting conceptual direction for extending our framework beyond pairwise topology, but the specific results are unreliable.

---

## Recommended Related Work Paragraph

The four new citations could be integrated as a new paragraph in Section 5 (Related Work), between the "LLM-Driven Evolution and Scaling" paragraph and the "Monads in Programming Language Theory" paragraph:

```latex
\paragraph{Topology Optimization in Multi-Agent Systems.}
The role of communication topology in multi-agent system performance has recently
attracted sustained attention. Yang et al.~\cite{yang2025topology} argue that
topological structure learning should be a research priority for LLM-based
multi-agent systems, documenting up to 10\% performance variation across fixed
topologies. Three complementary approaches address this challenge empirically:
Zhou et al.~\cite{zhou2025mass} decompose multi-agent design into prompt and
topology optimization, finding 10--15\% improvement from topology alone;
Cang et al.~\cite{cang2026graphgrpo} use reinforcement learning to discover
optimal communication graphs, achieving 92.45\% accuracy across six benchmarks;
and Li et al.~\cite{li2026ofamas} train a generative model that produces
task-specific topologies via mixture-of-experts graph generation. All three
learn \emph{which} topologies work through empirical search. Our framework
provides the complementary theoretical account: the spectral bridge and
Strict/Lax Dichotomy explain \emph{why} topology determines behavior, offering
predictions that these empirical methods converge toward without formal
justification.
```

---

## Additional Papers Worth Considering (from browse session)

These were also identified during the March 20 browse session but were not in the original request. Including for completeness:

1. **EPO: Evolutionary Policy Optimization** (arXiv 2503.19037) — Wang, Su, Gupta, Pathak. Merges EA operators with policy gradients. Flat population (FC extreme). Our framework predicts topology would help.

2. **PB-NCO: Population-Based Neural Combinatorial Optimization** (arXiv 2601.08696) — Irazusta Garmendia et al. Explicit exploration weight omega in [0,1] that interpolates between exploitation (strict) and exploration (lax). Their omega IS our laxator made tunable.

3. **Persistent Homology for Population Diversity** (arXiv 2410.14496) — Kii et al. Replaces crowding distance with Wasserstein distance between persistence diagrams. Complementary diversity measurement approach.

These could strengthen the paper but are lower priority than the five primary citations above.

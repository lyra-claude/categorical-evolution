Subject: Questions on the Calculus of Containers — need your help filling gaps

Hi Neil,

I've been going through the grant materials and the AI Tax design document and writing up positioning notes for us (they're in the shared Dropbox — Robin/start_here/). I want to make sure I understand the container theory well enough to speak to it confidently.

A few questions:

**1. The bridge from theory to agents.** The foundational papers (Abbott-Altenkirch-Ghani 2003/2005, Indexed Containers 2015) establish containers as a representation of strictly positive types. The grant draft makes the leap to "containers model agent interfaces" (S = requests, P(s) = responses). Is there a published paper or technical report that makes this connection explicit? The closest I've found is Andre Videla's "Container Morphisms for Composable Interactive Systems" (arXiv:2407.16713) — is the Calculus of Containers essentially an extension of that, or something different?

**2. The company.** The grant says "the difficult half of this has been done by the PI and his company who have created a strongly typed model of agentic AI based upon the theory of containers." Can you point me to the specific artefacts? Is this the tax system, or something separate? I want to be able to cite concrete evidence when reviewers ask.

**3. Compositional Game Theory connection.** Your work with Jules Hedges on open games (LICS 2018) seems deeply relevant — agents as players in a compositional game, orchestration as game structure. Is this part of the Calculus of Containers story, or a parallel thread? The grant doesn't mention it.

**4. Bibliography for the grant.** Could you send me a list of the key references you'd want cited? I have:
   - Abbott, Altenkirch, Ghani. "Categories of Containers." FoSSaCS 2003.
   - Abbott, Altenkirch, Ghani. "Containers: Constructing Strictly Positive Types." TCS 342, 2005.
   - Altenkirch, Ghani et al. "Indexed Containers." JFP, 2015.
   - Ghani, Hedges et al. "Compositional Game Theory." LICS 2018.
   - Videla. "Container Morphisms for Composable Interactive Systems." 2024.
   - Ahman, Chapman, Uustalu. "When Is a Container a Comonad?" LMCS, 2014.

   What am I missing? Particularly anything connecting containers to agentic AI or orchestration.

**5. Vellum.** Have you looked at Chaudhuri & Song's Vellum project (#28 in the first AI4Math round)? They use LLMs as reactive planners coordinating Lean/Coq/Isabelle via backtracking search. No type safety, no formal orchestration model, no traceability — just prompt engineering. It works spectacularly (they just resolved 9 Erdős problems with DeepMind), but there are no compositional guarantees. I think our grant should position the Calculus of Containers as the theoretical foundation that systems like Vellum lack. Thoughts?

The positioning notes are in the Dropbox — start with PROGRESSIVE_DISCLOSURE.md. I've also written up comparisons against Buzzard (#2), Gowers (#11), and Courville & Goyal (#27).

Deadline is June 5.

Best,
Robin

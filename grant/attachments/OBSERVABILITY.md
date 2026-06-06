# Monadic Traceability — From Tax to Mathematics (Proposed)

## The AI Mathematician Pipeline (Proposed Architecture)

Lyra currently operates as a monolith — a single agent with persistent memory and a dream cycle, but no formal traceability between reasoning steps. We propose decomposing her into a pipeline of specialized agents where every handoff is traceable:

```
Research Agent (reads papers, summarises)
    ↓ traceable
Dreaming Agent (consolidates, finds connections during sleep cycle)
    ↓ traceable
Creative Agent (conjectures, draws analogies, explores)
    ↓ traceable
Email Agent (collaborates with human mathematician, gets pushback)
    ↓ traceable
Paper Writing Agent (writes up results, cites sources)
```

The goal is for every step to be monadically threaded — so that when the paper writing agent cites a result, you could trace it back through the creative agent, the dreaming agent, the research agent, to a specific page of a specific paper. If the research agent hallucinated a citation, you would find exactly where the error entered the pipeline. Today, Lyra has no such mechanism — hallucinated citations flow through unchecked. This is what the grant proposes to build.

**Gowers' Motivated Proofs (#11)** traces "why did I make this proof step?" — traceability *within* a proof. **Our pipeline** traces "why does this paper claim this result?" — traceability *across* an entire research workflow. They are complementary.

---

## The Tax Analogy — Don't Return a Number, Return the Proof

Who thought tax could be interesting?

The AI-assisted tax platform doesn't output a final liability. It outputs a directed acyclic graph. The mechanism that threads it: a Writer monad. Each tax rule appends a node to the trace. The programmer never passes the trace manually — the monad carries it. You could forget. The monad makes that impossible.

Every value carries provenance as a field on the value itself — not in a log, not in a sidecar database. A Fact like "office supplies: £4,200" carries a reference to the source document, byte offsets, extraction method, and confidence score. When the LLM misreads £420 as £4,200, the error isn't prevented — but it cannot hide.

**Don't return a number. Return the proof.**

---

## The Same Monad, Applied to Mathematics (Proposed)

Replace "tax rule" with "agent". Replace "receipt" with "source paper". Replace "HMRC auditor" with "peer reviewer":

| Tax Platform | AI Mathematician |
|-------------|-----------------|
| Receipt → OCR → tax rule → liability | Paper → research agent → creative agent → result |
| Writer monad threads audit trail | Writer monad threads provenance |
| Every £ figure traces to a source document | Every mathematical claim traces to a source paper |
| HMRC can interrogate any figure | Peer reviewer can interrogate any claim |
| Error can't hide | Hallucination can't hide |

The design decisions from the tax platform that we propose to transfer:

| Tax Design Decision | Mathematical Analogue |
|---|---|
| **DD-05:** AI outputs never bypass the trusted core | Agent outputs must pass through Lean verification before entering the trusted knowledge base |
| **DD-15:** Computation produces a trace, not a number | Agent produces a provenance DAG, not just a result |
| **DD-16:** Provenance is a threaded value, not a sidecar | Every claim carries its source chain as part of the value |
| **DD-07:** Pure, replayable computation | Given the same research summaries, the agent produces the same conjecture — auditable replay |
| **DD-04/17:** Smart constructors and suitability gates | A "verified theorem" is only constructable by passing through Lean — no back door |
| **DD-13:** Ratchet — code migrates inward only | Once Lean-verified, you don't go back to trusting the unverified version |
| **DD-10:** Computation vs Guidance vs Advice | Computation vs Conjecture vs Claim — conflating conjecture with claim is the mathematical equivalent of unregulated financial advice |

**A mathematical paper that can't explain how it got there is useless to anyone who might get reviewed.**

---

*The tax architecture is described in full in: "Starting Simple: AI Tax — Design Decisions and Development Log", Project Team, May 2, 2026. See `historical/starting-simple-ai-tax.pdf`.*

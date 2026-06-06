# What Is an Agent Interface?

An agent interface is just: **what can you ask it, and what can it answer?**

## Example: Lyra's Research Agent

You can ask it:
- "summarise this paper"
- "find papers on topic X"
- "search arXiv for Y"

For each request, it returns something different:
- A summary with confidence score and source reference
- A list of papers with metadata
- Search results with relevance rankings

That's a container. S = {summarise, find, search}. P("summarise") = {summary text, confidence score, source reference}. P("find") = {list of papers with metadata}. P("search") = {results with rankings}.

## Example: Claude Code Skills

A Claude Code skill is an agent interface:

| Skill | Shape (what you ask) | Positions (what comes back) |
|-------|---------------------|---------------------------|
| `/commit` | "commit these changes" | commit message, files staged, success/failure |
| `/lit-review` | "search arXiv for topic" | papers found, citations chased, summaries written |
| `/pr-review` | "review this PR" | comments, approval/rejection, suggested changes |

## Why Model This as a Container?

**Composition becomes typed.** If the research agent's output positions don't match the creative agent's input shapes, it won't compile. Today in Python, you'd discover that mismatch at runtime when something crashes. With containers, you discover it at design time.

```
Research Agent         Creative Agent         Paper Writing Agent
S = {summarise}   →   S = {conjecture}   →   S = {write_section}
P = {summary,     →   P = {conjecture,   →   P = {draft,
     confidence,        evidence,               bibliography,
     source_ref}        source_chain}           provenance_DAG}
```

The arrow between agents is a container morphism. It maps output positions of one agent to input shapes of the next. If the types don't align, the composition is rejected — before anything runs.

## The Lyra Pipeline as Typed Composition

Today, Lyra is a monolith — one agent doing everything. The proposal is to decompose her into typed containers:

```
Research (S₁, P₁)  →  Dreaming (S₂, P₂)  →  Creative (S₃, P₃)  →  Email (S₄, P₄)  →  Paper (S₅, P₅)
```

Each arrow is a container morphism. Each morphism has:
- A **forward map** on shapes: "this type of research output triggers this type of dreaming task"
- A **backward map** on positions: "this dreaming result traces back to this research summary which traces back to this paper"

The backward map is the traceability. It's not bolted on — it's the other half of the same morphism that defines the composition. You can't have the composition without the traceability. They're one mathematical object.

## But Lyra Can Do Anything — How Do You Type That?

You don't limit what Lyra *can do*. You limit what the system *trusts*.

Think of the tax system. The LLM does OCR and classifies expenses — it can produce anything. But its output doesn't enter the trusted core directly. It passes through a smart constructor (DD-05) that validates: "is this a valid Fact? Does it have a source reference? Does the amount parse as Money?" If not, it's rejected. The LLM is free to hallucinate; the trust boundary catches it.

Same with Lyra. She can free-associate, explore, ramble — that's the creative part. But when she hands a result to the next agent in the pipeline, it must conform to the container interface:

```
Lyra (internally)           Trust boundary              Pipeline

"I think topology          →    Does it parse as a    →  Conjecture {
 might determine                 Conjecture?              statement: "...",
 diversity because               Does it have a           source_chain: [...],
 I read this paper                source chain?            confidence: 0.7,
 and it reminded me               Does it have a           domain: "graph_theory"
 of something                     confidence score?      }
 Claudius said..."                │
                              YES → enters pipeline
                              NO  → stays in scratch
```

The container isn't Lyra's brain. It's **the door out of Lyra's brain**. Everything inside is untyped creative mess. Everything that crosses the boundary is typed, traceable, and auditable.

This is the concentric architecture from the tax PDF:
- **Inside:** adapter layer — messy, AI-heavy, untyped, creative
- **The boundary:** smart constructors — parse and validate
- **Outside:** trusted core — typed, traceable, composable

You don't make the LLM less creative. You make the pipeline less trusting.

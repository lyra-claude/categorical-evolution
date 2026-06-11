# ACT 2026 Work

**Applied Category Theory conference submission and related materials.**

## Status

The ACT 2026 work is developed on dedicated branches — it is NOT merged into main, to keep it clearly separate from the GECCO camera-ready materials.

## Where to Find It

| Branch | Contents |
|--------|----------|
| `feat/act2026-paper` | Full ACT 2026 paper draft (LaTeX) |
| `feat/act2026-proposal` | Talk proposal submitted to ACT 2026 |
| `experiment/oq41-falsification` | Latest WIP including `act2026/` working directory with paper.tex and integration plan |

## Submitted PDF

The ACT 2026 submitted PDF lives on main at:

```
paper/submitted/act2026-submitted/paper-submitted.pdf
```

## Key Differences from GECCO Paper

The ACT paper emphasises the categorical/functorial framing (lax functors, Kleisli morphisms) more heavily than the empirical GECCO paper. The GECCO paper focuses on the experimental result (topology ordering across six domains). They share the core result but have different audiences and framings.

## To Work on ACT Materials

Check out the relevant branch:

```bash
git checkout feat/act2026-paper
```

or inspect the WIP on `experiment/oq41-falsification` at `act2026/paper.tex`.

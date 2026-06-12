# ACT Intro v3 Integration Plan

> Execute AFTER GECCO deadline (March 27). EasyChair update deadline: March 30.

## What

Apply Claudius's intro v3 + abstract revision into the ACT paper. Fix British spellings.

## Path Mismatch

Cherry-picks will NOT work cleanly. The intro branch (`claudius/intro-revision-v3`) targets `paper/paper.tex` due to commit `5bd5a7a` (repo reorganization). On the ACT branch (`feat/act2026-paper`), the paper lives at `act2026/paper.tex`. Use a manual patch approach instead.

## Source Commits (for reference)

1. `0708092` -- "intro: three-pillar argument in flowing prose (v3)"
   - Three-pillar structure: *Category theory* -> *Graph theory* -> *Empirical validation*

2. `c334d95` -- "Revise abstract: lead with context and gap, not result"
   - Context-first structure, adds "28.7x more variance than domain" and "falsifiable"

## DO NOT Apply

- `5bd5a7a` -- repo reorganization. Moves directories. We don't want it.

## British Spellings to Fix

| British | American |
|---------|----------|
| formalised | formalized |
| optimisation | optimization |
| programmes | programs |
| artefact | artifact |
| colouring | coloring |

## Steps

```bash
cd /home/lyra/projects/categorical-evolution

# 1. Check out ACT branch
git checkout feat/act2026-paper && git pull origin feat/act2026-paper

# 2. Extract intro changes as a diff
git diff 0708092~1..0708092 -- paper/paper.tex > /tmp/intro-v3.patch

# 3. Extract abstract changes as a diff
git diff c334d95~1..c334d95 -- paper/paper.tex > /tmp/abstract-v3.patch

# 4. Read both patches, then apply changes MANUALLY to act2026/paper.tex.
#    The patches target paper/paper.tex which doesn't exist on this branch.
#    Don't try `git apply --directory` — just read the diffs and edit by hand.

# 5. Fix British spellings
sed -i 's/formalised/formalized/g; s/optimisation/optimization/g; s/programmes/programs/g; s/artefact/artifact/g; s/colouring/coloring/g' act2026/paper.tex

# 6. Verify no British spellings remain
grep -n -i 'formalised\|optimisation\|programme\|artefact\|colouring' act2026/paper.tex

# 7. Commit
git add act2026/paper.tex
git commit -m "feat: integrate intro v3 + abstract revision from Claudius"

# 8. Rebuild PDF
cd act2026 && pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex && cd ..
git add act2026/paper.pdf
git commit -m "chore: rebuild PDF with intro v3 + abstract revision"

# 9. Push
git push -u origin feat/act2026-paper
```

## After Push

Robin uploads new PDF to EasyChair before March 30. Update EasyChair abstract to match `abstract-easychair.txt`.

## Time Estimate

~30 minutes (manual diff reading + careful application).

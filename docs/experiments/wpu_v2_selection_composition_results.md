# WPU V2 Working-Set Selection Composition

## Purpose

The interaction-density retrieval experiment improved deployed loss, but loss
alone does not explain the mechanism. This diagnostic measures what each
pre-tensor retriever actually admits into the working set.

Output:

- `scripts/analyze_working_set_selection.py`
- `docs/experiments/wpu_v2_selection_composition.csv`

## Setup

- N = 2048
- K = 8, 16, 32
- Budgets = 4, 8, 16, 32
- Samples = 256
- Selection modes = indexed, proximity, interaction
- Interaction mode = pairwise

## Budget 4 Result

Budget 4 is the critical under-complete setting from the K-expansion
experiments.

| K | selection | causal recall | obstacle recall | selected obstacles | hand hit | selected obstacle-pair density |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | indexed | 0.500 | 0.000 | 0.000 | 1.000 | 0.000 |
| 8 | proximity | 0.500 | 0.562 | 2.246 | 0.398 | 0.254 |
| 8 | interaction | 0.500 | 0.500 | 2.000 | 1.000 | 0.625 |
| 16 | indexed | 0.250 | 0.000 | 0.000 | 1.000 | 0.000 |
| 16 | proximity | 0.250 | 0.209 | 2.504 | 0.332 | 0.551 |
| 16 | interaction | 0.250 | 0.167 | 2.000 | 1.000 | 0.812 |
| 32 | indexed | 0.125 | 0.000 | 0.000 | 1.000 | 0.000 |
| 32 | proximity | 0.125 | 0.092 | 2.590 | 0.301 | 0.799 |
| 32 | interaction | 0.125 | 0.071 | 2.000 | 1.000 | 0.902 |

## Interpretation

This explains the retrieval result.

Indexed retrieval preserves the contact hand but admits no obstacles at budget
4, so it misses the pairwise interaction evidence. Proximity retrieval admits
obstacles, but often drops the hand anchor. Interaction-density retrieval keeps
the hand and selects obstacle pairs with much higher local pair density.

The key metric is not raw causal recall. At budget 4, every selector can only
cover a small fraction of K when K is large. The useful distinction is
composition:

```text
contact anchor retained + high-density obstacle pair selected
```

This is why interaction retrieval improves loss without increasing global
state processing.

## Consequence

The next WPU v2 retriever should report at least:

- selected object type composition,
- contact-anchor retention,
- obstacle/topical causal recall,
- local interaction density,
- downstream regret and branch accuracy.

This makes WPU retrieval falsifiable. A sparse working set is only meaningful
if it contains the right causal structure.

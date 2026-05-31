# WPU v2 Generated Candidate Retriever Reranker

Source CSV: `docs/experiments/wpu_v2_retriever_generated_candidates.csv`

## Purpose

The object-set reranker improved retrieval decisions across K=8,16,32, but it was still restricted to four fixed candidate selectors: indexed, proximity, interaction, and learned. This experiment tests whether WPU improves when the retrieval system also generates additional local state candidates.

The generated candidates are still state-native and sparse:

- they are sampled only from the event frontier;
- the event target is always included;
- candidates combine hand, table/edge anchors, and density-ranked obstacle subsets;
- no token sequence or full-state tensorization is used for candidate generation.

The object-set reranker then scores the fixed and generated candidates from explicit selected-object features.

## Command

```powershell
.\.venv\Scripts\python.exe scripts\retriever_generated_candidate_probe.py `
  --n-values 2048 `
  --k-values 8 16 32 `
  --seeds 11 13 17 19 23 `
  --budget 4 `
  --generated-candidates 4 `
  --propagation-steps 40 `
  --retriever-steps 400 `
  --reranker-steps 600 `
  --validation-samples 180 `
  --samples 90 `
  --batch-size 10 `
  --device cuda `
  --safe-margin 0.005 `
  --out docs\experiments\wpu_v2_retriever_generated_candidates.csv
```

## Main Results

| K | Policy | Loss | Accuracy | Excess over generated oracle | Selected generated rate |
|---:|---|---:|---:|---:|---:|
| 8 | static base validation choice | 0.987487 | 0.506667 | 0.031020 | 0.000000 |
| 8 | generated-set reranker | 0.973823 | 0.528889 | 0.017355 | 0.371111 |
| 8 | base oracle | 0.962094 | 0.551111 | 0.005626 | 0.000000 |
| 8 | generated oracle | 0.956468 | 0.557778 | 0.000000 | 0.442222 |
| 16 | static base validation choice | 0.958607 | 0.504444 | 0.052539 | 0.000000 |
| 16 | generated-set reranker | 0.951378 | 0.511111 | 0.045310 | 0.384444 |
| 16 | base oracle | 0.915192 | 0.566667 | 0.009125 | 0.000000 |
| 16 | generated oracle | 0.906068 | 0.580000 | 0.000000 | 0.477778 |
| 32 | static base validation choice | 1.004628 | 0.477778 | 0.035905 | 0.000000 |
| 32 | generated-set reranker | 0.999229 | 0.506667 | 0.030506 | 0.422222 |
| 32 | base oracle | 0.974522 | 0.575556 | 0.005798 | 0.000000 |
| 32 | generated oracle | 0.968723 | 0.577778 | 0.000000 | 0.513333 |

## Comparison to Previous Object-Set Reranker

| K | Fixed-candidate object-set reranker | Generated-candidate reranker | Change |
|---:|---:|---:|---:|
| 8 | 0.976833 | 0.973823 | -0.003010 |
| 16 | 0.957018 | 0.951378 | -0.005640 |
| 32 | 1.003592 | 0.999229 | -0.004363 |

Compared with the static base validation choice:

- K=8: loss improves by 0.013664 and accuracy improves by 0.022222.
- K=16: loss improves by 0.007229 and accuracy improves by 0.006667.
- K=32: loss improves by 0.005399 and accuracy improves by 0.028889.

## Interpretation

At this stage of the v2 sequence, this was the strongest generated-candidate
retrieval result. It shows two separate effects:

- Candidate generation expands the achievable oracle: generated oracle improves over base oracle by 0.005626 at K=8, 0.009125 at K=16, and 0.005798 at K=32.
- The learned object-set reranker recovers part of that new oracle gap while remaining deployed and state-native.

The generated candidates are selected frequently by the reranker: 37.1% at K=8, 38.4% at K=16, and 42.2% at K=32. This means the gain is not a bookkeeping artifact; the policy actively uses generated state subsets.

## Scientific Consequence

The WPU thesis becomes more precise:

> WPU is not just sparse propagation over a fixed graph. It needs learned state working-set generation, object-set scoring, and selective propagation.

This moves the work away from a weak claim that handcrafted state heuristics beat token processing. The stronger and more defensible claim is that explicit state enables a different learning problem: generate and score small causal working sets before expensive global tensor computation.

The result is still not a proof of broad superiority. The oracle gap remains nontrivial, and the synthetic task is narrow. However, the result gives a concrete mechanism that improves as the architecture becomes more state-native rather than more token-like.

## Next Step

The next experiment should test robustness of candidate generation:

- sweep the number of generated candidates;
- evaluate whether gains persist at larger N and larger K;
- replace heuristic generation with a learned generator;
- co-train generator, reranker, and propagation so retrieval is optimized for downstream dynamics instead of post-hoc validation loss.

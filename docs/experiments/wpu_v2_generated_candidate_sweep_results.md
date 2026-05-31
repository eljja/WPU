# WPU v2 Generated Candidate Count Sweep

Source CSV: `docs/experiments/wpu_v2_retriever_generated_candidate_sweep.csv`

## Purpose

The first generated-candidate experiment used four generated working-set candidates. This sweep tests whether that choice is robust. The goal is to separate two effects:

- whether more generated candidates expand the best achievable oracle;
- whether the deployed object-set reranker can reliably select among a larger candidate pool.

The sweep uses the same N=2048, K=8/16/32, five-seed setup as the previous generated-candidate experiment. New runs evaluate generated candidate counts 1, 2, and 8; the prior generated count 4 result is included for comparison.

## Command

```powershell
.\.venv\Scripts\python.exe scripts\retriever_generated_candidate_sweep.py `
  --n-values 2048 `
  --k-values 8 16 32 `
  --seeds 11 13 17 19 23 `
  --generated-candidate-values 1 2 8 `
  --budget 4 `
  --propagation-steps 40 `
  --retriever-steps 400 `
  --reranker-steps 600 `
  --validation-samples 180 `
  --samples 90 `
  --batch-size 10 `
  --device cuda `
  --safe-margin 0.005 `
  --out docs\experiments\wpu_v2_retriever_generated_candidate_sweep.csv
```

## Deployed Gain vs Static Base

| Generated candidates | K=8 loss delta | K=16 loss delta | K=32 loss delta |
|---:|---:|---:|---:|
| 1 | -0.010669 | +0.000111 | -0.002874 |
| 2 | -0.013166 | -0.004194 | -0.007733 |
| 4 | -0.013664 | -0.007229 | -0.005399 |
| 8 | -0.011548 | -0.004201 | -0.007399 |

Accuracy deltas versus the static base validation choice:

| Generated candidates | K=8 accuracy delta | K=16 accuracy delta | K=32 accuracy delta |
|---:|---:|---:|---:|
| 1 | +0.017778 | +0.000000 | +0.022222 |
| 2 | +0.022222 | +0.008889 | +0.040000 |
| 4 | +0.022222 | +0.006667 | +0.028889 |
| 8 | +0.015556 | +0.017778 | +0.024444 |

## Oracle Expansion

Generated oracle improvement over the fixed-candidate base oracle:

| Generated candidates | K=8 oracle delta | K=16 oracle delta | K=32 oracle delta |
|---:|---:|---:|---:|
| 1 | -0.001136 | -0.000859 | -0.000780 |
| 2 | -0.005295 | -0.008583 | -0.005337 |
| 4 | -0.005626 | -0.009124 | -0.005799 |
| 8 | -0.006834 | -0.011461 | -0.006780 |

## Interpretation

More generated candidates expand the oracle. The generated oracle improves as the candidate pool grows, especially at K=16. This confirms that local state candidate generation is not merely reshuffling the four fixed selectors; it discovers working sets that the fixed selectors miss.

The deployed reranker does not improve monotonically with candidate count. At K=8, four candidates are best. At K=16, four candidates are best by loss, while eight candidates give the highest accuracy. At K=32, two candidates are best by both loss and accuracy. This means candidate generation and candidate scoring are now separate bottlenecks.

The most practical current setting is 2-4 generated candidates:

- 2 candidates gives the best K=32 deployed result and strong K=8/K=16 gains.
- 4 candidates gives the best K=8/K=16 deployed loss and still improves K=32.
- 8 candidates expands the oracle further but makes deployed scoring harder.

## Scientific Consequence

This sweep strengthens the WPU v2 claim but narrows it appropriately:

> Explicit state enables useful local candidate generation, but the value of generation depends on a learned scorer that can handle candidate-pool growth.

The important distinction is that the bottleneck has moved. Earlier failures were caused by missing causal state. The generated-candidate pipeline now exposes enough useful state that the limiting factor is the reranker's scoring capacity and calibration.

This is a productive research position: WPU v2 now has a measurable regime where state-native candidate generation improves deployed loss, and a clear falsifiable next step.

## Next Step

The next mechanism should improve candidate scoring rather than simply generating more candidates:

- increase reranker capacity only modestly and measure overfitting;
- use pairwise ranking loss between candidates instead of only cross-entropy to the best candidate;
- add calibration or validation-margin selection over candidate count;
- train one cross-seed reranker instead of a per-seed reranker;
- eventually learn the candidate generator instead of using stochastic heuristics.

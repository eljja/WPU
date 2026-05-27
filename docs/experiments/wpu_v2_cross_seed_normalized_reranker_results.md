# WPU v2 Normalized Cross-Seed Reranker Probe

## Purpose

The first cross-seed reranker showed weak transfer. One likely cause was absolute loss-scale overfitting: each seed trains a different propagation model, so raw candidate cross-entropy values may not share a stable scale.

This experiment trains a leave-one-seed-out reranker using per-example normalized candidate losses:

- subtract the mean candidate loss within each sample;
- divide by the within-sample candidate-loss standard deviation;
- train the reranker on relative/listwise utility rather than raw loss magnitude.

The protocol remains strict: the held-out seed's validation examples are not used for training or selector choice.

## Command

```powershell
.\.venv\Scripts\python.exe scripts\retriever_cross_seed_normalized_reranker_probe.py `
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
  --out docs\experiments\wpu_v2_retriever_cross_seed_normalized_reranker.csv
```

## Results

| K | Standard cross-seed loss | Normalized cross-seed loss | Loss change | Standard acc. | Normalized acc. | Accuracy change |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.989754 | 0.988200 | -0.001554 | 0.506667 | 0.517778 | +0.011111 |
| 16 | 0.965120 | 0.962778 | -0.002342 | 0.495556 | 0.506667 | +0.011111 |
| 32 | 1.003401 | 1.003886 | +0.000485 | 0.500000 | 0.495556 | -0.004444 |

Compared with cross-seed static base choice:

| K | Static loss | Normalized reranker loss | Loss change | Static acc. | Normalized acc. | Accuracy change |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.989850 | 0.988200 | -0.001650 | 0.502222 | 0.517778 | +0.015556 |
| 16 | 0.966183 | 0.962778 | -0.003404 | 0.504444 | 0.506667 | +0.002222 |
| 32 | 1.004095 | 1.003886 | -0.000208 | 0.475556 | 0.495556 | +0.020000 |

Same-seed reranking remains substantially better:

| K | Same-seed loss | Normalized cross-seed loss | Gap |
|---:|---:|---:|---:|
| 8 | 0.973823 | 0.988200 | +0.014377 |
| 16 | 0.951378 | 0.962778 | +0.011400 |
| 32 | 0.999229 | 1.003886 | +0.004657 |

## Interpretation

Normalization helps, but it does not solve cross-seed transfer. The gains over the standard cross-seed reranker at K=8 and K=16 show that raw loss-scale overfitting was part of the problem. The weak or negative K=32 result shows that normalization alone is not sufficient.

The key conclusion is narrower:

> Cross-seed retrieval scoring needs both scale-invariant objectives and model-invariant candidate features.

The current reranker still depends on the held-out propagation model's loss landscape. Same-seed performance remains much stronger, so robust WPU retrieval requires either co-training retrieval with propagation or learning features that transfer across propagation models.

## Next Step

The next robust direction is not more heuristic candidate generation. It is calibration and shared training:

- train a single reranker over pooled seeds and evaluate on unseen seeds and unseen K;
- remove candidate identity one-hot features to test whether selector identity overfits;
- add model-state diagnostics as features, such as sparse confidence or route entropy;
- co-train the retriever/reranker with propagation so the score function is not fitted post-hoc to separate models.

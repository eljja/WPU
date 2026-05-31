# WPU v2 Cross-Seed Reranker Probe

Source CSV: `docs/experiments/wpu_v2_retriever_cross_seed_reranker.csv`

## Purpose

Previous generated-candidate reranker experiments trained and selected policies from validation examples produced by the same seed/model condition used for testing. That is useful for debugging, but it is not a strong generalization claim.

This experiment uses a stricter leave-one-seed-out protocol:

- train one WPU propagation model per seed as before;
- collect validation/test candidate examples for each seed;
- for each held-out seed, train the reranker only on validation examples from the other seeds;
- evaluate on the held-out seed test set;
- do not use the held-out seed validation loss for reranker training or static selector selection.

The candidate setting is the current practical setting: four generated candidates.

## Command

```powershell
.\.venv\Scripts\python.exe scripts\retriever_cross_seed_reranker_probe.py `
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
  --out docs\experiments\wpu_v2_retriever_cross_seed_reranker.csv
```

## Results

| K | Cross-seed static loss | Cross-seed reranker loss | Delta | Cross-seed static acc. | Cross-seed reranker acc. | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.989850 | 0.989754 | -0.000096 | 0.502222 | 0.506667 | +0.004444 |
| 16 | 0.966183 | 0.965120 | -0.001062 | 0.504444 | 0.495556 | -0.008889 |
| 32 | 1.004095 | 1.003401 | -0.000694 | 0.475556 | 0.500000 | +0.024444 |

Compared with the prior same-seed reranker:

| K | Same-seed reranker loss | Cross-seed reranker loss | Loss change | Same-seed acc. | Cross-seed acc. | Accuracy change |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.973823 | 0.989754 | +0.015931 | 0.528889 | 0.506667 | -0.022222 |
| 16 | 0.951378 | 0.965120 | +0.013743 | 0.511111 | 0.495556 | -0.015556 |
| 32 | 0.999229 | 1.003401 | +0.004172 | 0.506667 | 0.500000 | -0.006667 |

Oracle-match rate also falls:

| K | Same-seed oracle match | Cross-seed oracle match | Change |
|---:|---:|---:|---:|
| 8 | 0.482222 | 0.295556 | -0.186666 |
| 16 | 0.382222 | 0.308889 | -0.073334 |
| 32 | 0.371111 | 0.313333 | -0.057778 |

## Interpretation

This is an important negative result. The current generated-candidate reranker is not yet a robust cross-seed retrieval policy. It learns useful seed/model-specific validation behavior, but much of that advantage disappears when trained on other seeds.

The cross-seed reranker still gives tiny loss improvements over a cross-seed static choice, and it improves K=32 accuracy. However, those gains are too small to support a strong generalization claim.

The failure mode is now narrower:

- candidate generation is useful;
- same-seed object-set reranking is useful;
- cross-seed reranker transfer is weak.

This suggests that retrieval scoring is entangled with the specific propagation model and seed-level loss landscape.

## Consequence

The WPU v2 claim should not yet be "we have a universal learned retriever." The defensible claim is:

> Explicit state enables candidate generation and local working-set scoring, but robust deployment requires cross-condition calibration or co-training.

The next step should target generalization rather than more candidate generation:

- normalize candidate losses per seed/model before training the reranker;
- train a shared reranker with seed/model-invariant features;
- use cross-seed validation as the primary selection criterion;
- co-train retrieval and propagation so candidate scoring aligns with the deployed model rather than post-hoc validation loss.

# WPU v2 Pairwise Reranker Probe

## Purpose

The generated-candidate sweep showed that more candidates improve the oracle but make deployed scoring harder. This experiment tests a direct fix: train the object-set reranker with pairwise candidate ranking loss instead of relying mainly on best-candidate classification.

The test focuses on the hardest prior condition:

- N=2048;
- K=8,16,32;
- five seeds;
- eight generated candidates;
- fixed propagation and candidate generation.

## Command

```powershell
.\.venv\Scripts\python.exe scripts\retriever_pairwise_reranker_probe.py `
  --n-values 2048 `
  --k-values 8 16 32 `
  --seeds 11 13 17 19 23 `
  --budget 4 `
  --generated-candidates 8 `
  --propagation-steps 40 `
  --retriever-steps 400 `
  --reranker-steps 600 `
  --validation-samples 180 `
  --samples 90 `
  --batch-size 10 `
  --device cuda `
  --safe-margin 0.005 `
  --out docs\experiments\wpu_v2_retriever_pairwise_reranker.csv
```

## Results

| K | CE reranker loss | Pairwise reranker loss | Loss change | CE accuracy | Pairwise accuracy | Accuracy change |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.975939 | 0.973914 | -0.002025 | 0.522222 | 0.535556 | +0.013333 |
| 16 | 0.954406 | 0.957946 | +0.003540 | 0.522222 | 0.495556 | -0.026667 |
| 32 | 0.997229 | 0.998525 | +0.001296 | 0.502222 | 0.497778 | -0.004444 |

Oracle-match rate also decreases under pairwise ranking:

| K | CE oracle match | Pairwise oracle match | Change |
|---:|---:|---:|---:|
| 8 | 0.422222 | 0.355555 | -0.066667 |
| 16 | 0.313333 | 0.248889 | -0.064444 |
| 32 | 0.342222 | 0.271111 | -0.071111 |

Pairwise reranking selects generated candidates more often:

| K | CE selected generated rate | Pairwise selected generated rate |
|---:|---:|---:|
| 8 | 0.424444 | 0.548889 |
| 16 | 0.495555 | 0.593333 |
| 32 | 0.460000 | 0.504444 |

## Interpretation

Pairwise ranking is not a general fix. It improves the easy K=8 condition, but it worsens K=16 and K=32. The failure mode is clear: pairwise ranking over-selects generated candidates and lowers oracle-match rate. In this setup, pairwise loss increases local ordering pressure but weakens global calibration among heterogeneous candidate types.

This is a useful negative result. The scoring bottleneck is not solved by changing the objective alone. Larger candidate pools require better candidate representations and calibration, not merely pairwise preference supervision.

## Consequence

The current best deployed retrieval policy remains the CE/soft-utility object-set reranker with 2-4 generated candidates. Pairwise ranking can be considered only as an auxiliary loss after calibration is improved.

The next scoring experiments should avoid a simple objective swap and instead test:

- cross-seed reranker training to reduce per-seed validation noise;
- candidate-type calibration or generated-candidate prior penalties;
- listwise ranking objectives that preserve global utility scale;
- larger object-set encoder capacity with explicit overfitting checks.

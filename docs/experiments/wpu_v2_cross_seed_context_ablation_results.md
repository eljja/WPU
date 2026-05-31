# WPU v2 Cross-Seed Reranker Context Ablation

Source CSV: `docs/experiments/wpu_v2_retriever_cross_seed_context_ablation.csv`

## Purpose

The normalized cross-seed reranker improved over the standard cross-seed reranker, but it still lagged behind same-seed reranking. One possible explanation was that the reranker overfit to candidate identity one-hot features rather than learning transferable object-set structure.

This experiment ablates candidate context features under the same leave-one-seed-out protocol:

- `full`: prior normalized cross-seed reranker with candidate identity and generated/base type flag.
- `no_identity_keep_type`: zero candidate identity one-hot features, keep the generated/base type flag.
- `no_identity_no_type`: zero both candidate identity and generated/base type flag.

## Command

```powershell
.\.venv\Scripts\python.exe scripts\retriever_cross_seed_context_ablation_probe.py `
  --n-values 2048 `
  --k-values 8 16 32 `
  --seeds 11 13 17 19 23 `
  --budget 4 `
  --generated-candidates 4 `
  --ablation-modes no_identity_keep_type no_identity_no_type `
  --propagation-steps 40 `
  --retriever-steps 400 `
  --reranker-steps 600 `
  --validation-samples 180 `
  --samples 90 `
  --batch-size 10 `
  --device cuda `
  --safe-margin 0.005 `
  --out docs\experiments\wpu_v2_retriever_cross_seed_context_ablation.csv
```

## Results

| K | Context | Loss | Accuracy | Oracle match | Selected generated |
|---:|---|---:|---:|---:|---:|
| 8 | full normalized | 0.988200 | 0.517778 | 0.293333 | 0.380000 |
| 8 | no identity, keep type | 0.989829 | 0.511111 | 0.304444 | 0.304445 |
| 8 | no identity, no type | 0.991440 | 0.497778 | 0.288889 | 0.335556 |
| 16 | full normalized | 0.962778 | 0.506667 | 0.308889 | 0.424444 |
| 16 | no identity, keep type | 0.963587 | 0.488889 | 0.324444 | 0.382222 |
| 16 | no identity, no type | 0.956393 | 0.493333 | 0.353333 | 0.451111 |
| 32 | full normalized | 1.003886 | 0.495556 | 0.317778 | 0.433333 |
| 32 | no identity, keep type | 1.006385 | 0.493333 | 0.320000 | 0.331111 |
| 32 | no identity, no type | 1.003990 | 0.484445 | 0.315555 | 0.342222 |

## Interpretation

The simple identity-overfitting hypothesis is not sufficient. Removing candidate identity generally hurts loss and accuracy at K=8 and K=32. At K=16, removing both identity and type improves loss, but it lowers accuracy. This is a mixed result, not a clean fix.

The useful signal is that candidate identity/type features are doing two things at once:

- They help calibrate candidate families in some regimes.
- They can also bias the reranker toward regime-specific selector behavior.

The deeper cross-seed bottleneck remains model-conditioned scoring. The reranker needs features that describe whether a candidate is good for the current propagation model, not only what kind of candidate it is.

## Consequence

The next mechanism should add model-state diagnostics rather than remove context blindly. Candidate identity is not the main problem by itself. The missing information is likely confidence/uncertainty from the WPU forward pass:

- sparse branch entropy for each candidate;
- margin between top branch logits;
- route regret prediction or dense/sparse disagreement;
- local propagation activation statistics.

These diagnostics would make the candidate scorer depend on how the current WPU model processes the state candidate, which is exactly what cross-seed transfer is missing.

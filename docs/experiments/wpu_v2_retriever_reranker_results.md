# WPU v2 Retriever Reranker Probe

Source CSV: `docs/experiments/wpu_v2_retriever_reranker.csv`

## Purpose

The retriever-regret oracle probe showed that no single static working-set selector is optimal for all samples. This experiment tests whether a deployed state-native reranker can recover part of that oracle gap without using tokenization, labels, or test losses at inference.

The reranker receives explicit candidate working-set features for each selector:

- selector identity: indexed, proximity, interaction, learned;
- whether the hand is selected;
- selected obstacle ratio;
- selected obstacle pair density;
- causal `K`, budget, and total world size `N`.

It predicts a utility for each candidate working set and selects one candidate before tensorization. The WPU propagation model is still state-based and sparse-first; the reranker only chooses which explicit object ids enter the local state tensor.

## Command

```powershell
.\.venv\Scripts\python.exe scripts\retriever_reranker_probe.py `
  --n-values 2048 `
  --k-values 8 16 32 `
  --seeds 11 13 17 19 23 `
  --budget 4 `
  --propagation-steps 40 `
  --retriever-steps 400 `
  --reranker-steps 600 `
  --validation-samples 180 `
  --samples 90 `
  --batch-size 10 `
  --device cuda `
  --safe-margin 0.005 `
  --out docs\experiments\wpu_v2_retriever_reranker.csv
```

## Main Results

| K | Policy | Loss | Accuracy | Excess over oracle | Oracle match | Reranker use |
|---:|---|---:|---:|---:|---:|---:|
| 8 | static validation choice | 0.987487 | 0.506667 | 0.025394 | 0.306667 | 0.000 |
| 8 | deployed reranker | 0.980910 | 0.524445 | 0.018816 | 0.471111 | 1.000 |
| 8 | validation-safe reranker | 0.980876 | 0.526667 | 0.018783 | 0.415555 | 0.800 |
| 8 | margin-safe reranker | 0.983275 | 0.522222 | 0.021182 | 0.382222 | 0.400 |
| 8 | oracle over retrievers | 0.962094 | 0.551111 | 0.000000 | 1.000000 | 1.000 |
| 16 | static validation choice | 0.958607 | 0.504444 | 0.043414 | 0.244444 | 0.000 |
| 16 | deployed reranker | 0.962844 | 0.520000 | 0.047651 | 0.453334 | 1.000 |
| 16 | validation-safe reranker | 0.959733 | 0.502222 | 0.044540 | 0.320000 | 0.200 |
| 16 | margin-safe reranker | 0.958607 | 0.504444 | 0.043414 | 0.244444 | 0.000 |
| 16 | oracle over retrievers | 0.915192 | 0.566667 | 0.000000 | 1.000000 | 1.000 |
| 32 | static validation choice | 1.004628 | 0.477778 | 0.030107 | 0.300000 | 0.000 |
| 32 | deployed reranker | 1.006325 | 0.486667 | 0.031803 | 0.424445 | 1.000 |
| 32 | validation-safe reranker | 1.003357 | 0.477778 | 0.028835 | 0.353334 | 0.400 |
| 32 | margin-safe reranker | 1.004628 | 0.477778 | 0.030107 | 0.300000 | 0.000 |
| 32 | oracle over retrievers | 0.974522 | 0.575556 | 0.000000 | 1.000000 | 1.000 |

## Interpretation

The reranker is a real improvement in the easiest retriever-adaptation regime. At K=8, the deployed reranker reduces loss by 0.006577 relative to the validation-selected static selector and improves accuracy by 0.017778. The validation-safe reranker is slightly better: loss improves by 0.006611 and accuracy by 0.020000.

The improvement does not yet generalize cleanly to larger K. At K=16 and K=32, the deployed reranker improves accuracy but worsens loss relative to the per-seed static validation choice. This means the current candidate features are sufficient to match the oracle mode more often, but not sufficient to calibrate loss-sensitive decisions reliably.

The margin-safe gate prevents broad regression. With a 0.005 validation margin, the reranker is used in 40% of K=8 seeds and 0% of K=16/K=32 seeds. This preserves the K=8 gain while avoiding the K=16/K=32 loss regressions, but it also shows that the current reranker has not yet expanded the reliable WPU advantage regime.

## Scientific Consequence

This is a useful negative/partial-positive result.

It supports the architectural direction: WPU should use explicit state candidates and learn a pre-tensor selection policy from downstream consequences. It does not yet support a broad claim that the current reranker solves large-K state retrieval.

The next bottleneck is not tokenization. It is state-native credit assignment:

- The reranker only sees aggregate selected-set features, not the full candidate object feature multiset.
- Validation supervision is small and noisy per seed.
- The reranker is trained after the propagation model, so retrieval and propagation are not co-adapted.
- The candidate set lacks stochastic or counterfactual object subsets beyond four hand-built selectors.

## Next Step

The next v2 experiment should upgrade the reranker from set-summary scoring to object-set encoding:

- Encode each candidate working set as a small set/graph, not only as counts.
- Train a candidate utility head from validation downstream loss.
- Add stochastic candidate generation around indexed/proximity/interaction selections.
- Use a conservative validation margin before deployment.

This keeps WPU non-token-based while moving from handcrafted retrieval toward learned state working-set selection.

# WPU v2 Object-Set Retriever Reranker Probe

## Purpose

The aggregate reranker used only selected-set summaries such as hand presence, obstacle count, and pair density. It improved K=8 but was not reliable at larger K. This experiment tests the next state-native mechanism: encode each candidate working set as an explicit object set before scoring it.

The reranker still operates before tensorization and does not use tokens. For each candidate selector, it receives:

- per-object state features for the selected objects;
- a mask over the fixed retrieval budget;
- selector identity and compact context features;
- validation downstream loss as the training signal.

The set encoder pools the selected objects with masked mean and max pooling, then predicts candidate utility.

## Command

```powershell
.\.venv\Scripts\python.exe scripts\retriever_set_reranker_probe.py `
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
  --out docs\experiments\wpu_v2_retriever_set_reranker.csv
```

## Main Results

| K | Policy | Loss | Accuracy | Excess over oracle | Oracle match |
|---:|---|---:|---:|---:|---:|
| 8 | static validation choice | 0.987487 | 0.506667 | 0.025394 | 0.306667 |
| 8 | object-set reranker | 0.976833 | 0.517778 | 0.014739 | 0.502222 |
| 8 | oracle over retrievers | 0.962094 | 0.551111 | 0.000000 | 1.000000 |
| 16 | static validation choice | 0.958607 | 0.504444 | 0.043414 | 0.244444 |
| 16 | object-set reranker | 0.957018 | 0.504444 | 0.041825 | 0.471111 |
| 16 | oracle over retrievers | 0.915192 | 0.566667 | 0.000000 | 1.000000 |
| 32 | static validation choice | 1.004628 | 0.477778 | 0.030107 | 0.300000 |
| 32 | object-set reranker | 1.003592 | 0.506667 | 0.029071 | 0.420000 |
| 32 | oracle over retrievers | 0.974522 | 0.575556 | 0.000000 | 1.000000 |

## Comparison to Aggregate Reranker

The object-set reranker improves the reliable regime.

| K | Aggregate deployed loss | Object-set deployed loss | Change |
|---:|---:|---:|---:|
| 8 | 0.980910 | 0.976833 | -0.004077 |
| 16 | 0.962844 | 0.957018 | -0.005826 |
| 32 | 1.006325 | 1.003592 | -0.002733 |

Compared with the validation-selected static selector, the object-set reranker reduces loss at every tested K:

- K=8: loss improves by 0.010654 and accuracy improves by 0.011111.
- K=16: loss improves by 0.001589 with unchanged accuracy.
- K=32: loss improves by 0.001036 and accuracy improves by 0.028889.

## Interpretation

This is the first v2 result where a learned state-native retrieval policy improves loss across all tested K values. The gain is still small at K=16 and K=32, but the direction is important: explicit object-set encoding is more stable than aggregate selected-set features.

The result supports a narrower and stronger WPU claim:

> Large-N WPU advantage requires not only sparse state access, but learned state working-set selection over explicit object sets.

The experiment does not prove broad superiority. The oracle gap remains substantial, especially at K=16 and K=32. However, it identifies a concrete path that is consistent with the WPU thesis and does not collapse back to token processing.

## Next Step

The next improvement should move from candidate selection among four fixed selectors to candidate generation:

- generate stochastic local object subsets around indexed, proximity, and interaction frontiers;
- score each subset with the object-set reranker;
- train the reranker with more validation samples or cross-seed supervision;
- eventually co-train retrieval and propagation instead of training the reranker after the propagation model.

This would test whether WPU can expand its reliable regime by learning both what state to retrieve and how to propagate through it.

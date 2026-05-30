# WPU v2 Cross-Seed Regret Distillation

## Question

Same-seed regret distillation is the strongest current WPU v2 retrieval result.
It trains an object retriever from candidate working sets that minimize
downstream branch loss. The remaining question is whether this mechanism
transfers across seeds and model instances.

This experiment uses a leave-one-seed-out protocol. For each held-out seed, the
regret retriever is trained on validation regret labels from the other four
seeds and evaluated on the held-out seed/model.

## Protocol

```powershell
.\.venv\Scripts\python.exe scripts\retriever_cross_seed_regret_distillation_probe.py --n-values 2048 --k-values 8 16 32 --seeds 11 13 17 19 23 --budget 4 --generated-candidates 4 --propagation-steps 40 --retriever-steps 400 --regret-retriever-steps 600 --validation-samples 180 --samples 90 --batch-size 10 --device cuda --out docs\experiments\wpu_v2_retriever_cross_seed_regret_distillation.csv
```

The script also evaluates two structural selection constraints:

- `cross_seed_regret_min1_obstacle`: force at least one obstacle into the selected working set.
- `cross_seed_regret_min2_obstacles`: force at least two obstacles into the selected working set.

These constraints test whether large-K failure is caused by the cross-seed
retriever under-selecting obstacle objects.

## Results

Mean over five held-out seeds:

| K | policy | loss | accuracy | excess over generated oracle | selected obstacles |
|---:|---|---:|---:|---:|---:|
| 8 | static learned interaction | 0.988432 | 0.506667 | 0.031964 | 2.000 |
| 8 | cross-seed regret | 0.985377 | 0.508889 | 0.028909 | 1.142 |
| 8 | cross-seed regret + min1 obstacle | 0.985547 | 0.502222 | 0.029080 | 1.507 |
| 8 | cross-seed regret + min2 obstacles | 0.988361 | 0.504444 | 0.031893 | 2.089 |
| 8 | generated oracle | 0.956468 | 0.557778 | 0.000000 | 1.460 |
| 16 | static learned interaction | 0.966183 | 0.504444 | 0.060115 | 2.000 |
| 16 | cross-seed regret | 0.952990 | 0.513333 | 0.046922 | 1.018 |
| 16 | cross-seed regret + min1 obstacle | 0.960704 | 0.500000 | 0.054635 | 1.442 |
| 16 | cross-seed regret + min2 obstacles | 0.965176 | 0.504444 | 0.059108 | 2.076 |
| 16 | generated oracle | 0.906068 | 0.580000 | 0.000000 | 1.702 |
| 32 | static learned interaction | 1.004095 | 0.475556 | 0.035372 | 2.000 |
| 32 | cross-seed regret | 1.005222 | 0.513333 | 0.036499 | 0.462 |
| 32 | cross-seed regret + min1 obstacle | 1.005073 | 0.493333 | 0.036350 | 1.198 |
| 32 | cross-seed regret + min2 obstacles | 1.003801 | 0.484445 | 0.035078 | 2.038 |
| 32 | generated oracle | 0.968723 | 0.577778 | 0.000000 | 1.724 |

## Interpretation

The result is partial transfer, not a solved mechanism.

At K=8 and K=16, cross-seed regret distillation improves loss over static
learned interaction. At K=16, it also improves over the same-seed regret
distillation average from the previous experiment. This is a meaningful result:
downstream-regret labels are more transferable than the earlier diagnostic
reranker scores in at least some causal working-set regimes.

At K=32, unconstrained cross-seed regret retrieval fails in loss despite higher
accuracy. The diagnostic is clear: it selects only 0.462 obstacles on average,
while the generated oracle selects 1.724. This under-selection does not appear
in the same-seed regret retriever to the same degree.

The minimum-obstacle constraints confirm the failure mode but do not fully solve
it. For K=32, forcing two obstacles lowers loss below static learned interaction
by a small amount, but it hurts K=8 and K=16. A single global structural
constraint is therefore not enough. The selector needs a K-aware or
state-conditioned structural prior.

## Consequence for WPU v2

The current evidence now supports a narrower and stronger statement:

- Regret-aware state retrieval is better than teacher imitation.
- Some regret-retrieval signal transfers cross-seed.
- Large-K transfer fails because the selected working-set composition is not
  calibrated to the causal regime.

The next implementation target should be a state-conditioned constrained
retriever: predict both object scores and a working-set composition prior, such
as expected obstacle count, hand inclusion probability, and uncertainty-driven
expansion budget. This is still state-native and does not return to token
processing.

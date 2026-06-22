# World-Copy Causal Index Stress

This benchmark measures whether the WPU v3 causal index retrieves event-local causal slices under large `N` and relation noise.
It is causal-retrieval substrate evidence, not trained world-model accuracy.
Source CSV: `docs/experiments/world_copy_causal_index_stress.csv`.

## Summary

| min relation confidence | missing rate | false-positive rate | mean recall | mean precision | max selected K | max touch ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0 | 0.0 | 1.000000 | 1.000000 | 16 | 0.22377622 |
| 0.3 | 0.0 | 0.0 | 1.000000 | 1.000000 | 16 | 0.22377622 |
| 0.0 | 0.0 | 0.1 | 1.000000 | 0.925926 | 18 | 0.23448276 |
| 0.3 | 0.0 | 0.1 | 1.000000 | 1.000000 | 16 | 0.23448276 |
| 0.0 | 0.0 | 0.25 | 1.000000 | 0.800000 | 20 | 0.24489796 |
| 0.3 | 0.0 | 0.25 | 1.000000 | 1.000000 | 16 | 0.24489796 |
| 0.0 | 0.25 | 0.0 | 1.000000 | 1.000000 | 16 | 0.20714286 |
| 0.3 | 0.25 | 0.0 | 1.000000 | 1.000000 | 16 | 0.20714286 |
| 0.0 | 0.25 | 0.1 | 1.000000 | 0.925926 | 18 | 0.21830986 |
| 0.3 | 0.25 | 0.1 | 1.000000 | 1.000000 | 16 | 0.21830986 |
| 0.0 | 0.25 | 0.25 | 1.000000 | 0.800000 | 20 | 0.22916667 |
| 0.3 | 0.25 | 0.25 | 1.000000 | 1.000000 | 16 | 0.22916667 |
| 0.0 | 0.5 | 0.0 | 1.000000 | 1.000000 | 16 | 0.18382353 |
| 0.3 | 0.5 | 0.0 | 1.000000 | 1.000000 | 16 | 0.18382353 |
| 0.0 | 0.5 | 0.1 | 1.000000 | 0.925926 | 18 | 0.20143885 |
| 0.3 | 0.5 | 0.1 | 1.000000 | 1.000000 | 16 | 0.20143885 |
| 0.0 | 0.5 | 0.25 | 1.000000 | 0.800000 | 20 | 0.20714286 |
| 0.3 | 0.5 | 0.25 | 1.000000 | 1.000000 | 16 | 0.20714286 |

## Interpretation

- Region-scoped retrieval keeps touched units far below full-state scan as `N` grows.
- Missing true relations are partly recovered because the active region is a causal scope, but this assumes correct objectification/region assignment.
- Relation confidence gating suppresses low-confidence false-positive relations and recovers precision.
- The next failure boundary is low-confidence or miscalibrated true causal relations.

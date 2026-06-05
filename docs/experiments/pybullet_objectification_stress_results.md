# PyBullet Objectification Stress

This experiment is the first perception-like robustness test built on the
PyBullet cup benchmark. Models are trained on clean simulator-objectified
`WorldState` samples and evaluated on corrupted objectified state. Labels and
target deltas remain tied to the clean simulator rollout, so failures can be
attributed to the state interface rather than to a changed physical scenario.

Source CSV:

- `docs/experiments/pybullet_objectification_stress.csv`

## Protocol

- Simulator: PyBullet `DIRECT` rigid-body rollout.
- Base task: balanced cup impulse branch prediction.
- Training state: clean simulator-derived `WorldState`.
- Evaluation state: corrupted `WorldState`.
- Models: `wpu-cws-indexed-sparse`, `wpu-cws-indexed-local-dense`,
  `graph-transformer`.
- Seeds: `11, 13`.
- Background objects: `32`.
- Training: 20 steps, batch 8, hidden 64.
- Evaluation: 36 samples per condition.

Corruption presets:

- `drop_relations_light`: remove 25% of event-target relations.
- `drop_relations_heavy`: remove 60% of event-target relations.
- `position_noise`: add Gaussian position noise with std `0.08`.
- `low_confidence`: multiply object/relation confidence by `0.55`.
- `identity_swap`: swap non-target causal object attributes/types.
- `combined`: relation drop, object drop, position noise, lower confidence,
  and partial identity swap.

## Summary

| corruption | model | accuracy | objectification score | frontier recall | selected K |
| --- | --- | ---: | ---: | ---: | ---: |
| clean | graph-transformer | 0.500 | 0.944 | 1.000 | 4.463 |
| clean | wpu-cws-indexed-local-dense | 0.542 | 0.944 | 1.000 | 4.463 |
| clean | wpu-cws-indexed-sparse | 0.528 | 0.944 | 1.000 | 4.463 |
| drop_relations_light | graph-transformer | 0.500 | 0.943 | 0.848 | 4.463 |
| drop_relations_light | wpu-cws-indexed-local-dense | 0.542 | 0.943 | 0.848 | 3.775 |
| drop_relations_light | wpu-cws-indexed-sparse | 0.528 | 0.943 | 0.848 | 3.775 |
| drop_relations_heavy | graph-transformer | 0.500 | 0.941 | 0.602 | 4.463 |
| drop_relations_heavy | wpu-cws-indexed-local-dense | 0.514 | 0.941 | 0.602 | 2.700 |
| drop_relations_heavy | wpu-cws-indexed-sparse | 0.528 | 0.941 | 0.602 | 2.700 |
| position_noise | graph-transformer | 0.500 | 0.944 | 1.000 | 4.463 |
| position_noise | wpu-cws-indexed-local-dense | 0.542 | 0.944 | 1.000 | 4.463 |
| position_noise | wpu-cws-indexed-sparse | 0.542 | 0.944 | 1.000 | 4.463 |
| low_confidence | graph-transformer | 0.486 | 0.789 | 1.000 | 4.463 |
| low_confidence | wpu-cws-indexed-local-dense | 0.514 | 0.789 | 1.000 | 4.463 |
| low_confidence | wpu-cws-indexed-sparse | 0.500 | 0.789 | 1.000 | 4.463 |
| identity_swap | graph-transformer | 0.500 | 0.944 | 1.000 | 4.463 |
| identity_swap | wpu-cws-indexed-local-dense | 0.542 | 0.944 | 1.000 | 4.463 |
| identity_swap | wpu-cws-indexed-sparse | 0.528 | 0.944 | 1.000 | 4.463 |
| combined | graph-transformer | 0.500 | 0.840 | 0.751 | 4.175 |
| combined | wpu-cws-indexed-local-dense | 0.514 | 0.840 | 0.751 | 3.362 |
| combined | wpu-cws-indexed-sparse | 0.542 | 0.840 | 0.751 | 3.362 |

## Interpretation

The stress test exposes a clearer objectification boundary than the base
PyBullet benchmark. Relation corruption reduces the event-local frontier
recall and WPU selected K, while full-state graph baselines still see all
objects. This is exactly the expected WPU failure mode: if relations do not
connect the event to the causal working set, indexed sparse processing loses
state before propagation.

The current task is still too small to make this a decisive accuracy result.
Branch accuracy changes only modestly under corruption, and all models remain
near a narrow 0.49-0.54 band. The result should therefore be used as a
diagnostic, not as a dominance claim.

## Issues Found

- `ObjectificationReport.contract_score` barely changes under relation-drop
  corruption because relation validity checks malformed endpoints, not missing
  expected causal edges. The new `frontier_causal_recall_mean` metric is needed
  to expose missing frontier connectivity.
- `identity_swap` is not detected by the current objectification score because
  object IDs remain syntactically valid. Semantic identity consistency requires
  history, tracking, or role/affordance checks.
- The model-level `causal_recall_mean` computed after pre-tensor projection is
  not sufficient for stress tests. It can remain 1.0 because it measures recall
  inside the already selected subgraph. The benchmark now logs pre-projection
  `frontier_causal_recall_mean`.

## Next Steps

- Extend `ObjectificationReport` with expected-frontier completeness and
  semantic identity consistency checks.
- Add relation repair during evaluation and measure whether recovered edges
  restore WPU selected K and accuracy.
- Increase horizon and use closed-loop rollout so missing relations create
  compounding state errors rather than only one-step branch noise.
- Add stronger corruption that removes or aliases the actual event target, then
  require WPU to abstain or escalate instead of producing confident sparse
  updates.

# WPU v2 Cross-Seed Composition-Regret Retriever

Source CSV: `docs/experiments/wpu_v2_retriever_cross_seed_composition_regret.csv`

## Question

The cross-seed regret-distilled retriever transferred at K=8 and K=16, but
failed at K=32 because it under-selected obstacle objects. The generated oracle
selected about 1.7 obstacles, while the unconstrained cross-seed retriever
selected only 0.46.

This experiment tests whether the retriever should predict working-set
composition, not only per-object scores.

## Method

The model keeps the cross-seed regret-distilled object scorer and adds a
state-conditioned composition prior. The prior receives aggregate state/event
features from the candidate frontier and predicts:

- obstacle count in the selected working set;
- hand inclusion probability.

Three selection policies are evaluated:

- `composition_regret_argmax`: use argmax obstacle-count prediction and hand
  inclusion prediction.
- `composition_regret_expected`: use rounded expected obstacle count and hand
  inclusion prediction.
- `composition_regret_count_only`: use obstacle-count prediction only, without
  forced hand inclusion.

Command:

```powershell
.\.venv\Scripts\python.exe scripts\retriever_cross_seed_composition_regret_probe.py --n-values 2048 --k-values 8 16 32 --seeds 11 13 17 19 23 --budget 4 --generated-candidates 4 --propagation-steps 40 --retriever-steps 400 --regret-retriever-steps 600 --composition-steps 600 --validation-samples 180 --samples 90 --batch-size 10 --device cuda --out docs\experiments\wpu_v2_retriever_cross_seed_composition_regret.csv
```

## Results

Mean over five held-out seeds:

| K | policy | loss | accuracy | excess over generated oracle | selected obstacles | hand rate |
|---:|---|---:|---:|---:|---:|---:|
| 8 | static learned interaction | 0.988432 | 0.506667 | 0.031964 | 2.000 | 1.000 |
| 8 | cross-seed regret | 0.985377 | 0.508889 | 0.028909 | 1.142 | 0.840 |
| 8 | composition argmax | 0.984911 | 0.515556 | 0.028443 | 1.544 | 0.711 |
| 8 | composition count-only | 0.984766 | 0.522222 | 0.028299 | 1.636 | 0.602 |
| 8 | generated oracle | 0.956468 | 0.557778 | 0.000000 | 1.460 | 0.653 |
| 16 | static learned interaction | 0.966183 | 0.504444 | 0.060115 | 2.000 | 1.000 |
| 16 | cross-seed regret | 0.952990 | 0.513333 | 0.046922 | 1.018 | 0.904 |
| 16 | composition argmax | 0.950800 | 0.533333 | 0.044732 | 1.758 | 0.633 |
| 16 | composition count-only | 0.951783 | 0.531111 | 0.045715 | 1.873 | 0.511 |
| 16 | generated oracle | 0.906068 | 0.580000 | 0.000000 | 1.702 | 0.567 |
| 32 | static learned interaction | 1.004095 | 0.475556 | 0.035372 | 2.000 | 1.000 |
| 32 | cross-seed regret | 1.005222 | 0.513333 | 0.036499 | 0.462 | 0.956 |
| 32 | composition argmax | 1.001625 | 0.517778 | 0.032902 | 1.460 | 0.693 |
| 32 | composition expected | 1.001793 | 0.506667 | 0.033070 | 1.696 | 0.864 |
| 32 | generated oracle | 0.968723 | 0.577778 | 0.000000 | 1.724 | 0.580 |

## Interpretation

This is a material v2 improvement.

The composition prior improves loss over static learned interaction and over
the unconstrained cross-seed regret retriever at K=8, K=16, and K=32. It also
fixes the K=32 obstacle under-selection failure: selected obstacles rise from
0.462 to 1.46--1.70, close to the generated oracle's 1.724.

The result supports a more precise mechanism claim: WPU retrieval should be
state-conditioned working-set construction, not independent per-object ranking.
Explicit state makes this mechanism natural because object scores, relation
features, event features, and working-set composition can be learned before
propagation.

## Remaining Failure

The composition prior still does not close the generated-oracle gap. At K=32,
loss improves from 1.005222 to 1.001625, but the generated oracle is 0.968723.
This means candidate selection is better calibrated, but the system still needs
either better generated candidates, joint retriever-propagator training, or a
stronger scorer that evaluates candidate sets rather than only object count and
object scores.

## Consequence

The next v2 target is no longer just "learn a retriever." It is:

```text
State-conditioned constrained working-set construction
  = object scores
  + composition prior
  + candidate-set evaluation
  + downstream regret training
```

This remains fully state-native and strengthens the distinction between WPU and
token serialization.

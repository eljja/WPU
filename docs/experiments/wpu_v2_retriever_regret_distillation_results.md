# WPU v2 Regret-Distilled Retriever

## Question

Earlier learned retrievers imitated the hand-built interaction selector. That
objective is state-native, but it is not the same as downstream branch loss. The
retriever-regret oracle probe showed that the interaction selector is rarely the
true downstream oracle.

This experiment asks whether a retriever trained from downstream-regret oracle
candidate sets performs better than one trained to imitate the interaction
teacher.

## Method

For each seed, the WPU propagation model is trained as before. On a validation
split, the script evaluates base candidates and generated candidates:

- `indexed`
- `proximity`
- `interaction`
- `learned`
- `generated_0..generated_3`

For each validation sample, the candidate set with the lowest downstream branch
cross-entropy is treated as a pseudo-label object set. A small state-native
object scorer is then trained to select objects from the event frontier whose
features match that pseudo-label set.

Command:

```powershell
.\.venv\Scripts\python.exe scripts\retriever_regret_distillation_probe.py --n-values 2048 --k-values 8 16 32 --seeds 11 13 17 19 23 --budget 4 --generated-candidates 4 --propagation-steps 40 --retriever-steps 400 --regret-retriever-steps 600 --validation-samples 180 --samples 90 --batch-size 10 --device cuda --out docs\experiments\wpu_v2_retriever_regret_distillation.csv
```

This is same-seed validation-to-test distillation. It does not yet prove
cross-seed generalization.

## Results

Mean over five seeds:

| K | policy | loss | accuracy | excess over generated oracle |
|---:|---|---:|---:|---:|
| 8 | static learned interaction | 0.988432 | 0.506667 | 0.031964 |
| 8 | regret-distilled retriever | 0.977017 | 0.542222 | 0.020550 |
| 8 | generated oracle | 0.956468 | 0.557778 | 0.000000 |
| 16 | static learned interaction | 0.966183 | 0.504444 | 0.060115 |
| 16 | regret-distilled retriever | 0.955077 | 0.513333 | 0.049009 |
| 16 | generated oracle | 0.906068 | 0.580000 | 0.000000 |
| 32 | static learned interaction | 1.004095 | 0.475556 | 0.035372 |
| 32 | regret-distilled retriever | 0.999112 | 0.513333 | 0.030389 |
| 32 | generated oracle | 0.968723 | 0.577778 | 0.000000 |

Per-seed loss deltas versus static learned interaction:

| K | seed deltas |
|---:|---|
| 8 | -0.007449, -0.014767, -0.004063, -0.019995, -0.010797 |
| 16 | -0.001948, -0.001358, -0.006320, -0.005728, -0.040176 |
| 32 | -0.001156, -0.009434, -0.005430, +0.000964, -0.009858 |

## Interpretation

This is the strongest v2 retrieval mechanism so far.

The regret-distilled retriever improves downstream loss at every K on average
and wins 14 of 15 seed/K conditions. It also changes the selected working-set
composition: it selects fewer obstacles and does not always force the hand into
the working set. That behavior matches the generated oracle more closely than
the interaction teacher, which always selects the hand and two obstacles under
the current budget.

This supports a stronger WPU claim: explicit state is useful not only because it
enables sparse propagation, but because it exposes retrieval as a trainable
pre-propagation control problem. Token baselines do not naturally expose this
object-level intervention point.

## Limitations

The experiment is still same-seed validation-to-test distillation. It does not
resolve the cross-seed scoring problem identified in the diagnostic reranker
experiments.

The pseudo-labels are also constrained by the candidate pool. If generated
candidates miss the true causal set, the distillation target inherits that
limitation.

## Next Step

The next required test is cross-seed regret distillation: train the object scorer
from downstream-regret labels across several seeds and evaluate it on a held-out
seed/model. If that transfers, WPU v2 can claim a concrete path from explicit
state to robust learned working-set selection.

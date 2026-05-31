# WPU v2 Conservative Set-Evaluator Gate

## Question

The cross-seed set evaluator found better candidate working sets in the pool,
but its deployed selections hurt `K=8` and `K=16`. This experiment tests a
minimal safety mechanism:

```text
Use the set evaluator only when its top-score margin is high;
otherwise fall back to the static learned-interaction selector.
```

Two gates are evaluated:

- `conservative_margin_gate`: choose the score-margin threshold that minimizes
  train-seed validation loss.
- `robust_per_seed_margin_gate`: choose a threshold only if it does not harm
  any individual train seed relative to static learned interaction.

## Protocol

```powershell
.\.venv\Scripts\python.exe scripts\retriever_cross_seed_conservative_set_evaluator_probe.py --n-values 2048 --k-values 8 16 32 --seeds 11 13 17 19 23 --budget 4 --generated-candidates 4 --propagation-steps 40 --retriever-steps 400 --regret-retriever-steps 600 --composition-steps 600 --set-evaluator-steps 600 --validation-samples 180 --samples 90 --batch-size 10 --device cuda --out docs\experiments\wpu_v2_retriever_conservative_set_evaluator.csv
```

The experiment uses the same candidate pool as the cross-seed set-evaluator
probe: base selectors, generated local candidates, and composition-regret
candidates.

## Results

Mean over five held-out seeds:

| K | policy | loss | accuracy | excess over oracle | oracle match | evaluator use |
|---:|---|---:|---:|---:|---:|---:|
| 8 | static learned interaction | 0.988432 | 0.506667 | 0.032897 | 0.006667 | - |
| 8 | set evaluator | 0.989208 | 0.502222 | 0.033674 | 0.137778 | 1.000000 |
| 8 | conservative margin gate | 0.989036 | 0.502222 | 0.033502 | 0.137778 | 0.988889 |
| 8 | robust per-seed margin gate | 0.989036 | 0.502222 | 0.033502 | 0.137778 | 0.988889 |
| 8 | candidate oracle | 0.955534 | 0.557778 | 0.000000 | 1.000000 | - |
| 16 | static learned interaction | 0.966183 | 0.504444 | 0.060800 | 0.002222 | - |
| 16 | set evaluator | 0.969430 | 0.497778 | 0.064047 | 0.133333 | 1.000000 |
| 16 | conservative margin gate | 0.969162 | 0.497778 | 0.063779 | 0.124444 | 0.937778 |
| 16 | robust per-seed margin gate | 0.969162 | 0.497778 | 0.063779 | 0.124444 | 0.937778 |
| 16 | candidate oracle | 0.905383 | 0.580000 | 0.000000 | 1.000000 | - |
| 32 | static learned interaction | 1.004095 | 0.475556 | 0.035643 | 0.004444 | - |
| 32 | set evaluator | 1.001607 | 0.511111 | 0.033156 | 0.162222 | 1.000000 |
| 32 | conservative margin gate | 1.001143 | 0.508889 | 0.032692 | 0.160000 | 0.966667 |
| 32 | robust per-seed margin gate | 1.001143 | 0.508889 | 0.032692 | 0.160000 | 0.966667 |
| 32 | candidate oracle | 0.968451 | 0.580000 | 0.000000 | 1.000000 | - |

## Interpretation

This is another negative result for the current set-evaluator family.

Margin gating slightly improves over the raw set evaluator, but it does not
solve the deployment problem. It still hurts `K=8` and `K=16` relative to the
static learned-interaction selector, and it helps only at `K=32`.

The more important finding is that the robust per-seed gate collapses to the
same thresholds as the train-loss gate. Even when the threshold is selected to
avoid harm on every training seed, the selected high-margin cases do not
transfer safely to held-out seeds.

Therefore, the evaluator's score margin is not a reliable epistemic confidence
signal. The missing mechanism is not just conservative fallback. The scorer
needs features or training objectives that make its confidence invariant across
world seeds and propagator instances.

## Consequence

The deployed v2 path should still use composition-regret retrieval rather than
set-evaluator gating. The next candidate-scoring work should focus on:

- invariant candidate descriptors rather than score-margin confidence;
- calibration under seed/layout perturbations;
- joint retriever-propagator objectives that penalize high-confidence
  wrong candidate choices.


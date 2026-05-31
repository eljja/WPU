# WPU v2 Cross-Seed Candidate-Set Evaluator

## Question

Composition-regret retrieval improved cross-seed working-set construction, but
it still left a large gap to the generated oracle. The next natural mechanism is
a candidate-set evaluator: generate base, generated, and composition-aware
candidate sets, then train a set-level scorer to select the candidate with the
lowest downstream branch loss.

This experiment tests that mechanism under leave-one-seed-out transfer.

## Protocol

```powershell
.\.venv\Scripts\python.exe scripts\retriever_cross_seed_set_evaluator_probe.py --n-values 2048 --k-values 8 16 32 --seeds 11 13 17 19 23 --budget 4 --generated-candidates 4 --propagation-steps 40 --retriever-steps 400 --regret-retriever-steps 600 --composition-steps 600 --set-evaluator-steps 600 --validation-samples 180 --samples 90 --batch-size 10 --device cuda --out docs\experiments\wpu_v2_retriever_cross_seed_set_evaluator.csv
```

The candidate pool contains:

- base selectors: `indexed`, `proximity`, `interaction`, `learned`;
- generated local candidates: `generated_0..generated_3`;
- composition-regret candidates: `composition_argmax`,
  `composition_expected`, `composition_count_only`.

The evaluator is trained on validation candidate losses from four seeds and
tested on a held-out seed/model.

## Results

Mean over five held-out seeds:

| K | policy | loss | accuracy | excess over candidate oracle | oracle match rate |
|---:|---|---:|---:|---:|---:|
| 8 | static learned interaction | 0.988432 | 0.506667 | 0.032897 | 0.006667 |
| 8 | set evaluator | 0.989208 | 0.502222 | 0.033674 | 0.137778 |
| 8 | candidate oracle | 0.955534 | 0.557778 | 0.000000 | 1.000000 |
| 16 | static learned interaction | 0.966183 | 0.504444 | 0.060800 | 0.002222 |
| 16 | set evaluator | 0.969430 | 0.497778 | 0.064047 | 0.133333 |
| 16 | candidate oracle | 0.905383 | 0.580000 | 0.000000 | 1.000000 |
| 32 | static learned interaction | 1.004095 | 0.475556 | 0.035643 | 0.004444 |
| 32 | set evaluator | 1.001607 | 0.511111 | 0.033156 | 0.162222 |
| 32 | candidate oracle | 0.968451 | 0.580000 | 0.000000 | 1.000000 |

## Interpretation

This is a negative result for the current set evaluator.

The expanded candidate pool is useful: the candidate oracle is slightly better
than the previous generated oracle at all K values. However, the learned
cross-seed set evaluator does not reliably select useful candidates. It hurts
K=8 and K=16, and only helps K=32.

A simple train-loss gate also fails. The evaluator improves training-seed loss
in all held-out runs, so the gate always chooses it, but that improvement does
not transfer to K=8 or K=16. The failure is therefore not lack of train signal;
it is cross-seed overfitting or missing invariant features.

## Consequence

The next step should not be "add more candidates" alone. The candidate oracle
shows that better choices exist, but the scorer cannot yet transfer. The
required mechanism is a cross-seed-regularized candidate evaluator with stronger
state-invariant features, or joint retriever-propagator training where candidate
selection is optimized against held-out-like perturbations.

At this stage, the best deployed v2 mechanism remained composition-regret
retrieval, not the set evaluator. Later invariant-descriptor experiments narrow
the conclusion further: opaque set evaluation is not enough, while
risk-adjusted mechanism selection over explicit state descriptors is the
stronger cross-seed path.

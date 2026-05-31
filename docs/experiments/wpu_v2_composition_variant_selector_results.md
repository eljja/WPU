# WPU v2 Composition Variant Selector

Source CSVs:

- `docs/experiments/wpu_v2_retriever_cross_seed_composition_regret.csv`
- `docs/experiments/wpu_v2_composition_variant_selector_summary.csv`

## Question

The composition-regret retriever introduced three deployable policies:

- `composition_regret_argmax`
- `composition_regret_expected`
- `composition_regret_count_only`

The best policy varies slightly by K and seed. This analysis asks whether a
held-out-safe selector can choose among them without using the target seed's
test performance.

## Method

Input:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_composition_variant_selector.py --input docs\experiments\wpu_v2_retriever_cross_seed_composition_regret.csv --out docs\experiments\wpu_v2_composition_variant_selector_summary.csv
```

For each held-out seed, the selector chooses a composition policy using only the
other held-out seeds at the same K. This is not a new model-training run; it is
a post-hoc deployment-policy audit over the completed cross-seed composition
experiment. It tests whether the policy choice is stable enough to be selected
without looking at the target seed.

## Results

| K | selector criterion | loss | accuracy | delta loss vs static learned | excess over generated oracle |
|---:|---|---:|---:|---:|---:|
| 8 | lowest other-seed loss | 0.986041 | 0.520000 | -0.002391 | 0.029573 |
| 8 | highest other-seed accuracy | 0.984766 | 0.522222 | -0.003665 | 0.028299 |
| 16 | lowest other-seed loss | 0.950800 | 0.533333 | -0.015383 | 0.044732 |
| 16 | highest other-seed accuracy | 0.952300 | 0.524444 | -0.013883 | 0.046232 |
| 32 | lowest other-seed loss | 1.002516 | 0.513334 | -0.001578 | 0.033793 |
| 32 | highest other-seed accuracy | 1.001494 | 0.506667 | -0.002600 | 0.032771 |

## Interpretation

The selector audit strengthens the composition-regret result.

Every listed selector improves loss over static learned interaction at K=8,
K=16, and K=32. The gains are modest at K=8 and K=32, but the important point is
that the composition policy can be selected without inspecting the target
seed's test loss.

This does not close the generated-oracle gap. It does show that the v2
retrieval direction is becoming less brittle: the working-set construction
policy is no longer a single hand-chosen setting, and the selection rule remains
state-native.

## Consequence

The next step should move from policy selection to policy learning:

```text
candidate-set evaluator
  input: object scores + composition prior + candidate-set features
  target: downstream regret
  protocol: leave-one-seed-out
```

This would directly test whether WPU can learn to choose among candidate sets,
rather than relying on a small menu of hand-defined composition policies.

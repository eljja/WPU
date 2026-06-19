# Joint Selector-Propagator Results

This report summarizes a P1 ablation that swaps the joint selector-propagator propagation model to a relation-conditioned WPU. It tests whether the K=16/32 bottleneck is partly caused by insufficient transition dynamics. If closure improves but harmful accept remains high, relation-aware dynamics are useful but no-harm candidate selection is still insufficient.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_relation.csv`

The best closure is `0.266805` (`K=32`, `confidence_selected_joint_selector_propagator`). P1 evaluates whether joint selector-propagator deployment closes the candidate-oracle gap while controlling harmful accepts. The conservative best under harmful-accept <= `0.25` is `0.064077` (`confidence_selected_joint_selector_propagator`). The train-selected deployed best is `0.266805` (`K=32`).

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `confidence_selected_joint_selector_propagator` | 1.002028 | 0.493333 | 0.037102 | 0.002377 | 0.064077 | 0.315556 | 0.084445 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.967449 | 0.484444 | 0.016754 | 0.004470 | 0.266805 | 0.906667 | 0.333333 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `train_selected_joint_selector_propagator` | 0.967449 | 0.484444 | 0.016754 | 0.004470 | 0.266805 | 0.906667 | 0.333333 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `joint_selector_propagator` | 0.968075 | 0.484444 | 0.016754 | 0.003844 | 0.229429 | 1.000000 | 0.395555 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `train_selected_joint_selector_propagator` | 1.002028 | 0.493333 | 0.037102 | 0.002377 | 0.064077 | 0.315556 | 0.084445 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_selector_propagator` | 1.003795 | 0.497778 | 0.037102 | 0.000611 | 0.016463 | 1.000000 | 0.422222 | 0.000000 | `insufficient_no_harm_rejection` |

## Interpretation

- The CSV keeps all reject-margin/risk-penalty deployment sweep points.
- The table below shows the best policy per K and the strongest overall policies.
- A useful deployed policy needs both high closure and low harmful accepts.
- If this probe underperforms the direct candidate-regret gate, simply coupling selector scores to propagation branch loss is not enough; object retrieval, candidate generation, and transition dynamics need deeper joint training.
- If relation-conditioned propagation raises closure but not harmful-accept safety, the next step is joint relation-aware transition learning with no-harm candidate selection.

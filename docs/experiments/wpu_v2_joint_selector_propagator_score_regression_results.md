# Score-Regression Joint Selector-Propagator Results

This report summarizes a P1 ablation that directly aligns joint selector-propagator scores with each candidate's relative propagation utility. It tests whether structured-candidate headroom is missed because argmax/ranking objectives do not learn candidate loss magnitudes well enough.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_score_regression.csv`

The best closure is `0.186333` (`K=16`, `joint_selector_propagator`). P1 evaluates whether joint selector-propagator deployment closes the candidate-oracle gap while controlling harmful accepts. The conservative best under harmful-accept <= `0.25` is `0.095543` (`confidence_selected_joint_selector_propagator`). The train-selected deployed best is `0.110287` (`K=16`).

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `joint_selector_propagator` | 0.999804 | 0.502222 | 0.051792 | 0.009651 | 0.186333 | 1.000000 | 0.435556 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.992320 | 0.511111 | 0.045062 | 0.004305 | 0.095543 | 0.484444 | 0.186667 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `confidence_selected_joint_selector_propagator` | 1.003742 | 0.502222 | 0.051792 | 0.005712 | 0.110287 | 0.871111 | 0.382222 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `train_selected_joint_selector_propagator` | 1.003742 | 0.502222 | 0.051792 | 0.005712 | 0.110287 | 0.871111 | 0.382222 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `train_selected_joint_selector_propagator` | 0.992320 | 0.511111 | 0.045062 | 0.004305 | 0.095543 | 0.484444 | 0.186667 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `joint_selector_propagator` | 0.993329 | 0.511111 | 0.045062 | 0.003296 | 0.073152 | 1.000000 | 0.511111 | 0.000000 | `insufficient_no_harm_rejection` |

## Interpretation

- The CSV keeps all reject-margin/risk-penalty deployment sweep points.
- The table below shows the best policy per K and the strongest overall policies.
- A useful deployed policy needs both high closure and low harmful accepts.
- If this probe underperforms the direct candidate-regret gate, simply coupling selector scores to propagation branch loss is not enough; object retrieval, candidate generation, and transition dynamics need deeper joint training.
- If score regression does not improve closure or raises harmful accepts, matching candidate loss magnitudes is not enough; safe generation needs verification-aware supervision.

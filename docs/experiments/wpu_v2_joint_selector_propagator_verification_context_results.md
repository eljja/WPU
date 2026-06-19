# Verification-Context Joint Selector-Propagator Results

This report summarizes a P1 ablation that appends label-free propagation verification signatures to the joint selector-propagator selector input. The signatures include candidate branch confidence, entropy, logit margin, and delta-norm signals computed without ground-truth labels. It tests whether the selector can reject harmful candidates by observing propagator behavior rather than only static candidate descriptions.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_verification_context.csv`

The best closure is `0.409420` (`K=16`, `joint_selector_propagator`). P1 evaluates whether joint selector-propagator deployment closes the candidate-oracle gap while controlling harmful accepts. The conservative best under harmful-accept <= `0.25` is `0.269216` (`confidence_selected_joint_selector_propagator`). The train-selected deployed best is `0.269216` (`K=16`).

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `joint_selector_propagator` | 0.962804 | 0.537778 | 0.081009 | 0.033167 | 0.409420 | 1.000000 | 0.342222 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.991791 | 0.502222 | 0.038259 | 0.005868 | 0.153386 | 0.306667 | 0.111111 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `confidence_selected_joint_selector_propagator` | 0.974162 | 0.524445 | 0.081009 | 0.021809 | 0.269216 | 0.440000 | 0.115555 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_joint_selector_propagator` | 0.974162 | 0.524445 | 0.081009 | 0.021809 | 0.269216 | 0.440000 | 0.115555 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `train_selected_joint_selector_propagator` | 0.991791 | 0.502222 | 0.038259 | 0.005868 | 0.153386 | 0.306667 | 0.111111 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `joint_selector_propagator` | 0.996778 | 0.506667 | 0.038259 | 0.000882 | 0.023059 | 1.000000 | 0.533333 | 0.000000 | `insufficient_no_harm_rejection` |

## Interpretation

- The CSV keeps all reject-margin/risk-penalty deployment sweep points.
- The table below shows the best policy per K and the strongest overall policies.
- A useful deployed policy needs both high closure and low harmful accepts.
- If this probe underperforms the direct candidate-regret gate, simply coupling selector scores to propagation branch loss is not enough; object retrieval, candidate generation, and transition dynamics need deeper joint training.
- If verification context does not improve closure and harmful accept together, label-free propagation signatures may be useful observations, but appending them to selector input is still insufficient; the verification objective must be trained with no-harm candidate generation.

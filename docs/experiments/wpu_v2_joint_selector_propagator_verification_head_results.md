# Verification-Head Joint Selector-Propagator Results

This report summarizes a P1 ablation that adds an explicit harmful-candidate verification head on top of label-free propagation signatures. The head learns whether a candidate is worse than the learned baseline and subtracts unsafe probability from deployment scores. It tests whether verification helps larger-K deployment when trained as a no-harm objective rather than merely concatenated as an input feature.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_verification_head.csv`

The best closure is `0.345395` (`K=16`, `joint_selector_propagator`). P1 evaluates whether joint selector-propagator deployment closes the candidate-oracle gap while controlling harmful accepts. The conservative best under harmful-accept <= `0.25` is `0.193197` (`confidence_selected_joint_selector_propagator`). The train-selected deployed best is `0.193197` (`K=16`).

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `joint_selector_propagator` | 0.970314 | 0.533333 | 0.080096 | 0.027665 | 0.345395 | 1.000000 | 0.391111 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.993600 | 0.506667 | 0.036576 | 0.002216 | 0.060597 | 0.222222 | 0.088889 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `confidence_selected_joint_selector_propagator` | 0.982505 | 0.511111 | 0.080096 | 0.015474 | 0.193197 | 0.355556 | 0.102222 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_joint_selector_propagator` | 0.982505 | 0.511111 | 0.080096 | 0.015474 | 0.193197 | 0.355556 | 0.102222 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `train_selected_joint_selector_propagator` | 0.993600 | 0.506667 | 0.036576 | 0.002216 | 0.060597 | 0.222222 | 0.088889 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `joint_selector_propagator` | 0.994742 | 0.511111 | 0.036576 | 0.001075 | 0.029385 | 1.000000 | 0.537778 | 0.000000 | `insufficient_no_harm_rejection` |

## Interpretation

- The CSV keeps all reject-margin/risk-penalty deployment sweep points.
- The table below shows the best policy per K and the strongest overall policies.
- A useful deployed policy needs both high closure and low harmful accepts.
- If this probe underperforms the direct candidate-regret gate, simply coupling selector scores to propagation branch loss is not enough; object retrieval, candidate generation, and transition dynamics need deeper joint training.
- If the verification head lowers harmful accept only by collapsing closure, explicit no-harm prediction is still insufficient without better candidate generation and propagation dynamics.

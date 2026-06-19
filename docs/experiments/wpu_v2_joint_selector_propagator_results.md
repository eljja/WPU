# Joint Selector-Propagator Results

This report summarizes a P1 probe that optimizes candidate working-set selector scores and WPU sparse propagation branch losses in the same computation graph. It couples candidate choice to propagation dynamics more directly than post-hoc selectors, fixed-propagator verifiers, or shallow branch-logit adapters, but hard object retrieval is still not a fully differentiable end-to-end generator.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator.csv`

The best closure is `0.877854` (`K=8`, `confidence_selected_joint_selector_propagator`). P1 evaluates whether joint selector-propagator deployment closes the candidate-oracle gap while controlling harmful accepts. The conservative best under harmful-accept <= `0.25` is `0.877854` (`confidence_selected_joint_selector_propagator`). The train-selected deployed best is `0.877854` (`K=8`).

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | `confidence_selected_joint_selector_propagator` | 0.893248 | 0.608889 | 0.079168 | 0.069498 | 0.877854 | 1.000000 | 0.075555 | 0.000000 | `passes_current_p1_threshold` |
| 16 | `joint_selector_propagator` | 0.985026 | 0.515555 | 0.063807 | 0.010953 | 0.171659 | 1.000000 | 0.444444 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `joint_selector_propagator` | 0.972520 | 0.493333 | 0.043295 | 0.002611 | 0.060311 | 1.000000 | 0.511111 | 0.000000 | `insufficient_no_harm_rejection` |
| 8 | `joint_selector_propagator` | 0.893248 | 0.608889 | 0.079168 | 0.069498 | 0.877854 | 1.000000 | 0.075555 | 0.000000 | `passes_current_p1_threshold` |
| 8 | `train_selected_joint_selector_propagator` | 0.893248 | 0.608889 | 0.079168 | 0.069498 | 0.877854 | 1.000000 | 0.075555 | 0.000000 | `passes_current_p1_threshold` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.972537 | 0.520000 | 0.043295 | 0.002594 | 0.059919 | 0.448889 | 0.200000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `train_selected_joint_selector_propagator` | 0.972537 | 0.520000 | 0.043295 | 0.002594 | 0.059919 | 0.448889 | 0.200000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `confidence_selected_joint_selector_propagator` | 0.997686 | 0.511111 | 0.063807 | -0.001707 | -0.026759 | 0.484444 | 0.222222 | 0.000000 | `harmful_transfer` |
| 16 | `train_selected_joint_selector_propagator` | 0.997686 | 0.511111 | 0.063807 | -0.001707 | -0.026759 | 0.484444 | 0.222222 | 0.000000 | `harmful_transfer` |

## Interpretation

- The CSV keeps all reject-margin/risk-penalty deployment sweep points.
- The table below shows the best policy per K and the strongest overall policies.
- A useful deployed policy needs both high closure and low harmful accepts.
- If this probe underperforms the direct candidate-regret gate, simply coupling selector scores to propagation branch loss is not enough; object retrieval, candidate generation, and transition dynamics need deeper joint training.

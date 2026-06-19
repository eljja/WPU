# Pairwise No-Harm Joint Selector-Propagator Results

This report summarizes a P1 ablation that adds a pairwise no-harm score margin to the joint selector-propagator objective. It tests whether high harmful accept at K=16/32 is not merely a confidence-threshold problem, but a baseline-safe ordering problem in the selector scores themselves.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_pairwise_noharm_w01.csv`

The best closure is `0.239301` (`K=32`, `joint_selector_propagator`). P1 evaluates whether joint selector-propagator deployment closes the candidate-oracle gap while controlling harmful accepts. The conservative best under harmful-accept <= `0.25` is `0.084224` (`confidence_selected_joint_selector_propagator`). The train-selected deployed best is `0.191273` (`K=32`).

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `joint_selector_propagator` | 0.991877 | 0.480000 | 0.052722 | 0.006145 | 0.116552 | 1.000000 | 0.333333 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `joint_selector_propagator` | 0.966575 | 0.502222 | 0.019081 | 0.004566 | 0.239301 | 0.933333 | 0.417778 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.967491 | 0.493333 | 0.019081 | 0.003650 | 0.191273 | 0.737778 | 0.328889 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `train_selected_joint_selector_propagator` | 0.967491 | 0.493333 | 0.019081 | 0.003650 | 0.191273 | 0.737778 | 0.328889 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `confidence_selected_joint_selector_propagator` | 0.993582 | 0.484444 | 0.052722 | 0.004440 | 0.084224 | 0.613333 | 0.173333 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_joint_selector_propagator` | 0.993582 | 0.484444 | 0.052722 | 0.004440 | 0.084224 | 0.613333 | 0.173333 | 0.000000 | `partial_but_insufficient_gap_closure` |

## Interpretation

- The CSV keeps all reject-margin/risk-penalty deployment sweep points.
- The table below shows the best policy per K and the strongest overall policies.
- A useful deployed policy needs both high closure and low harmful accepts.
- If this probe underperforms the direct candidate-regret gate, simply coupling selector scores to propagation branch loss is not enough; object retrieval, candidate generation, and transition dynamics need deeper joint training.
- If the pairwise no-harm margin lowers harmful accept but also collapses closure, P1 is not solved by safety regularization alone; safe candidate generation and relation-aware propagation quality must improve together.

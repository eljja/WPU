# Pairwise No-Harm Joint Selector-Propagator Results

This report summarizes a P1 ablation that adds a pairwise no-harm score margin to the joint selector-propagator objective. It tests whether high harmful accept at K=16/32 is not merely a confidence-threshold problem, but a baseline-safe ordering problem in the selector scores themselves.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_pairwise_noharm_w03.csv`

The best closure is `0.200230` (`K=32`, `joint_selector_propagator`). P1 evaluates whether joint selector-propagator deployment closes the candidate-oracle gap while controlling harmful accepts. The conservative best under harmful-accept <= `0.25` is `0.200230` (`joint_selector_propagator`). The train-selected deployed best is `0.192596` (`K=32`).

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `joint_selector_propagator` | 0.990521 | 0.511111 | 0.050670 | 0.008874 | 0.175125 | 0.764444 | 0.195555 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `joint_selector_propagator` | 0.966322 | 0.497778 | 0.020697 | 0.004144 | 0.200230 | 0.226667 | 0.053333 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.966480 | 0.497778 | 0.020697 | 0.003986 | 0.192596 | 0.160000 | 0.026667 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `train_selected_joint_selector_propagator` | 0.966480 | 0.497778 | 0.020697 | 0.003986 | 0.192596 | 0.160000 | 0.026667 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `confidence_selected_joint_selector_propagator` | 0.992009 | 0.506667 | 0.050670 | 0.007385 | 0.145747 | 0.622222 | 0.155556 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_joint_selector_propagator` | 0.992009 | 0.506667 | 0.050670 | 0.007385 | 0.145747 | 0.622222 | 0.155556 | 0.000000 | `partial_but_insufficient_gap_closure` |

## Interpretation

- The CSV keeps all reject-margin/risk-penalty deployment sweep points.
- The table below shows the best policy per K and the strongest overall policies.
- A useful deployed policy needs both high closure and low harmful accepts.
- If this probe underperforms the direct candidate-regret gate, simply coupling selector scores to propagation branch loss is not enough; object retrieval, candidate generation, and transition dynamics need deeper joint training.
- If the pairwise no-harm margin lowers harmful accept but also collapses closure, P1 is not solved by safety regularization alone; safe candidate generation and relation-aware propagation quality must improve together.

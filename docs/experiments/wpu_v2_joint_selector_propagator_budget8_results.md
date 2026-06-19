# Joint Selector-Propagator Results

This report summarizes a P1 ablation that increases the joint selector-propagator working-set budget. It tests whether K=16/32 fails mainly because budget=4 cuts too much causal state.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_budget8.csv`

The best closure is `0.109276` (`K=32`, `joint_selector_propagator`). P1 evaluates whether joint selector-propagator deployment closes the candidate-oracle gap while controlling harmful accepts. The conservative best under harmful-accept <= `0.25` is `0.092290` (`confidence_selected_joint_selector_propagator`). The train-selected deployed best is `0.092290` (`K=32`).

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `joint_selector_propagator` | 1.005292 | 0.506667 | 0.046077 | 0.003394 | 0.073651 | 1.000000 | 0.417778 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `joint_selector_propagator` | 0.970192 | 0.537778 | 0.026716 | 0.002919 | 0.109276 | 1.000000 | 0.360000 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.970646 | 0.542222 | 0.026716 | 0.002466 | 0.092290 | 0.582222 | 0.160000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `train_selected_joint_selector_propagator` | 0.970646 | 0.542222 | 0.026716 | 0.002466 | 0.092290 | 0.582222 | 0.160000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `confidence_selected_joint_selector_propagator` | 1.005881 | 0.515556 | 0.046077 | 0.002804 | 0.060864 | 0.657778 | 0.240000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_joint_selector_propagator` | 1.005881 | 0.515556 | 0.046077 | 0.002804 | 0.060864 | 0.657778 | 0.240000 | 0.000000 | `partial_but_insufficient_gap_closure` |

## Interpretation

- The CSV keeps all reject-margin/risk-penalty deployment sweep points.
- The table below shows the best policy per K and the strongest overall policies.
- A useful deployed policy needs both high closure and low harmful accepts.
- If this probe underperforms the direct candidate-regret gate, simply coupling selector scores to propagation branch loss is not enough; object retrieval, candidate generation, and transition dynamics need deeper joint training.
- If increasing the budget gives only small gains, larger-K failure is not just working-set size; candidate quality and transition dynamics remain bottlenecks.

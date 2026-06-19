# Learned-Safe-Candidate Joint Selector-Propagator Results

This report summarizes a P1 ablation that adds train-fold learned object-level safe candidate generators to the joint selector-propagator candidate pool. The generators imitate diverse interaction, proximity, density, and axis teachers to test whether candidate generation itself improves before any stronger rejection head is applied. This is still a teacher-supervised diagnostic, not a fully differentiable generator.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_learned_safe_candidates.csv`

The best closure is `0.271116` (`K=16`, `joint_selector_propagator`). P1 evaluates whether joint selector-propagator deployment closes the candidate-oracle gap while controlling harmful accepts. The conservative best under harmful-accept <= `0.25` is `0.246071` (`confidence_selected_joint_selector_propagator`). The train-selected deployed best is `0.246071` (`K=16`).

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `joint_selector_propagator` | 0.998987 | 0.560000 | 0.089258 | 0.024199 | 0.271116 | 1.000000 | 0.342222 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.993054 | 0.475556 | 0.039324 | 0.005639 | 0.143398 | 0.377778 | 0.133333 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `confidence_selected_joint_selector_propagator` | 1.001222 | 0.542222 | 0.089258 | 0.021964 | 0.246071 | 0.666667 | 0.222222 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_joint_selector_propagator` | 1.001222 | 0.542222 | 0.089258 | 0.021964 | 0.246071 | 0.666667 | 0.222222 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `train_selected_joint_selector_propagator` | 0.993054 | 0.475556 | 0.039324 | 0.005639 | 0.143398 | 0.377778 | 0.133333 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `joint_selector_propagator` | 1.000165 | 0.475556 | 0.039324 | -0.001473 | -0.037453 | 1.000000 | 0.537778 | 0.000000 | `harmful_transfer` |

## Interpretation

- The CSV keeps all reject-margin/risk-penalty deployment sweep points.
- The table below shows the best policy per K and the strongest overall policies.
- A useful deployed policy needs both high closure and low harmful accepts.
- If this probe underperforms the direct candidate-regret gate, simply coupling selector scores to propagation branch loss is not enough; object retrieval, candidate generation, and transition dynamics need deeper joint training.
- If learned-safe candidates do not raise closure, teacher-supervised object scoring is insufficient; candidate generation must be trained directly against propagation loss and no-harm objectives.

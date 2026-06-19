# Structured-Candidate Joint Selector-Propagator Results

This report summarizes a P1 ablation that adds deterministic structured candidates to the joint selector-propagator probe. It tests whether K=16/32 fails because the system lacks safe high-quality candidates, rather than only because it fails to reject unsafe ones.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_structured_candidates_no_margin.csv`

The best closure is `0.327084` (`K=16`, `joint_selector_propagator`). P1 evaluates whether joint selector-propagator deployment closes the candidate-oracle gap while controlling harmful accepts. The conservative best under harmful-accept <= `0.25` is `0.241624` (`confidence_selected_joint_selector_propagator`). The train-selected deployed best is `0.241624` (`K=16`).

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `joint_selector_propagator` | 0.967612 | 0.551111 | 0.079496 | 0.026002 | 0.327084 | 1.000000 | 0.355555 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.990694 | 0.502222 | 0.030397 | 0.003785 | 0.124512 | 0.235556 | 0.080000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `confidence_selected_joint_selector_propagator` | 0.974406 | 0.542222 | 0.079496 | 0.019208 | 0.241624 | 0.653333 | 0.200000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_joint_selector_propagator` | 0.974406 | 0.542222 | 0.079496 | 0.019208 | 0.241624 | 0.653333 | 0.200000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `train_selected_joint_selector_propagator` | 0.990694 | 0.502222 | 0.030397 | 0.003785 | 0.124512 | 0.235556 | 0.080000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `joint_selector_propagator` | 0.993703 | 0.497778 | 0.030397 | 0.000775 | 0.025496 | 1.000000 | 0.488889 | 0.000000 | `insufficient_no_harm_rejection` |

## Interpretation

- The CSV keeps all reject-margin/risk-penalty deployment sweep points.
- The table below shows the best policy per K and the strongest overall policies.
- A useful deployed policy needs both high closure and low harmful accepts.
- If this probe underperforms the direct candidate-regret gate, simply coupling selector scores to propagation branch loss is not enough; object retrieval, candidate generation, and transition dynamics need deeper joint training.
- If structured candidates do not improve oracle or deployed closure, hand-built diversity is not enough; candidate generation needs learned propagation-aware verification.

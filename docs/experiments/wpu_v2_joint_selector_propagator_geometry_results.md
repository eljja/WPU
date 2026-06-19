# Joint Selector-Propagator Results

This report summarizes a geometry-context ablation of the joint selector-propagator probe. It appends candidate-level geometry and force descriptors to test whether the K=16/32 failure is just a descriptor bottleneck. Weak results imply that larger-K P1 needs deeper joint retrieval, generation, and propagation dynamics rather than feature concatenation.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_geometry.csv`

The best closure is `0.087030` (`K=16`, `joint_selector_propagator`). P1 evaluates whether joint selector-propagator deployment closes the candidate-oracle gap while controlling harmful accepts. The conservative best under harmful-accept <= `0.25` is `0.044841` (`confidence_selected_joint_selector_propagator`). The train-selected deployed best is `0.044841` (`K=16`).

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `joint_selector_propagator` | 0.994853 | 0.511111 | 0.053853 | 0.004687 | 0.087030 | 1.000000 | 0.408889 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `joint_selector_propagator` | 0.973978 | 0.488889 | 0.038579 | 0.000598 | 0.015495 | 1.000000 | 0.511111 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `confidence_selected_joint_selector_propagator` | 0.997125 | 0.511111 | 0.053853 | 0.002415 | 0.044841 | 0.213333 | 0.080000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_joint_selector_propagator` | 0.997125 | 0.511111 | 0.053853 | 0.002415 | 0.044841 | 0.213333 | 0.080000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.974169 | 0.493333 | 0.038579 | 0.000407 | 0.010550 | 0.280000 | 0.124444 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `train_selected_joint_selector_propagator` | 0.974169 | 0.493333 | 0.038579 | 0.000407 | 0.010550 | 0.280000 | 0.124444 | 0.000000 | `partial_but_insufficient_gap_closure` |

## Interpretation

- The CSV keeps all reject-margin/risk-penalty deployment sweep points.
- The table below shows the best policy per K and the strongest overall policies.
- A useful deployed policy needs both high closure and low harmful accepts.
- If this probe underperforms the direct candidate-regret gate, simply coupling selector scores to propagation branch loss is not enough; object retrieval, candidate generation, and transition dynamics need deeper joint training.
- If geometry/force descriptors do not improve K=16/32, the next bottleneck is candidate generation and relation-conditioned transition learning, not more post-hoc descriptor concatenation.

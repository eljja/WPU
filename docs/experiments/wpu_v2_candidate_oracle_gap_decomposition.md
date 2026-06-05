# WPU v2 Candidate-Oracle Gap Decomposition

This report decomposes priority 1 by feature variant, causal working-set
size, and deployed policy family. It is diagnostic evidence: it does not
claim the candidate-oracle gap is solved.

Source CSV:

- `docs/experiments/wpu_v2_retriever_invariant_set_scorer.csv`

Derived CSV:

- `docs/experiments/wpu_v2_candidate_oracle_gap_decomposition.csv`

## Decomposition Table

| feature variant | K | oracle gain | best non-oracle policy | best closure | risk closure | oracle match | negative policies | failure mode |
|---|---:|---:|---|---:|---:|---:|---:|---|
| role_geometry_family | 8 | 0.032895 | risk_adjusted_selected_mechanism | 0.195451 | 0.195451 | 0.228889 | 0 | partial_gap_closure_only |
| role_geometry_family | 16 | 0.061174 | risk_adjusted_selected_mechanism | 0.244220 | 0.244220 | 0.077778 | 0 | partial_gap_closure_only |
| role_geometry_family | 32 | 0.035547 | risk_adjusted_selected_mechanism | 0.042131 | 0.042131 | 0.113333 | 3 | selection_signal_absent_or_miscalibrated |
| role_geometry_only | 8 | 0.032895 | risk_adjusted_selected_mechanism | 0.113548 | 0.113548 | 0.140000 | 0 | partial_gap_closure_only |
| role_geometry_only | 16 | 0.061174 | risk_adjusted_selected_mechanism | 0.244220 | 0.244220 | 0.077778 | 2 | partial_gap_closure_only |
| role_geometry_only | 32 | 0.035547 | train_selected_mechanism | 0.047178 | -0.012226 | 0.142222 | 4 | selection_signal_absent_or_miscalibrated |

## Interpretation

The best non-oracle closure found in the committed aggregate evidence is `0.244220`.
This remains below the dashboard threshold of `0.5`. The failure is
not simply that one policy was omitted from the dashboard: across the
available policy families, several variants have negative closure and
the large-`K` condition remains weak.

Weak cells:

- `role_geometry_family`, K=32: selection_signal_absent_or_miscalibrated
- `role_geometry_only`, K=32: selection_signal_absent_or_miscalibrated

## Next Technical Target

The next P1 implementation should move below aggregate policy selection:
per-candidate uncertainty, sample-level no-harm gating, and regret
targets must be attached to the candidate scorer before deployment.
Changing only the aggregate policy selector is unlikely to close the gap.

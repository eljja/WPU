# PyBullet Calibration-Cost Frontier Audit

This derived audit normalizes existing uncertainty-gated recompute, learned gate, and mechanism-adaptive policy results onto common accuracy, calibration, and cost axes. The purpose is not to claim P5 solved, but to test whether accuracy improvement, calibration improvement, and low added compute are simultaneously achieved.

Source CSV: `docs/experiments/pybullet_calibration_cost_frontier.csv`

## Summary

- Non-reference calibration-safe policies within the low-cost budget (`cost_proxy <= 0.25`): `1`.
- The most accurate low-cost policy is `source_learned_p0.12` with accuracy delta `0.052910`, ECE delta `0.010769`, and cost proxy `0.205027`.
- The strongest ECE improvement is `mechanism_aware_adaptive_policy` with ECE delta `-0.099347`, accuracy delta `0.198412`, and cost proxy `1.000000`.
- The lowest-cost non-reference calibration-safe policy is `mechanism_selective_best_safe` with cost proxy `0.247355`.
- Pareto-efficient policies: wpu_sparse_uncertainty_probe, wpu_sparse_learned_gate_probe, wpu_gated_t0.34, source_learned_p0.12, mechanism_selective_best_safe, source_learned_p0.08, fewshot_learned_p0.12, source_learned_p0.04, fewshot_learned_p0.08, source_learned_p0.02, source_learned_p0.01, source_learned_p0.00, fewshot_learned_p0.04, fewshot_learned_p0.02, fewshot_learned_p0.01, fewshot_learned_p0.00, wpu_gated_t0.40, wpu_gated_t0.45, mechanism_aware_adaptive_policy.

## Interpretation

Global confidence thresholds and sparse-output benefit gates still do not solve low-cost calibration-safe routing. The new mechanism-selective calibration gate, however, finds a non-reference policy that improves accuracy, ECE, and Brier at low average cost. P5 is therefore narrowed from an impossible-looking tradeoff to a mechanism-identification and calibration-aware policy-selection problem. The caveat remains decisive: this positive result is mechanism-specific and adapted, not zero-shot calibration-safe routing.

## Frontier Rows

| family | policy | protocol | cost | acc_delta | ece_delta | brier_delta | safe | low_cost | pareto |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| learned_sparse_output_gate | wpu_sparse_learned_gate_probe | sparse_reference | 0.000000 | 0.000000 | 0.000000 | 0.000000 | True | True | True |
| uncertainty_threshold | wpu_sparse_uncertainty_probe | sparse_reference | 0.000000 | 0.000000 | 0.000000 | 0.000000 | True | True | True |
| uncertainty_threshold | wpu_gated_t0.34 | zero_shot_threshold_gate | 0.025132 | 0.009260 | 0.005395 | -0.001779 | False | True | True |
| learned_sparse_output_gate | source_learned_p0.12 | source_learned_gate | 0.205027 | 0.052910 | 0.010769 | -0.033549 | False | True | True |
| mechanism_selective_calibration_gate | mechanism_selective_best_safe | mechanism_selective_detect_and_adapt | 0.247355 | 0.029100 | -0.001652 | -0.030758 | True | True | True |
| learned_sparse_output_gate | source_learned_p0.08 | source_learned_gate | 0.280423 | 0.064815 | 0.020740 | -0.043976 | False | False | True |
| learned_sparse_output_gate | fewshot_learned_p0.12 | fewshot_learned_gate | 0.292328 | 0.089947 | 0.041513 | -0.050933 | False | False | True |
| learned_sparse_output_gate | source_learned_p0.04 | source_learned_gate | 0.361111 | 0.071429 | 0.023684 | -0.049750 | False | False | True |
| learned_sparse_output_gate | fewshot_learned_p0.08 | fewshot_learned_gate | 0.387566 | 0.100529 | 0.046772 | -0.056375 | False | False | True |
| learned_sparse_output_gate | source_learned_p0.02 | source_learned_gate | 0.423280 | 0.075397 | 0.030267 | -0.052832 | False | False | True |
| learned_sparse_output_gate | source_learned_p0.01 | source_learned_gate | 0.466931 | 0.079365 | 0.031195 | -0.054161 | False | False | True |
| learned_sparse_output_gate | source_learned_p0.00 | source_learned_gate | 0.511905 | 0.080688 | 0.028263 | -0.057034 | False | False | True |
| learned_sparse_output_gate | fewshot_learned_p0.04 | fewshot_learned_gate | 0.517196 | 0.092592 | 0.035174 | -0.060443 | False | False | True |
| learned_sparse_output_gate | fewshot_learned_p0.02 | fewshot_learned_gate | 0.596561 | 0.091270 | 0.025773 | -0.062870 | False | False | True |
| learned_sparse_output_gate | fewshot_learned_p0.01 | fewshot_learned_gate | 0.634921 | 0.088624 | 0.025530 | -0.062387 | False | False | True |
| learned_sparse_output_gate | fewshot_learned_p0.00 | fewshot_learned_gate | 0.671958 | 0.091270 | 0.020279 | -0.064075 | False | False | True |
| uncertainty_threshold | wpu_gated_t0.40 | zero_shot_threshold_gate | 0.867725 | 0.071428 | -0.009109 | -0.021645 | True | False | True |
| uncertainty_threshold | wpu_gated_t0.45 | zero_shot_threshold_gate | 0.985450 | 0.071428 | -0.016396 | -0.023272 | True | False | True |
| learned_sparse_output_gate | wpu_local_dense_learned_probe | full_local_dense_reference | 1.000000 | 0.074074 | -0.014312 | -0.063911 | True | False | False |
| mechanism_adaptive_policy | mechanism_aware_adaptive_policy | detect_and_adapt | 1.000000 | 0.198412 | -0.099347 | -0.155443 | True | False | True |
| uncertainty_threshold | wpu_gated_t0.50 | zero_shot_threshold_gate | 1.000000 | 0.071428 | -0.014473 | -0.023864 | True | False | False |
| uncertainty_threshold | wpu_gated_t0.55 | zero_shot_threshold_gate | 1.000000 | 0.071428 | -0.014473 | -0.023864 | True | False | False |
| uncertainty_threshold | wpu_gated_t0.60 | zero_shot_threshold_gate | 1.000000 | 0.071428 | -0.014473 | -0.023864 | True | False | False |
| uncertainty_threshold | wpu_gated_t0.65 | zero_shot_threshold_gate | 1.000000 | 0.071428 | -0.014473 | -0.023864 | True | False | False |

# PyBullet Calibration-Cost Frontier Audit

이 파생 감사는 기존 uncertainty-gated recompute, learned gate, mechanism-adaptive policy 결과를 같은 축으로 정규화한다. 목표는 P5를 성공처럼 포장하는 것이 아니라, 정확도 개선, calibration 개선, 낮은 추가 계산이 동시에 가능한지를 검증하는 것이다.

Source CSV: `docs/experiments/pybullet_calibration_cost_frontier.csv`

## Summary

- Low-cost budget(`cost_proxy <= 0.25`) 안에서 non-reference calibration-safe policy는 1개다.
- 가장 정확한 low-cost policy는 `source_learned_p0.12`이며 accuracy delta `0.052910`, ECE delta `0.010769`, cost proxy `0.205027`이다.
- 가장 큰 ECE 개선은 `mechanism_aware_adaptive_policy`이며 ECE delta `-0.099347`, accuracy delta `0.198412`, cost proxy `1.000000`이다.
- 최저 비용 non-reference calibration-safe policy는 `mechanism_selective_best_safe`이며 cost proxy `0.247355`이다.
- Pareto-efficient policies: wpu_sparse_uncertainty_probe, wpu_sparse_learned_gate_probe, wpu_gated_t0.34, source_learned_p0.12, mechanism_selective_best_safe, source_learned_p0.08, fewshot_learned_p0.12, source_learned_p0.04, fewshot_learned_p0.08, source_learned_p0.02, source_learned_p0.01, source_learned_p0.00, fewshot_learned_p0.04, fewshot_learned_p0.02, fewshot_learned_p0.01, fewshot_learned_p0.00, wpu_gated_t0.40, wpu_gated_t0.45, mechanism_aware_adaptive_policy.

## Interpretation

현재 증거에서 전역 confidence threshold와 sparse-output benefit gate는 아직 low-cost calibration-safe routing을 해결하지 못한다. 하지만 새 mechanism-selective calibration gate는 낮은 평균 cost에서 accuracy, ECE, Brier를 동시에 개선하는 non-reference policy를 만든다. 따라서 P5는 불가능한 문제가 아니라, mechanism 식별과 calibration-aware policy selection이 필요한 문제로 좁혀졌다. 단, 이 positive 결과는 mechanism-specific adapted setting이며 zero-shot calibration-safe routing은 아직 아니다.

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

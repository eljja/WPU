# Verified Candidate Controller 결과

이 문서는 후보 working set의 object/context feature에 sparse 및 local-dense propagation의 label-free verification signature를 추가한 P1 probe를 요약한다. signature는 branch confidence, entropy, sparse/dense disagreement, delta norm gap처럼 정답 label 없이 계산 가능한 값만 사용한다.

Source CSV: `docs/experiments/wpu_v2_verified_candidate_controller.csv`

최고 closure는 `0.024989` (`K=16`, `train_selected_verified_candidate_controller`)다. P1 목표 `0.5`를 기준으로 verified-controller deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건의 conservative best는 `0.023029` (`verified_uncertainty_regret_gate_r0p5_m0p005`)다. Train-selected deployed best는 `0.024989` (`K=16`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | `verified_uncertainty_regret_gate_r0p5_m0p005` | 0.987666 | 0.513333 | 0.033246 | 0.000766 | 0.023029 | 0.557778 | 0.248889 | 0.092232 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_verified_candidate_controller` | 0.964643 | 0.506667 | 0.061627 | 0.001540 | 0.024989 | 0.835556 | 0.424444 | -0.055012 | `insufficient_no_harm_rejection` |
| 32 | `verified_uncertainty_regret_gate_r0p75_m0p01` | 1.003407 | 0.482222 | 0.035925 | 0.000688 | 0.019140 | 0.215556 | 0.102222 | 0.049777 | `partial_but_insufficient_gap_closure` |
| 16 | `verified_uncertainty_regret_gate_r0p5_m0p0025` | 0.964780 | 0.506667 | 0.061627 | 0.001402 | 0.022756 | 0.628889 | 0.315556 | -0.055012 | `insufficient_no_harm_rejection` |
| 8 | `verified_candidate_regret_gate` | 0.987743 | 0.511111 | 0.033246 | 0.000689 | 0.020725 | 0.942222 | 0.415556 | 0.092232 | `insufficient_no_harm_rejection` |
| 32 | `verified_uncertainty_regret_gate_r1_m0` | 1.003447 | 0.482222 | 0.035925 | 0.000648 | 0.018026 | 0.188889 | 0.091111 | 0.049777 | `partial_but_insufficient_gap_closure` |
| 32 | `verified_uncertainty_regret_gate_r0p75_m0p02` | 1.003449 | 0.482222 | 0.035925 | 0.000646 | 0.017971 | 0.142222 | 0.066667 | 0.049777 | `partial_but_insufficient_gap_closure` |
| 8 | `verified_uncertainty_regret_gate_r0p5_m0p01` | 0.987851 | 0.508889 | 0.033246 | 0.000581 | 0.017464 | 0.473333 | 0.211111 | 0.092232 | `partial_but_insufficient_gap_closure` |
| 8 | `verified_uncertainty_regret_gate_r1p5_m0p05` | 0.987859 | 0.504444 | 0.033246 | 0.000573 | 0.017223 | 0.046667 | 0.015556 | 0.092232 | `partial_but_insufficient_gap_closure` |
| 16 | `verified_uncertainty_regret_gate_r3_m0p05` | 0.965223 | 0.504444 | 0.061627 | 0.000960 | 0.015571 | 0.011111 | 0.002222 | -0.055012 | `partial_but_insufficient_gap_closure` |
| 8 | `verified_uncertainty_regret_gate_r0p5_m0p0025` | 0.987930 | 0.513333 | 0.033246 | 0.000501 | 0.015076 | 0.611111 | 0.277778 | 0.092232 | `insufficient_no_harm_rejection` |
| 8 | `verified_candidate_regret_gate_m0p0025` | 0.987975 | 0.511111 | 0.033246 | 0.000457 | 0.013746 | 0.931111 | 0.413333 | 0.092232 | `insufficient_no_harm_rejection` |
| 16 | `verified_uncertainty_regret_gate_r0p5_m0p005` | 0.965392 | 0.506667 | 0.061627 | 0.000790 | 0.012826 | 0.580000 | 0.286667 | -0.055012 | `insufficient_no_harm_rejection` |
| 8 | `verified_uncertainty_regret_gate_r1_m0p02` | 0.988008 | 0.506667 | 0.033246 | 0.000423 | 0.012736 | 0.151111 | 0.064444 | 0.092232 | `partial_but_insufficient_gap_closure` |
| 32 | `verified_uncertainty_regret_gate_r1_m0p0025` | 1.003643 | 0.480000 | 0.035925 | 0.000451 | 0.012559 | 0.173333 | 0.082222 | 0.049777 | `partial_but_insufficient_gap_closure` |
| 32 | `verified_candidate_regret_gate_m0p05` | 1.003701 | 0.482222 | 0.035925 | 0.000394 | 0.010967 | 0.195556 | 0.095556 | 0.049777 | `partial_but_insufficient_gap_closure` |
| 8 | `verified_uncertainty_regret_gate_r1p5_m0p02` | 0.988077 | 0.504444 | 0.033246 | 0.000355 | 0.010666 | 0.080000 | 0.031111 | 0.092232 | `partial_but_insufficient_gap_closure` |
| 16 | `verified_uncertainty_regret_gate` | 0.965526 | 0.506667 | 0.061627 | 0.000657 | 0.010664 | 0.664445 | 0.335556 | -0.055012 | `insufficient_no_harm_rejection` |

## 해석

- CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.
- 이 실험이 direct candidate-regret gate보다 약하면, label-free sparse/dense verification signature를 후처리 feature로 추가하는 것만으로는 P1을 해결하지 못한다고 해석한다.

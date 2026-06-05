# Candidate Regret Gate 결과

이 문서는 candidate별 `candidate_loss - learned_loss`를 직접 예측하고, 예측 regret이 충분히 낮을 때만 baseline 대신 선택하는 P1 probe를 요약한다.

Source CSV: `docs/experiments/wpu_v2_candidate_regret_gate_penalty.csv`

최고 closure는 `0.123585` (`K=16`, `candidate_regret_gate`)다. P1 목표 `0.5`를 기준으로 candidate-regret deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건의 conservative best는 `0.083764` (`uncertainty_regret_gate`)다. Train-selected deployed best는 `0.081253` (`K=16`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | `candidate_regret_gate_m0p0025` | 0.986621 | 0.504445 | 0.032763 | 0.001811 | 0.055275 | 0.964444 | 0.428889 | 0.055405 | `insufficient_no_harm_rejection` |
| 16 | `candidate_regret_gate` | 0.958672 | 0.506667 | 0.060773 | 0.007511 | 0.123585 | 0.908889 | 0.422222 | 0.076345 | `insufficient_no_harm_rejection` |
| 32 | `train_selected_candidate_regret_gate` | 1.003477 | 0.506667 | 0.035577 | 0.000618 | 0.017360 | 0.375555 | 0.197778 | 0.050925 | `partial_but_insufficient_gap_closure` |
| 16 | `candidate_regret_gate_m0p0025` | 0.959073 | 0.506667 | 0.060773 | 0.007109 | 0.116983 | 0.871111 | 0.406667 | 0.076345 | `insufficient_no_harm_rejection` |
| 16 | `candidate_regret_gate_m0p005` | 0.959331 | 0.506667 | 0.060773 | 0.006852 | 0.112748 | 0.851111 | 0.400000 | 0.076345 | `insufficient_no_harm_rejection` |
| 16 | `candidate_regret_gate_m0p01` | 0.960389 | 0.504445 | 0.060773 | 0.005794 | 0.095342 | 0.764444 | 0.366667 | 0.076345 | `insufficient_no_harm_rejection` |
| 16 | `uncertainty_regret_gate` | 0.961092 | 0.508889 | 0.060773 | 0.005091 | 0.083764 | 0.222222 | 0.075556 | 0.076345 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_candidate_regret_gate` | 0.961245 | 0.504444 | 0.060773 | 0.004938 | 0.081253 | 0.235556 | 0.088889 | 0.076345 | `partial_but_insufficient_gap_closure` |
| 16 | `uncertainty_regret_gate_r0p5_m0p0025` | 0.961633 | 0.508889 | 0.060773 | 0.004549 | 0.074859 | 0.166667 | 0.046667 | 0.076345 | `partial_but_insufficient_gap_closure` |
| 16 | `candidate_regret_gate_m0p02` | 0.961942 | 0.504445 | 0.060773 | 0.004241 | 0.069788 | 0.611111 | 0.302222 | 0.076345 | `insufficient_no_harm_rejection` |
| 8 | `candidate_regret_gate` | 0.986702 | 0.504445 | 0.032763 | 0.001730 | 0.052791 | 0.977778 | 0.435556 | 0.055405 | `insufficient_no_harm_rejection` |
| 8 | `candidate_regret_gate_m0p005` | 0.986936 | 0.502222 | 0.032763 | 0.001496 | 0.045649 | 0.948889 | 0.426667 | 0.055405 | `insufficient_no_harm_rejection` |
| 16 | `uncertainty_regret_gate_r0p5_m0p005` | 0.964260 | 0.504444 | 0.060773 | 0.001923 | 0.031642 | 0.115556 | 0.033333 | 0.076345 | `partial_but_insufficient_gap_closure` |
| 8 | `candidate_regret_gate_m0p01` | 0.987405 | 0.497778 | 0.032763 | 0.001026 | 0.031322 | 0.864444 | 0.397778 | 0.055405 | `insufficient_no_harm_rejection` |
| 16 | `uncertainty_regret_gate_r0p5_m0p01` | 0.964742 | 0.504444 | 0.060773 | 0.001441 | 0.023705 | 0.057778 | 0.020000 | 0.076345 | `partial_but_insufficient_gap_closure` |
| 16 | `uncertainty_regret_gate_r0p75_m0` | 0.965190 | 0.504444 | 0.060773 | 0.000993 | 0.016340 | 0.020000 | 0.002222 | 0.076345 | `partial_but_insufficient_gap_closure` |
| 8 | `candidate_regret_gate_m0p02` | 0.988037 | 0.502222 | 0.032763 | 0.000395 | 0.012056 | 0.566667 | 0.271111 | 0.055405 | `insufficient_no_harm_rejection` |
| 8 | `train_selected_candidate_regret_gate` | 0.988103 | 0.506667 | 0.032763 | 0.000328 | 0.010023 | 0.620000 | 0.282222 | 0.055405 | `insufficient_no_harm_rejection` |

## 해석

- CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.

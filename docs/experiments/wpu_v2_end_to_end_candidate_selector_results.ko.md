# End-to-End Candidate Selector 결과

이 문서는 후보 working set selector를 downstream propagation loss와 baseline보다 나빠지는 no-harm mass에 직접 맞춰 학습한 P1 probe를 요약한다. 목표는 oracle label imitation이 아니라 선택 정책의 실제 expected loss를 줄이는 것이다.

Source CSV: `docs/experiments/wpu_v2_end_to_end_candidate_selector.csv`

최고 closure는 `0.106927` (`K=16`, `end_to_end_selector_pg0p2_pmin0`)다. P1 목표 `0.5`를 기준으로 end-to-end selector deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건을 만족하는 deployed policy는 없다. Train-selected deployed best는 `0.096833` (`K=16`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | `end_to_end_selector_pg0_pmin0p6` | 0.988301 | 0.495556 | 0.033003 | 0.000131 | 0.003969 | 0.851111 | 0.355556 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0p2_pmin0` | 0.959662 | 0.502222 | 0.060987 | 0.006521 | 0.106927 | 0.982222 | 0.382222 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `end_to_end_selector_pg0_pmin0p4` | 1.004137 | 0.486667 | 0.035586 | -0.000043 | -0.001197 | 0.975556 | 0.473333 | 0.000000 | `harmful_transfer` |
| 16 | `end_to_end_selector_pg0p2_pmin0p3` | 0.959662 | 0.502222 | 0.060987 | 0.006521 | 0.106927 | 0.982222 | 0.382222 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0p1_pmin0` | 0.959920 | 0.502222 | 0.060987 | 0.006263 | 0.102697 | 0.991111 | 0.384444 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0p1_pmin0p3` | 0.959920 | 0.502222 | 0.060987 | 0.006263 | 0.102697 | 0.991111 | 0.384444 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_candidate_selector` | 0.960019 | 0.502222 | 0.060987 | 0.006163 | 0.101061 | 0.995556 | 0.386667 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0_pmin0` | 0.960019 | 0.502222 | 0.060987 | 0.006163 | 0.101061 | 0.995556 | 0.386667 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0_pmin0p3` | 0.960019 | 0.502222 | 0.060987 | 0.006163 | 0.101061 | 0.995556 | 0.386667 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0p02_pmin0` | 0.960019 | 0.502222 | 0.060987 | 0.006163 | 0.101061 | 0.995556 | 0.386667 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0p02_pmin0p3` | 0.960019 | 0.502222 | 0.060987 | 0.006163 | 0.101061 | 0.995556 | 0.386667 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0p05_pmin0` | 0.960019 | 0.502222 | 0.060987 | 0.006163 | 0.101061 | 0.993333 | 0.386667 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0p05_pmin0p3` | 0.960019 | 0.502222 | 0.060987 | 0.006163 | 0.101061 | 0.993333 | 0.386667 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0p2_pmin0p4` | 0.960218 | 0.500000 | 0.060987 | 0.005965 | 0.097811 | 0.960000 | 0.377778 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `train_selected_end_to_end_candidate_selector` | 0.960277 | 0.502222 | 0.060987 | 0.005906 | 0.096833 | 0.977778 | 0.382222 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0_pmin0p5` | 0.960292 | 0.502222 | 0.060987 | 0.005891 | 0.096591 | 0.902222 | 0.353333 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0p02_pmin0p5` | 0.960292 | 0.502222 | 0.060987 | 0.005891 | 0.096591 | 0.902222 | 0.353333 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0p05_pmin0p5` | 0.960292 | 0.502222 | 0.060987 | 0.005891 | 0.096591 | 0.902222 | 0.353333 | 0.000000 | `insufficient_no_harm_rejection` |

## 해석

- CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.
- 이 실험도 direct candidate-regret gate보다 약하면, P1 병목은 후처리 threshold가 아니라 후보 생성/전파 모델과 selector의 더 깊은 공동학습 문제로 해석한다.

# Joint Object-Set Candidate Gate 결과

이 문서는 후보 working set의 명시적 object-set feature와 compact context를 함께 인코딩하는 P1 probe를 요약한다. 목표는 post-hoc threshold가 아니라 후보 state 자체를 보고 candidate regret와 no-harm accept를 예측할 수 있는지 평가하는 것이다.

Source CSV: `docs/experiments/wpu_v2_candidate_joint_gate_regression_heavy_k16.csv`

최고 closure는 `0.034751` (`K=16`, `joint_gate_p0p7_m0_r0p5`)다. P1 목표 `0.5`를 기준으로 joint object-set deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건의 conservative best는 `0.033842` (`joint_gate_p0p6_m0_r1`)다. Train-selected deployed best는 `-0.003089` (`K=16`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `joint_gate_p0p7_m0_r0p5` | 0.964059 | 0.513333 | 0.061120 | 0.002124 | 0.034751 | 0.635555 | 0.257778 | 0.020121 | `insufficient_no_harm_rejection` |
| 16 | `joint_gate_p0p6_m0_r1` | 0.964114 | 0.513333 | 0.061120 | 0.002068 | 0.033842 | 0.344445 | 0.140000 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p6_m0p01_r1` | 0.964119 | 0.508889 | 0.061120 | 0.002064 | 0.033770 | 0.251111 | 0.100000 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p001_r0p5` | 0.964136 | 0.515555 | 0.061120 | 0.002047 | 0.033485 | 0.615556 | 0.248889 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p55_m0p01_r1` | 0.964178 | 0.508889 | 0.061120 | 0.002005 | 0.032804 | 0.253333 | 0.102222 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p5_m0p01_r1` | 0.964178 | 0.508889 | 0.061120 | 0.002005 | 0.032804 | 0.253333 | 0.102222 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0_r1` | 0.964188 | 0.513333 | 0.061120 | 0.001995 | 0.032637 | 0.337778 | 0.137778 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p01_r1` | 0.964192 | 0.508889 | 0.061120 | 0.001990 | 0.032565 | 0.244445 | 0.097778 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p65_m0_r1` | 0.964236 | 0.513333 | 0.061120 | 0.001947 | 0.031852 | 0.342222 | 0.140000 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p65_m0p01_r1` | 0.964240 | 0.508889 | 0.061120 | 0.001942 | 0.031780 | 0.248889 | 0.100000 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p6_m0p001_r1` | 0.964274 | 0.511111 | 0.061120 | 0.001909 | 0.031234 | 0.335556 | 0.137778 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p55_m0p001_r1` | 0.964333 | 0.511111 | 0.061120 | 0.001850 | 0.030268 | 0.337778 | 0.140000 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p5_m0p001_r1` | 0.964333 | 0.511111 | 0.061120 | 0.001850 | 0.030268 | 0.337778 | 0.140000 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p6_m0_r0p5` | 0.964333 | 0.511111 | 0.061120 | 0.001850 | 0.030268 | 0.653333 | 0.266667 | 0.020121 | `insufficient_no_harm_rejection` |
| 16 | `joint_gate_p0p7_m0p001_r1` | 0.964347 | 0.511111 | 0.061120 | 0.001835 | 0.030029 | 0.328889 | 0.135556 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p65_m0p001_r1` | 0.964395 | 0.511111 | 0.061120 | 0.001787 | 0.029244 | 0.333334 | 0.137778 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p6_m0p001_r0p5` | 0.964411 | 0.513333 | 0.061120 | 0.001772 | 0.028995 | 0.633333 | 0.257778 | 0.020121 | `insufficient_no_harm_rejection` |
| 16 | `joint_gate_p0p6_m0p0025_r1` | 0.964469 | 0.508889 | 0.061120 | 0.001714 | 0.028037 | 0.324444 | 0.131111 | 0.020121 | `partial_but_insufficient_gap_closure` |

## 해석

- CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.
- 이 실험이 direct candidate-regret gate보다 약하면, 병목은 object-set feature 부재만이 아니라 cross-seed regret target 자체의 안정성 부족으로 해석한다.

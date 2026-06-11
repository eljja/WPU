# Joint Object-Set Candidate Gate 결과

이 문서는 후보 working set의 명시적 object-set feature와 compact context를 함께 인코딩하는 P1 probe를 요약한다. 목표는 post-hoc threshold가 아니라 후보 state 자체를 보고 candidate regret와 no-harm accept를 예측할 수 있는지 평가하는 것이다.

Source CSV: `docs/experiments/wpu_v2_candidate_joint_gate.csv`

최고 closure는 `0.101454` (`K=16`, `joint_gate_p0p7_m0p0025_r0p75`)다. P1 목표 `0.5`를 기준으로 joint object-set deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건의 conservative best는 `0.101454` (`joint_gate_p0p7_m0p0025_r0p75`)다. Train-selected deployed best는 `0.072167` (`K=16`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | `joint_gate_p0p6_m0p0025_r1` | 0.987104 | 0.502222 | 0.032854 | 0.001328 | 0.040415 | 0.391111 | 0.166667 | -0.020419 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p0025_r0p75` | 0.959996 | 0.502222 | 0.060983 | 0.006187 | 0.101454 | 0.446667 | 0.191111 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 32 | `joint_gate_p0p5_m0p02_r0` | 1.002218 | 0.502222 | 0.035570 | 0.001877 | 0.052763 | 0.637778 | 0.295556 | -0.012262 | `insufficient_no_harm_rejection` |
| 16 | `joint_gate_p0p7_m0p005_r0p75` | 0.960104 | 0.500000 | 0.060983 | 0.006079 | 0.099677 | 0.440000 | 0.188889 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p01_r0p75` | 0.960104 | 0.500000 | 0.060983 | 0.006079 | 0.099677 | 0.440000 | 0.188889 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p02_r0p75` | 0.960156 | 0.502222 | 0.060983 | 0.006027 | 0.098824 | 0.422222 | 0.180000 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p001_r0p75` | 0.960277 | 0.502222 | 0.060983 | 0.005906 | 0.096846 | 0.448889 | 0.193333 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0_r0p75` | 0.960427 | 0.502222 | 0.060983 | 0.005756 | 0.094383 | 0.451111 | 0.195556 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p01_r0p25` | 0.960811 | 0.513333 | 0.060983 | 0.005372 | 0.088093 | 0.475555 | 0.206667 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p005_r0p25` | 0.960824 | 0.513333 | 0.060983 | 0.005359 | 0.087880 | 0.477778 | 0.208889 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p02_r0p25` | 0.960839 | 0.513333 | 0.060983 | 0.005344 | 0.087631 | 0.471111 | 0.206667 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0_r0p25` | 0.960850 | 0.513333 | 0.060983 | 0.005333 | 0.087454 | 0.482222 | 0.211111 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p001_r0p25` | 0.960911 | 0.513333 | 0.060983 | 0.005272 | 0.086443 | 0.480000 | 0.211111 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p0025_r0p25` | 0.960911 | 0.513333 | 0.060983 | 0.005272 | 0.086443 | 0.480000 | 0.211111 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0_r0` | 0.961016 | 0.513333 | 0.060983 | 0.005167 | 0.084732 | 0.475556 | 0.208889 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p001_r0` | 0.961016 | 0.513333 | 0.060983 | 0.005167 | 0.084732 | 0.475556 | 0.208889 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p0025_r0` | 0.961016 | 0.513333 | 0.060983 | 0.005167 | 0.084732 | 0.475556 | 0.208889 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p005_r0` | 0.961016 | 0.513333 | 0.060983 | 0.005167 | 0.084732 | 0.473333 | 0.208889 | -0.000180 | `partial_but_insufficient_gap_closure` |

## 해석

- CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.
- 이 실험이 direct candidate-regret gate보다 약하면, 병목은 object-set feature 부재만이 아니라 cross-seed regret target 자체의 안정성 부족으로 해석한다.

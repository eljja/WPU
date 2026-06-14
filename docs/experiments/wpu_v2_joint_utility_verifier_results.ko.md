# Joint Utility Verifier 결과

이 문서는 후보 object set, compact context, sparse/local-dense verification signature를 함께 인코딩하고 candidate regret, uncertainty, no-harm safety를 동시에 예측하는 P1 probe를 요약한다. 이는 post-hoc feature 추가보다 더 직접적인 joint utility/safety head지만, propagation model 자체는 아직 고정되어 있다.

Source CSV: `docs/experiments/wpu_v2_joint_utility_verifier.csv`

최고 closure는 `0.097845` (`K=8`, `joint_utility_uncertainty_regret_gate_r1p5_m0_s0p35`)다. P1 목표 `0.5`를 기준으로 joint utility-verifier deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건의 conservative best는 `0.097845` (`joint_utility_uncertainty_regret_gate_r1p5_m0_s0p35`)다. Train-selected deployed best는 `0.077781` (`K=8`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0_s0p35` | 0.985173 | 0.520000 | 0.033304 | 0.003259 | 0.097845 | 0.451111 | 0.182222 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_utility_uncertainty_regret_gate_r2_m0_s0p35` | 0.964810 | 0.500000 | 0.061485 | 0.001373 | 0.022327 | 0.253333 | 0.097778 | -0.023260 | `partial_but_insufficient_gap_closure` |
| 32 | `joint_utility_uncertainty_regret_gate_r1p5_m0p01_s0p35` | 1.002293 | 0.491111 | 0.035809 | 0.001802 | 0.050318 | 0.411111 | 0.195556 | -0.003019 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0_s0p5` | 0.985235 | 0.517778 | 0.033304 | 0.003197 | 0.095995 | 0.440000 | 0.180000 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0p0025_s0p35` | 0.985301 | 0.522222 | 0.033304 | 0.003131 | 0.094007 | 0.442222 | 0.180000 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0p005_s0p35` | 0.985306 | 0.522222 | 0.033304 | 0.003125 | 0.093839 | 0.437778 | 0.177778 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0p0025_s0p5` | 0.985362 | 0.520000 | 0.033304 | 0.003069 | 0.092158 | 0.431111 | 0.177778 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0p005_s0p5` | 0.985368 | 0.520000 | 0.033304 | 0.003064 | 0.091990 | 0.426666 | 0.175555 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0p02_s0p35` | 0.985427 | 0.520000 | 0.033304 | 0.003004 | 0.090206 | 0.402222 | 0.162222 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0p02_s0p5` | 0.985489 | 0.517778 | 0.033304 | 0.002942 | 0.088350 | 0.391111 | 0.160000 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0p01_s0p35` | 0.985494 | 0.520000 | 0.033304 | 0.002938 | 0.088218 | 0.424444 | 0.175555 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0_s0p8` | 0.985545 | 0.517778 | 0.033304 | 0.002887 | 0.086687 | 0.420000 | 0.173333 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0p01_s0p5` | 0.985555 | 0.517778 | 0.033304 | 0.002876 | 0.086363 | 0.413333 | 0.173333 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0_s0p65` | 0.985634 | 0.517778 | 0.033304 | 0.002797 | 0.083996 | 0.426667 | 0.177778 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_candidate_regret_gate_m0p05_s0p35` | 0.985696 | 0.513333 | 0.033304 | 0.002735 | 0.082129 | 0.735556 | 0.313333 | 0.050741 | `insufficient_no_harm_rejection` |
| 8 | `joint_utility_candidate_regret_gate_m0p02_s0p8` | 0.985744 | 0.511111 | 0.033304 | 0.002688 | 0.080706 | 0.691111 | 0.297778 | 0.050741 | `insufficient_no_harm_rejection` |
| 8 | `joint_utility_candidate_regret_gate_m0p05_s0p5` | 0.985748 | 0.513333 | 0.033304 | 0.002684 | 0.080585 | 0.711111 | 0.306667 | 0.050741 | `insufficient_no_harm_rejection` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0p0025_s0p8` | 0.985751 | 0.517778 | 0.033304 | 0.002680 | 0.080483 | 0.413333 | 0.173333 | 0.050741 | `partial_but_insufficient_gap_closure` |

## 해석

- CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.
- 이 실험이 direct candidate-regret gate보다 약하면, object set, verification signature, utility/safety head를 결합해도 propagation model이 고정된 상태에서는 P1 병목을 해결하지 못한다고 해석한다.

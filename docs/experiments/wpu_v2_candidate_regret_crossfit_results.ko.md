# Cross-Fit Candidate Regret Gate 결과

이 문서는 candidate별 `candidate_loss - learned_loss`를 직접 예측하되, deployment threshold를 in-sample train prediction이 아니라 out-of-source-seed cross-fit prediction으로 선택하는 P1 probe를 요약한다.

Source CSV: `docs/experiments/wpu_v2_candidate_regret_crossfit.csv`

최고 closure는 `0.287268` (`K=16`, `crossfit_regret_gate_m0p0025_r0_d0_v0`)다. P1 목표 `0.5`를 기준으로 candidate-regret deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건의 conservative best는 `0.279738` (`crossfit_regret_gate_m0_r0_d0_v1`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | `crossfit_regret_gate_m0p005_r0_d0_v1` | 0.984248 | 0.517778 | 0.033040 | 0.004183 | 0.126617 | 0.677778 | 0.260000 | 0.082187 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0p0025_r0_d0_v0` | 0.948649 | 0.537778 | 0.061036 | 0.017534 | 0.287268 | 0.928889 | 0.284444 | 0.117792 | `insufficient_no_harm_rejection` |
| 32 | `crossfit_regret_gate_m0p0025_r0_d0_v0p75` | 1.000466 | 0.524445 | 0.035612 | 0.003628 | 0.101881 | 0.753333 | 0.320000 | 0.087270 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0p0025_r0_d0_v0p5` | 0.948649 | 0.537778 | 0.061036 | 0.017534 | 0.287268 | 0.928889 | 0.284444 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0_r0_d0_v0p75` | 0.948958 | 0.537778 | 0.061036 | 0.017225 | 0.282206 | 0.920000 | 0.288889 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0_r0_d0_v0` | 0.949015 | 0.537778 | 0.061036 | 0.017168 | 0.281278 | 0.953333 | 0.300000 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0_r0_d0_v0p5` | 0.949018 | 0.537778 | 0.061036 | 0.017165 | 0.281229 | 0.951111 | 0.300000 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0_r0_d0_v1` | 0.949109 | 0.531111 | 0.061036 | 0.017074 | 0.279738 | 0.824444 | 0.244445 | 0.117792 | `partial_but_insufficient_gap_closure` |
| 16 | `crossfit_regret_gate_m0p0025_r0_d0_v0p75` | 0.949260 | 0.537778 | 0.061036 | 0.016922 | 0.277255 | 0.877778 | 0.273333 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0p005_r0_d0_v0p5` | 0.949452 | 0.537778 | 0.061036 | 0.016730 | 0.274109 | 0.880000 | 0.271111 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0p005_r0_d0_v0` | 0.949467 | 0.537778 | 0.061036 | 0.016715 | 0.273863 | 0.882222 | 0.273333 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0_r0_d0p5_v0p75` | 0.949598 | 0.537778 | 0.061036 | 0.016585 | 0.271720 | 0.913333 | 0.282222 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0_r0_d0p5_v1` | 0.949639 | 0.531111 | 0.061036 | 0.016543 | 0.271045 | 0.837778 | 0.251111 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_selected_candidate_regret_gate` | 0.949643 | 0.531111 | 0.061036 | 0.016540 | 0.270989 | 0.782222 | 0.222222 | 0.117792 | `partial_but_insufficient_gap_closure` |
| 16 | `crossfit_regret_gate_m0_r0_d1_v1` | 0.949736 | 0.531111 | 0.061036 | 0.016447 | 0.269462 | 0.840000 | 0.251111 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0p005_r0_d0_v0p75` | 0.949784 | 0.531111 | 0.061036 | 0.016398 | 0.268669 | 0.833333 | 0.257778 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0_r0_d0p5_v0` | 0.949786 | 0.537778 | 0.061036 | 0.016397 | 0.268650 | 0.922222 | 0.284444 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0_r0_d0p5_v0p5` | 0.949786 | 0.537778 | 0.061036 | 0.016397 | 0.268650 | 0.922222 | 0.284444 | 0.117792 | `insufficient_no_harm_rejection` |

## 해석

- CSV에는 모든 reject-margin, risk-penalty, disagreement-penalty, vote-threshold deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 이 결과는 개선이 아니라 negative result다. Cross-fit은 train-selection optimism을 줄이지만 direct candidate-regret gate보다 closure를 낮춘다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.

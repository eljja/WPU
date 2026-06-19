# Joint Selector-Propagator 결과

이 문서는 후보 working set selector와 WPU sparse propagation branch loss를 같은 학습 그래프에서 최적화한 P1 probe를 요약한다. 기존 post-hoc selector, 고정 propagation verifier, branch-logit adapter보다 candidate choice와 propagation dynamics를 더 직접적으로 결합하지만, hard object retrieval 자체는 아직 완전한 미분 가능 end-to-end 생성기가 아니다.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator.csv`

최고 closure는 `0.877854` (`K=8`, `confidence_selected_joint_selector_propagator`)다. P1 목표 `0.5`를 기준으로 joint selector-propagator deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건의 conservative best는 `0.877854` (`confidence_selected_joint_selector_propagator`)다. Train-selected deployed best는 `0.877854` (`K=8`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | `confidence_selected_joint_selector_propagator` | 0.893248 | 0.608889 | 0.079168 | 0.069498 | 0.877854 | 1.000000 | 0.075555 | 0.000000 | `passes_current_p1_threshold` |
| 16 | `joint_selector_propagator` | 0.985026 | 0.515555 | 0.063807 | 0.010953 | 0.171659 | 1.000000 | 0.444444 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `joint_selector_propagator` | 0.972520 | 0.493333 | 0.043295 | 0.002611 | 0.060311 | 1.000000 | 0.511111 | 0.000000 | `insufficient_no_harm_rejection` |
| 8 | `joint_selector_propagator` | 0.893248 | 0.608889 | 0.079168 | 0.069498 | 0.877854 | 1.000000 | 0.075555 | 0.000000 | `passes_current_p1_threshold` |
| 8 | `train_selected_joint_selector_propagator` | 0.893248 | 0.608889 | 0.079168 | 0.069498 | 0.877854 | 1.000000 | 0.075555 | 0.000000 | `passes_current_p1_threshold` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.972537 | 0.520000 | 0.043295 | 0.002594 | 0.059919 | 0.448889 | 0.200000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `train_selected_joint_selector_propagator` | 0.972537 | 0.520000 | 0.043295 | 0.002594 | 0.059919 | 0.448889 | 0.200000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `confidence_selected_joint_selector_propagator` | 0.997686 | 0.511111 | 0.063807 | -0.001707 | -0.026759 | 0.484444 | 0.222222 | 0.000000 | `harmful_transfer` |
| 16 | `train_selected_joint_selector_propagator` | 0.997686 | 0.511111 | 0.063807 | -0.001707 | -0.026759 | 0.484444 | 0.222222 | 0.000000 | `harmful_transfer` |

## 해석

- CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.
- 이 실험이 direct candidate-regret gate보다 약하면, selector와 propagation branch loss를 같은 루프로 묶는 것만으로는 부족하며 object retrieval, candidate generation, transition dynamics까지 더 깊게 공동학습해야 한다고 해석한다.

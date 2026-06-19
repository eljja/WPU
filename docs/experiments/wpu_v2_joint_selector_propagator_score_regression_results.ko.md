# Score-Regression Joint Selector-Propagator 결과

이 문서는 joint selector-propagator selector score를 후보별 propagation loss의 상대 utility에 직접 맞추는 P1 ablation을 요약한다. 목적은 structured candidate가 만든 headroom을 argmax/ranking objective가 충분히 활용하지 못하는지 검사하는 것이다.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_score_regression.csv`

최고 closure는 `0.186333` (`K=16`, `joint_selector_propagator`)다. P1 목표 `0.5`를 기준으로 joint selector-propagator deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건의 conservative best는 `0.095543` (`confidence_selected_joint_selector_propagator`)다. Train-selected deployed best는 `0.110287` (`K=16`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `joint_selector_propagator` | 0.999804 | 0.502222 | 0.051792 | 0.009651 | 0.186333 | 1.000000 | 0.435556 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.992320 | 0.511111 | 0.045062 | 0.004305 | 0.095543 | 0.484444 | 0.186667 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `confidence_selected_joint_selector_propagator` | 1.003742 | 0.502222 | 0.051792 | 0.005712 | 0.110287 | 0.871111 | 0.382222 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `train_selected_joint_selector_propagator` | 1.003742 | 0.502222 | 0.051792 | 0.005712 | 0.110287 | 0.871111 | 0.382222 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `train_selected_joint_selector_propagator` | 0.992320 | 0.511111 | 0.045062 | 0.004305 | 0.095543 | 0.484444 | 0.186667 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `joint_selector_propagator` | 0.993329 | 0.511111 | 0.045062 | 0.003296 | 0.073152 | 1.000000 | 0.511111 | 0.000000 | `insufficient_no_harm_rejection` |

## 해석

- CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.
- 이 실험이 direct candidate-regret gate보다 약하면, selector와 propagation branch loss를 같은 루프로 묶는 것만으로는 부족하며 object retrieval, candidate generation, transition dynamics까지 더 깊게 공동학습해야 한다고 해석한다.
- Score regression이 closure를 올리지 못하거나 harmful accept를 키우면, 후보 loss magnitude를 맞추는 것만으로는 부족하며 검증 가능한 safe generation이 필요하다고 해석한다.

# Learned-Safe-Candidate Joint Selector-Propagator 결과

이 문서는 train fold에서 학습한 object-level safe candidate generator를 joint selector-propagator 후보 pool에 추가한 P1 ablation을 요약한다. Generator는 interaction, proximity, density, axis teacher를 다양하게 모방해 rejection 이전에 후보 pool 자체가 개선되는지 검사한다. 이는 아직 full differentiable generator가 아니라 teacher-supervised candidate-generation diagnostic이다.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_learned_safe_candidates.csv`

최고 closure는 `0.271116` (`K=16`, `joint_selector_propagator`)다. P1 목표 `0.5`를 기준으로 joint selector-propagator deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건의 conservative best는 `0.246071` (`confidence_selected_joint_selector_propagator`)다. Train-selected deployed best는 `0.246071` (`K=16`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `joint_selector_propagator` | 0.998987 | 0.560000 | 0.089258 | 0.024199 | 0.271116 | 1.000000 | 0.342222 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.993054 | 0.475556 | 0.039324 | 0.005639 | 0.143398 | 0.377778 | 0.133333 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `confidence_selected_joint_selector_propagator` | 1.001222 | 0.542222 | 0.089258 | 0.021964 | 0.246071 | 0.666667 | 0.222222 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_joint_selector_propagator` | 1.001222 | 0.542222 | 0.089258 | 0.021964 | 0.246071 | 0.666667 | 0.222222 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `train_selected_joint_selector_propagator` | 0.993054 | 0.475556 | 0.039324 | 0.005639 | 0.143398 | 0.377778 | 0.133333 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `joint_selector_propagator` | 1.000165 | 0.475556 | 0.039324 | -0.001473 | -0.037453 | 1.000000 | 0.537778 | 0.000000 | `harmful_transfer` |

## 해석

- CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.
- 이 실험이 direct candidate-regret gate보다 약하면, selector와 propagation branch loss를 같은 루프로 묶는 것만으로는 부족하며 object retrieval, candidate generation, transition dynamics까지 더 깊게 공동학습해야 한다고 해석한다.
- Learned-safe candidate가 closure를 올리지 못하면, teacher-supervised object scorer만으로는 충분하지 않고 candidate generation을 propagation loss 및 no-harm objective와 직접 연결해야 한다.

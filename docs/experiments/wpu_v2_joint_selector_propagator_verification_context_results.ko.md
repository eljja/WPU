# Verification-Context Joint Selector-Propagator 결과

이 문서는 joint selector-propagator selector 입력에 label-free propagation verification signature를 추가한 P1 ablation을 요약한다. Signature는 후보별 branch confidence, entropy, logit margin, delta norm처럼 정답 label 없이 계산되는 전파 결과 신호다. 목적은 selector가 정적 candidate description만이 아니라 실제 propagator behavior를 보고 harmful candidate를 거부할 수 있는지 검사하는 것이다.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_verification_context.csv`

최고 closure는 `0.409420` (`K=16`, `joint_selector_propagator`)다. P1 목표 `0.5`를 기준으로 joint selector-propagator deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건의 conservative best는 `0.269216` (`confidence_selected_joint_selector_propagator`)다. Train-selected deployed best는 `0.269216` (`K=16`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `joint_selector_propagator` | 0.962804 | 0.537778 | 0.081009 | 0.033167 | 0.409420 | 1.000000 | 0.342222 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.991791 | 0.502222 | 0.038259 | 0.005868 | 0.153386 | 0.306667 | 0.111111 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `confidence_selected_joint_selector_propagator` | 0.974162 | 0.524445 | 0.081009 | 0.021809 | 0.269216 | 0.440000 | 0.115555 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_joint_selector_propagator` | 0.974162 | 0.524445 | 0.081009 | 0.021809 | 0.269216 | 0.440000 | 0.115555 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `train_selected_joint_selector_propagator` | 0.991791 | 0.502222 | 0.038259 | 0.005868 | 0.153386 | 0.306667 | 0.111111 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `joint_selector_propagator` | 0.996778 | 0.506667 | 0.038259 | 0.000882 | 0.023059 | 1.000000 | 0.533333 | 0.000000 | `insufficient_no_harm_rejection` |

## 해석

- CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.
- 이 실험이 direct candidate-regret gate보다 약하면, selector와 propagation branch loss를 같은 루프로 묶는 것만으로는 부족하며 object retrieval, candidate generation, transition dynamics까지 더 깊게 공동학습해야 한다고 해석한다.
- Verification context가 closure와 harmful accept를 동시에 개선하지 못하면, label-free 전파 signature는 유용한 관측 신호일 수 있지만 selector 입력 추가만으로는 충분하지 않고 verification objective 자체를 no-harm candidate generation과 함께 학습해야 한다.

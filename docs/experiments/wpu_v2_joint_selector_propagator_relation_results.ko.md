# Joint Selector-Propagator 결과

이 문서는 joint selector-propagator의 propagation model을 relation-conditioned WPU로 바꾼 P1 ablation을 요약한다. 목적은 K=16/32 병목이 transition dynamics 부족인지 검사하는 것이다. Closure가 오르지만 harmful accept가 남으면 relation-aware dynamics는 방향이지만 no-harm candidate selection이 아직 부족하다고 해석한다.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_relation.csv`

최고 closure는 `0.266805` (`K=32`, `confidence_selected_joint_selector_propagator`)다. P1 목표 `0.5`를 기준으로 joint selector-propagator deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건의 conservative best는 `0.064077` (`confidence_selected_joint_selector_propagator`)다. Train-selected deployed best는 `0.266805` (`K=32`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `confidence_selected_joint_selector_propagator` | 1.002028 | 0.493333 | 0.037102 | 0.002377 | 0.064077 | 0.315556 | 0.084445 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.967449 | 0.484444 | 0.016754 | 0.004470 | 0.266805 | 0.906667 | 0.333333 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `train_selected_joint_selector_propagator` | 0.967449 | 0.484444 | 0.016754 | 0.004470 | 0.266805 | 0.906667 | 0.333333 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `joint_selector_propagator` | 0.968075 | 0.484444 | 0.016754 | 0.003844 | 0.229429 | 1.000000 | 0.395555 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `train_selected_joint_selector_propagator` | 1.002028 | 0.493333 | 0.037102 | 0.002377 | 0.064077 | 0.315556 | 0.084445 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_selector_propagator` | 1.003795 | 0.497778 | 0.037102 | 0.000611 | 0.016463 | 1.000000 | 0.422222 | 0.000000 | `insufficient_no_harm_rejection` |

## 해석

- CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.
- 이 실험이 direct candidate-regret gate보다 약하면, selector와 propagation branch loss를 같은 루프로 묶는 것만으로는 부족하며 object retrieval, candidate generation, transition dynamics까지 더 깊게 공동학습해야 한다고 해석한다.
- Relation-conditioned propagation이 closure를 올리지만 harmful accept를 낮추지 못하면, 다음 단계는 relation-aware transition과 no-harm candidate selection의 공동학습이다.

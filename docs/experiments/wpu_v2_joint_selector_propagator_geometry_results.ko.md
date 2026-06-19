# Joint Selector-Propagator 결과

이 문서는 joint selector-propagator에 후보별 geometry/force context를 추가한 P1 ablation을 요약한다. 목적은 K=16/32 실패가 단순 descriptor 부족인지 검사하는 것이다. 결과가 약하면 larger-K 병목은 feature concatenation이 아니라 retrieval, generation, propagation dynamics의 더 깊은 공동학습 문제로 해석한다.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_geometry.csv`

최고 closure는 `0.087030` (`K=16`, `joint_selector_propagator`)다. P1 목표 `0.5`를 기준으로 joint selector-propagator deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건의 conservative best는 `0.044841` (`confidence_selected_joint_selector_propagator`)다. Train-selected deployed best는 `0.044841` (`K=16`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `joint_selector_propagator` | 0.994853 | 0.511111 | 0.053853 | 0.004687 | 0.087030 | 1.000000 | 0.408889 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `joint_selector_propagator` | 0.973978 | 0.488889 | 0.038579 | 0.000598 | 0.015495 | 1.000000 | 0.511111 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `confidence_selected_joint_selector_propagator` | 0.997125 | 0.511111 | 0.053853 | 0.002415 | 0.044841 | 0.213333 | 0.080000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_joint_selector_propagator` | 0.997125 | 0.511111 | 0.053853 | 0.002415 | 0.044841 | 0.213333 | 0.080000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.974169 | 0.493333 | 0.038579 | 0.000407 | 0.010550 | 0.280000 | 0.124444 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `train_selected_joint_selector_propagator` | 0.974169 | 0.493333 | 0.038579 | 0.000407 | 0.010550 | 0.280000 | 0.124444 | 0.000000 | `partial_but_insufficient_gap_closure` |

## 해석

- CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.
- 이 실험이 direct candidate-regret gate보다 약하면, selector와 propagation branch loss를 같은 루프로 묶는 것만으로는 부족하며 object retrieval, candidate generation, transition dynamics까지 더 깊게 공동학습해야 한다고 해석한다.
- Geometry/force descriptor 추가가 K=16/32를 개선하지 못하면, 단순 feature 추가가 아니라 candidate 생성 및 relation-conditioned transition learning이 다음 병목이다.

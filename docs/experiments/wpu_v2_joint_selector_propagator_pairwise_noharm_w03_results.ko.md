# Pairwise No-Harm Joint Selector-Propagator 결과

이 문서는 joint selector-propagator에 pairwise no-harm score margin을 추가한 P1 ablation을 요약한다. 목적은 K=16/32에서 높은 harmful accept가 단순 confidence threshold 문제가 아니라 selector score 자체의 baseline-safe ordering 문제인지 검사하는 것이다.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_pairwise_noharm_w03.csv`

최고 closure는 `0.200230` (`K=32`, `joint_selector_propagator`)다. P1 목표 `0.5`를 기준으로 joint selector-propagator deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건의 conservative best는 `0.200230` (`joint_selector_propagator`)다. Train-selected deployed best는 `0.192596` (`K=32`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `joint_selector_propagator` | 0.990521 | 0.511111 | 0.050670 | 0.008874 | 0.175125 | 0.764444 | 0.195555 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `joint_selector_propagator` | 0.966322 | 0.497778 | 0.020697 | 0.004144 | 0.200230 | 0.226667 | 0.053333 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.966480 | 0.497778 | 0.020697 | 0.003986 | 0.192596 | 0.160000 | 0.026667 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `train_selected_joint_selector_propagator` | 0.966480 | 0.497778 | 0.020697 | 0.003986 | 0.192596 | 0.160000 | 0.026667 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `confidence_selected_joint_selector_propagator` | 0.992009 | 0.506667 | 0.050670 | 0.007385 | 0.145747 | 0.622222 | 0.155556 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_joint_selector_propagator` | 0.992009 | 0.506667 | 0.050670 | 0.007385 | 0.145747 | 0.622222 | 0.155556 | 0.000000 | `partial_but_insufficient_gap_closure` |

## 해석

- CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.
- 이 실험이 direct candidate-regret gate보다 약하면, selector와 propagation branch loss를 같은 루프로 묶는 것만으로는 부족하며 object retrieval, candidate generation, transition dynamics까지 더 깊게 공동학습해야 한다고 해석한다.
- Pairwise no-harm margin이 harmful accept를 낮추지만 closure도 크게 낮추면, P1 병목은 안전 제약 부재만이 아니라 안전한 후보 생성과 relation-aware propagation 품질의 결합 문제다.

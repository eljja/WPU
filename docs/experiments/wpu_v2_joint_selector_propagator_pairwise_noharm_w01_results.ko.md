# Pairwise No-Harm Joint Selector-Propagator 결과

이 문서는 joint selector-propagator에 pairwise no-harm score margin을 추가한 P1 ablation을 요약한다. 목적은 K=16/32에서 높은 harmful accept가 단순 confidence threshold 문제가 아니라 selector score 자체의 baseline-safe ordering 문제인지 검사하는 것이다.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_pairwise_noharm_w01.csv`

최고 closure는 `0.239301` (`K=32`, `joint_selector_propagator`)다. P1 목표 `0.5`를 기준으로 joint selector-propagator deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건의 conservative best는 `0.084224` (`confidence_selected_joint_selector_propagator`)다. Train-selected deployed best는 `0.191273` (`K=32`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `joint_selector_propagator` | 0.991877 | 0.480000 | 0.052722 | 0.006145 | 0.116552 | 1.000000 | 0.333333 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `joint_selector_propagator` | 0.966575 | 0.502222 | 0.019081 | 0.004566 | 0.239301 | 0.933333 | 0.417778 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.967491 | 0.493333 | 0.019081 | 0.003650 | 0.191273 | 0.737778 | 0.328889 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `train_selected_joint_selector_propagator` | 0.967491 | 0.493333 | 0.019081 | 0.003650 | 0.191273 | 0.737778 | 0.328889 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `confidence_selected_joint_selector_propagator` | 0.993582 | 0.484444 | 0.052722 | 0.004440 | 0.084224 | 0.613333 | 0.173333 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_joint_selector_propagator` | 0.993582 | 0.484444 | 0.052722 | 0.004440 | 0.084224 | 0.613333 | 0.173333 | 0.000000 | `partial_but_insufficient_gap_closure` |

## 해석

- CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.
- 이 실험이 direct candidate-regret gate보다 약하면, selector와 propagation branch loss를 같은 루프로 묶는 것만으로는 부족하며 object retrieval, candidate generation, transition dynamics까지 더 깊게 공동학습해야 한다고 해석한다.
- Pairwise no-harm margin이 harmful accept를 낮추지만 closure도 크게 낮추면, P1 병목은 안전 제약 부재만이 아니라 안전한 후보 생성과 relation-aware propagation 품질의 결합 문제다.

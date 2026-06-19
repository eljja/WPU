# Structured-Candidate Joint Selector-Propagator 결과

이 문서는 joint selector-propagator에 deterministic structured candidate를 추가한 P1 ablation을 요약한다. 목적은 K=16/32 병목이 나쁜 후보를 거부하는 문제가 아니라 안전하고 좋은 후보 자체가 부족한 문제인지 검사하는 것이다.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_structured_candidates.csv`

최고 closure는 `0.066958` (`K=32`, `confidence_selected_joint_selector_propagator`)다. P1 목표 `0.5`를 기준으로 joint selector-propagator deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건의 conservative best는 `0.066958` (`confidence_selected_joint_selector_propagator`)다. Train-selected deployed best는 `0.066958` (`K=32`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `confidence_selected_joint_selector_propagator` | 1.004592 | 0.493333 | 0.046598 | 0.002108 | 0.045246 | 0.084445 | 0.008889 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.995727 | 0.502222 | 0.040222 | 0.002693 | 0.066958 | 0.275556 | 0.102222 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `train_selected_joint_selector_propagator` | 0.995727 | 0.502222 | 0.040222 | 0.002693 | 0.066958 | 0.275556 | 0.102222 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `joint_selector_propagator` | 0.996112 | 0.502222 | 0.040222 | 0.002308 | 0.057382 | 0.528889 | 0.235556 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_joint_selector_propagator` | 1.004592 | 0.493333 | 0.046598 | 0.002108 | 0.045246 | 0.084445 | 0.008889 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_selector_propagator` | 1.006789 | 0.493333 | 0.046598 | -0.000089 | -0.001901 | 0.106667 | 0.017778 | 0.000000 | `harmful_transfer` |

## 해석

- CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.
- 이 실험이 direct candidate-regret gate보다 약하면, selector와 propagation branch loss를 같은 루프로 묶는 것만으로는 부족하며 object retrieval, candidate generation, transition dynamics까지 더 깊게 공동학습해야 한다고 해석한다.
- Structured candidate 추가가 oracle 또는 deployed closure를 올리지 못하면, 손으로 만든 다양성보다 학습된 candidate generation과 propagation-aware verification이 필요하다고 해석한다.

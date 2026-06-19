# Structured-Candidate Joint Selector-Propagator 결과

이 문서는 joint selector-propagator에 deterministic structured candidate를 추가한 P1 ablation을 요약한다. 목적은 K=16/32 병목이 나쁜 후보를 거부하는 문제가 아니라 안전하고 좋은 후보 자체가 부족한 문제인지 검사하는 것이다.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_structured_candidates_no_margin.csv`

최고 closure는 `0.327084` (`K=16`, `joint_selector_propagator`)다. P1 목표 `0.5`를 기준으로 joint selector-propagator deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건의 conservative best는 `0.241624` (`confidence_selected_joint_selector_propagator`)다. Train-selected deployed best는 `0.241624` (`K=16`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `joint_selector_propagator` | 0.967612 | 0.551111 | 0.079496 | 0.026002 | 0.327084 | 1.000000 | 0.355555 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.990694 | 0.502222 | 0.030397 | 0.003785 | 0.124512 | 0.235556 | 0.080000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `confidence_selected_joint_selector_propagator` | 0.974406 | 0.542222 | 0.079496 | 0.019208 | 0.241624 | 0.653333 | 0.200000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_joint_selector_propagator` | 0.974406 | 0.542222 | 0.079496 | 0.019208 | 0.241624 | 0.653333 | 0.200000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `train_selected_joint_selector_propagator` | 0.990694 | 0.502222 | 0.030397 | 0.003785 | 0.124512 | 0.235556 | 0.080000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `joint_selector_propagator` | 0.993703 | 0.497778 | 0.030397 | 0.000775 | 0.025496 | 1.000000 | 0.488889 | 0.000000 | `insufficient_no_harm_rejection` |

## 해석

- CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.
- 이 실험이 direct candidate-regret gate보다 약하면, selector와 propagation branch loss를 같은 루프로 묶는 것만으로는 부족하며 object retrieval, candidate generation, transition dynamics까지 더 깊게 공동학습해야 한다고 해석한다.
- Structured candidate 추가가 oracle 또는 deployed closure를 올리지 못하면, 손으로 만든 다양성보다 학습된 candidate generation과 propagation-aware verification이 필요하다고 해석한다.

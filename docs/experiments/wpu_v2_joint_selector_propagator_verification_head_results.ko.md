# Verification-Head Joint Selector-Propagator 결과

이 문서는 label-free propagation signature 위에 explicit harmful-candidate verification head를 추가한 P1 ablation을 요약한다. Head는 learned baseline보다 나쁜 후보를 auxiliary target으로 학습하고 deployment score에서 unsafe probability를 감점한다. 목적은 verification을 단순 입력 feature가 아니라 no-harm objective로 분리했을 때 larger-K safe deployment가 개선되는지 검사하는 것이다.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_verification_head.csv`

최고 closure는 `0.345395` (`K=16`, `joint_selector_propagator`)다. P1 목표 `0.5`를 기준으로 joint selector-propagator deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건의 conservative best는 `0.193197` (`confidence_selected_joint_selector_propagator`)다. Train-selected deployed best는 `0.193197` (`K=16`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `joint_selector_propagator` | 0.970314 | 0.533333 | 0.080096 | 0.027665 | 0.345395 | 1.000000 | 0.391111 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.993600 | 0.506667 | 0.036576 | 0.002216 | 0.060597 | 0.222222 | 0.088889 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `confidence_selected_joint_selector_propagator` | 0.982505 | 0.511111 | 0.080096 | 0.015474 | 0.193197 | 0.355556 | 0.102222 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_joint_selector_propagator` | 0.982505 | 0.511111 | 0.080096 | 0.015474 | 0.193197 | 0.355556 | 0.102222 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `train_selected_joint_selector_propagator` | 0.993600 | 0.506667 | 0.036576 | 0.002216 | 0.060597 | 0.222222 | 0.088889 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `joint_selector_propagator` | 0.994742 | 0.511111 | 0.036576 | 0.001075 | 0.029385 | 1.000000 | 0.537778 | 0.000000 | `insufficient_no_harm_rejection` |

## 해석

- CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.
- 이 실험이 direct candidate-regret gate보다 약하면, selector와 propagation branch loss를 같은 루프로 묶는 것만으로는 부족하며 object retrieval, candidate generation, transition dynamics까지 더 깊게 공동학습해야 한다고 해석한다.
- Verification head가 harmful accept를 낮추면서 closure를 유지하지 못하면, explicit no-harm 판정도 후보 생성과 propagation dynamics 품질이 부족한 상태에서는 충분하지 않다고 해석한다.

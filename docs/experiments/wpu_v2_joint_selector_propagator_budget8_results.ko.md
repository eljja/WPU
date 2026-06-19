# Joint Selector-Propagator 결과

이 문서는 joint selector-propagator의 working-set budget을 확대한 P1 ablation을 요약한다. 목적은 K=16/32 실패가 budget=4의 과도한 causal-state 절단 때문인지 검사하는 것이다.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_budget8.csv`

최고 closure는 `0.109276` (`K=32`, `joint_selector_propagator`)다. P1 목표 `0.5`를 기준으로 joint selector-propagator deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건의 conservative best는 `0.092290` (`confidence_selected_joint_selector_propagator`)다. Train-selected deployed best는 `0.092290` (`K=32`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `joint_selector_propagator` | 1.005292 | 0.506667 | 0.046077 | 0.003394 | 0.073651 | 1.000000 | 0.417778 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `joint_selector_propagator` | 0.970192 | 0.537778 | 0.026716 | 0.002919 | 0.109276 | 1.000000 | 0.360000 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `confidence_selected_joint_selector_propagator` | 0.970646 | 0.542222 | 0.026716 | 0.002466 | 0.092290 | 0.582222 | 0.160000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 32 | `train_selected_joint_selector_propagator` | 0.970646 | 0.542222 | 0.026716 | 0.002466 | 0.092290 | 0.582222 | 0.160000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `confidence_selected_joint_selector_propagator` | 1.005881 | 0.515556 | 0.046077 | 0.002804 | 0.060864 | 0.657778 | 0.240000 | 0.000000 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_joint_selector_propagator` | 1.005881 | 0.515556 | 0.046077 | 0.002804 | 0.060864 | 0.657778 | 0.240000 | 0.000000 | `partial_but_insufficient_gap_closure` |

## 해석

- CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.
- 이 실험이 direct candidate-regret gate보다 약하면, selector와 propagation branch loss를 같은 루프로 묶는 것만으로는 부족하며 object retrieval, candidate generation, transition dynamics까지 더 깊게 공동학습해야 한다고 해석한다.
- Budget 확장만으로 closure가 크게 오르지 않으면, larger-K 실패는 working-set 크기만의 문제가 아니라 후보 품질과 transition dynamics의 문제다.

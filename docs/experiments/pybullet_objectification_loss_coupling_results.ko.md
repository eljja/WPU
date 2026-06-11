# PyBullet Objectification-Loss Coupling Audit

이 파생 감사는 PyBullet objectification quality metric과 downstream propagation degradation을 연결한다. 목표는 objectification score가 존재한다는 사실을 넘어서, 어떤 component가 branch accuracy와 MSE 변화를 설명하는지 확인하는 것이다.

Source CSV: `docs/experiments/pybullet_objectification_loss_coupling.csv`

## Summary

- 가장 큰 평균 accuracy drop은 `wpu-cws-indexed-local-dense` / `combined`에서 `0.027778`이다.
- 가장 큰 평균 MSE increase는 `wpu-cws-indexed-sparse` / `drop_relations_heavy`에서 `0.087356`이다.
- MSE degradation과 가장 강하게 연결된 component deficit은 `selected_k_mean`이며 |r|=`0.481851`이다.
- Accuracy degradation과 가장 강하게 연결된 component deficit은 `relation_confidence`이며 |r|=`0.352431`이다.

## Interpretation

현재 stress task는 branch accuracy 변화가 작아서 accuracy coupling은 약하다. 반면 MSE와 selected-K/frontier 관련 component는 objectification failure가 propagation 손실로 이어지는 경로를 더 잘 보여준다. 따라서 P7은 부분적으로 개선됐지만, 더 강한 closed-loop/horizon corruption 실험이 필요하다.

## Rows

| row_type | model | corruption | predictor | n | acc_drop | mse_increase | r_acc | r_mse |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| corruption_summary | graph-transformer | clean |  | 2 | 0.000000 | 0.000000 |  |  |
| corruption_summary | graph-transformer | combined |  | 2 | 0.000000 | 0.002870 |  |  |
| corruption_summary | graph-transformer | drop_relations_heavy |  | 2 | 0.000000 | -0.000929 |  |  |
| corruption_summary | graph-transformer | drop_relations_light |  | 2 | 0.000000 | -0.000414 |  |  |
| corruption_summary | graph-transformer | identity_swap |  | 2 | 0.000000 | 0.000135 |  |  |
| corruption_summary | graph-transformer | low_confidence |  | 2 | 0.013889 | 0.009951 |  |  |
| corruption_summary | graph-transformer | position_noise |  | 2 | 0.000000 | 0.000138 |  |  |
| corruption_summary | wpu-cws-indexed-local-dense | clean |  | 2 | 0.000000 | 0.000000 |  |  |
| corruption_summary | wpu-cws-indexed-local-dense | combined |  | 2 | 0.027778 | -0.011876 |  |  |
| corruption_summary | wpu-cws-indexed-local-dense | drop_relations_heavy |  | 2 | 0.027778 | 0.027078 |  |  |
| corruption_summary | wpu-cws-indexed-local-dense | drop_relations_light |  | 2 | 0.000000 | -0.025027 |  |  |
| corruption_summary | wpu-cws-indexed-local-dense | identity_swap |  | 2 | 0.000000 | 0.000000 |  |  |
| corruption_summary | wpu-cws-indexed-local-dense | low_confidence |  | 2 | 0.027778 | 0.001755 |  |  |
| corruption_summary | wpu-cws-indexed-local-dense | position_noise |  | 2 | 0.000000 | -0.000051 |  |  |
| corruption_summary | wpu-cws-indexed-sparse | clean |  | 2 | 0.000000 | 0.000000 |  |  |
| corruption_summary | wpu-cws-indexed-sparse | combined |  | 2 | -0.013889 | 0.026208 |  |  |
| corruption_summary | wpu-cws-indexed-sparse | drop_relations_heavy |  | 2 | 0.000000 | 0.087356 |  |  |
| corruption_summary | wpu-cws-indexed-sparse | drop_relations_light |  | 2 | 0.000000 | 0.008038 |  |  |
| corruption_summary | wpu-cws-indexed-sparse | identity_swap |  | 2 | 0.000000 | 0.000000 |  |  |
| corruption_summary | wpu-cws-indexed-sparse | low_confidence |  | 2 | 0.027778 | -0.001589 |  |  |
| corruption_summary | wpu-cws-indexed-sparse | position_noise |  | 2 | -0.013889 | 0.000109 |  |  |
| predictor_correlation | all | non_clean | selected_k_mean | 36 |  |  | 0.119908 | 0.481851 |
| predictor_correlation | all | non_clean | quality_frontier_recall | 30 |  |  | 0.033095 | 0.392100 |
| predictor_correlation | all | non_clean | quality_frontier_completeness_report | 30 |  |  | 0.033095 | 0.392100 |
| predictor_correlation | all | non_clean | frontier_causal_recall_mean | 36 |  |  | 0.024152 | 0.374627 |
| predictor_correlation | all | non_clean | quality_semantic_identity_consistency | 30 |  |  | -0.307929 | -0.212320 |
| predictor_correlation | all | non_clean | quality_semantic_identity_consistency_report | 30 |  |  | -0.307929 | -0.212320 |
| predictor_correlation | all | non_clean | quality_relation_recall | 30 |  |  | -0.014439 | 0.141054 |
| predictor_correlation | all | non_clean | quality_identity_recall | 30 |  |  | -0.041487 | -0.058326 |
| predictor_correlation | all | non_clean | object_confidence | 36 |  |  | 0.346120 | -0.057876 |
| predictor_correlation | all | non_clean | objectification_score | 36 |  |  | 0.348867 | -0.049799 |
| predictor_correlation | all | non_clean | relation_confidence | 36 |  |  | 0.352431 | -0.038696 |
| predictor_correlation | all | non_clean | identity_coverage | 36 |  |  |  |  |
| predictor_correlation | all | non_clean | relation_validity | 36 |  |  |  |  |
| predictor_correlation | all | non_clean | quality_relation_precision | 30 |  |  |  |  |

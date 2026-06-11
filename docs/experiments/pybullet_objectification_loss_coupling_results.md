# PyBullet Objectification-Loss Coupling Audit

This derived audit links PyBullet objectification quality metrics to downstream propagation degradation. The goal is to move beyond reporting an objectification score and identify which components explain branch-accuracy and MSE changes.

Source CSV: `docs/experiments/pybullet_objectification_loss_coupling.csv`

## Summary

- Largest mean accuracy drop: `wpu-cws-indexed-local-dense` / `combined` at `0.027778`.
- Largest mean MSE increase: `wpu-cws-indexed-sparse` / `drop_relations_heavy` at `0.087356`.
- Strongest component deficit for MSE degradation: `selected_k_mean` with |r|=`0.481851`.
- Strongest component deficit for accuracy degradation: `relation_confidence` with |r|=`0.352431`.

## Interpretation

The current stress task has small branch-accuracy movement, so accuracy coupling is weak. MSE and selected-K/frontier components give a clearer path from objectification failure to propagation degradation. P7 is therefore improved but not solved; a stronger closed-loop or multi-horizon corruption experiment is still needed.

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

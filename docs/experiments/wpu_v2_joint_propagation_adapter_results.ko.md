# Joint Propagation Adapter 결과

이 문서는 후보별 sparse/local-dense verification feature로 branch-logit propagation adapter를 학습한 뒤, adapted propagation loss 위에서 candidate-regret/no-harm deployment를 평가한 P1 probe를 요약한다. 이는 full retriever-propagator end-to-end training은 아니지만, selector와 propagation output correction을 같은 supervision 아래 묶는 중간 단계다.

Source CSV: `docs/experiments/wpu_v2_joint_propagation_adapter.csv`

최고 closure는 `0.092185` (`K=8`, `joint_adapter_uncertainty_regret_gate_r0p75_m0p05`)다. P1 목표 `0.5`를 기준으로 joint propagation-adapter deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건의 conservative best는 `0.092185` (`joint_adapter_uncertainty_regret_gate_r0p75_m0p05`)다. Train-selected deployed best는 `0.069911` (`K=32`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | `joint_adapter_uncertainty_regret_gate_r0p75_m0p05` | 1.850384 | 0.671111 | 1.723180 | 0.158852 | 0.092185 | 0.197778 | 0.035555 | 0.134374 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_adapter_uncertainty_regret_gate_r2_m0` | 3.012316 | 0.455555 | 2.564796 | 0.025470 | 0.009931 | 0.004444 | 0.000000 | -0.051454 | `partial_but_insufficient_gap_closure` |
| 32 | `train_selected_joint_propagation_adapter` | 3.861638 | 0.508889 | 3.288203 | 0.229881 | 0.069911 | 0.662222 | 0.248889 | 0.061837 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_adapter_uncertainty_regret_gate` | 1.855904 | 0.704444 | 1.723180 | 0.153331 | 0.088981 | 0.471111 | 0.120000 | 0.134374 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_adapter_uncertainty_regret_gate_r0p5_m0p0025` | 1.858923 | 0.702222 | 1.723180 | 0.150312 | 0.087230 | 0.462222 | 0.120000 | 0.134374 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_adapter_uncertainty_regret_gate_r0p5_m0p05` | 1.866634 | 0.693333 | 1.723180 | 0.142601 | 0.082755 | 0.380000 | 0.095556 | 0.134374 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_adapter_uncertainty_regret_gate_r1_m0` | 1.895837 | 0.660000 | 1.723180 | 0.113398 | 0.065807 | 0.166667 | 0.031111 | 0.134374 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_adapter_uncertainty_regret_gate_r1_m0p0025` | 1.895837 | 0.660000 | 1.723180 | 0.113398 | 0.065807 | 0.166667 | 0.031111 | 0.134374 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_adapter_uncertainty_regret_gate_r1_m0p005` | 1.895837 | 0.660000 | 1.723180 | 0.113398 | 0.065807 | 0.164444 | 0.031111 | 0.134374 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_adapter_uncertainty_regret_gate_r1_m0p01` | 1.898363 | 0.662222 | 1.723180 | 0.110872 | 0.064342 | 0.146666 | 0.022222 | 0.134374 | `partial_but_insufficient_gap_closure` |
| 8 | `train_selected_joint_propagation_adapter` | 1.905562 | 0.706667 | 1.723180 | 0.103673 | 0.060164 | 0.708889 | 0.213333 | 0.134374 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_adapter_uncertainty_regret_gate_r0p5_m0p005` | 1.907159 | 0.697778 | 1.723180 | 0.102076 | 0.059237 | 0.448889 | 0.117778 | 0.134374 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_adapter_uncertainty_regret_gate_r0p75_m0p01` | 1.907723 | 0.666667 | 1.723180 | 0.101513 | 0.058910 | 0.273333 | 0.071111 | 0.134374 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_adapter_uncertainty_regret_gate_r1_m0p02` | 1.914322 | 0.655556 | 1.723180 | 0.094913 | 0.055080 | 0.135555 | 0.022222 | 0.134374 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_adapter_uncertainty_regret_gate_r0p75_m0p005` | 1.916211 | 0.666667 | 1.723180 | 0.093025 | 0.053984 | 0.282222 | 0.075555 | 0.134374 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_adapter_uncertainty_regret_gate_r0p75_m0p0025` | 1.916289 | 0.666667 | 1.723180 | 0.092947 | 0.053939 | 0.284444 | 0.077778 | 0.134374 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_adapter_uncertainty_regret_gate_r0p5_m0p02` | 1.917157 | 0.695555 | 1.723180 | 0.092078 | 0.053435 | 0.422222 | 0.108889 | 0.134374 | `partial_but_insufficient_gap_closure` |
| 32 | `joint_adapter_uncertainty_regret_gate_r0p75_m0p02` | 3.916088 | 0.460000 | 3.288203 | 0.175431 | 0.053352 | 0.197778 | 0.066667 | 0.061837 | `partial_but_insufficient_gap_closure` |

## 해석

- CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.
- 이 실험이 direct candidate-regret gate보다 약하면, 단순 branch-logit adapter도 P1 병목을 풀기에 부족하며 retrieval, propagation dynamics, no-harm objective를 더 깊게 공동학습해야 한다.

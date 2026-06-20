# Loss-Supervised Safe Candidate Generation 결과

이 문서는 P1 joint selector-propagator 후속 실험을 요약한다. 이전
`learned_safe_*` 진단은 hand-built teacher를 모방했지만, 이번 run은
train-fold propagation loss와 no-harm transfer label에서 learned safe candidate
generator를 학습한다.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_loss_safe_candidates.csv`

Protocol:

- 전체 world size: `N=2048`.
- causal working-set size: `K=16`, `K=32`.
- held-out seed: `11, 13, 17, 19, 23`.
- candidate pool: base selector, generated candidate, structured candidate,
  네 개의 loss-supervised `learned_safe_*` candidate.
- selector context: label-free propagation verification signature 사용.
- safety objective: pairwise no-harm score margin, weight `0.3`.

## Aggregate Results

| K | Policy | Loss | Accuracy | Gap closure | Accept | Harmful accept |
|---:|---|---:|---:|---:|---:|---:|
| 16 | `static_learned_interaction` | 0.898381 | 0.520000 | 0.000000 | 0.000000 | 0.000000 |
| 16 | `train_selected_joint_selector_propagator` | 0.877104 | 0.573333 | 0.302097 | 0.653333 | 0.146667 |
| 16 | `joint_selector_propagator` | 0.879296 | 0.577778 | 0.270969 | 0.786667 | 0.220000 |
| 16 | `generated_plus_composition_oracle` | 0.827949 | 0.626667 | 1.000000 | 0.997778 | 0.000000 |
| 32 | `static_learned_interaction` | 0.982548 | 0.488889 | 0.000000 | 0.000000 | 0.000000 |
| 32 | `train_selected_joint_selector_propagator` | 0.966607 | 0.515555 | 0.357010 | 0.326667 | 0.055555 |
| 32 | `joint_selector_propagator` | 0.973588 | 0.491111 | 0.200666 | 0.637778 | 0.224444 |
| 32 | `generated_plus_composition_oracle` | 0.937897 | 0.542222 | 1.000000 | 0.993333 | 0.000000 |

## Interpretation

Loss-supervised safe candidate generation은 larger-K deployment boundary를
개선한다. 기존 K=32 safe closure의 강한 기준은 verification context의
`0.153386`, pairwise no-harm scoring의 `0.200230` 정도였다. 이번 run은
train-selected K=32 closure를 `0.357010`까지 올리면서 harmful accept를
`0.055555`로 낮게 유지한다.

하지만 P1이 해결된 것은 아니다. P1 목표는 safe closure `0.5`이고, K=16은
`0.302097`에 머문다. 따라서 결론은 제한적이다. candidate generation은
hand-built teacher imitation만으로는 부족하고, propagation-loss/no-harm
label에서 직접 학습해야 한다. 다음 실험은 loss-supervised generator를
frozen pre-step으로 두지 말고, candidate generation과 propagation
verification을 더 직접적으로 joint objective 안에 묶어야 한다.

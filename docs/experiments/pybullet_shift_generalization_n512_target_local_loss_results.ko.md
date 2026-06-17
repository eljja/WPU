# N=512 Target-Local Delta Supervision 감사

이 실험은 이전 large-N mechanism-composition 실패가 loss 정렬 문제 때문인지 확인한다. 기존 학습 objective는 tensorized object 전체에 대한 global object-delta MSE를 사용했다. `N_bg=512`에서는 event target이 수백 개 객체 중 하나뿐이므로, target-state update 신호가 대부분의 background zero-delta 객체에 의해 희석될 수 있다.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization_n512_mechanism_factorized_shuffled_multitrain_5seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_target_local_loss_w025_5seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_target_local_loss_w05_5seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_target_local_loss_multitrain_5seed.csv`

## 프로토콜

- 도메인: PyBullet cup/table/hand/edge branch prediction.
- 학습 mechanism: `nominal`, `high_force`, `edge_shift`, `catch_heavy`, `no_catch`.
- 평가 mechanism: 위 다섯 family와 `edge_high_force`, `edge_catch_heavy`.
- World size: `background_objects=512`, total objects `N=517`.
- Seed: `11`, `13`, `17`, `19`, `23`.
- 모델: `wpu-cws-indexed-mechanism-factorized`, `graph-transformer`, `serialized-token`.
- 추가 objective: `batch.target_indices`에 대한 target-local delta MSE.

## 결과

Target-local objective는 단독 해결책으로는 negative result다.

| target-local weight | WPU macro branch accuracy | WPU target-object MSE | WPU win/tie/loss vs target-loss baselines | Mean margin |
|---:|---:|---:|---:|---:|
| 0.00 | 0.497143 | not measured | 3/0/4 | +0.002857 |
| 0.25 | 0.470000 | 0.366811 | 3/0/4 | -0.024286 |
| 0.50 | 0.445714 | 0.345698 | 2/0/5 | -0.048571 |
| 1.00 | 0.418571 | 0.324363 | 2/0/5 | -0.075714 |

Weight `1.0`에서는 target-object MSE가 낮아지지만 branch accuracy는 떨어진다. 같은 run에서 가장 강한 baseline은 `graph-transformer`이며 macro branch accuracy `0.494286`이다. WPU는 `0.418571`이다.

Weight `1.0`의 mechanism별 결과:

| mechanism | WPU accuracy | best baseline | margin |
|---|---:|---:|---:|
| `catch_heavy` | 0.720000 | 0.590000 | +0.130000 |
| `nominal` | 0.590000 | 0.530000 | +0.060000 |
| `edge_shift` | 0.370000 | 0.430000 | -0.060000 |
| `no_catch` | 0.410000 | 0.520000 | -0.110000 |
| `high_force` | 0.380000 | 0.510000 | -0.130000 |
| `edge_catch_heavy` | 0.260000 | 0.420000 | -0.160000 |
| `edge_high_force` | 0.200000 | 0.460000 | -0.260000 |

## 해석

이 실험은 실제 구현 문제를 확인한다. Global delta MSE는 large `N`에서 sparse state processing과 잘 맞지 않는다. 그러나 target-local delta loss를 단순히 추가하는 것만으로 edge-conditioned branch composition이 해결되지는 않는다. 이 loss는 state-delta 측정 목적은 개선하지만, 현재 얕은 transition head에서는 branch objective와 충돌한다.

다음 architecture 변경은 또 다른 scalar loss-weight sweep이 아니어야 한다. 증거는 branch-conditioned 또는 mechanism-specific transition dynamics가 필요하다는 쪽을 가리킨다. 모델은 target object의 다음 state vector를 더 가깝게 맞추는 것을 넘어서, local physical delta가 `edge_high_force` 같은 composed mechanism에서 어떤 branch outcome으로 이어지는지 학습해야 한다.

## Claim Boundary

이 결과는 실패 양상을 명확하게 만들어 논문을 더 강하게 만든다. 하지만 WPU의 보편적 우월성을 지지하지는 않는다. 방어 가능한 주장은 여전히 조건부다. 작은 causal working set을 식별할 수 있을 때 WPU식 sparse state processing은 계산적으로 매력적이지만, robust branch composition에는 현재 v2 구현보다 강한 transition dynamics가 필요하다.

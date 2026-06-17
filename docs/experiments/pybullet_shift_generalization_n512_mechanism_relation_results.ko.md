# N=512 Relation-Conditioned Sparse Propagation 감사

이 실험은 branch-expert negative result에서 나온 architecture 결론을 검증한다. Missing mechanism은 branch-logit capacity만이 아니라 relation-conditioned local propagation이다. 새 `wpu-cws-indexed-mechanism-relation` 모델은 indexed sparse execution과 zero dense fallback을 유지하고, mechanism-conditioned object modulation 이후 selected working-set relation을 따라 learned message를 scatter한다. Message input은 source hidden state, target hidden state, relation feature, route physics feature를 포함한다.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_trainpool40_steps16_samples40_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_expert_trainpool40_steps16_samples40_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_relation_trainpool40_steps16_samples40_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_relation_h64_trainpool40_steps16_samples40_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_baselines_h64_trainpool40_steps16_samples40_3seed.csv`

## 프로토콜

- 도메인: PyBullet cup/table/hand/edge branch prediction.
- 학습 mechanism: `nominal`, `high_force`, `edge_shift`, `catch_heavy`, `no_catch`.
- 평가 mechanism: 위 다섯 family와 `edge_high_force`, `edge_catch_heavy`.
- World size: `background_objects=512`, total objects `N=517`.
- Stress setting: `train_samples_per_mechanism=40`, `steps=16`, `samples=40`.
- Seeds: `11`, `13`, `17`.

## 결과

### Hidden size 32

| model | macro branch accuracy | ECE | dense compute ratio |
|---|---:|---:|---:|
| `wpu-cws-indexed-mechanism-relation` | 0.644048 | 0.267468 | 0.000000 |
| `graph-transformer` | 0.598810 | 0.265100 | 1.000000 |
| `serialized-token` | 0.526190 | 0.212392 | 1.000000 |
| `wpu-cws-indexed-mechanism-branch` | 0.534524 | 0.187826 | 0.000000 |
| `wpu-cws-indexed-mechanism-branch-expert` | 0.505952 | 0.191405 | 0.000000 |

h32에서 relation-conditioned WPU는 best h32 non-WPU baseline 대비 7개 mechanism을 모두 이긴다. Win/tie/loss는 `7/0/0`, mean margin은 `+0.045238`이다.

### Hidden size 64

| model | macro branch accuracy | ECE | dense compute ratio |
|---|---:|---:|---:|
| `wpu-cws-indexed-mechanism-relation` | 0.678571 | 0.285929 | 0.000000 |
| `serialized-token` | 0.622619 | 0.262267 | 1.000000 |
| `graph-transformer` | 0.582143 | 0.249364 | 1.000000 |

h64에서도 relation-conditioned WPU는 macro accuracy에서 best h64 baseline을 이긴다. Win/tie/loss는 `4/1/2`, mean margin은 `+0.021428`이다. 남은 negative mechanism은 `edge_catch_heavy`, `edge_shift`이며 calibration도 best baseline보다 약하다.

## 해석

이 결과는 현재 WPU v2 방향에 대한 가장 강한 증거다. Output-only branch expert는 실패했지만, 같은 stress protocol에서 relation-conditioned sparse propagation은 성공했다. 따라서 더 정확한 주장은 다음과 같다. Objectified world state에서 핵심 primitive는 전체 token attention도 아니고 branch-level classification만도 아니며, 작은 causal working set 위의 local relation-conditioned state propagation이다.

단, 주장은 여전히 제한적이다. 이는 3-seed PyBullet synthetic stress screen이지 broad superiority result가 아니다. 5-seed 확장, 더 큰 `N`, long-horizon rollout이 필요하다. 그래도 dense compute를 정확히 0으로 유지하면서 accuracy가 개선됐다는 점에서 WPU story를 실질적으로 강화한다.

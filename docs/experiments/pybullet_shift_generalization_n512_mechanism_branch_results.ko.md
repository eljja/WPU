# N=512 Mechanism-Branch Transition 감사

이 실험은 두 가지 negative diagnostic 이후의 다음 architecture 변경을 검증한다.

- Shuffled factorized mechanism adapter는 5 seeds에서 graph-transformer baseline을 이기지 못했다.
- Target-local delta supervision은 large-N loss dilution을 드러냈지만, scalar loss reweighting은 branch accuracy를 낮췄다.

새 모델 `wpu-cws-indexed-mechanism-branch`는 sparse indexed execution과 zero dense fallback을 유지하면서, mechanism-conditioned branch transition head를 추가한다. 이 head는 pooled causal working-set state, predicted local delta summary, explicit route physics features를 받아 branch-logit correction을 만든다. 목적은 branch outcome에 단순한 next-state vector supervision이 아니라 mechanism-specific transition dynamics가 필요한지 검증하는 것이다.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_multitrain_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_multitrain_5seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_factorized_shuffled_multitrain_5seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_target_local_loss_multitrain_5seed.csv`

## 프로토콜

- 도메인: PyBullet cup/table/hand/edge branch prediction.
- 학습 mechanism: `nominal`, `high_force`, `edge_shift`, `catch_heavy`, `no_catch`.
- 평가 mechanism: 위 다섯 family와 composed shifts `edge_high_force`, `edge_catch_heavy`.
- World size: `background_objects=512`, total objects `N=517`.
- Seeds: `11`, `13`, `17`, `19`, `23`.
- 모델: `wpu-cws-indexed-mechanism-branch`, `graph-transformer`, `serialized-token`.

## 결과

5 seeds에서 mechanism-branch WPU는 positive screen이다.

| model | macro branch accuracy | ECE | dense compute ratio |
|---|---:|---:|---:|
| `wpu-cws-indexed-mechanism-branch` | 0.568571 | 0.247101 | 0.000000 |
| `graph-transformer` | 0.548571 | 0.254194 | 1.000000 |
| `serialized-token` | 0.394286 | 0.256186 | 1.000000 |

Best non-WPU baseline 대비 mechanism별 비교:

| mechanism | WPU accuracy | best baseline accuracy | margin |
|---|---:|---:|---:|
| `nominal` | 0.660000 | 0.570000 | +0.090000 |
| `no_catch` | 0.590000 | 0.470000 | +0.120000 |
| `high_force` | 0.530000 | 0.520000 | +0.010000 |
| `edge_catch_heavy` | 0.470000 | 0.450000 | +0.020000 |
| `catch_heavy` | 0.690000 | 0.720000 | -0.030000 |
| `edge_shift` | 0.510000 | 0.540000 | -0.030000 |
| `edge_high_force` | 0.530000 | 0.570000 | -0.040000 |

5-seed win/tie/loss는 `4/0/3`, mean margin은 `+0.020000`이다. 3-seed screen은 더 강했지만 (`6/0/1`, WPU macro `0.626191`, graph-transformer `0.540476`), 보수적인 증거는 5-seed 결과다.

## 해석

이 결과는 shuffled multi-mechanism protocol을 통과하면서 sparse execution을 유지한 첫 large-N mechanism-composition positive screen이다. Target-local loss audit에서 나온 architecture 가설, 즉 branch outcome에는 local delta accuracy만이 아니라 mechanism-conditioned transition dynamics가 필요하다는 해석을 지지한다.

하지만 이것은 WPU의 보편적 우월성 주장이 아니다. 세 mechanism은 여전히 best dense baseline보다 낮고, 실험은 synthetic PyBullet domain, small model, short training 조건이다. 방어 가능한 주장은 더 좁다. Large `N`에서 작은 causal working set이 식별 가능할 때, explicit mechanism-conditioned branch transition dynamics를 추가하면 dense compute를 0으로 유지하면서 accuracy를 회복할 수 있다.

## 다음 단계

다음 우선순위는 이 결과를 현재 short training screen 밖으로 확장하는 것이다.

- step budget과 sample count를 늘려 margin이 유지되는지 확인한다.
- 더 넓은 `N` sweep으로 이득이 large-N sparse execution과 연결되는지 검증한다.
- calibration 또는 branch-prior control을 추가해 branch head가 hidden prior shortcut이 되지 않게 한다.
- 같은 mechanism-branch 설계가 long-horizon rollout consistency도 개선하는지 확인한다.

# N=512 Branch-Specific Expert 감사

이 실험은 mechanism-branch stress failure를 하나의 additive branch correction head 대신 branch-specific transition expert로 해결할 수 있는지 확인한다. 새 `wpu-cws-indexed-mechanism-branch-expert` 모델은 indexed sparse execution과 zero dense fallback을 유지하면서, 각 branch에 learned branch query와 branch-specific expert logit correction을 둔다.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_trainpool40_steps16_samples40_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_expert_trainpool40_steps16_samples40_3seed.csv`

## 프로토콜

- 도메인: PyBullet cup/table/hand/edge branch prediction.
- 학습 mechanism: `nominal`, `high_force`, `edge_shift`, `catch_heavy`, `no_catch`.
- 평가 mechanism: 위 다섯 family와 `edge_high_force`, `edge_catch_heavy`.
- World size: `background_objects=512`, total objects `N=517`.
- Stress setting: `train_samples_per_mechanism=40`, `steps=16`, `samples=40`.
- Seeds: `11`, `13`, `17`.
- Hidden size: `32`.

## 결과

Branch-specific expert는 단독 해결책으로 negative다.

| model | macro branch accuracy | ECE | dense compute ratio |
|---|---:|---:|---:|
| `graph-transformer` | 0.598810 | 0.265100 | 1.000000 |
| `serialized-token` | 0.526190 | 0.212392 | 1.000000 |
| `wpu-cws-indexed-mechanism-branch` | 0.534524 | 0.187826 | 0.000000 |
| `wpu-cws-indexed-mechanism-branch-expert` | 0.505952 | 0.191405 | 0.000000 |

Best h32 non-WPU baseline 대비 mechanism별 비교:

| mechanism | expert WPU | prior branch WPU | best baseline | expert margin |
|---|---:|---:|---:|---:|
| `edge_catch_heavy` | 0.475000 | 0.441667 | 0.400000 | +0.075000 |
| `edge_shift` | 0.558333 | 0.491667 | 0.500000 | +0.058333 |
| `no_catch` | 0.500000 | 0.483333 | 0.466667 | +0.033333 |
| `edge_high_force` | 0.533333 | 0.566667 | 0.641667 | -0.108334 |
| `nominal` | 0.500000 | 0.558333 | 0.650000 | -0.150000 |
| `high_force` | 0.416667 | 0.491667 | 0.650000 | -0.233333 |
| `catch_heavy` | 0.558333 | 0.708333 | 0.883333 | -0.325000 |

Expert는 일부 edge/catch composition case를 개선하지만 macro score는 기존 mechanism-branch head보다 낮고, dense graph baseline도 여전히 앞선다.

## 해석

이 실패는 유용하다. Branch-logit layer에 branch-specific output expert를 추가하는 것만으로는 underlying local physical transition이 개선되지 않는다. 모델은 몇몇 composed-edge gain을 얻는 대신 일반 mechanism accuracy를 잃는 것으로 보인다.

다음 수정은 branch logit 아래, sparse state update 자체로 내려가야 한다. `near_edge`, `impulse_source`, `catch_region`, `on_top_of` relation에 대해 relation-type-conditioned local message를 넣어야 한다. 즉 branch expert가 pooled context만 보고 composition을 추론하는 것이 아니라, local propagation path가 branch prediction 전에 relation-specific causal update를 인코딩해야 한다.

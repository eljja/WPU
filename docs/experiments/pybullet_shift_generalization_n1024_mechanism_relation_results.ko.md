# N=1024 Relation-Conditioned Sparse Propagation 감사

이 감사는 N=512 relation-conditioned sparse propagation 결과를 더 큰 distractor
world로 확장한다. PyBullet cup/table/hand/edge mechanism protocol은 그대로 두고
`background_objects`만 `512`에서 `1024`로 늘려 total objects를 `N=517`에서
`N=1029`로 키운다.

Source CSV:

- `docs/experiments/pybullet_shift_generalization_n1024_mechanism_relation_trainpool40_steps16_samples40_3seed.csv`

## 프로토콜

- 도메인: PyBullet cup/table/hand/edge branch prediction.
- 학습 mechanism: `nominal`, `high_force`, `edge_shift`, `catch_heavy`,
  `no_catch`.
- 평가 mechanism: 위 다섯 family와 `edge_high_force`, `edge_catch_heavy`.
- World size: `background_objects=1024`, total objects `N=1029`.
- Stress setting: `train_samples_per_mechanism=40`, `steps=16`, `samples=40`.
- Seeds: `11`, `13`, `17`.
- Models: `wpu-cws-indexed-mechanism-relation`, `graph-transformer`,
  `serialized-token`.

## 결과

| model | macro branch accuracy | ECE | dense compute ratio |
|---|---:|---:|---:|
| `wpu-cws-indexed-mechanism-relation` | 0.644048 | 0.267468 | 0.000000 |
| `graph-transformer` | 0.559524 | 0.255787 | 1.000000 |
| `serialized-token` | 0.515476 | 0.219203 | 1.000000 |

Best non-WPU baseline 대비 mechanism별 비교:

| eval mechanism | WPU accuracy | best baseline | baseline accuracy | margin |
|---|---:|---|---:|---:|
| `catch_heavy` | 0.900000 | `graph-transformer` | 0.808333 | +0.091667 |
| `edge_catch_heavy` | 0.416667 | `graph-transformer` | 0.408333 | +0.008334 |
| `edge_high_force` | 0.658333 | `graph-transformer` | 0.491667 | +0.166666 |
| `edge_shift` | 0.583333 | `graph-transformer` | 0.475000 | +0.108333 |
| `high_force` | 0.716667 | `graph-transformer` | 0.625000 | +0.091667 |
| `no_catch` | 0.533333 | `graph-transformer` | 0.475000 | +0.058333 |
| `nominal` | 0.700000 | `graph-transformer` | 0.633333 | +0.066667 |

3-seed N=1029 screen에서 WPU는 best baseline 대비 win/tie/loss `7/0/0`,
mean margin `+0.084524`를 보인다.

## 해석

이 결과는 WPU claim의 large-state 부분을 강화한다. Relation-conditioned WPU route는
zero dense fallback을 유지하면서 N=517 3-seed screen과 같은 macro accuracy를 보존한다.
반면 dense graph/token baseline은 non-causal background object 수가 커질수록 낮아진다.

단, 해석은 좁게 해야 한다. 이 결과는 causal working set이 작고 tensorization 전에
식별 가능할 때 더 큰 non-causal distractor state에 견딘다는 증거다. Broad large-N
superiority, long-horizon stability, real-world grounding, calibration dominance를
증명하지는 않는다. 다음 확장은 5-seed N=1029 run, N=2053 이상, long-horizon rollout,
calibration-aware evaluation이다.

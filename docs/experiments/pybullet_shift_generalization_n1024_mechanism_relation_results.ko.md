# N=1024 Relation-Conditioned Sparse Propagation 감사

이 감사는 N=512 relation-conditioned sparse propagation 결과를 더 큰 distractor
world로 확장한다. PyBullet cup/table/hand/edge mechanism protocol은 그대로 두고
`background_objects`만 `512`에서 `1024`로 늘려 total objects를 `N=517`에서
`N=1029`로 키운다.

Source CSV:

- `docs/experiments/pybullet_shift_generalization_n1024_mechanism_relation_trainpool40_steps16_samples40_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n1024_mechanism_relation_trainpool40_steps16_samples40_5seed.csv`

## 프로토콜

- 도메인: PyBullet cup/table/hand/edge branch prediction.
- 학습 mechanism: `nominal`, `high_force`, `edge_shift`, `catch_heavy`,
  `no_catch`.
- 평가 mechanism: 위 다섯 family와 `edge_high_force`, `edge_catch_heavy`.
- World size: `background_objects=1024`, total objects `N=1029`.
- Stress setting: `train_samples_per_mechanism=40`, `steps=16`, `samples=40`.
- Seeds: primary 3-seed screen은 `11`, `13`, `17`을 사용하고, 5-seed 확장은
  `19`, `23`을 추가한다.
- Models: `wpu-cws-indexed-mechanism-relation`, `graph-transformer`,
  `serialized-token`.

## 결과

### 3-seed screen

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

### 5-seed 확장

| model | macro branch accuracy | ECE | dense compute ratio |
|---|---:|---:|---:|
| `wpu-cws-indexed-mechanism-relation` | 0.639286 | 0.257334 | 0.000000 |
| `graph-transformer` | 0.577143 | 0.263510 | 1.000000 |
| `serialized-token` | 0.515714 | 0.206709 | 1.000000 |

Best non-WPU baseline 대비 mechanism별 비교:

| eval mechanism | WPU accuracy | best baseline | baseline accuracy | margin |
|---|---:|---|---:|---:|
| `catch_heavy` | 0.890000 | `serialized-token` | 0.750000 | +0.140000 |
| `edge_catch_heavy` | 0.435000 | `graph-transformer` | 0.420000 | +0.015000 |
| `edge_high_force` | 0.665000 | `graph-transformer` | 0.555000 | +0.110000 |
| `edge_shift` | 0.595000 | `graph-transformer` | 0.555000 | +0.040000 |
| `high_force` | 0.725000 | `graph-transformer` | 0.630000 | +0.095000 |
| `no_catch` | 0.490000 | `graph-transformer` | 0.575000 | -0.085000 |
| `nominal` | 0.675000 | `graph-transformer` | 0.635000 | +0.040000 |

5-seed N=1029 확장에서 WPU는 best baseline 대비 win/tie/loss `6/0/1`,
mean margin `+0.050714`를 보인다. 남은 negative mechanism은 `no_catch`다.

## 해석

이 결과는 WPU claim의 large-state 부분을 강화한다. Relation-conditioned WPU route는
zero dense fallback을 유지하면서 5-seed N=1029 확장에서도 positive를 유지한다. Dense
graph/token baseline은 non-causal background object 수가 커진 조건에서 여전히 WPU보다
낮다.

단, 해석은 좁게 해야 한다. 이 결과는 causal working set이 작고 tensorization 전에
식별 가능할 때 더 큰 non-causal distractor state에 견딘다는 증거다. Broad large-N
superiority, long-horizon stability, real-world grounding, calibration dominance를
증명하지는 않는다. `no_catch` loss는 sparse relation propagation에 더 나은
mechanism/prior handling이 필요함도 보여준다. 다음 확장은 N=2053 이상, long-horizon
rollout, calibration-aware evaluation이다.

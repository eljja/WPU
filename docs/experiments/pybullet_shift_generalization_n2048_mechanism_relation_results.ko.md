# N=2048 Relation-Conditioned Sparse Propagation 감사

이 감사는 relation-conditioned sparse propagation large-state screen을
`background_objects=2048`까지 확장해 total objects를 `N=2053`으로 키운다. N=512와
N=1024 감사에서 사용한 PyBullet cup/table/hand/edge mechanism protocol은 그대로
유지한다.

Source CSV:

- `docs/experiments/pybullet_shift_generalization_n2048_mechanism_relation_trainpool40_steps16_samples40_3seed.csv`

## 프로토콜

- 도메인: PyBullet cup/table/hand/edge branch prediction.
- 학습 mechanism: `nominal`, `high_force`, `edge_shift`, `catch_heavy`,
  `no_catch`.
- 평가 mechanism: 위 다섯 family와 `edge_high_force`, `edge_catch_heavy`.
- World size: `background_objects=2048`, total objects `N=2053`.
- Stress setting: `train_samples_per_mechanism=40`, `steps=16`, `samples=40`.
- Seeds: `11`, `13`, `17`.
- Models: `wpu-cws-indexed-mechanism-relation`, `graph-transformer`,
  `serialized-token`.

## 결과

| model | macro branch accuracy | ECE | dense compute ratio |
|---|---:|---:|---:|
| `wpu-cws-indexed-mechanism-relation` | 0.644048 | 0.267468 | 0.000000 |
| `graph-transformer` | 0.516667 | 0.236740 | 1.000000 |
| `serialized-token` | 0.484524 | 0.217951 | 1.000000 |

Best non-WPU baseline 대비 mechanism별 비교:

| eval mechanism | WPU accuracy | best baseline | baseline accuracy | margin |
|---|---:|---|---:|---:|
| `catch_heavy` | 0.900000 | `serialized-token` | 0.691667 | +0.208333 |
| `edge_catch_heavy` | 0.416667 | `graph-transformer` | 0.400000 | +0.016667 |
| `edge_high_force` | 0.658333 | `graph-transformer` | 0.575000 | +0.083333 |
| `edge_shift` | 0.583333 | `serialized-token` | 0.441667 | +0.141666 |
| `high_force` | 0.716667 | `graph-transformer` | 0.625000 | +0.091667 |
| `no_catch` | 0.533333 | `graph-transformer` | 0.475000 | +0.058333 |
| `nominal` | 0.700000 | `serialized-token` | 0.541667 | +0.158333 |

N=2053 3-seed screen에서 WPU는 best baseline 대비 win/tie/loss `7/0/0`,
mean margin `+0.108333`을 보인다.

## 해석

현재까지 가장 강한 distractor-scaling evidence다. WPU route는 zero dense fallback과
stable macro accuracy를 유지하는 반면, dense graph/token baseline은 non-causal
background state가 N=517, N=1029, N=2053으로 커질수록 낮아진다.

단, 주장은 여전히 제한적이다. 이 결과는 WPU가 모든 large N에서 보편적으로 낫다는 뜻이
아니다. Objectification이 tensorization 전에 작고 식별 가능한 causal working set을
노출할 때, relation-conditioned sparse propagation이 dense/token baseline이 처리해야
하는 큰 non-causal state를 무시할 수 있음을 보인다. 다음 검증은 non-causal distractor가
아니라 더 어려운 causal large-N 설정으로 이동해야 한다. 예를 들어 긴 causal chain,
여러 상호작용 object, 더 크거나 변하는 working set, long-horizon rollout이 필요하다.

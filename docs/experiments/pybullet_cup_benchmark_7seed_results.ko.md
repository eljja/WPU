# PyBullet Cup Benchmark 7-Seed 확장

이 run은 simulator-backed PyBullet cup benchmark를 5개 seed에서 7개 seed로 확장하고,
두 world size에서 평가한다. 아직 작은 benchmark지만 현재 WPU v2 dashboard의 seed
fragility를 줄인다.

Source CSV:

- `docs/experiments/pybullet_cup_benchmark_7seed.csv`

## Protocol

- Models: `wpu-cws-indexed-sparse`, `wpu-cws-indexed-local-dense`,
  `graph-transformer`, `serialized-token`.
- Seeds: `11, 13, 17, 19, 23, 29, 31`.
- Background objects: `0, 128`.
- Training steps: `20`.
- Eval samples: seed/condition마다 `36`.
- Hidden dim: `64`.
- Runtime repeats: `3`.

## 요약

| N | model | branch accuracy | forward ms/sample | CUDA peak MB |
|---:|---|---:|---:|---:|
| 5 | graph-transformer | 0.579365 | 2.032552 | 19.234 |
| 5 | serialized-token | 0.551587 | 0.121180 | 19.068 |
| 5 | wpu-cws-indexed-local-dense | 0.531746 | 1.259456 | 19.223 |
| 5 | wpu-cws-indexed-sparse | 0.547619 | 1.107382 | 17.661 |
| 133 | graph-transformer | 0.492063 | 37.241540 | 22.938 |
| 133 | serialized-token | 0.539683 | 0.440278 | 40.744 |
| 133 | wpu-cws-indexed-local-dense | 0.531746 | 1.277999 | 19.223 |
| 133 | wpu-cws-indexed-sparse | 0.547619 | 1.126503 | 17.661 |

## 해석

`N=133`에서 sparse WPU는 이 작은 7-seed run의 최고 평균 branch accuracy
(`0.547619`)를 보이고, `N=5` 대비 runtime도 거의 평평하게 유지한다. 하지만
serialized-token baseline은 여전히 가장 빠르므로 보편 latency dominance를 주장할
수는 없다. 방어 가능한 주장은 더 좁다. Explicit state와 pre-tensor indexing은
irrelevant background object가 늘어날 때 accuracy를 유지할 수 있고, full-state graph
processing은 훨씬 느려진다.

다음 P3 단계는 같은 cup scene에서 seed만 늘리는 것이 아니라 더 많은 mechanism과
long-horizon rollout evaluation을 추가하는 것이다.

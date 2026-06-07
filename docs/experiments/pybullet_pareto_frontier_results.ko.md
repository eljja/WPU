# PyBullet Pareto Frontier Audit

이 보고서는 parameter-matched PyBullet benchmark에서 accuracy-latency Pareto frontier를 계산한다.

Source CSV:

- `docs/experiments/pybullet_matched_baseline_benchmark.csv`

Derived CSV:

- `docs/experiments/pybullet_pareto_frontier.csv`

| N | model | WPU | accuracy | ms/sample | Pareto | dominated by |
|---:|---|---|---:|---:|---|---|
| 5 | `graph-transformer` | False | 0.527778 | 2.198570 | False | `serialized-token` |
| 5 | `serialized-token` | False | 0.569445 | 0.144650 | True | `` |
| 5 | `wpu-cws-indexed-local-dense` | True | 0.500000 | 1.429921 | False | `serialized-token` |
| 5 | `wpu-cws-indexed-sparse` | True | 0.569445 | 1.299167 | False | `serialized-token` |
| 133 | `graph-transformer` | False | 0.472222 | 41.766361 | False | `wpu-cws-indexed-sparse` |
| 133 | `serialized-token` | False | 0.472222 | 0.293472 | True | `` |
| 133 | `wpu-cws-indexed-local-dense` | True | 0.500000 | 1.437547 | True | `` |
| 133 | `wpu-cws-indexed-sparse` | True | 0.569445 | 2.177138 | True | `` |

## Interpretation

- WPU가 Pareto frontier에 있는 N은 `[133]`이다.
- 이 audit은 best-accuracy baseline 대비 speedup과 다른 질문을 다룬다. WPU가 어떤 baseline보다 더 정확하고 빠른 지점이 있어도, 더 낮은 accuracy에서 훨씬 빠른 token baseline이 있으면 전체 Pareto dominance는 아니다.
- 따라서 P6 주장은 large-N matched-or-better evidence와 Pareto-frontier evidence를 분리해야 한다.

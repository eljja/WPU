# PyBullet Pareto Frontier Audit

This report computes accuracy-latency Pareto frontiers for the parameter-matched PyBullet benchmark.

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

- WPU lies on the Pareto frontier at N values `[133]`.
- This audit asks a different question from speedup against the best-accuracy baseline. WPU can be more accurate and faster than one baseline while still not Pareto-dominating a faster, lower-accuracy token baseline.
- P6 claims must therefore separate large-N matched-or-better evidence from full Pareto-frontier evidence.

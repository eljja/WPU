# PyBullet Matched-Accuracy Speedup Audit

This audit compares the best WPU and best non-WPU baseline at each N in the parameter-matched PyBullet benchmark. Speedup is interpreted only when the accuracy gap is within the configured tolerance.

Source CSVs:

- `docs/experiments/pybullet_matched_speedup_audit.csv`
- `docs/experiments/pybullet_matched_baseline_benchmark.csv`
- `docs/experiments/pybullet_system_profile_cuda.csv`

Accuracy tolerance: `0.03`

| N | best WPU | best baseline | WPU acc | baseline acc | gap | matched | speedup | WPU ms | baseline ms | WPU/baseline peak mem |
|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| 5 | `wpu-cws-indexed-sparse` | `serialized-token` | 0.569445 | 0.569445 | 0.000000 | True | 0.111341 | 1.299167 | 0.144650 | 0.985685 |
| 133 | `wpu-cws-indexed-sparse` | `graph-transformer` | 0.569445 | 0.472222 | 0.097223 | False | 19.184067 | 2.177138 | 41.766361 | 0.740589 |

## Interpretation

The result enforces a stricter claim: WPU speedup only matters in regimes where accuracy is matched. The CUDA systems profile gives an upper-bound random-model latency signal at large N, while the benchmark table reports trained small-model runtime.

## CUDA Profile Context

At max profiled N `2052.4`, random-model sparse forward latency reduction is `0.996216` and sparse peak-memory reduction is `0.304080`.

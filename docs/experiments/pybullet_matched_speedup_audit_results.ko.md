# PyBullet Matched-Accuracy Speedup Audit

이 audit은 parameter-matched PyBullet benchmark에서 best WPU와 best non-WPU baseline을 같은 N별로 비교하고, WPU accuracy가 baseline보다 tolerance 이상 낮지 않을 때만 speedup을 해석한다.

Source CSVs:

- `docs/experiments/pybullet_matched_speedup_audit.csv`
- `docs/experiments/pybullet_matched_baseline_benchmark.csv`
- `docs/experiments/pybullet_system_profile_cuda.csv`

Accuracy tolerance: `0.03`

| N | best WPU | best baseline | WPU acc | baseline acc | gap | relation | matched-or-better | speedup | WPU ms | baseline ms | WPU/baseline peak mem |
|---:|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| 5 | `wpu-cws-indexed-sparse` | `serialized-token` | 0.569445 | 0.569445 | 0.000000 | matched | True | 0.111341 | 1.299167 | 0.144650 | 0.985685 |
| 133 | `wpu-cws-indexed-sparse` | `graph-transformer` | 0.569445 | 0.472222 | 0.097223 | wpu_higher | True | 19.184067 | 2.177138 | 41.766361 | 0.740589 |

## Interpretation

이 결과는 matched-or-better accuracy가 성립하는 구간에서만 speedup을 주장해야 함을 보여준다. CUDA systems profile은 큰 N에서 random-model sparse forward latency가 크게 줄어드는 상한 근거를 제공하지만, benchmark 표의 speedup은 실제 학습된 small model 조건이다.

## CUDA Profile Context

At max profiled N `2052.4`, random-model sparse forward latency reduction is `0.996216` and sparse peak-memory reduction is `0.304080`.

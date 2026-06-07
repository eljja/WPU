# PyBullet Matched-or-Better Speedup Tolerance Sweep

이 문서는 matched-or-better speedup 주장이 accuracy tolerance에 얼마나 민감한지 보여준다.

Source CSV:

- `docs/experiments/pybullet_matched_baseline_benchmark.csv`

Derived CSV:

- `docs/experiments/pybullet_matched_speedup_tolerance.csv`

| tolerance | N | matched-or-better | acc gap | speedup | WPU acc | baseline acc |
|---:|---:|---|---:|---:|---:|---:|
| 0.000 | 5 | True | 0.000000 | 0.111341 | 0.569445 | 0.569445 |
| 0.000 | 133 | True | 0.097223 | 19.184067 | 0.569445 | 0.472222 |
| 0.010 | 5 | True | 0.000000 | 0.111341 | 0.569445 | 0.569445 |
| 0.010 | 133 | True | 0.097223 | 19.184067 | 0.569445 | 0.472222 |
| 0.030 | 5 | True | 0.000000 | 0.111341 | 0.569445 | 0.569445 |
| 0.030 | 133 | True | 0.097223 | 19.184067 | 0.569445 | 0.472222 |
| 0.050 | 5 | True | 0.000000 | 0.111341 | 0.569445 | 0.569445 |
| 0.050 | 133 | True | 0.097223 | 19.184067 | 0.569445 | 0.472222 |
| 0.075 | 5 | True | 0.000000 | 0.111341 | 0.569445 | 0.569445 |
| 0.075 | 133 | True | 0.097223 | 19.184067 | 0.569445 | 0.472222 |
| 0.100 | 5 | True | 0.000000 | 0.111341 | 0.569445 | 0.569445 |
| 0.100 | 133 | True | 0.097223 | 19.184067 | 0.569445 | 0.472222 |
| 0.125 | 5 | True | 0.000000 | 0.111341 | 0.569445 | 0.569445 |
| 0.125 | 133 | True | 0.097223 | 19.184067 | 0.569445 | 0.472222 |

## Interpretation

- `matched`는 WPU accuracy가 baseline보다 tolerance 이상 낮지 않다는 뜻이다. WPU가 더 정확하면 tolerance 0에서도 통과한다.
- N=133 large-N 조건은 WPU accuracy가 더 높기 때문에 tolerance `0.000`에서도 matched-or-better로 통과한다.
- 따라서 현재 P6는 best-accuracy non-WPU baseline 대비 large-N matched-or-better speedup은 보이지만, 모든 baseline에 대한 Pareto 우월성이나 에너지 증명은 아니다.

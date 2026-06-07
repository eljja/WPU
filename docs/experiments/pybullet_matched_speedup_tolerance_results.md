# PyBullet Matched-or-Better Speedup Tolerance Sweep

This report shows how sensitive matched-or-better speedup claims are to the chosen accuracy tolerance.

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

- `Matched` means WPU accuracy is not below the baseline by more than the tolerance. If WPU is more accurate, it passes even at tolerance 0.
- The N=133 large-N point passes even at tolerance `0.000` because WPU accuracy is higher.
- P6 therefore shows large-N matched-or-better speedup against the best-accuracy non-WPU baseline, but not Pareto dominance over every baseline or real energy evidence.

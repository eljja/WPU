# PyBullet Prior-Strength Sweep

이 실험은 mechanism-aware branch prior를 한 번 학습한 모델에 여러 강도로 적용한다. 목표는 prior adaptation이 정확도 개선뿐 아니라 calibration-safe하게 적용될 수 있는지 확인하는 것이다.

Source CSV:

- `docs/experiments/pybullet_shift_generalization_prior_strength_sweep.csv`

Derived CSV:

- `docs/experiments/pybullet_prior_strength_sweep_summary.csv`

| strength | shifted WPU win-rate | WPU acc | baseline acc | WPU-baseline | WPU ECE | WPU Brier | prior-dominated shifts |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.333333 | 0.456349 | 0.460317 | -0.003968 | 0.263753 | 0.677425 | 1 |
| 0.25 | 0.666667 | 0.574074 | 0.548942 | 0.025132 | 0.275497 | 0.651754 | 1 |
| 0.50 | 0.666667 | 0.595238 | 0.583333 | 0.011905 | 0.290511 | 0.643619 | 1 |
| 0.75 | 0.666667 | 0.601852 | 0.592592 | 0.009259 | 0.293264 | 0.653145 | 0 |
| 1.00 | 0.666667 | 0.595238 | 0.595238 | 0.000000 | 0.288572 | 0.671209 | 0 |

## Shift Detail

| strength | mechanism | best WPU | best baseline | WPU acc | baseline acc | WPU-baseline | WPU ECE | prior-dominated |
|---:|---|---|---|---:|---:|---:|---:|---|
| 0.00 | catch_heavy | wpu-cws-indexed-local-dense | serialized-token | 0.408730 | 0.349206 | 0.059524 | 0.264469 | True |
| 0.00 | edge_shift | wpu-cws-indexed-sparse | serialized-token | 0.527778 | 0.571428 | -0.043650 | 0.212969 | False |
| 0.00 | high_force | wpu-cws-indexed-local-dense | serialized-token | 0.432540 | 0.460318 | -0.027778 | 0.313820 | False |
| 0.25 | catch_heavy | wpu-cws-indexed-sparse | serialized-token | 0.674603 | 0.579365 | 0.095238 | 0.254560 | True |
| 0.25 | edge_shift | wpu-cws-indexed-sparse | serialized-token | 0.619048 | 0.638889 | -0.019841 | 0.194883 | False |
| 0.25 | high_force | wpu-cws-indexed-local-dense | graph-transformer | 0.428571 | 0.428571 | 0.000000 | 0.377047 | False |
| 0.50 | catch_heavy | wpu-cws-indexed-sparse | serialized-token | 0.730159 | 0.674603 | 0.055556 | 0.233865 | True |
| 0.50 | edge_shift | wpu-cws-indexed-local-dense | serialized-token | 0.630953 | 0.650794 | -0.019841 | 0.209004 | False |
| 0.50 | high_force | wpu-cws-indexed-local-dense | graph-transformer | 0.424603 | 0.424603 | 0.000000 | 0.428664 | False |
| 0.75 | catch_heavy | wpu-cws-indexed-sparse | serialized-token | 0.753968 | 0.710317 | 0.043651 | 0.182402 | False |
| 0.75 | edge_shift | wpu-cws-indexed-local-dense | serialized-token | 0.626984 | 0.642857 | -0.015873 | 0.226969 | False |
| 0.75 | high_force | wpu-cws-indexed-local-dense | graph-transformer | 0.424603 | 0.424603 | 0.000000 | 0.470420 | False |
| 1.00 | catch_heavy | wpu-cws-indexed-sparse | serialized-token | 0.753968 | 0.738095 | 0.015873 | 0.123364 | False |
| 1.00 | edge_shift | wpu-cws-indexed-local-dense | graph-transformer | 0.607143 | 0.623016 | -0.015873 | 0.240886 | False |
| 1.00 | high_force | wpu-cws-indexed-local-dense | graph-transformer | 0.424603 | 0.424603 | 0.000000 | 0.501466 | False |

## Interpretation

- 정확도 기준 best strength는 `0.75`이며 shifted WPU win-rate는 `0.666667`이다.
- Calibration-safe 판정: `strength=0` 대비 win-rate를 유지/개선하면서 ECE를 악화시키지 않는 비영점 강도는 발견되지 않았다.
- 따라서 v2의 다음 개선점은 단순 prior bias가 아니라 confidence-aware strength selection, mechanism uncertainty, 또는 per-class calibration이다.

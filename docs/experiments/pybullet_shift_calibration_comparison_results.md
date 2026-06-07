# PyBullet Shift Calibration Comparison

This report compares temperature calibration with temperature+bias calibration on the composition-shift stress probe.

Source CSVs:

- `docs/experiments/pybullet_shift_composition_stress_summary.csv`
- `docs/experiments/pybullet_shift_composition_stress_bias_calibrated_summary.csv`

Derived CSV:

- `docs/experiments/pybullet_shift_calibration_comparison.csv`

| mechanism | base acc delta | bias acc delta | acc change | base ECE ratio | bias ECE ratio | ECE change |
|---|---:|---:|---:|---:|---:|---:|
| edge_catch_heavy | 0.120370 | 0.000000 | -0.120370 | 0.598070 | 1.269938 | 0.671868 |
| edge_high_force | 0.000000 | 0.185185 | 0.185185 | 1.022954 | 1.099548 | 0.076594 |
| no_catch | 0.250000 | 0.203704 | -0.046296 | 2.362081 | 0.960054 | -1.402027 |

## Interpretation

- Mean accuracy-delta change is `0.006173` and mean ECE-ratio change is `-0.217855`.
- ECE ratio improves on `1/3` mechanisms.
- Branch-bias calibration strongly helps `no_catch`, but it can degrade accuracy or ECE on other shifts. P5 therefore remains a mechanism-aware uncertainty/calibration problem, not a solved post-hoc calibration problem.

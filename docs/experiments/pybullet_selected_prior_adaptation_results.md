# PyBullet Calibration-Selected Prior Adaptation

This experiment selects a prior strength on a small held-out calibration set for each mechanism, then applies the selected strength to the evaluation set. It tests whether branch-prior adaptation can be made more calibration-safe than a fixed prior bias.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization.csv`
- `docs/experiments/pybullet_shift_generalization_selected_prior.csv`

Derived CSV:

- `docs/experiments/pybullet_selected_prior_adaptation_summary.csv`

| mechanism | selected strength | base WPU acc | selected WPU acc | WPU acc change | base WPU-baseline | selected WPU-baseline | WPU ECE change | WPU Brier change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catch_heavy | 1.000000 | 0.408730 | 0.753968 | 0.345238 | 0.059524 | 0.027778 | -0.141105 | -0.275406 |
| edge_shift | 0.571429 | 0.527778 | 0.619048 | 0.091270 | -0.043650 | -0.039682 | 0.002492 | -0.041003 |
| high_force | 0.000000 | 0.432540 | 0.432540 | 0.000000 | -0.027778 | -0.019841 | 0.000000 | 0.000000 |
| nominal | 0.357143 | 0.444445 | 0.424603 | -0.019842 | -0.055555 | -0.083333 | 0.000352 | 0.018304 |

## Interpretation

- Shifted WPU win-rate changes from `0.333333` to `0.333333`.
- Mean shifted WPU accuracy change is `0.145503`.
- Mean shifted WPU ECE change is `-0.046204` and mean Brier change is `-0.105470`; negative means better.
- Prior-dominated shifted mechanisms fall from `1` to `0`.
- Selected priors improve P5 calibration evidence but do not improve P4 baseline win-rate. The next step is to learn model confidence and mechanism uncertainty jointly rather than only selecting a scalar prior strength.

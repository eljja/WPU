# PyBullet Mechanism-Prior Adaptation

This experiment estimates a branch-label prior from a small calibration set for each held-out mechanism and adds a log-prior bias relative to the training prior. It is not a test-label oracle; it is a diagnostic for mechanism-aware prior adaptation.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization.csv`
- `docs/experiments/pybullet_shift_generalization_mechanism_prior.csv`

Derived CSV:

- `docs/experiments/pybullet_mechanism_prior_adaptation_summary.csv`

| mechanism | base WPU acc | adapted WPU acc | WPU acc change | base WPU-baseline | adapted WPU-baseline | majority acc | base gap | adapted gap | WPU ECE change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| catch_heavy | 0.408730 | 0.753968 | 0.345238 | 0.059524 | 0.015873 | 0.753968 | 0.345238 | 0.000000 | -0.141105 |
| edge_shift | 0.527778 | 0.607143 | 0.079365 | -0.043650 | -0.015873 | 0.515873 | -0.011905 | -0.091270 | 0.027917 |
| high_force | 0.432540 | 0.424603 | -0.007937 | -0.027778 | 0.000000 | 0.424603 | -0.007937 | 0.000000 | 0.187646 |
| nominal | 0.444445 | 0.448413 | 0.003968 | -0.055555 | -0.051587 | 0.468254 | 0.023809 | 0.019841 | 0.111761 |

## Interpretation

- WPU win-rate over shifted mechanisms changes from `0.333333` to `0.666667`.
- Mean WPU accuracy change over shifted mechanisms is `0.138889`.
- Mean WPU ECE change over shifted mechanisms is `0.024819`; positive means worse calibration.
- Prior-dominated shifted mechanisms fall from `1` to `0`.
- `catch_heavy` improves strongly, but other shifts can lose accuracy or calibration. Mechanism-aware branch priors are necessary, but a simple prior bias does not solve P4/P5.

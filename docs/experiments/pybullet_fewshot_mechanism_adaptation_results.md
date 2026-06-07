# PyBullet Few-Shot Mechanism Adaptation

This experiment fine-tunes nominal-trained models for a few steps on a small calibration set from each held-out mechanism, then evaluates on the same shift benchmark. The same adaptation is applied to WPU and non-WPU baselines.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization.csv`
- `docs/experiments/pybullet_fewshot_mechanism_adaptation.csv`

Derived CSV:

- `docs/experiments/pybullet_fewshot_mechanism_adaptation_summary.csv`

| mechanism | base WPU acc | adapted WPU acc | WPU acc change | baseline acc change | base WPU-baseline | adapted WPU-baseline | margin change | WPU ECE change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| catch_heavy | 0.408730 | 0.623016 | 0.214286 | 0.269842 | 0.059524 | 0.003968 | -0.055556 | -0.009088 |
| edge_shift | 0.527778 | 0.642857 | 0.115079 | 0.039683 | -0.043650 | 0.031746 | 0.075396 | -0.023790 |
| high_force | 0.432540 | 0.567460 | 0.134920 | 0.003968 | -0.027778 | 0.103174 | 0.130952 | -0.133147 |
| nominal | 0.444445 | 0.543651 | 0.099206 | 0.011905 | -0.055555 | 0.031746 | 0.087301 | 0.057890 |

## Interpretation

- Shifted WPU win-rate changes from `0.333333` to `1.000000`.
- Mean shifted WPU accuracy change is `0.154762`; baseline accuracy change is `0.104498`.
- Mean shifted WPU-baseline margin change is `0.050264`.
- Mean shifted WPU ECE change is `-0.055342` and Brier change is `-0.103932`.
- This is a P4 follow-up: it tests whether mechanism shift that is not solved by branch priors can be reduced by parameter adaptation.

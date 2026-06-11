# PyBullet Mechanism-Aware Adaptive Policy

This analysis combines the existing selected-prior and few-shot adaptation results. If the calibration-selected prior strength exceeds the threshold, the policy uses branch-prior adaptation; otherwise it uses few-shot parameter adaptation for shifted mechanisms. Nominal evaluation keeps the base policy.

This is an adapted protocol that uses mechanism-specific calibration/adaptation samples. It is not zero-shot generalization evidence. It is evidence that improving P4 requires a mechanism-shift detector plus a selective adaptation policy.

Source CSVs:

- `docs/experiments/pybullet_selected_prior_adaptation_summary.csv`
- `docs/experiments/pybullet_fewshot_mechanism_adaptation_summary.csv`

Derived CSV:

- `docs/experiments/pybullet_mechanism_adaptive_policy_summary.csv`

| mechanism | selected policy | prior strength | base WPU acc | policy WPU acc | acc change | base margin | policy margin | margin change | ECE change | Brier change |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| catch_heavy | `selected_prior` | 1.000000 | 0.408730 | 0.753968 | 0.345238 | 0.059524 | 0.027778 | -0.031746 | -0.141105 | -0.275406 |
| edge_shift | `fewshot_adaptation` | 0.571429 | 0.527778 | 0.642857 | 0.115079 | -0.043650 | 0.031746 | 0.075396 | -0.023790 | -0.065298 |
| high_force | `fewshot_adaptation` | 0.000000 | 0.432540 | 0.567460 | 0.134920 | -0.027778 | 0.103174 | 0.130952 | -0.133147 | -0.125624 |
| nominal | `base_nominal` | 0.357143 | 0.444445 | 0.444445 | 0.000000 | -0.055555 | -0.055555 | 0.000000 | 0.000000 | 0.000000 |

## Interpretation

- Policy WPU win-rate over shifted mechanisms is `1.000000`.
- Mean shifted WPU accuracy change is `0.198412`.
- Mean shifted WPU-baseline margin change is `0.058201`.
- Mean shifted WPU ECE change is `-0.099347` and Brier change is `-0.155443`; negative means better.
- The result suggests a stronger P4/P5 direction than a single scalar prior or unconditional few-shot adaptation, while strengthening the condition that mechanism-aware adaptation is required.

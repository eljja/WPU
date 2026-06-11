# PyBullet Shift-Detector Adaptive Policy

This audit tests whether adaptation can be selected from calibration statistics instead of direct mechanism identity. If base WPU ECE is below a threshold, the detector keeps the base policy. Otherwise it selects branch-prior adaptation when the majority-prior gap is large and few-shot adaptation otherwise.

This is still not zero-shot: it uses calibration labels and mechanism-specific adaptation samples. It is stricter than the previous mechanism-aware policy because the decision is made from observable statistics rather than the mechanism name.

Source CSVs:

- `docs/experiments/pybullet_selected_prior_adaptation_summary.csv`
- `docs/experiments/pybullet_fewshot_mechanism_adaptation_summary.csv`

Derived CSV:

- `docs/experiments/pybullet_shift_detector_policy.csv`

| ECE threshold | prior-gap threshold | shifted win-rate | acc change | margin change | ECE change | Brier change | nominal false adaptation | score | decisions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.12 | 0.00 | 1.000000 | 0.198412 | 0.058201 | -0.099347 | -0.155443 | 0 | 1.083037 | `catch_heavy=selected_prior; edge_shift=fewshot_adaptation; high_force=fewshot_adaptation; nominal=base` |
| 0.12 | 0.05 | 1.000000 | 0.198412 | 0.058201 | -0.099347 | -0.155443 | 0 | 1.083037 | `catch_heavy=selected_prior; edge_shift=fewshot_adaptation; high_force=fewshot_adaptation; nominal=base` |
| 0.12 | 0.10 | 1.000000 | 0.198412 | 0.058201 | -0.099347 | -0.155443 | 0 | 1.083037 | `catch_heavy=selected_prior; edge_shift=fewshot_adaptation; high_force=fewshot_adaptation; nominal=base` |
| 0.12 | 0.20 | 1.000000 | 0.198412 | 0.058201 | -0.099347 | -0.155443 | 0 | 1.083037 | `catch_heavy=selected_prior; edge_shift=fewshot_adaptation; high_force=fewshot_adaptation; nominal=base` |
| 0.12 | 0.30 | 1.000000 | 0.198412 | 0.058201 | -0.099347 | -0.155443 | 0 | 1.083037 | `catch_heavy=selected_prior; edge_shift=fewshot_adaptation; high_force=fewshot_adaptation; nominal=base` |
| 0.15 | 0.00 | 1.000000 | 0.198412 | 0.058201 | -0.099347 | -0.155443 | 0 | 1.083037 | `catch_heavy=selected_prior; edge_shift=fewshot_adaptation; high_force=fewshot_adaptation; nominal=base` |
| 0.15 | 0.05 | 1.000000 | 0.198412 | 0.058201 | -0.099347 | -0.155443 | 0 | 1.083037 | `catch_heavy=selected_prior; edge_shift=fewshot_adaptation; high_force=fewshot_adaptation; nominal=base` |
| 0.15 | 0.10 | 1.000000 | 0.198412 | 0.058201 | -0.099347 | -0.155443 | 0 | 1.083037 | `catch_heavy=selected_prior; edge_shift=fewshot_adaptation; high_force=fewshot_adaptation; nominal=base` |
| 0.15 | 0.20 | 1.000000 | 0.198412 | 0.058201 | -0.099347 | -0.155443 | 0 | 1.083037 | `catch_heavy=selected_prior; edge_shift=fewshot_adaptation; high_force=fewshot_adaptation; nominal=base` |
| 0.15 | 0.30 | 1.000000 | 0.198412 | 0.058201 | -0.099347 | -0.155443 | 0 | 1.083037 | `catch_heavy=selected_prior; edge_shift=fewshot_adaptation; high_force=fewshot_adaptation; nominal=base` |
| 0.18 | 0.00 | 1.000000 | 0.198412 | 0.058201 | -0.099347 | -0.155443 | 0 | 1.083037 | `catch_heavy=selected_prior; edge_shift=fewshot_adaptation; high_force=fewshot_adaptation; nominal=base` |
| 0.18 | 0.05 | 1.000000 | 0.198412 | 0.058201 | -0.099347 | -0.155443 | 0 | 1.083037 | `catch_heavy=selected_prior; edge_shift=fewshot_adaptation; high_force=fewshot_adaptation; nominal=base` |

## Best Decisions

- `catch_heavy`: `selected_prior` (base ECE `0.264469`, prior gap `0.345238`)
- `edge_shift`: `fewshot_adaptation` (base ECE `0.212969`, prior gap `-0.011905`)
- `high_force`: `fewshot_adaptation` (base ECE `0.313820`, prior gap `-0.007937`)
- `nominal`: `base` (base ECE `0.118385`, prior gap `0.023809`)

## Interpretation

- Best detector score occurs at thresholds `(ECE=0.12, gap=0.00)` with score `1.083037`.
- Its shifted win-rate is `1.000000`, mean accuracy change is `0.198412`, and mean ECE change is `-0.099347`.
- The best-safe policy with no nominal false adaptation also reaches detector score `1.083037`.
- This supports the next P4 direction: calibration-statistic detection plus selective adaptation, not a mechanism-name oracle.

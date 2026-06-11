# PyBullet Shift-Detector Adaptive Policy

이 audit는 mechanism 이름을 직접 사용하지 않고 calibration statistics로 adaptation policy를 선택할 수 있는지 검사한다. Base WPU ECE가 threshold보다 낮으면 base policy를 유지하고, shift로 판단되면 majority-prior gap이 큰 경우 selected-prior를, 그 외에는 few-shot adaptation을 사용한다.

이 결과도 zero-shot은 아니다. Calibration labels와 mechanism-specific adaptation sample을 사용한다. 다만 기존 mechanism-aware policy보다 엄격하게 mechanism identity 대신 observable statistics로 detect-and-adapt 결정을 만든다.

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

## 최선 결정

- `catch_heavy`: `selected_prior` (base ECE `0.264469`, prior gap `0.345238`)
- `edge_shift`: `fewshot_adaptation` (base ECE `0.212969`, prior gap `-0.011905`)
- `high_force`: `fewshot_adaptation` (base ECE `0.313820`, prior gap `-0.007937`)
- `nominal`: `base` (base ECE `0.118385`, prior gap `0.023809`)

## 해석

- Best detector score는 threshold `(ECE=0.12, gap=0.00)`에서 `1.083037`이다.
- Best shifted win-rate는 `1.000000`, mean accuracy change는 `0.198412`, mean ECE change는 `-0.099347`이다.
- Nominal false adaptation이 없는 best-safe policy도 detector score `1.083037`를 달성한다.
- 이는 P4의 다음 방향이 mechanism 이름 oracle이 아니라 calibration-statistic detector와 selective adaptation임을 지지한다.

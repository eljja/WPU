# PyBullet Mechanism-Aware Adaptive Policy

이 분석은 기존 selected-prior 결과와 few-shot adaptation 결과를 결합한다. Calibration-selected prior strength가 threshold 이상이면 branch-prior adaptation을 사용하고, 그 외 shifted mechanism에는 few-shot parameter adaptation을 사용한다. Nominal은 base policy를 유지한다.

이는 mechanism별 calibration/adaptation sample을 사용하는 adapted protocol이다. 따라서 zero-shot generalization 증거가 아니라, P4를 개선하려면 mechanism shift detector와 selective adaptation policy가 필요하다는 증거로 해석해야 한다.

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

## 해석

- Shifted mechanism 기준 policy WPU win-rate는 `1.000000`이다.
- Shifted 평균 WPU accuracy 변화는 `0.198412`이다.
- Shifted 평균 WPU-baseline margin 변화는 `0.058201`이다.
- Shifted 평균 WPU ECE 변화는 `-0.099347`, Brier 변화는 `-0.155443`이다. 음수는 개선이다.
- 이 결과는 단일 scalar prior나 무조건 few-shot보다 더 나은 P4/P5 결합 방향을 제시하지만, mechanism-aware adaptation이 필요하다는 조건을 강화한다.

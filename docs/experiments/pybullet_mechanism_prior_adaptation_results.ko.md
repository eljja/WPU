# PyBullet Mechanism-Prior Adaptation

이 실험은 held-out mechanism별 작은 calibration set으로 branch label prior를 추정하고, train prior 대비 log-prior bias를 branch logits에 더한다. 이는 test label oracle이 아니라 mechanism-aware prior adaptation의 작은 진단 실험이다.

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

- Shift mechanism 기준 WPU win-rate는 `0.333333`에서 `0.666667`로 변했다.
- Shift mechanism 기준 평균 WPU accuracy 변화는 `0.138889`이다.
- Shift mechanism 기준 평균 WPU ECE 변화는 `0.024819`이다. 양수면 calibration이 악화된 것이다.
- Prior-dominated shifted mechanism은 `1`개에서 `0`개로 줄었다.
- `catch_heavy`는 크게 개선되지만, 다른 shift에서는 ECE와 accuracy가 악화될 수 있다. 따라서 branch prior adaptation은 필요하지만, 단순 prior bias만으로 P4/P5가 해결되지는 않는다.

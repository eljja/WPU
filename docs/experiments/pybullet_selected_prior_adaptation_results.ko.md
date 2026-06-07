# PyBullet Calibration-Selected Prior Adaptation

이 실험은 test set이 아니라 held-out mechanism별 작은 calibration set에서 후보 prior strength를 선택한 뒤 evaluation set에 적용한다. 목표는 고정 prior bias보다 더 calibration-safe한 branch-prior adaptation이 가능한지 확인하는 것이다.

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

- Shifted WPU win-rate는 `0.333333`에서 `0.333333`로 변했다.
- Shifted 평균 WPU accuracy 변화는 `0.145503`이다.
- Shifted 평균 WPU ECE 변화는 `-0.046204`이고, 평균 Brier 변화는 `-0.105470`이다. 음수는 개선이다.
- Prior-dominated shifted mechanism은 `1`개에서 `0`개로 줄었다.
- 따라서 selected prior는 P5 calibration에는 실제 개선을 보이지만, P4 baseline win-rate를 올리지는 못한다. 다음 단계는 prior strength 선택이 아니라 model confidence와 mechanism uncertainty를 함께 학습하는 것이다.

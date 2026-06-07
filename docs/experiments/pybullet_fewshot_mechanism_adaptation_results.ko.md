# PyBullet Few-Shot Mechanism Adaptation

이 실험은 nominal로 학습한 모델을 held-out mechanism별 작은 calibration set에서 몇 step fine-tune한 뒤 evaluation set에 적용한다. Baseline에도 같은 adaptation을 적용하므로, WPU만 유리하게 만든 실험이 아니다.

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

- Shifted WPU win-rate는 `0.333333`에서 `1.000000`로 변했다.
- Shifted 평균 WPU accuracy 변화는 `0.154762`이고 baseline accuracy 변화는 `0.104498`이다.
- Shifted 평균 WPU-baseline margin 변화는 `0.050264`이다.
- Shifted 평균 WPU ECE 변화는 `-0.055342`, Brier 변화는 `-0.103932`이다.
- 이 실험은 branch prior만으로 부족한 mechanism shift를 모델 파라미터 적응으로 줄일 수 있는지 보는 P4 follow-up이다.

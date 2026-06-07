# PyBullet Shift Calibration Comparison

이 문서는 composition-shift stress에서 temperature calibration과 temperature+bias calibration을 비교한다.

Source CSVs:

- `docs/experiments/pybullet_shift_composition_stress_summary.csv`
- `docs/experiments/pybullet_shift_composition_stress_bias_calibrated_summary.csv`

Derived CSV:

- `docs/experiments/pybullet_shift_calibration_comparison.csv`

| mechanism | base acc delta | bias acc delta | acc change | base ECE ratio | bias ECE ratio | ECE change |
|---|---:|---:|---:|---:|---:|---:|
| edge_catch_heavy | 0.120370 | 0.000000 | -0.120370 | 0.598070 | 1.269938 | 0.671868 |
| edge_high_force | 0.000000 | 0.185185 | 0.185185 | 1.022954 | 1.099548 | 0.076594 |
| no_catch | 0.250000 | 0.203704 | -0.046296 | 2.362081 | 0.960054 | -1.402027 |

## Interpretation

- 평균 accuracy-delta 변화는 `0.006173`이고 평균 ECE-ratio 변화는 `-0.217855`이다.
- ECE ratio가 개선된 mechanism은 `1/3`개다.
- Branch-bias calibration은 `no_catch` calibration을 크게 개선하지만, 다른 shift에서는 accuracy 또는 ECE를 악화시킬 수 있다. 따라서 P5는 보편 해결이 아니라 mechanism-aware uncertainty/calibration 문제로 남는다.

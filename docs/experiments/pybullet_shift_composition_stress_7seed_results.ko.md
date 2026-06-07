# PyBullet Composition-Shift Stress

이 실험은 nominal/high_force/edge_shift/catch_heavy로 학습한 모델을 새로운 조합형 mechanism에서 평가한다.

Source CSV:

- `docs/experiments/pybullet_shift_composition_stress_7seed.csv`

Derived CSV:

- `docs/experiments/pybullet_shift_composition_stress_7seed_summary.csv`

| mechanism | best WPU | best baseline | WPU acc | baseline acc | delta | WPU ECE | baseline ECE | ECE ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|
| edge_catch_heavy | `wpu-cws-indexed-local-dense` | `graph-transformer` | 0.412698 | 0.353175 | 0.059523 | 0.101415 | 0.098169 | 1.033065 |
| edge_high_force | `wpu-cws-indexed-local-dense` | `graph-transformer` | 0.615079 | 0.611111 | 0.003968 | 0.138419 | 0.163713 | 0.845498 |
| no_catch | `wpu-cws-indexed-local-dense` | `graph-transformer` | 0.599206 | 0.448413 | 0.150793 | 0.206403 | 0.177007 | 1.166073 |

## Interpretation

- WPU win-rate는 `1.000000`, 평균 accuracy delta는 `0.071428`다.
- 평균 ECE ratio는 `1.014879`이며, 1보다 작으면 best WPU의 ECE가 best baseline보다 낮다는 뜻이다.
- 이 stress test는 단일 held-out family보다 어렵다. Accuracy-positive 결과는 compound-shift P4를 강화하지만, ECE가 1보다 크면 P5 calibration은 별도 문제로 남는다.

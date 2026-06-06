# PyBullet Composition-Shift Stress

이 실험은 nominal/high_force/edge_shift/catch_heavy로 학습한 모델을 새로운 조합형 mechanism에서 평가한다.

Source CSV:

- `docs/experiments/pybullet_shift_composition_stress.csv`

Derived CSV:

- `docs/experiments/pybullet_shift_composition_stress_summary.csv`

| mechanism | best WPU | best baseline | WPU acc | baseline acc | delta | WPU ECE | baseline ECE | ECE ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|
| edge_catch_heavy | `wpu-cws-indexed-sparse` | `graph-transformer` | 0.453704 | 0.333334 | 0.120370 | 0.070473 | 0.117834 | 0.598070 |
| edge_high_force | `wpu-cws-indexed-local-dense` | `graph-transformer` | 0.583333 | 0.583333 | 0.000000 | 0.196624 | 0.192212 | 1.022954 |
| no_catch | `wpu-cws-indexed-sparse` | `graph-transformer` | 0.759259 | 0.509259 | 0.250000 | 0.337050 | 0.142692 | 2.362081 |

## Interpretation

- WPU win-rate는 `1.000000`, 평균 accuracy delta는 `0.123457`다.
- 평균 ECE ratio는 `1.327702`이며, 1보다 작으면 best WPU의 ECE가 best baseline보다 낮다는 뜻이다.
- 이 stress test는 단일 held-out family보다 어렵다. compound shift에서 지면 WPU 주장은 local-state regime으로 더 좁혀야 한다.

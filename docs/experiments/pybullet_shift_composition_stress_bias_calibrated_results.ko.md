# PyBullet Composition-Shift Stress

이 실험은 nominal/high_force/edge_shift/catch_heavy로 학습한 모델을 새로운 조합형 mechanism에서 평가한다.

Source CSV:

- `docs/experiments/pybullet_shift_composition_stress_bias_calibrated.csv`

Derived CSV:

- `docs/experiments/pybullet_shift_composition_stress_bias_calibrated_summary.csv`

| mechanism | best WPU | best baseline | WPU acc | baseline acc | delta | WPU ECE | baseline ECE | ECE ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|
| edge_catch_heavy | `wpu-cws-indexed-sparse` | `graph-transformer` | 0.490741 | 0.490741 | 0.000000 | 0.118183 | 0.093062 | 1.269938 |
| edge_high_force | `wpu-cws-indexed-local-dense` | `serialized-token` | 0.546296 | 0.361111 | 0.185185 | 0.118594 | 0.107857 | 1.099548 |
| no_catch | `wpu-cws-indexed-local-dense` | `graph-transformer` | 0.685185 | 0.481481 | 0.203704 | 0.245334 | 0.255542 | 0.960054 |

## Interpretation

- WPU win-rate는 `1.000000`, 평균 accuracy delta는 `0.129630`다.
- 평균 ECE ratio는 `1.109847`이며, 1보다 작으면 best WPU의 ECE가 best baseline보다 낮다는 뜻이다.
- 이 stress test는 단일 held-out family보다 어렵다. compound shift에서 지면 WPU 주장은 local-state regime으로 더 좁혀야 한다.

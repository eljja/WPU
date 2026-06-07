# PyBullet Composition-Shift Stress

This experiment trains on nominal/high_force/edge_shift/catch_heavy and evaluates on unseen composition mechanisms.

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

- WPU win-rate is `1.000000` and mean accuracy delta is `0.129630`.
- Mean ECE ratio is `1.109847`; values below 1 mean the best WPU has lower ECE than the best baseline.
- This stress test is harder than single held-out families. Failures here narrow the WPU claim to local-state regimes rather than broad shift generalization.

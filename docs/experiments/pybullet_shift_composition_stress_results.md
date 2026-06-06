# PyBullet Composition-Shift Stress

This experiment trains on nominal/high_force/edge_shift/catch_heavy and evaluates on unseen composition mechanisms.

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

- WPU win-rate is `1.000000` and mean accuracy delta is `0.123457`.
- Mean ECE ratio is `1.327702`; values below 1 mean the best WPU has lower ECE than the best baseline.
- This stress test is harder than single held-out families. Failures here narrow the WPU claim to local-state regimes rather than broad shift generalization.

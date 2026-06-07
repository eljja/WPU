# PyBullet Composition-Shift Stress

This experiment trains on nominal/high_force/edge_shift/catch_heavy and evaluates on unseen composition mechanisms.

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

- WPU win-rate is `1.000000` and mean accuracy delta is `0.071428`.
- Mean ECE ratio is `1.014879`; values below 1 mean the best WPU has lower ECE than the best baseline.
- This stress test is harder than single held-out families. Accuracy-positive results strengthen compound-shift P4, but ECE above 1 keeps P5 calibration separate.

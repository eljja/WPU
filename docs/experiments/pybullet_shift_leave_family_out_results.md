# PyBullet Leave-Family-Out Shift Results

This experiment holds out one mechanism family at a time, trains on the remaining three families, and evaluates on the held-out family. It is a stricter cross-mechanism generalization probe than nominal-only shift.

Source CSVs:

- `docs/experiments/pybullet_shift_leave_family_nominal.csv`
- `docs/experiments/pybullet_shift_leave_family_high_force.csv`
- `docs/experiments/pybullet_shift_leave_family_edge_shift.csv`
- `docs/experiments/pybullet_shift_leave_family_catch_heavy.csv`

| held-out mechanism | train mechanisms | best WPU | best baseline | WPU acc | baseline acc | gap | WPU ECE | baseline ECE | ECE ratio | WPU win |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| catch_heavy | `nominal+high_force+edge_shift` | `wpu-cws-indexed-sparse` | `graph-transformer` | 0.175926 | 0.296296 | -0.120370 | 0.299501 | 0.313708 | 0.954713 | False |
| edge_shift | `nominal+high_force+catch_heavy` | `wpu-cws-indexed-sparse` | `serialized-token` | 0.509260 | 0.444445 | 0.064815 | 0.133161 | 0.106670 | 1.248345 | True |
| high_force | `nominal+edge_shift+catch_heavy` | `wpu-cws-indexed-local-dense` | `graph-transformer` | 0.500000 | 0.490741 | 0.009259 | 0.167035 | 0.207701 | 0.804208 | True |
| nominal | `high_force+edge_shift+catch_heavy` | `wpu-cws-indexed-sparse` | `graph-transformer` | 0.462963 | 0.407407 | 0.055556 | 0.127846 | 0.144669 | 0.883714 | True |

## Interpretation

Leave-family-out results show that WPU can help in some geometry-driven shifts, but it does not solve branch-prior shift in general. Accuracy wins must be read together with ECE ratios because calibration remains unstable.

WPU leave-family-out win rate: `0.750000`.
Mean WPU/baseline ECE ratio: `0.972745`.

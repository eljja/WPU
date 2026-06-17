# N=512 Target-Local Delta Supervision Audit

This audit tests whether the previous large-N mechanism-composition failure is caused by a loss-alignment bug. The old training objective used a global object-delta MSE over all tensorized objects. At `N_bg=512`, the event target is only one object among hundreds, so the target-state update can be numerically diluted by background zero-delta objects.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization_n512_mechanism_factorized_shuffled_multitrain_5seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_target_local_loss_w025_5seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_target_local_loss_w05_5seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_target_local_loss_multitrain_5seed.csv`

## Protocol

- Domain: PyBullet cup/table/hand/edge branch prediction.
- Training mechanisms: `nominal`, `high_force`, `edge_shift`, `catch_heavy`, `no_catch`.
- Held-out/evaluation mechanisms: the five training families plus `edge_high_force` and `edge_catch_heavy`.
- World size: `background_objects=512`, total objects `N=517`.
- Seeds: `11`, `13`, `17`, `19`, `23`.
- Models: `wpu-cws-indexed-mechanism-factorized`, `graph-transformer`, `serialized-token`.
- Added objective: target-local delta MSE on `batch.target_indices`.

## Result

The target-local objective is a negative result as a standalone fix.

| target-local weight | WPU macro branch accuracy | WPU target-object MSE | WPU win/tie/loss vs target-loss baselines | Mean margin |
|---:|---:|---:|---:|---:|
| 0.00 | 0.497143 | not measured | 3/0/4 | +0.002857 |
| 0.25 | 0.470000 | 0.366811 | 3/0/4 | -0.024286 |
| 0.50 | 0.445714 | 0.345698 | 2/0/5 | -0.048571 |
| 1.00 | 0.418571 | 0.324363 | 2/0/5 | -0.075714 |

At weight `1.0`, target-object MSE improves monotonically relative to lower target-local weights, but branch accuracy falls. The strongest baseline in the matched run is `graph-transformer` with macro branch accuracy `0.494286`; the WPU reaches `0.418571`.

Per-mechanism behavior at weight `1.0`:

| mechanism | WPU accuracy | best baseline | margin |
|---|---:|---:|---:|
| `catch_heavy` | 0.720000 | 0.590000 | +0.130000 |
| `nominal` | 0.590000 | 0.530000 | +0.060000 |
| `edge_shift` | 0.370000 | 0.430000 | -0.060000 |
| `no_catch` | 0.410000 | 0.520000 | -0.110000 |
| `high_force` | 0.380000 | 0.510000 | -0.130000 |
| `edge_catch_heavy` | 0.260000 | 0.420000 | -0.160000 |
| `edge_high_force` | 0.200000 | 0.460000 | -0.260000 |

## Interpretation

The experiment confirms a real implementation issue: global delta MSE is poorly aligned with sparse state processing at large `N`. However, simply adding a target-local delta loss does not solve edge-conditioned branch composition. It improves the state-delta measurement objective but competes with the branch objective in the current shallow transition head.

The next architecture change should not be another scalar loss-weight sweep. The evidence points to branch-conditioned or mechanism-specific transition dynamics: the model needs to predict how a local physical delta maps to branch outcomes under composed mechanisms such as `edge_high_force`, rather than only making the target object's next-state vector closer.

## Claim Boundary

This result strengthens the paper by making the failure mode explicit. It does not support broad WPU superiority. The defensible claim remains conditional: WPU-style sparse state processing is computationally attractive when a small causal working set is identifiable, but robust branch composition requires stronger transition dynamics than the current v2 implementation.

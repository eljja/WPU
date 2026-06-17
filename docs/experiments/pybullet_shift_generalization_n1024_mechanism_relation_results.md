# N=1024 Relation-Conditioned Sparse Propagation Audit

This audit extends the N=512 relation-conditioned sparse propagation result to a
larger distractor world. It keeps the same PyBullet cup/table/hand/edge
mechanism protocol and increases only `background_objects` from `512` to `1024`,
raising total objects from `N=517` to `N=1029`.

Source CSV:

- `docs/experiments/pybullet_shift_generalization_n1024_mechanism_relation_trainpool40_steps16_samples40_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n1024_mechanism_relation_trainpool40_steps16_samples40_5seed.csv`

## Protocol

- Domain: PyBullet cup/table/hand/edge branch prediction.
- Training mechanisms: `nominal`, `high_force`, `edge_shift`, `catch_heavy`,
  `no_catch`.
- Evaluation mechanisms: the five training families plus `edge_high_force` and
  `edge_catch_heavy`.
- World size: `background_objects=1024`, total objects `N=1029`.
- Stress setting: `train_samples_per_mechanism=40`, `steps=16`, `samples=40`.
- Seeds: primary 3-seed screen uses `11`, `13`, `17`; the 5-seed expansion adds
  `19`, `23`.
- Models: `wpu-cws-indexed-mechanism-relation`, `graph-transformer`,
  `serialized-token`.

## Results

### 3-seed screen

| model | macro branch accuracy | ECE | dense compute ratio |
|---|---:|---:|---:|
| `wpu-cws-indexed-mechanism-relation` | 0.644048 | 0.267468 | 0.000000 |
| `graph-transformer` | 0.559524 | 0.255787 | 1.000000 |
| `serialized-token` | 0.515476 | 0.219203 | 1.000000 |

Per-mechanism comparison against the best non-WPU baseline:

| eval mechanism | WPU accuracy | best baseline | baseline accuracy | margin |
|---|---:|---|---:|---:|
| `catch_heavy` | 0.900000 | `graph-transformer` | 0.808333 | +0.091667 |
| `edge_catch_heavy` | 0.416667 | `graph-transformer` | 0.408333 | +0.008334 |
| `edge_high_force` | 0.658333 | `graph-transformer` | 0.491667 | +0.166666 |
| `edge_shift` | 0.583333 | `graph-transformer` | 0.475000 | +0.108333 |
| `high_force` | 0.716667 | `graph-transformer` | 0.625000 | +0.091667 |
| `no_catch` | 0.533333 | `graph-transformer` | 0.475000 | +0.058333 |
| `nominal` | 0.700000 | `graph-transformer` | 0.633333 | +0.066667 |

The 3-seed N=1029 screen gives win/tie/loss `7/0/0` against the best baseline,
with mean margin `+0.084524`.

### 5-seed expansion

| model | macro branch accuracy | ECE | dense compute ratio |
|---|---:|---:|---:|
| `wpu-cws-indexed-mechanism-relation` | 0.639286 | 0.257334 | 0.000000 |
| `graph-transformer` | 0.577143 | 0.263510 | 1.000000 |
| `serialized-token` | 0.515714 | 0.206709 | 1.000000 |

Per-mechanism comparison against the best non-WPU baseline:

| eval mechanism | WPU accuracy | best baseline | baseline accuracy | margin |
|---|---:|---|---:|---:|
| `catch_heavy` | 0.890000 | `serialized-token` | 0.750000 | +0.140000 |
| `edge_catch_heavy` | 0.435000 | `graph-transformer` | 0.420000 | +0.015000 |
| `edge_high_force` | 0.665000 | `graph-transformer` | 0.555000 | +0.110000 |
| `edge_shift` | 0.595000 | `graph-transformer` | 0.555000 | +0.040000 |
| `high_force` | 0.725000 | `graph-transformer` | 0.630000 | +0.095000 |
| `no_catch` | 0.490000 | `graph-transformer` | 0.575000 | -0.085000 |
| `nominal` | 0.675000 | `graph-transformer` | 0.635000 | +0.040000 |

The 5-seed N=1029 expansion gives win/tie/loss `6/0/1` against the best
baseline, with mean margin `+0.050714`. The remaining negative mechanism is
`no_catch`.

## Interpretation

This strengthens the large-state part of the WPU claim. The relation-conditioned
WPU route keeps zero dense fallback and stays positive under the 5-seed N=1029
expansion, while dense graph/token baselines remain below it as the number of
non-causal background objects grows.

The result should be interpreted narrowly. It shows robustness to larger
non-causal distractor state when the causal working set remains small and
identifiable before tensorization. It does not prove broad large-N superiority,
long-horizon stability, real-world grounding, or calibration dominance. The
`no_catch` loss also shows that sparse relation propagation still needs better
mechanism/prior handling. The next required extensions are N=2053 or larger,
long-horizon rollout, and calibration-aware evaluation.

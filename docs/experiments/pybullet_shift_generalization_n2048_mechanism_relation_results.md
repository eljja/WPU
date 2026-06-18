# N=2048 Relation-Conditioned Sparse Propagation Audit

This audit extends the relation-conditioned sparse propagation large-state
screen to `background_objects=2048`, raising total objects to `N=2053`. It keeps
the same PyBullet cup/table/hand/edge mechanism protocol used in the N=512 and
N=1024 audits.

Source CSV:

- `docs/experiments/pybullet_shift_generalization_n2048_mechanism_relation_trainpool40_steps16_samples40_3seed.csv`

## Protocol

- Domain: PyBullet cup/table/hand/edge branch prediction.
- Training mechanisms: `nominal`, `high_force`, `edge_shift`, `catch_heavy`,
  `no_catch`.
- Evaluation mechanisms: the five training families plus `edge_high_force` and
  `edge_catch_heavy`.
- World size: `background_objects=2048`, total objects `N=2053`.
- Stress setting: `train_samples_per_mechanism=40`, `steps=16`, `samples=40`.
- Seeds: `11`, `13`, `17`.
- Models: `wpu-cws-indexed-mechanism-relation`, `graph-transformer`,
  `serialized-token`.

## Results

| model | macro branch accuracy | ECE | dense compute ratio |
|---|---:|---:|---:|
| `wpu-cws-indexed-mechanism-relation` | 0.644048 | 0.267468 | 0.000000 |
| `graph-transformer` | 0.516667 | 0.236740 | 1.000000 |
| `serialized-token` | 0.484524 | 0.217951 | 1.000000 |

Per-mechanism comparison against the best non-WPU baseline:

| eval mechanism | WPU accuracy | best baseline | baseline accuracy | margin |
|---|---:|---|---:|---:|
| `catch_heavy` | 0.900000 | `serialized-token` | 0.691667 | +0.208333 |
| `edge_catch_heavy` | 0.416667 | `graph-transformer` | 0.400000 | +0.016667 |
| `edge_high_force` | 0.658333 | `graph-transformer` | 0.575000 | +0.083333 |
| `edge_shift` | 0.583333 | `serialized-token` | 0.441667 | +0.141666 |
| `high_force` | 0.716667 | `graph-transformer` | 0.625000 | +0.091667 |
| `no_catch` | 0.533333 | `graph-transformer` | 0.475000 | +0.058333 |
| `nominal` | 0.700000 | `serialized-token` | 0.541667 | +0.158333 |

The N=2053 3-seed screen gives win/tie/loss `7/0/0` against the best baseline,
with mean margin `+0.108333`.

## Interpretation

This is the strongest distractor-scaling evidence so far. The WPU route keeps
zero dense fallback and stable macro accuracy while dense graph/token baselines
fall as non-causal background state grows from N=517 to N=1029 to N=2053.

The claim remains bounded. This does not show that WPU is universally better at
large N. It shows that when objectification exposes a small, identifiable causal
working set before tensorization, relation-conditioned sparse propagation can
ignore large non-causal state that dense/token baselines still process. The next
test should move from non-causal distractors to harder causal large-N settings:
longer causal chains, multiple interacting cups/objects, larger or changing
working sets, and long-horizon rollout.

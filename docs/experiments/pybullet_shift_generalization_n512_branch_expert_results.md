# N=512 Branch-Specific Expert Audit

This audit tests whether the mechanism-branch stress failure can be fixed by replacing a single additive branch correction head with branch-specific transition experts. The new `wpu-cws-indexed-mechanism-branch-expert` model keeps indexed sparse execution and zero dense fallback, but gives each branch a learned branch query and a branch-specific expert logit correction.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_trainpool40_steps16_samples40_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_expert_trainpool40_steps16_samples40_3seed.csv`

## Protocol

- Domain: PyBullet cup/table/hand/edge branch prediction.
- Training mechanisms: `nominal`, `high_force`, `edge_shift`, `catch_heavy`, `no_catch`.
- Evaluation mechanisms: the five training families plus `edge_high_force` and `edge_catch_heavy`.
- World size: `background_objects=512`, total objects `N=517`.
- Stress setting: `train_samples_per_mechanism=40`, `steps=16`, `samples=40`.
- Seeds: `11`, `13`, `17`.
- Hidden size: `32`.

## Result

Branch-specific experts are a negative standalone fix.

| model | macro branch accuracy | ECE | dense compute ratio |
|---|---:|---:|---:|
| `graph-transformer` | 0.598810 | 0.265100 | 1.000000 |
| `serialized-token` | 0.526190 | 0.212392 | 1.000000 |
| `wpu-cws-indexed-mechanism-branch` | 0.534524 | 0.187826 | 0.000000 |
| `wpu-cws-indexed-mechanism-branch-expert` | 0.505952 | 0.191405 | 0.000000 |

Per-mechanism comparison against the best h32 non-WPU baseline:

| mechanism | expert WPU | prior branch WPU | best baseline | expert margin |
|---|---:|---:|---:|---:|
| `edge_catch_heavy` | 0.475000 | 0.441667 | 0.400000 | +0.075000 |
| `edge_shift` | 0.558333 | 0.491667 | 0.500000 | +0.058333 |
| `no_catch` | 0.500000 | 0.483333 | 0.466667 | +0.033333 |
| `edge_high_force` | 0.533333 | 0.566667 | 0.641667 | -0.108334 |
| `nominal` | 0.500000 | 0.558333 | 0.650000 | -0.150000 |
| `high_force` | 0.416667 | 0.491667 | 0.650000 | -0.233333 |
| `catch_heavy` | 0.558333 | 0.708333 | 0.883333 | -0.325000 |

The expert improves some edge/catch composition cases, but the macro score falls below the prior mechanism-branch head and the dense graph baseline remains ahead.

## Interpretation

The failure is informative. Simply adding branch-specific output experts increases freedom at the branch-logit layer, but it does not improve the underlying local physical transition. The model appears to trade general mechanism accuracy for a few composed-edge gains.

The next fix should move below branch logits into the sparse state update itself: relation-type-conditioned local messages for `near_edge`, `impulse_source`, `catch_region`, and `on_top_of` relations. In other words, the branch expert should not infer composition only from pooled context; the local propagation path should encode relation-specific causal updates before branch prediction.

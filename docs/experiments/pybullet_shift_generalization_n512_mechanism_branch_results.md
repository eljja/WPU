# N=512 Mechanism-Branch Transition Audit

This audit tests the next architectural step after two negative diagnostics:

- The shuffled factorized mechanism adapter did not beat the graph-transformer baseline at 5 seeds.
- Target-local delta supervision exposed large-N loss dilution, but scalar loss reweighting reduced branch accuracy.

The new model, `wpu-cws-indexed-mechanism-branch`, keeps sparse indexed execution and zero dense fallback, but adds a mechanism-conditioned branch transition head. The head receives the pooled causal working-set state, predicted local delta summary, and explicit route physics features before producing branch-logit corrections. This tests whether branch outcomes need mechanism-specific transition dynamics rather than only better next-state vector supervision.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_multitrain_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_multitrain_5seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_factorized_shuffled_multitrain_5seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_target_local_loss_multitrain_5seed.csv`

## Protocol

- Domain: PyBullet cup/table/hand/edge branch prediction.
- Training mechanisms: `nominal`, `high_force`, `edge_shift`, `catch_heavy`, `no_catch`.
- Evaluation mechanisms: the five training families plus composed shifts `edge_high_force` and `edge_catch_heavy`.
- World size: `background_objects=512`, total objects `N=517`.
- Seeds: `11`, `13`, `17`, `19`, `23`.
- Models: `wpu-cws-indexed-mechanism-branch`, `graph-transformer`, `serialized-token`.

## Result

At 5 seeds, mechanism-branch WPU is a positive screen:

| model | macro branch accuracy | ECE | dense compute ratio |
|---|---:|---:|---:|
| `wpu-cws-indexed-mechanism-branch` | 0.568571 | 0.247101 | 0.000000 |
| `graph-transformer` | 0.548571 | 0.254194 | 1.000000 |
| `serialized-token` | 0.394286 | 0.256186 | 1.000000 |

Per-mechanism comparison against the best non-WPU baseline:

| mechanism | WPU accuracy | best baseline accuracy | margin |
|---|---:|---:|---:|
| `nominal` | 0.660000 | 0.570000 | +0.090000 |
| `no_catch` | 0.590000 | 0.470000 | +0.120000 |
| `high_force` | 0.530000 | 0.520000 | +0.010000 |
| `edge_catch_heavy` | 0.470000 | 0.450000 | +0.020000 |
| `catch_heavy` | 0.690000 | 0.720000 | -0.030000 |
| `edge_shift` | 0.510000 | 0.540000 | -0.030000 |
| `edge_high_force` | 0.530000 | 0.570000 | -0.040000 |

The 5-seed win/tie/loss is `4/0/3`, with mean margin `+0.020000`. The 3-seed screen was stronger (`6/0/1`, macro WPU `0.626191` versus graph-transformer `0.540476`), but the 5-seed result is the more conservative evidence.

## Interpretation

This is the first large-N mechanism-composition screen that survives the shuffled multi-mechanism protocol while preserving sparse execution. It supports the architectural hypothesis from the target-local loss audit: branch outcomes require mechanism-conditioned transition dynamics, not only local delta accuracy.

The result is still not a broad WPU superiority claim. Three mechanisms remain negative against the best dense baseline, and the experiment uses a synthetic PyBullet domain with small models and short training. The defensible claim is narrower: when a small causal working set is identifiable at large `N`, adding explicit mechanism-conditioned branch transition dynamics can recover accuracy while keeping dense compute at zero.

## Next Step

The next priority is to stress this result beyond the current short training screen:

- increase step budget and sample count to test whether the margin persists;
- run a wider `N` sweep to verify that the benefit is tied to large-N sparse execution;
- add calibration or branch-prior controls so the branch head does not become a hidden prior shortcut;
- test whether the same mechanism-branch design improves long-horizon rollout consistency.

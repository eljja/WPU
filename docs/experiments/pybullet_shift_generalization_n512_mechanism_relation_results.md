# N=512 Relation-Conditioned Sparse Propagation Audit

This audit tests the architectural conclusion from the branch-expert negative result: the missing mechanism is not branch-logit capacity alone, but relation-conditioned local propagation. The new `wpu-cws-indexed-mechanism-relation` model keeps indexed sparse execution and zero dense fallback, applies mechanism-conditioned object modulation, then scatters learned messages over selected working-set relations. The message input includes source hidden state, target hidden state, relation features, and route physics features.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_trainpool40_steps16_samples40_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_expert_trainpool40_steps16_samples40_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_relation_trainpool40_steps16_samples40_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_relation_trainpool40_steps16_samples40_5seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_relation_h64_trainpool40_steps16_samples40_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_baselines_h64_trainpool40_steps16_samples40_3seed.csv`

## Protocol

- Domain: PyBullet cup/table/hand/edge branch prediction.
- Training mechanisms: `nominal`, `high_force`, `edge_shift`, `catch_heavy`, `no_catch`.
- Evaluation mechanisms: the five training families plus `edge_high_force` and `edge_catch_heavy`.
- World size: `background_objects=512`, total objects `N=517`.
- Stress setting: `train_samples_per_mechanism=40`, `steps=16`, `samples=40`.
- Seeds: primary 3-seed screen uses `11`, `13`, `17`; the h32 expansion adds
  `19`, `23` for 5-seed evidence.

## Results

### Hidden size 32, 3 seeds

| model | macro branch accuracy | ECE | dense compute ratio |
|---|---:|---:|---:|
| `wpu-cws-indexed-mechanism-relation` | 0.644048 | 0.267468 | 0.000000 |
| `graph-transformer` | 0.598810 | 0.265100 | 1.000000 |
| `serialized-token` | 0.526190 | 0.212392 | 1.000000 |
| `wpu-cws-indexed-mechanism-branch` | 0.534524 | 0.187826 | 0.000000 |
| `wpu-cws-indexed-mechanism-branch-expert` | 0.505952 | 0.191405 | 0.000000 |

At h32, relation-conditioned WPU wins all seven mechanisms against the best h32 non-WPU baseline: win/tie/loss `7/0/0`, mean margin `+0.045238`.

### Hidden size 32, 5 seeds

| model | macro branch accuracy | ECE | dense compute ratio |
|---|---:|---:|---:|
| `wpu-cws-indexed-mechanism-relation` | 0.639286 | 0.257334 | 0.000000 |
| `graph-transformer` | 0.597143 | 0.257880 | 1.000000 |
| `serialized-token` | 0.518571 | 0.203011 | 1.000000 |

The 5-seed expansion remains positive but is more conservative than the 3-seed screen. Relation-conditioned WPU wins/ties/loses `5/0/2` against the best baseline, with mean margin `+0.042143`. The remaining negative mechanisms are `no_catch` (`-0.035000`) and `nominal` (`-0.015000`).

### Hidden size 64

| model | macro branch accuracy | ECE | dense compute ratio |
|---|---:|---:|---:|
| `wpu-cws-indexed-mechanism-relation` | 0.678571 | 0.285929 | 0.000000 |
| `serialized-token` | 0.622619 | 0.262267 | 1.000000 |
| `graph-transformer` | 0.582143 | 0.249364 | 1.000000 |

At h64, relation-conditioned WPU also beats the best h64 baseline in macro accuracy, with win/tie/loss `4/1/2` and mean margin `+0.021428`. The remaining negative mechanisms are `edge_catch_heavy` and `edge_shift`; calibration is also weaker than the best baseline.

## Interpretation

This is the strongest current evidence for the WPU v2 direction. Output-only branch experts failed, but relation-conditioned sparse propagation succeeds under the same stress protocol. The result supports a more precise claim: for objectified world state, the critical primitive is not attention over all tokens, nor only branch-level classification, but local relation-conditioned state propagation over a small causal working set.

The claim remains bounded. The 5-seed h32 expansion strengthens the evidence, but the benchmark is still PyBullet synthetic and single-step. It should be expanded to larger `N`, calibration-aware evaluation, and long-horizon rollout. Still, it materially improves the WPU story because accuracy improves while dense compute remains exactly zero.

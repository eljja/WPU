# N=4096 Relation-Conditioned Sparse Propagation Boundary

This audit extends the relation-conditioned sparse propagation distractor
screen to `background_objects=4096`, raising total objects to `N=4101`.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization_n4096_mechanism_relation_trainpool40_steps16_samples40_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n4096_baseline_feasibility_smoke.csv`

## Protocol

- Domain: PyBullet cup/table/hand/edge branch prediction.
- Training mechanisms: `nominal`, `high_force`, `edge_shift`, `catch_heavy`,
  `no_catch`.
- Evaluation mechanisms: the five training families plus `edge_high_force` and
  `edge_catch_heavy`.
- World size: `background_objects=4096`, total objects `N=4101`.
- Stress setting for the WPU run: `train_samples_per_mechanism=40`,
  `steps=16`, `samples=40`.
- Seeds for the WPU run: `11`, `13`, `17`.
- WPU model: `wpu-cws-indexed-mechanism-relation`.

The matched dense/token sweep was attempted with `graph-transformer` and
`serialized-token`, but did not complete under the full stress protocol after
the WPU rows were written. A separate minimal CPU baseline smoke was therefore
run only to check feasibility, not to provide comparable accuracy evidence.

## WPU Results

| model | macro branch accuracy | ECE | selected K | dense compute ratio |
|---|---:|---:|---:|---:|
| `wpu-cws-indexed-mechanism-relation` | 0.638095 | 0.187201 | 4.446429 | 0.000000 |

Per-mechanism WPU summary:

| eval mechanism | branch accuracy | majority accuracy | ECE | selected K |
|---|---:|---:|---:|---:|
| `catch_heavy` | 0.683333 | 0.783333 | 0.240258 | 4.816667 |
| `edge_catch_heavy` | 0.508333 | 0.583333 | 0.254894 | 4.816667 |
| `edge_high_force` | 0.725000 | 0.583333 | 0.185066 | 4.366667 |
| `edge_shift` | 0.633333 | 0.558333 | 0.181751 | 4.366667 |
| `high_force` | 0.633333 | 0.400000 | 0.177789 | 4.366667 |
| `no_catch` | 0.625000 | 0.683333 | 0.114365 | 4.025000 |
| `nominal` | 0.658333 | 0.458333 | 0.156283 | 4.366667 |

## Baseline Feasibility Smoke

The minimal CPU smoke used one seed, one training step, one eval mechanism,
eight eval samples, `hidden_dim=16`, one layer, two heads, and batch size `2`.
It completed for `graph-transformer` and `serialized-token`, confirming that
the code path is feasible, but the result is not comparable to the WPU stress
protocol.

| model | eval mechanism | branch accuracy | ECE | dense compute ratio |
|---|---|---:|---:|---:|
| `graph-transformer` | `nominal` | 0.625000 | 0.317135 | 1.000000 |
| `serialized-token` | `nominal` | 0.625000 | 0.207959 | 1.000000 |

## Interpretation

This is a systems boundary result, not a baseline victory. It shows that the
relation-conditioned WPU sparse path can still operate with a small selected
working set at `N=4101`, while the full matched dense/token comparison is not
yet available in this environment.

The claim remains bounded: WPU is not proven better at all large world sizes.
The positive evidence is that explicit objectification plus relation-conditioned
sparse propagation keeps compute tied to the causal working set rather than the
total number of non-causal distractor objects. The unresolved issue is whether
the same advantage remains under baseline-complete large-N experiments and
under harder causal large-N settings where the working set itself grows.

## Next Step

- Add a resumed or checkpointed large-N benchmark runner so dense/token baselines
  can be executed separately without losing completed WPU rows.
- Report large-N runs as three separate categories: baseline-complete,
  WPU-only sparse feasibility, and baseline feasibility smoke.
- Move from non-causal distractors to causal large-N stress: multiple cups,
  longer relation chains, changing working sets, and long-horizon rollouts.

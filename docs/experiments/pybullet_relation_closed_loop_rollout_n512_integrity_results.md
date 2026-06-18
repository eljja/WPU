# PyBullet Relation-Conditioned Closed-Loop Rollout Audit

This audit tests whether the strongest current sparse WPU route,
`wpu-cws-indexed-mechanism-relation`, remains stable when its predicted
`DeltaState` is repeatedly overlaid without simulator resynchronization. It is
not a physics-accuracy benchmark. It is a state-integrity diagnostic: repeated
world-state updates should remain finite, bounded, and branch-stable if WPU is
to claim persistent state as more than a one-step execution primitive.

Source CSVs:

- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_multihorizon_4_8_12_w1_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_multihorizon_4_8_12_w5_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_multihorizon_4_8_12_w1_clip1_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_train_stride4_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_train_stride8_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_train_stride4_delta1_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_train_stride4_branch01_delta1_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_delta_scale025_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_delta_scale010_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_delta_norm_strong_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_consistency_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_validity_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_finite_projection_3seed.csv`

Derived CSV:

- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_integrity_summary.csv`

## Summary

| run | model | H | violations/step | delta norm | flip rate | reject rate | correction rate | corrected objects | rollback rate | escalation rate | escalation success | integrity score | low-disruption score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| relation_consistency | wpu-cws-indexed-mechanism-relation | 5 | 4.354167 | 1000000000.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.100000 | 0.100000 |
| relation_consistency | wpu-cws-indexed-mechanism-relation | 10 | 4.354167 | 1000000000.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.100000 | 0.100000 |
| relation_consistency | wpu-cws-indexed-mechanism-relation | 25 | 4.354167 | 1000000000.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.100000 | 0.100000 |
| relation_delta_norm_strong | wpu-cws-indexed-mechanism-relation | 5 | 0.166667 | 0.838266 | 0.062500 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.866494 | 0.866494 |
| relation_delta_norm_strong | wpu-cws-indexed-mechanism-relation | 10 | 0.366667 | 3.804898 | 0.166667 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.631829 | 0.631829 |
| relation_delta_norm_strong | wpu-cws-indexed-mechanism-relation | 25 | 3.115833 | 2421948.183622 | 0.064236 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.087153 | 0.087153 |
| relation_delta_scale010 | wpu-cws-indexed-mechanism-relation | 5 | 0.000000 | 0.173915 | 0.026042 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.988705 | 0.988705 |
| relation_delta_scale010 | wpu-cws-indexed-mechanism-relation | 10 | 0.102083 | 0.461472 | 0.138889 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.899925 | 0.899925 |
| relation_delta_scale010 | wpu-cws-indexed-mechanism-relation | 25 | 2.646667 | 245997.938780 | 0.052083 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.089583 | 0.089583 |
| relation_delta_scale025 | wpu-cws-indexed-mechanism-relation | 5 | 0.033333 | 0.417601 | 0.041667 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.958717 | 0.958717 |
| relation_delta_scale025 | wpu-cws-indexed-mechanism-relation | 10 | 0.172917 | 1.141403 | 0.141204 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.836706 | 0.836706 |
| relation_delta_scale025 | wpu-cws-indexed-mechanism-relation | 25 | 2.849167 | 610325.001919 | 0.052952 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.089410 | 0.089410 |
| relation_finite_projection | wpu-cws-indexed-mechanism-relation | 5 | 0.000000 | 1.643664 | 0.078125 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.926847 | 0.926847 |
| relation_finite_projection | wpu-cws-indexed-mechanism-relation | 10 | 0.000000 | 2.483041 | 0.113426 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.890408 | 0.890408 |
| relation_finite_projection | wpu-cws-indexed-mechanism-relation | 25 | 0.200833 | 4.051999 | 0.043403 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.739041 | 0.739041 |
| relation_multihorizon_w1 | wpu-cws-indexed-mechanism-relation | 5 | 4.458333 | 1000000000.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.100000 | 0.100000 |
| relation_multihorizon_w1 | wpu-cws-indexed-mechanism-relation | 10 | 4.458333 | 1000000000.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.100000 | 0.100000 |
| relation_multihorizon_w1 | wpu-cws-indexed-mechanism-relation | 25 | 4.458333 | 1000000000.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.100000 | 0.100000 |
| relation_multihorizon_w1_clip1 | wpu-cws-indexed-mechanism-relation | 5 | 4.458333 | 1000000000.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.100000 | 0.100000 |
| relation_multihorizon_w1_clip1 | wpu-cws-indexed-mechanism-relation | 10 | 4.458333 | 1000000000.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.100000 | 0.100000 |
| relation_multihorizon_w1_clip1 | wpu-cws-indexed-mechanism-relation | 25 | 4.458333 | 1000000000.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.100000 | 0.100000 |
| relation_multihorizon_w5 | wpu-cws-indexed-mechanism-relation | 5 | 4.458333 | 1000000000.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.100000 | 0.100000 |
| relation_multihorizon_w5 | wpu-cws-indexed-mechanism-relation | 10 | 4.458333 | 1000000000.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.100000 | 0.100000 |
| relation_multihorizon_w5 | wpu-cws-indexed-mechanism-relation | 25 | 4.458333 | 1000000000.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.100000 | 0.100000 |
| relation_raw | graph-transformer | 5 | 0.170833 | 23.833869 | 0.255208 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.505000 | 0.505000 |
| relation_raw | graph-transformer | 10 | 0.300000 | 26.306219 | 0.300926 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.424815 | 0.424815 |
| relation_raw | graph-transformer | 25 | 6.304167 | 30.461124 | 0.112847 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.077431 | 0.077431 |
| relation_raw | serialized-token | 5 | 0.187500 | 21.977465 | 0.067708 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.533333 | 0.533333 |
| relation_raw | serialized-token | 10 | 0.427083 | 28.337312 | 0.150463 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.385012 | 0.385012 |
| relation_raw | serialized-token | 25 | 0.554167 | 27.408027 | 0.056424 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.333924 | 0.333924 |
| relation_raw | wpu-cws-indexed-mechanism-relation | 5 | 0.379167 | 1.662182 | 0.088542 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.715574 | 0.715574 |
| relation_raw | wpu-cws-indexed-mechanism-relation | 10 | 0.483333 | 4.552682 | 0.111111 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.552601 | 0.552601 |
| relation_raw | wpu-cws-indexed-mechanism-relation | 25 | 3.180833 | 2379159.471470 | 0.043403 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.091319 | 0.091319 |
| relation_train_stride4 | wpu-cws-indexed-mechanism-relation | 5 | 0.379167 | 2.053519 | 0.067708 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.706043 | 0.706043 |
| relation_train_stride4 | wpu-cws-indexed-mechanism-relation | 10 | 0.541667 | 5.267674 | 0.104167 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.496881 | 0.496881 |
| relation_train_stride4 | wpu-cws-indexed-mechanism-relation | 25 | 3.236667 | 2425517.613951 | 0.052952 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.089410 | 0.089410 |
| relation_train_stride4_branch01_delta1 | wpu-cws-indexed-mechanism-relation | 5 | 0.212500 | 1.608456 | 0.114583 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.803912 | 0.803912 |
| relation_train_stride4_branch01_delta1 | wpu-cws-indexed-mechanism-relation | 10 | 0.497917 | 4.672586 | 0.136574 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.535290 | 0.535290 |
| relation_train_stride4_branch01_delta1 | wpu-cws-indexed-mechanism-relation | 25 | 3.215000 | 2455636.864591 | 0.065104 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.086979 | 0.086979 |
| relation_train_stride4_delta1 | wpu-cws-indexed-mechanism-relation | 5 | 0.287500 | 1.663140 | 0.140625 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.755540 | 0.755540 |
| relation_train_stride4_delta1 | wpu-cws-indexed-mechanism-relation | 10 | 0.554167 | 4.710854 | 0.157407 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.498847 | 0.498847 |
| relation_train_stride4_delta1 | wpu-cws-indexed-mechanism-relation | 25 | 3.237500 | 2460938.605532 | 0.073785 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.085243 | 0.085243 |
| relation_train_stride8 | wpu-cws-indexed-mechanism-relation | 5 | 0.383333 | 1.776338 | 0.140625 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.698870 | 0.698870 |
| relation_train_stride8 | wpu-cws-indexed-mechanism-relation | 10 | 0.543750 | 5.025681 | 0.141204 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.496798 | 0.496798 |
| relation_train_stride8 | wpu-cws-indexed-mechanism-relation | 25 | 3.257500 | 2424016.166901 | 0.065972 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.086806 | 0.086806 |
| relation_validity | wpu-cws-indexed-mechanism-relation | 5 | 0.358333 | 1.666083 | 0.088542 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.726895 | 0.726895 |
| relation_validity | wpu-cws-indexed-mechanism-relation | 10 | 0.472917 | 4.551994 | 0.111111 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.558354 | 0.558354 |
| relation_validity | wpu-cws-indexed-mechanism-relation | 25 | 3.176667 | 2379929.526420 | 0.043403 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.091319 | 0.091319 |

## Interpretation

The audit confirms that one-step branch accuracy is not enough for a
world-state processor. The relation-conditioned WPU route is strong in one-step
and distractor-scaling screens, but raw H=25 rollout integrity is only
`0.091319` despite selected `K = 4.354167`. The failure is not caused by full
graph recomputation cost; it is a learned-transition stability failure.

Simple scalar fixes do not solve it. Strong delta-norm regularization reaches
H=25 integrity `0.087153`, state-validity training remains at `0.091319`, and
rollout-consistency training produces non-finite deltas that are penalized as
delta norm `1000000000.000000`. Temporal delta scaling also fails as a standalone
fix: scale `0.25` reaches H=25 integrity `0.089410`, and scale `0.10` reaches
`0.089583`. Scaling reduces average delta magnitude but does not convert a
one-step target into a stable multi-step transition operator.

Short-stride simulator targets are also negative in this implementation.
Training the relation WPU on shorter supervised simulator transitions
(`train_sim_steps=4` or `8`, evaluated against `sim_steps=80`) gives H=25
integrity `0.089410` and `0.086806`, respectively. Making the short-stride run
delta-focused does not help: stride-4 delta-only training reaches `0.085243`,
and branch `0.1` plus delta `1.0` reaches `0.086979`.

The first explicit multi-horizon simulator-resynchronized target is also
negative with the current transition head. Supervising simulator horizons
`4/8/12` with weight `1.0` or `5.0` produces non-finite long-horizon deltas
penalized as H=25 integrity `0.100000`; gradient clipping at `1.0` does not
change that outcome. This narrows the next step: WPU needs a transition
architecture or training objective that is stable under repeated application,
not just more one-step targets attached to the same head.

Finite projection lifts applied-state H=25 integrity to `0.739041` and keeps
delta norm bounded at `4.051999`. This is useful as a state-store safety guard,
but it is not evidence of learned long-horizon dynamics. Future WPU claims must
separate raw model deltas from guarded state-store deltas.

This makes state integrity a first-class WPU metric:

```text
state-integrity = constraint validity + bounded delta drift + branch stability
```

Future WPU rollout claims should report this score or its components next to
accuracy and latency. The next architecture/training step is a genuinely stable
recurrent/local transition operator, likely with bounded delta parameterization
or simulator-resynchronized rollout training that constrains every unrolled
step.

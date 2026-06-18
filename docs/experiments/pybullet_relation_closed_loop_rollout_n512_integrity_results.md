# PyBullet Relation-Conditioned Closed-Loop Rollout Audit

This audit tests whether the strongest current one-step route,
`wpu-cws-indexed-mechanism-relation`, remains stable under repeated
`DeltaState` overlays. It does not resynchronize to the simulator; it evaluates
whether objectified state remains within simple validity bounds.

Source CSVs:

- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_3seed.csv`
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
| relation_finite_projection | wpu-cws-indexed-mechanism-relation | 5 | 0.000000 | 1.643664 | 0.078125 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.926847 | 0.926847 |
| relation_finite_projection | wpu-cws-indexed-mechanism-relation | 10 | 0.000000 | 2.483041 | 0.113426 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.890408 | 0.890408 |
| relation_finite_projection | wpu-cws-indexed-mechanism-relation | 25 | 0.200833 | 4.051999 | 0.043403 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.739041 | 0.739041 |
| relation_raw | graph-transformer | 5 | 0.170833 | 23.833869 | 0.255208 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.505000 | 0.505000 |
| relation_raw | graph-transformer | 10 | 0.300000 | 26.306219 | 0.300926 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.424815 | 0.424815 |
| relation_raw | graph-transformer | 25 | 6.304167 | 30.461124 | 0.112847 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.077431 | 0.077431 |
| relation_raw | serialized-token | 5 | 0.187500 | 21.977465 | 0.067708 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.533333 | 0.533333 |
| relation_raw | serialized-token | 10 | 0.427083 | 28.337312 | 0.150463 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.385012 | 0.385012 |
| relation_raw | serialized-token | 25 | 0.554167 | 27.408027 | 0.056424 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.333924 | 0.333924 |
| relation_raw | wpu-cws-indexed-mechanism-relation | 5 | 0.379167 | 1.662182 | 0.088542 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.715574 | 0.715574 |
| relation_raw | wpu-cws-indexed-mechanism-relation | 10 | 0.483333 | 4.552682 | 0.111111 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.552601 | 0.552601 |
| relation_raw | wpu-cws-indexed-mechanism-relation | 25 | 3.180833 | 2379159.471470 | 0.043403 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.091319 | 0.091319 |
| relation_validity | wpu-cws-indexed-mechanism-relation | 5 | 0.358333 | 1.666083 | 0.088542 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.726895 | 0.726895 |
| relation_validity | wpu-cws-indexed-mechanism-relation | 10 | 0.472917 | 4.551994 | 0.111111 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.558354 | 0.558354 |
| relation_validity | wpu-cws-indexed-mechanism-relation | 25 | 3.176667 | 2379929.526420 | 0.043403 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.091319 | 0.091319 |

## Interpretation

The audit confirms that one-step branch accuracy is not enough for a
world-state processor. Relation-conditioned WPU keeps a small selected `K` and
is stronger than dense/token baselines at H=5 and H=10, but raw H=25 deltas
explode. The raw relation WPU H=25 integrity score is `0.091319`, with mean
delta norm `2379159.471470`.

The first learned-stability ablations do not solve the collapse. Strong
delta-norm regularization reaches H=25 integrity `0.087153`; state-validity
training remains at `0.091319`; rollout-consistency training produces non-finite
delta behavior and is penalized to delta norm `1000000000.000000`. These are
negative diagnostics, not improvements.

Finite delta clamp plus integrity projection protects the applied state and
lifts relation WPU H=25 integrity to `0.739041`, with applied delta norm
`4.051999`. This is useful as a state-store safety guard, but it is not learned
long-horizon dynamics: raw model deltas can still be unstable.

This makes state integrity a first-class WPU metric:

```text
state-integrity = constraint validity + bounded delta drift + branch stability
```

Future WPU rollout claims should report this score or its components next
to accuracy and latency.

The next architecture step is multi-step or simulator-resynchronized transition
training, not another one-step classifier head or a simple scalar regularizer.

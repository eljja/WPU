# PyBullet Relation-Conditioned Closed-Loop Rollout Audit

This audit tests whether the strongest current relation-conditioned WPU route,
`wpu-cws-indexed-mechanism-relation`, remains stable under repeated
`DeltaState` overlays at large distractor count (`background_objects=512`,
total `N=517`). It now reports both state-integrity and
simulator-resynchronized trajectory metrics. The trajectory metrics compare the
H-step accumulated WPU state against a PyBullet target generated from the same
initial condition at `H * sim_steps`.

Source CSVs:

- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta05_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta025_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta01_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta005_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_finite_projection_3seed.csv`

Derived CSV:

- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_integrity_summary.csv`

## Summary

| run | model | H | violations/step | delta norm | flip rate | branch acc | trajectory MSE | target-object MSE | reject rate | correction rate | rollback rate | escalation rate | integrity score | low-disruption score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| relation_bounded_delta005 | wpu-cws-indexed-mechanism-relation | 5 | 0.000000 | 0.235083 | 0.020833 | 0.833333 | 0.012121 | 6.089423 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.987605 | 0.987605 |
| relation_bounded_delta005 | wpu-cws-indexed-mechanism-relation | 10 | 0.000000 | 0.235170 | 0.009259 | 0.833333 | 0.081619 | 41.490147 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.989917 | 0.989917 |
| relation_bounded_delta005 | wpu-cws-indexed-mechanism-relation | 25 | 0.217500 | 0.234325 | 0.009549 | 0.729167 | 0.707117 | 361.358309 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.870264 | 0.870264 |
| relation_bounded_delta01 | wpu-cws-indexed-mechanism-relation | 5 | 0.000000 | 0.429045 | 0.020833 | 0.833333 | 0.013191 | 6.289925 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.980817 | 0.980817 |
| relation_bounded_delta01 | wpu-cws-indexed-mechanism-relation | 10 | 0.179167 | 0.428478 | 0.025463 | 0.729167 | 0.085312 | 42.132881 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.881369 | 0.881369 |
| relation_bounded_delta01 | wpu-cws-indexed-mechanism-relation | 25 | 0.342500 | 0.437645 | 0.015625 | 0.708333 | 0.720660 | 360.705327 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.793182 | 0.793182 |
| relation_bounded_delta025 | wpu-cws-indexed-mechanism-relation | 5 | 0.258333 | 0.872737 | 0.062500 | 0.729167 | 0.017777 | 7.271356 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.814871 | 0.814871 |
| relation_bounded_delta025 | wpu-cws-indexed-mechanism-relation | 10 | 0.377083 | 0.890806 | 0.048611 | 0.583333 | 0.101334 | 44.700227 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.751704 | 0.751704 |
| relation_bounded_delta025 | wpu-cws-indexed-mechanism-relation | 25 | 0.559167 | 0.967416 | 0.028646 | 0.458333 | 0.833795 | 374.035886 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.652870 | 0.652870 |
| relation_bounded_delta05 | wpu-cws-indexed-mechanism-relation | 5 | 0.325000 | 1.307840 | 0.072917 | 0.687500 | 0.024792 | 9.061604 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.760892 | 0.760892 |
| relation_bounded_delta05 | wpu-cws-indexed-mechanism-relation | 10 | 0.441667 | 1.378109 | 0.067129 | 0.520833 | 0.132672 | 51.401310 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.695424 | 0.695424 |
| relation_bounded_delta05 | wpu-cws-indexed-mechanism-relation | 25 | 0.593333 | 1.556419 | 0.036458 | 0.541667 | 1.079404 | 424.235526 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.611900 | 0.611900 |
| relation_finite_projection | wpu-cws-indexed-mechanism-relation | 5 | 0.000000 | 1.749332 | 0.114583 | 0.541667 | 0.033942 | 11.609245 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.915857 | 0.915857 |
| relation_finite_projection | wpu-cws-indexed-mechanism-relation | 10 | 0.000000 | 2.008994 | 0.074074 | 0.458333 | 0.193246 | 66.185466 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.914870 | 0.914870 |
| relation_finite_projection | wpu-cws-indexed-mechanism-relation | 25 | 0.034167 | 2.686625 | 0.052083 | 0.250000 | 1.695024 | 538.635690 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.876760 | 0.876760 |
| relation_raw | wpu-cws-indexed-mechanism-relation | 5 | 0.416667 | 1.783713 | 0.119792 | 0.562500 | 0.035057 | 12.370823 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.684445 | 0.684445 |
| relation_raw | wpu-cws-indexed-mechanism-relation | 10 | 0.441667 | 2.062501 | 0.081018 | 0.479167 | 0.201422 | 71.182410 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.668692 | 0.668692 |
| relation_raw | wpu-cws-indexed-mechanism-relation | 25 | 1.184167 | 5.401195 | 0.052951 | 0.208333 | 6.975125 | 922.699696 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.250368 | 0.250368 |

## Interpretation

The audit confirms that one-step branch accuracy and small selected `K` are not
enough for a world-state processor. The raw relation WPU keeps selected
`K = 4.354167`, but H=25 integrity is only `0.250368`, trajectory MSE is
`6.975125`, target-object trajectory MSE is `922.699696`, and final branch
accuracy is `0.208333`.

Finite projection is a strong memory-safety guard but a weak predictive
rollout. It raises H=25 integrity to `0.876760`, but trajectory MSE is
`1.695024`, target-object trajectory MSE is `538.635690`, and branch accuracy is
only `0.250000`.

Bounded delta parameterization is the first raw transition change that improves
both stability and simulator-resynchronized prediction. At bound `0.05`, H=25
integrity is `0.870264`, trajectory MSE is `0.707117`, target-object trajectory
MSE is `361.358309`, and branch accuracy is `0.729167`. At bound `0.1`, H=25
integrity is lower (`0.793182`) but trajectory MSE remains close (`0.720660`)
and branch accuracy is `0.708333`. These results substantially reduce the
under-update concern: the smallest bound is not merely preserving validity by
doing nothing; it also improves final branch prediction and whole-state
trajectory error relative to raw and finite-projected rollouts.

The remaining weakness is target-object trajectory error. Even the best bounded
run still has large cup-centric error (`361.358309`), so the claim is not
"learned physics is solved." The correct P2 conclusion is narrower: bounded
state-native transition parameterization turns the relation WPU from an
unstable long-horizon model into a usable sparse rollout baseline, but it needs
adaptive per-feature/per-relation bounds and simulator-resynchronized
trajectory training to become a high-fidelity dynamics model.

This makes state integrity a first-class WPU metric:

```text
state-integrity = constraint validity + bounded delta drift + branch stability
```

Future WPU rollout claims should report this score or its components next
to accuracy and latency.

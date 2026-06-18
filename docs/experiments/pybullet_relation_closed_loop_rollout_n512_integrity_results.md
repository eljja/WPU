# PyBullet State-Integrity Audit

This audit derives long-horizon state-integrity metrics from the PyBullet
closed-loop rollout results. It does not resynchronize to the simulator;
it evaluates whether repeated `DeltaState` overlays keep object state
within simple validity bounds.

Source CSVs:

- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta05_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta025_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta01_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta005_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta005_lr3e4_clip1_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta005_unrolled_h2_4_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_mechanism_target_bounded_delta005_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_adaptive_delta025_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_split_delta_p010_v005_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta005_targetloss1_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_finite_projection_3seed.csv`

Derived CSV:

- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_integrity_summary.csv`

## Summary

| run | model | H | violations/step | delta norm | flip rate | branch acc | trajectory MSE | target-object MSE | target pos MSE | target vel MSE | nonfinite grad | reject rate | correction rate | rollback rate | escalation rate | integrity score | low-disruption score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| relation_adaptive_delta025 | wpu-cws-indexed-mechanism-relation | 5 | 0.041667 | 0.549475 | 0.036458 | 0.770833 | 0.014159 | 6.419266 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.950560 | 0.950560 |
| relation_adaptive_delta025 | wpu-cws-indexed-mechanism-relation | 10 | 0.137500 | 0.549388 | 0.037037 | 0.687500 | 0.088870 | 42.536227 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.897739 | 0.897739 |
| relation_adaptive_delta025 | wpu-cws-indexed-mechanism-relation | 25 | 0.300833 | 0.562927 | 0.033854 | 0.333333 | 0.739266 | 361.306078 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.808068 | 0.808068 |
| relation_bounded_delta005 | wpu-cws-indexed-mechanism-relation | 5 | 0.000000 | 0.235083 | 0.020833 | 0.833333 | 0.012121 | 6.089423 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.987605 | 0.987605 |
| relation_bounded_delta005 | wpu-cws-indexed-mechanism-relation | 10 | 0.000000 | 0.235170 | 0.009259 | 0.833333 | 0.081619 | 41.490147 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.989917 | 0.989917 |
| relation_bounded_delta005 | wpu-cws-indexed-mechanism-relation | 25 | 0.217500 | 0.234325 | 0.009549 | 0.729167 | 0.707117 | 361.358309 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.870264 | 0.870264 |
| relation_bounded_delta005_lr3e4_clip1 | wpu-cws-indexed-mechanism-relation | 5 | 0.000000 | 0.238561 | 0.041667 | 0.687500 | 0.012133 | 6.082024 | 9.192385 | 2.971664 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.983317 | 0.983317 |
| relation_bounded_delta005_lr3e4_clip1 | wpu-cws-indexed-mechanism-relation | 10 | 0.000000 | 0.238644 | 0.037037 | 0.645833 | 0.081700 | 41.463994 | 79.305106 | 3.622881 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.984240 | 0.984240 |
| relation_bounded_delta005_lr3e4_clip1 | wpu-cws-indexed-mechanism-relation | 25 | 0.163333 | 0.237314 | 0.032986 | 0.500000 | 0.711506 | 363.187130 | 721.641456 | 4.732804 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.895263 | 0.895263 |
| relation_bounded_delta005_targetloss1 | wpu-cws-indexed-mechanism-relation | 5 | 0.000000 | 0.239155 | 0.026042 | 0.791667 | 0.012201 | 6.115898 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.986421 | 0.986421 |
| relation_bounded_delta005_targetloss1 | wpu-cws-indexed-mechanism-relation | 10 | 0.000000 | 0.237540 | 0.018518 | 0.812500 | 0.082033 | 41.636998 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.987982 | 0.987982 |
| relation_bounded_delta005_targetloss1 | wpu-cws-indexed-mechanism-relation | 25 | 0.000833 | 0.236673 | 0.017361 | 0.687500 | 0.711838 | 363.368470 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.987786 | 0.987786 |
| relation_bounded_delta005_unrolled_h2_4 | wpu-cws-indexed-mechanism-relation | 5 | 0.000000 | 0.239224 | 0.026042 | 0.666667 | 0.012136 | 6.082119 | 9.192612 | 2.971625 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.986419 | 0.986419 |
| relation_bounded_delta005_unrolled_h2_4 | wpu-cws-indexed-mechanism-relation | 10 | 0.000000 | 0.238967 | 0.030093 | 0.645833 | 0.081705 | 41.465050 | 79.306661 | 3.623439 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.985618 | 0.985618 |
| relation_bounded_delta005_unrolled_h2_4 | wpu-cws-indexed-mechanism-relation | 25 | 0.180000 | 0.237049 | 0.030382 | 0.500000 | 0.710751 | 362.829388 | 720.911891 | 4.746885 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.886627 | 0.886627 |
| relation_bounded_delta01 | wpu-cws-indexed-mechanism-relation | 5 | 0.000000 | 0.429045 | 0.020833 | 0.833333 | 0.013191 | 6.289925 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.980817 | 0.980817 |
| relation_bounded_delta01 | wpu-cws-indexed-mechanism-relation | 10 | 0.179167 | 0.428478 | 0.025463 | 0.729167 | 0.085312 | 42.132881 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.881369 | 0.881369 |
| relation_bounded_delta01 | wpu-cws-indexed-mechanism-relation | 25 | 0.342500 | 0.437645 | 0.015625 | 0.708333 | 0.720660 | 360.705327 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.793182 | 0.793182 |
| relation_bounded_delta025 | wpu-cws-indexed-mechanism-relation | 5 | 0.258333 | 0.872737 | 0.062500 | 0.729167 | 0.017777 | 7.271356 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.814871 | 0.814871 |
| relation_bounded_delta025 | wpu-cws-indexed-mechanism-relation | 10 | 0.377083 | 0.890806 | 0.048611 | 0.583333 | 0.101334 | 44.700227 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.751704 | 0.751704 |
| relation_bounded_delta025 | wpu-cws-indexed-mechanism-relation | 25 | 0.559167 | 0.967416 | 0.028646 | 0.458333 | 0.833795 | 374.035886 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.652870 | 0.652870 |
| relation_bounded_delta05 | wpu-cws-indexed-mechanism-relation | 5 | 0.325000 | 1.307840 | 0.072917 | 0.687500 | 0.024792 | 9.061604 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.760892 | 0.760892 |
| relation_bounded_delta05 | wpu-cws-indexed-mechanism-relation | 10 | 0.441667 | 1.378109 | 0.067129 | 0.520833 | 0.132672 | 51.401310 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.695424 | 0.695424 |
| relation_bounded_delta05 | wpu-cws-indexed-mechanism-relation | 25 | 0.593333 | 1.556419 | 0.036458 | 0.541667 | 1.079404 | 424.235526 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.611900 | 0.611900 |
| relation_finite_projection | wpu-cws-indexed-mechanism-relation | 5 | 0.000000 | 1.749332 | 0.114583 | 0.541667 | 0.033942 | 11.609245 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.915857 | 0.915857 |
| relation_finite_projection | wpu-cws-indexed-mechanism-relation | 10 | 0.000000 | 2.008994 | 0.074074 | 0.458333 | 0.193246 | 66.185466 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.914870 | 0.914870 |
| relation_finite_projection | wpu-cws-indexed-mechanism-relation | 25 | 0.034167 | 2.686625 | 0.052083 | 0.250000 | 1.695024 | 538.635690 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.876760 | 0.876760 |
| relation_mechanism_target_bounded_delta005 | wpu-cws-indexed-mechanism-target | 5 | 0.000000 | 0.237980 | 0.010417 | 0.854167 | 0.011951 | 5.992753 | 9.019743 | 2.965763 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.989587 | 0.989587 |
| relation_mechanism_target_bounded_delta005 | wpu-cws-indexed-mechanism-target | 10 | 0.000000 | 0.237494 | 0.018518 | 0.770833 | 0.080643 | 40.940546 | 78.283679 | 3.597411 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.987984 | 0.987984 |
| relation_mechanism_target_bounded_delta005 | wpu-cws-indexed-mechanism-target | 25 | 0.230833 | 0.235406 | 0.011285 | 0.750000 | 0.699230 | 357.220733 | 709.683800 | 4.757665 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.862545 | 0.862545 |
| relation_raw | wpu-cws-indexed-mechanism-relation | 5 | 0.416667 | 1.783713 | 0.119792 | 0.562500 | 0.035057 | 12.370823 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.684445 | 0.684445 |
| relation_raw | wpu-cws-indexed-mechanism-relation | 10 | 0.441667 | 2.062501 | 0.081018 | 0.479167 | 0.201422 | 71.182410 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.668692 | 0.668692 |
| relation_raw | wpu-cws-indexed-mechanism-relation | 25 | 1.184167 | 5.401195 | 0.052951 | 0.208333 | 6.975125 | 922.699696 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.250368 | 0.250368 |
| relation_split_delta_p010_v005 | wpu-cws-indexed-mechanism-relation | 5 | 0.000000 | 0.357943 | 0.026042 | 0.833333 | 0.012622 | 6.124859 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.982264 | 0.982264 |
| relation_split_delta_p010_v005 | wpu-cws-indexed-mechanism-relation | 10 | 0.100000 | 0.354002 | 0.037037 | 0.687500 | 0.083435 | 41.645770 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.925202 | 0.925202 |
| relation_split_delta_p010_v005 | wpu-cws-indexed-mechanism-relation | 25 | 0.187500 | 0.351914 | 0.023437 | 0.541667 | 0.715922 | 361.626218 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.879871 | 0.879871 |

## Interpretation

The audit confirms that one-step branch accuracy is not enough for a
world-state processor. The sparse WPU path can keep a small selected
`K`, but repeated raw deltas can still create invalid state. Guarded
state-store projection can protect the applied state and lift WPU
H=25 integrity above the dashboard threshold, but it does not prove
the underlying delta model is stable. Future reports must distinguish
raw model deltas from guarded state-store deltas.

When simulator-resynchronized trajectory columns are present, they
compare the accumulated rollout delta against a PyBullet target
generated from the same initial condition at `horizon * sim_steps`.
These metrics prevent a bounded transition from being overclaimed:
high integrity is only useful if trajectory error and final branch
accuracy remain competitive.

The branch-weighted target-local transition head is the first
positive follow-up on the target-object bottleneck. At H=25 it
changes branch accuracy from 0.729167 to 0.750000, trajectory MSE from 0.707117 to 0.699230,
and target-object MSE from 361.358309 to 357.220733. This is a small architectural
improvement, not solved high-fidelity dynamics; target-object position
MSE remains 709.683800.

This makes state integrity a first-class WPU metric:

```text
state-integrity = constraint validity + bounded delta drift + branch stability
```

Future WPU rollout claims should report this score or its components next
to accuracy and latency.

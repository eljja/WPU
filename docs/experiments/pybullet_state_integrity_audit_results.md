# PyBullet State-Integrity Audit

This audit derives long-horizon state-integrity metrics from the PyBullet
closed-loop rollout results. It does not resynchronize to the simulator;
it evaluates whether repeated `DeltaState` overlays keep object state
within simple validity bounds.

Source CSVs:

- `docs/experiments/pybullet_closed_loop_rollout.csv`
- `docs/experiments/pybullet_closed_loop_rollout_clipped.csv`

Derived CSV:

- `docs/experiments/pybullet_state_integrity_audit.csv`

## Summary

| run | model | H | violations/step | delta norm | flip rate | integrity score |
|---|---|---:|---:|---:|---:|---:|
| clipped | graph-transformer | 25 | 0.253333 | 8.363635 | 0.054688 | 0.557002 |
| clipped | wpu-cws-indexed-local-dense | 25 | 0.314167 | 2.809900 | 0.048611 | 0.719139 |
| clipped | wpu-cws-indexed-sparse | 25 | 0.785000 | 1939290.233702 | 0.082465 | 0.201757 |
| raw | graph-transformer | 5 | 0.045833 | 4.162467 | 0.062500 | 0.816605 |
| raw | graph-transformer | 10 | 0.139583 | 6.106724 | 0.150463 | 0.679401 |
| raw | graph-transformer | 25 | 0.272500 | 8.409911 | 0.056424 | 0.544493 |
| raw | wpu-cws-indexed-local-dense | 5 | 0.066667 | 2.788842 | 0.145834 | 0.836557 |
| raw | wpu-cws-indexed-local-dense | 10 | 0.204167 | 2.648499 | 0.148148 | 0.765381 |
| raw | wpu-cws-indexed-local-dense | 25 | 0.499166 | 2.744688 | 0.055556 | 0.618283 |
| raw | wpu-cws-indexed-sparse | 5 | 0.200000 | 0.799330 | 0.125000 | 0.837023 |
| raw | wpu-cws-indexed-sparse | 10 | 0.531250 | 3.534072 | 0.203704 | 0.543379 |
| raw | wpu-cws-indexed-sparse | 25 | 3.374166 | 1958877.607881 | 0.076389 | 0.084722 |

## Interpretation

The audit confirms that one-step branch accuracy is not enough for a
world-state processor. The sparse WPU path can keep a small selected
`K`, but repeated raw deltas can still create invalid state. Delta
clipping improves the integrity score by reducing violations, but it
does not prove the underlying delta model is stable.

This makes state integrity a first-class WPU metric:

```text
state-integrity = constraint validity + bounded delta drift + branch stability
```

Future WPU rollout claims should report this score or its components next
to accuracy and latency.

# PyBullet Closed-Loop Rollout

This experiment tests whether one-step PyBullet state predictions remain stable
when model-predicted deltas are repeatedly applied back into `WorldState`.
It is a state-integrity diagnostic, not a simulator-resynchronized physics
benchmark.

Source CSVs:

- `docs/experiments/pybullet_closed_loop_rollout.csv`
- `docs/experiments/pybullet_closed_loop_rollout_clipped.csv`

## Protocol

- Simulator: PyBullet `DIRECT` rigid-body rollout for initial objectified state.
- Training: clean one-step PyBullet cup samples.
- Rollout: repeatedly apply predicted object deltas to `WorldState`.
- Horizons: `5, 10, 25`.
- Models: `wpu-cws-indexed-sparse`, `wpu-cws-indexed-local-dense`,
  `graph-transformer`.
- Seeds: `11, 13`.
- Background objects: `32`.
- Metrics: branch flip rate, branch entropy, raw predicted delta norm,
  constraint violations per step, selected K.

## Unclipped Rollout Summary

| horizon | model | branch flip | violations/step | entropy | raw delta norm | selected K |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 5 | graph-transformer | 0.063 | 0.046 | 0.858 | 4.163 | 36.417 |
| 5 | wpu-cws-indexed-local-dense | 0.146 | 0.067 | 0.922 | 2.789 | 4.417 |
| 5 | wpu-cws-indexed-sparse | 0.125 | 0.200 | 1.061 | 0.799 | 4.417 |
| 10 | graph-transformer | 0.151 | 0.140 | 0.891 | 6.107 | 36.417 |
| 10 | wpu-cws-indexed-local-dense | 0.148 | 0.204 | 0.951 | 2.649 | 4.417 |
| 10 | wpu-cws-indexed-sparse | 0.204 | 0.531 | 1.026 | 3.534 | 4.417 |
| 25 | graph-transformer | 0.056 | 0.273 | 0.890 | 8.410 | 36.417 |
| 25 | wpu-cws-indexed-local-dense | 0.056 | 0.499 | 0.881 | 2.745 | 4.417 |
| 25 | wpu-cws-indexed-sparse | 0.076 | 3.374 | 0.423 | 1,958,877.608 | 4.417 |

## Delta-Clipped H=25 Summary

The second run applies a per-object delta-vector norm clip of `0.25` before
updating `WorldState`.

| model | unclipped violations/step | clipped violations/step | clipped branch flip | raw delta norm after clipping run |
| --- | ---: | ---: | ---: | ---: |
| graph-transformer | 0.273 | 0.253 | 0.055 | 8.364 |
| wpu-cws-indexed-local-dense | 0.499 | 0.314 | 0.049 | 2.810 |
| wpu-cws-indexed-sparse | 3.374 | 0.785 | 0.083 | 1,939,290.234 |

## Interpretation

The result is a negative but important WPU finding. One-step WPU sparse can look
reasonable, but repeated delta application can produce catastrophic state
explosion by horizon 25. Local-dense WPU is more stable than sparse WPU in this
diagnostic, but still accumulates more constraint violations than the graph
baseline in the unclipped run.

Delta clipping reduces violations, especially for WPU sparse, but it does not
solve the underlying model instability. The raw predicted delta norm remains
huge in the clipped run because the clamp is applied only before state update.
Therefore clipping is a safety layer, not a learned long-horizon solution.

## Design Consequence

WPU needs an explicit state-integrity loop:

```text
predict delta -> verify constraints -> clip or reject unsafe delta
-> expand K or run local dense -> update branch uncertainty
```

This supports the v2 direction that WPU cannot be evaluated only by one-step
branch accuracy. It must report state integrity, rollout drift, uncertainty,
and correction/escalation frequency.

## Next Steps

- Train with rollout consistency losses, not only one-step branch/delta loss.
- Add branch-specific delta trajectories instead of reusing the same one-step
  event repeatedly.
- Add constraint-aware delta heads or verifier-guided rejection.
- Re-synchronize selected rollout states with PyBullet to compare model drift
  against simulator ground truth.

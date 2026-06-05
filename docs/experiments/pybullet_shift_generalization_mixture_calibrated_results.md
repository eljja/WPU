# PyBullet Mixture-Trained Shift and Calibration Probe

This probe complements the 7-seed nominal-only shift benchmark. It trains on a
mixture of all four mechanism families and applies post-hoc temperature
calibration before evaluating the same mechanisms. The run is intentionally
reported as a 3-seed probe because the full 7-seed calibrated mixture run is
too expensive for the current iteration.

Source CSV:

- `docs/experiments/pybullet_shift_generalization_mixture_calibrated.csv`

## Protocol

- Train mechanisms: `nominal`, `high_force`, `edge_shift`, `catch_heavy`.
- Eval mechanisms: `nominal`, `high_force`, `edge_shift`, `catch_heavy`.
- Models: `wpu-cws-indexed-sparse`, `wpu-cws-indexed-local-dense`,
  `graph-transformer`, `serialized-token`.
- Seeds: `11, 13, 17`.
- Background objects: `32`.
- Training steps: `16`.
- Eval samples: `36` per seed/mechanism.
- Calibration: post-hoc scalar temperature fitted on held-out training-family
  samples.

## Summary

| eval mechanism | best WPU accuracy | best baseline accuracy | best WPU ECE | best baseline ECE | interpretation |
|---|---:|---:|---:|---:|---|
| nominal | 0.472222 | 0.407407 | 0.107388 | 0.143710 | WPU improves nominal accuracy and calibration in this small probe. |
| high_force | 0.444444 | 0.444444 | 0.204216 | 0.182570 | Accuracy ties, but baseline calibration is better. |
| edge_shift | 0.546297 | 0.388889 | 0.157736 | 0.093222 | WPU local-dense improves accuracy, but graph-transformer calibration is better. |
| catch_heavy | 0.333333 | 0.481481 | 0.295547 | 0.202056 | Serialized-token baseline is clearly stronger. |

Mean WPU ECE is `0.208404`; mean non-WPU baseline ECE is `0.183805`, so the
calibrated-mixture ECE ratio is `1.133834`. This reverses the small ECE
advantage observed in the 7-seed nominal-only benchmark.

## Interpretation

Mixture training is not a universal fix. It helps WPU on `edge_shift`, where
explicit object geometry and local relation propagation are useful, but it does
not solve `catch_heavy`, where the branch prior changes sharply. Calibration is
also not solved by post-hoc temperature alone: WPU can gain accuracy in some
mechanisms while still being less calibrated than dense/token baselines.

The next useful P4/P5 step is not more threshold tuning. It should add a
mechanism-aware branch prior, uncertainty-gated fallback, and multi-step
calibration losses that are trained inside the WPU branch/rollout model.

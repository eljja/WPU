# PyBullet Route-Regret Training Smoke

Source CSVs:

- `docs/experiments/pybullet_route_regret_training_smoke.csv`
- `docs/experiments/pybullet_route_regret_training_smoke_untrained.csv`
- `docs/experiments/pybullet_route_regret_training_smoke_threshold.csv`

This smoke test verifies that PyBullet mechanism-shift experiments can train and
measure WPU route-regret heads explicitly. It follows the staged-regret lesson:
a regret-hybrid model should not be inserted into PyBullet shift evaluation with
an untrained route head.

## What Changed

- `scripts/pybullet_shift_generalization.py` now supports
  `--route-regret-loss-weight`, `--route-regret-compute-cost`, and
  `--route-regret-threshold`.
- During training, CWS models can receive a counterfactual target
  `dense_loss - sparse_loss + compute_cost` through `model.route_regret_loss`.
- Evaluation records `route_regret_mean`, `route_regret_negative_ratio`,
  `route_regret_loss_weight`, `route_regret_compute_cost`, and
  `route_regret_threshold`.
- `CausalWorkingSetProcessor` now exposes a configurable internal
  `route_regret_threshold` instead of hard-coding route-dense if predicted
  regret is below zero.

## Smoke Results

| condition | threshold | nominal dense compute | high-force dense compute | nominal accuracy | high-force accuracy |
|---|---:|---:|---:|---:|---:|
| untrained route head | 0.0 | 1.0 | 1.0 | 0.625 | 0.125 |
| trained route head | 0.0 | 1.0 | 1.0 | 0.625 | 0.125 |
| trained route head | -0.5 | 0.0 | 0.0 | 0.625 | 0.125 |

## Interpretation

This is infrastructure evidence, not a mechanism-shift performance result. The
small run shows two things:

- Explicit route-regret training is wired into the PyBullet path and emits route
  metrics.
- A configurable route threshold is necessary because a zero threshold can
  collapse to all-dense routing in this tiny setting.

The next real experiment should sweep route-regret threshold or select it on a
validation split, then compare sparse, local-dense, regret-hybrid, graph, and
serialized-token baselines under the same mechanism-shift protocol.

# WPU V2 Staged Regret Context Probe

Source CSVs:

- `docs/experiments/wpu_v2_staged_regret_context_samples.csv`
- `docs/experiments/wpu_v2_staged_regret_context_probe.csv`
- `docs/experiments/wpu_v2_staged_regret_context_probe_summary.csv`

## Purpose

The margin-policy analysis showed that K alone is not enough to choose a robust
sparse-favoring margin. This probe exports sample-level state and model
diagnostic features from the staged internal-regret WPU, then trains
seed-heldout MLP regressors to predict dense-vs-sparse regret.

The tested feature sets are:

- `mlp_state_regret`: physical state features only.
- `mlp_state_selector_regret`: state features plus selector confidence and
  selected fraction.
- `mlp_state_regret_scalar`: state features plus the model's predicted regret
  scalar.
- `mlp_sparse_diagnostics_regret`: state features plus sparse entropy, margin,
  confidence, delta norm, and uncertainty.
- `mlp_route_context_regret`: all exported route context features.

## Setup

- N = 2048
- K = 8, 16, 32
- Seeds = 11, 13, 17, 19, 23
- Samples = 90 per K/seed, 1350 total samples
- Staged propagation/regret training as in the previous margin sweeps
- Probe split = leave-one-seed-out
- Probe model = one-hidden-layer MLP, 600 steps

## Results at Compute Cost 0.05

| feature set | regret pearson | regret R2 | dense rate | policy loss | loss delta | oracle excess | policy acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| state only | 0.400 | 0.103 | 0.293 | 0.970 | -0.019 | 0.051 | 0.484 |
| state + regret scalar | 0.371 | 0.066 | 0.303 | 0.973 | -0.015 | 0.055 | 0.479 |
| state + selector | 0.344 | 0.012 | 0.324 | 0.976 | -0.012 | 0.058 | 0.486 |
| sparse diagnostics | 0.312 | -0.405 | 0.366 | 0.984 | -0.005 | 0.065 | 0.481 |
| all route context | 0.195 | -1.961 | 0.350 | 0.996 | 0.008 | 0.078 | 0.478 |

## Interpretation

The result is a useful correction to the scheduler hypothesis. More internal
diagnostics do not automatically improve routing under seed shift. The best
generalizing probe is the simplest one: physical state features. Adding the
model's own predicted regret, sparse diagnostics, or all route context features
causes overfitting or miscalibration in this small regime.

This means the next v2 scheduler should not simply concatenate every available
diagnostic into a learned router. It should separate invariant state evidence
from model-specific confidence signals and regularize the latter strongly.

The constructive conclusion is:

```text
state evidence predicts useful dense-vs-sparse regret;
raw model diagnostics are not yet seed-stable enough to be trusted directly.
```

## Revised Scheduler Direction

The next scheduler should use a two-level design:

```text
base_margin = f(physical_state_features)
diagnostic_adjustment = g(selector_confidence, entropy, uncertainty, rollout_drift)
execute_dense if predicted_regret < -(base_margin + clipped_adjustment)
```

`g` should be constrained or calibrated so that diagnostics can adjust a
state-derived margin but cannot dominate it under distribution shift.

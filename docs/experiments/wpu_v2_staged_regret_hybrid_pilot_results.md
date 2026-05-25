# WPU V2 Staged Regret-Hybrid Pilot

Source CSV: `docs/experiments/wpu_v2_staged_regret_hybrid_pilot.csv`

## Purpose

This pilot tests whether the WPU scheduler can learn a cost-sensitive
sparse-versus-local-dense routing decision inside the model. The previous
jointly trained route-regret head collapsed to always-sparse. This version
separates optimization:

```text
train sparse and local-dense propagation
freeze propagation core
train route-regret head on counterfactual losses
calibrate dense threshold on validation samples
evaluate hard selective execution on held-out test samples
```

The target is not universal accuracy superiority. The target is narrower and
more falsifiable: reduce expected prediction loss while executing dense
recompute on only a minority of samples.

## Setup

- N = 2048 total objects
- K = 8, 16, 32 causal working-set objects
- Seeds = 11, 13
- Hidden dim = 128
- Layers = 1
- Propagation steps = 40
- Regret-head steps = 80
- Validation samples = 90 per seed
- Test samples = 90 per seed
- Interaction mode = pairwise
- Dense compute cost = 0.05

## Aggregate Results

| K | sparse acc | dense acc | routed acc | sparse loss | dense loss | routed loss | oracle loss | dense compute | routed loss delta | oracle excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 0.490 | 0.472 | 0.490 | 0.977 | 1.033 | 0.953 | 0.903 | 0.133 | -0.024 | 0.050 |
| 16 | 0.500 | 0.411 | 0.500 | 0.958 | 1.033 | 0.912 | 0.858 | 0.272 | -0.047 | 0.054 |
| 32 | 0.483 | 0.467 | 0.500 | 1.019 | 1.058 | 0.989 | 0.941 | 0.272 | -0.030 | 0.048 |

## Interpretation

The staged route head fixes the always-sparse failure mode. Across all tested K
values, routed loss is lower than always-sparse loss while dense recompute is
used for roughly 13-27% of samples. This is the first internal selective-dense
result that supports the WPU v2 scheduler direction.

The result is not strong enough to claim broad WPU superiority. Accuracy is tied
with sparse at K=8 and K=16 and only slightly higher at K=32. Routed loss also
remains about 0.05 above the oracle per-sample sparse/dense choice. The main
remaining problem is therefore route-regret calibration and sample-level
identifiability, not just compute reduction.

## Next Required Fix

The next scheduler should make threshold and compute cost trainable or
validation-stable across seeds, then rerun:

- five-seed K sweep;
- denser N sweep at fixed causal K;
- distractor ambiguity sweep;
- rollout-horizon sweep with accumulated delta errors.

The v2 paper claim should remain conditional:

```text
WPU is favorable when N is large, K is small or moderate, causal retrieval is
reliable, and selective dense fallback can reduce expected loss at bounded
compute.
```

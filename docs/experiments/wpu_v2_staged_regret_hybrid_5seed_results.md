# WPU V2 Staged Regret-Hybrid 5-Seed Audit

Source CSV: `docs/experiments/wpu_v2_staged_regret_hybrid_5seed.csv`

## Purpose

The two-seed staged regret pilot showed that internal selective dense routing
can reduce loss relative to always-sparse. This audit makes the test stricter:

- increase to five seeds;
- record calibrated-threshold routing and fixed zero-threshold routing;
- record oracle selective dense usage;
- measure correlation between predicted route regret and realized dense-vs-sparse
  regret.

This separates three questions:

```text
1. Does selective dense routing still reduce expected loss?
2. Is validation threshold calibration necessary?
3. Is the route-regret signal ranking samples in the right direction?
```

## Setup

- N = 2048 total objects
- K = 8, 16, 32 causal working-set objects
- Seeds = 11, 13, 17, 19, 23
- Hidden dim = 128
- Layers = 1
- Propagation steps = 40
- Regret-head steps = 80
- Validation samples = 90 per seed
- Test samples = 90 per seed
- Interaction mode = pairwise
- Dense compute cost = 0.05

## Aggregate Results

Values are mean +/- 95% confidence interval over five seeds.

| K | sparse acc | routed acc | sparse loss | routed loss | zero-threshold loss | dense compute | oracle dense | routed delta | oracle excess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 0.482 +/- 0.015 | 0.482 +/- 0.015 | 0.991 +/- 0.013 | 0.960 +/- 0.021 | 0.972 +/- 0.018 | 0.204 +/- 0.078 | 0.482 +/- 0.049 | -0.031 +/- 0.014 | 0.042 +/- 0.010 |
| 16 | 0.504 +/- 0.024 | 0.504 +/- 0.024 | 0.967 +/- 0.022 | 0.949 +/- 0.040 | 0.949 +/- 0.029 | 0.204 +/- 0.070 | 0.471 +/- 0.075 | -0.017 +/- 0.025 | 0.049 +/- 0.013 |
| 32 | 0.478 +/- 0.027 | 0.493 +/- 0.025 | 1.008 +/- 0.020 | 0.979 +/- 0.018 | 1.001 +/- 0.024 | 0.302 +/- 0.205 | 0.562 +/- 0.186 | -0.028 +/- 0.010 | 0.043 +/- 0.016 |

Route-regret fit:

| K | calibration gain vs zero | zero-threshold delta | regret corr | regret MSE |
| --- | --- | --- | --- | --- |
| 8 | 0.012 +/- 0.009 | -0.018 +/- 0.012 | 0.377 +/- 0.079 | 0.056 +/- 0.021 |
| 16 | -0.000 +/- 0.012 | -0.017 +/- 0.017 | 0.289 +/- 0.192 | 0.088 +/- 0.035 |
| 32 | 0.022 +/- 0.017 | -0.007 +/- 0.024 | 0.388 +/- 0.130 | 0.068 +/- 0.030 |

## Interpretation

The positive result survives the five-seed audit in the loss metric. Calibrated
selective routing lowers expected loss versus always-sparse at K=8, K=16, and
K=32 while using dense recompute on only 20-30% of samples. The route-regret
correlation is positive in all K groups, so the head is learning a real ranking
signal rather than random threshold noise.

The result is still not sufficient for a broad accuracy claim. Accuracy is
unchanged at K=8 and K=16 and only modestly better at K=32. The routed loss also
remains about 0.04-0.05 above oracle selective routing. This means the remaining
problem is not whether sparse/dense routing has a signal; it is how to turn the
signal into a robust decision boundary.

Calibration is not fully stable. Validation threshold selection improves over a
fixed zero threshold at K=8 and K=32, but it gives no average benefit at K=16 and
hurts some individual seeds. A WPU v2 paper should not claim that threshold
calibration is solved. It should claim that route-regret prediction is a
promising scheduler signal and report threshold sensitivity as an open systems
problem.

## Revised Scientific Claim

The defensible claim after this audit is:

```text
For large-N, bounded-K world-state prediction, WPU-style selective propagation
can reduce expected prediction loss by routing a minority of samples through a
local dense recompute path. The current implementation has a repeatable regret
signal, but not yet a robust threshold-free scheduler or broad accuracy
superiority.
```

## Next Required Fix

The next implementation should remove brittle threshold dependence. Strong
candidates are:

- learn a cost-conditioned route probability and evaluate expected utility;
- train a margin around zero regret so near-tie samples stay sparse;
- calibrate route regret with held-out seed/domain augmentation;
- use constraint-violation and rollout-drift features as additional route
  evidence;
- evaluate the same route head across denser N/K/D sweeps.

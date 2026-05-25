# WPU V2 Staged Regret Margin Sweep

Source CSV: `docs/experiments/wpu_v2_staged_regret_margin_sweep.csv`

## Purpose

The five-seed audit showed that validation-calibrated route thresholds reduce
loss, but threshold calibration is not uniformly stable. This sweep asks whether
a fixed sparse-favoring margin can provide a more realistic scheduler rule:

```text
execute dense if predicted_regret < -margin
```

The goal is not to beat calibrated routing. The goal is to find whether WPU can
use a threshold-stable conservative policy that still improves over
always-sparse while reducing dense compute.

## Setup

- N = 2048 total objects
- K = 8, 16, 32 causal working-set objects
- Seeds = 11, 13, 17, 19, 23
- Hidden dim = 128
- Layers = 1
- Propagation steps = 40
- Regret-head steps = 80
- Validation/test samples = 90 per seed
- Interaction mode = pairwise
- Dense compute cost = 0.05
- Fixed margins = 0, 0.02, 0.05, 0.1, 0.2

Each model is trained once per N/K/seed condition and then evaluated under all
policies. This isolates scheduler-policy effects from training randomness.

## Overall Policy Comparison

Aggregated over K = 8, 16, 32 and five seeds.

| policy | routed loss | sparse loss | loss delta | dense compute | routed acc | oracle excess |
| --- | --- | --- | --- | --- | --- | --- |
| calibrated | 0.960 | 0.990 | -0.030 | 0.240 | 0.490 | 0.040 |
| margin 0.02 | 0.970 | 0.990 | -0.020 | 0.209 | 0.487 | 0.054 |
| margin 0.05 | 0.972 | 0.990 | -0.018 | 0.127 | 0.487 | 0.054 |
| margin 0.00 | 0.974 | 0.990 | -0.014 | 0.264 | 0.485 | 0.056 |
| margin 0.10 | 0.975 | 0.990 | -0.013 | 0.063 | 0.488 | 0.057 |
| margin 0.20 | 0.987 | 0.990 | -0.001 | 0.003 | 0.488 | 0.069 |

## K-Specific Results

| K | best calibrated loss | best fixed-margin loss | best fixed margin | fixed dense compute | fixed loss delta |
| --- | --- | --- | --- | --- | --- |
| 8 | 0.960 | 0.972 | 0.02 | 0.180 | -0.019 |
| 16 | 0.949 | 0.949 | 0.00 | 0.104 | -0.017 |
| 32 | 0.979 | 0.985 | 0.10 | 0.111 | -0.023 |

## Interpretation

The sweep identifies a useful but limited result. A fixed sparse-favoring
margin can improve over always-sparse without validation-specific threshold
calibration, but no single margin matches calibrated routing across all K. The
most practical global policy in this sweep is `margin=0.05`: it keeps dense
compute to about 13% while preserving most of the loss improvement. If the
system prioritizes maximum loss reduction, `margin=0.02` is better but spends
more dense compute.

This changes the v2 scheduler requirement. The next scheduler should not simply
learn a raw threshold. It should expose a controllable utility knob:

```text
small margin  -> better loss, more dense compute
larger margin -> lower compute, smaller but more conservative loss gain
```

The remaining weakness is K-dependence. K=8 prefers a smaller margin, K=16
prefers zero margin, and K=32 prefers a larger margin. This suggests the margin
should be state- or regime-conditioned rather than globally fixed.

## Revised Engineering Direction

The next implementation should add a cost-conditioned margin scheduler:

```text
margin = f(K, selector_confidence, interaction_density, rollout_drift, compute_budget)
execute dense if predicted_regret < -margin
```

This preserves the WPU principle: stay sparse by default, invoke dense recompute
only when the state evidence justifies the extra work.

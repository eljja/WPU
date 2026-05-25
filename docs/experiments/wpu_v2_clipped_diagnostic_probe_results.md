# WPU V2 Clipped Diagnostic Adjustment Probe

Source CSVs:

- `docs/experiments/wpu_v2_clipped_diagnostic_probe.csv`
- `docs/experiments/wpu_v2_clipped_diagnostic_probe_summary.csv`

## Purpose

The state-evidence context probe showed that physical state features generalize
better than raw model diagnostics. This follow-up tests a safer architecture:

```text
prediction = state_base_prediction + clipped_diagnostic_residual
```

The hypothesis is that diagnostics might be useful if they are constrained to
small residual adjustments rather than allowed to dominate the route decision.

## Setup

- Input: `docs/experiments/wpu_v2_staged_regret_context_samples.csv`
- Split: leave-one-seed-out
- Base model: MLP over physical state features
- Residual model: MLP over route diagnostics
- Residual clips: 0, 0.01, 0.02, 0.05, 0.1
- Compute costs: 0.02, 0.05, 0.1

## Results at Compute Cost 0.05

| residual clip | regret pearson | regret R2 | dense rate | policy loss | loss delta | oracle excess |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 0.402 | 0.105 | 0.294 | 0.968 | -0.020 | 0.050 |
| 0.01 | 0.399 | 0.099 | 0.300 | 0.969 | -0.019 | 0.051 |
| 0.02 | 0.395 | 0.091 | 0.299 | 0.969 | -0.019 | 0.051 |
| 0.05 | 0.380 | 0.054 | 0.329 | 0.973 | -0.016 | 0.054 |
| 0.10 | 0.351 | -0.029 | 0.356 | 0.978 | -0.010 | 0.060 |

## Interpretation

The clipped diagnostic residual does not improve over the state-only base model.
Even small residual clips slightly hurt held-out performance, and larger clips
increase dense routing while reducing the loss benefit. This means the current
diagnostics are not just too strong when unconstrained; their residual signal is
not reliably aligned under seed shift.

The practical scheduler conclusion is:

```text
use physical state evidence for the route/margin decision;
use diagnostics as safety triggers or abstention signals, not as direct regret
adjustments, until they are calibrated under broader shifts.
```

This strengthens the WPU v2 claim by narrowing it. The useful path is not a
generic learned router over all internal activations. It is a state-grounded
scheduler with separately validated consistency checks.

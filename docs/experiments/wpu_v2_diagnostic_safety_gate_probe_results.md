# WPU V2 Diagnostic Safety Gate Probe

Source CSVs:

- `docs/experiments/wpu_v2_diagnostic_safety_gate_probe.csv`
- `docs/experiments/wpu_v2_diagnostic_safety_gate_probe_summary.csv`

## Purpose

Previous probes showed that raw diagnostics should not directly adjust the
state-based regret prediction. This probe tests a safer use: diagnostics as a
safety gate. The state model first decides whether dense recompute is useful;
then a diagnostic threshold may veto dense execution.

Gate selection is evaluated strictly:

```text
train state regret predictor on four seeds
select diagnostic gate threshold on those four seeds
evaluate the selected gate on the held-out seed
```

The probe also reports a test-oracle diagnostic gate as an upper bound, selected
on the held-out test seed itself. This upper bound is not deployable; it only
indicates whether diagnostic gates contain any signal at all.

## Results

| policy | cost | dense rate | policy loss | loss delta | oracle excess | policy acc |
| --- | --- | --- | --- | --- | --- | --- |
| state route, no gate | 0.02 | 0.387 | 0.960 | -0.028 | 0.058 | 0.484 |
| LOSO diagnostic gate | 0.02 | 0.318 | 0.962 | -0.026 | 0.059 | 0.487 |
| test-oracle diagnostic gate | 0.02 | 0.306 | 0.954 | -0.034 | 0.052 | 0.495 |
| state route, no gate | 0.05 | 0.296 | 0.969 | -0.019 | 0.051 | 0.483 |
| LOSO diagnostic gate | 0.05 | 0.254 | 0.972 | -0.017 | 0.053 | 0.485 |
| test-oracle diagnostic gate | 0.05 | 0.202 | 0.963 | -0.025 | 0.045 | 0.487 |
| state route, no gate | 0.10 | 0.191 | 0.979 | -0.010 | 0.038 | 0.486 |
| LOSO diagnostic gate | 0.10 | 0.155 | 0.980 | -0.008 | 0.039 | 0.487 |
| test-oracle diagnostic gate | 0.10 | 0.113 | 0.973 | -0.015 | 0.032 | 0.488 |

## Interpretation

The deployable leave-one-seed-out diagnostic gate does not improve over the
state route without a gate. It reduces dense compute, but the loss improvement
shrinks. This means diagnostic thresholds do not currently transfer reliably
across seeds.

The test-oracle gate is better than no-gate, so diagnostics do contain some
useful safety information. The problem is not absence of signal; it is
threshold transfer and calibration.

The v2 scheduler implication is narrow:

```text
Do not use single diagnostic thresholds as deployed safety gates yet.
Use diagnostics to trigger explicit verification mechanisms:
K expansion, consistency checks, uncertainty increase, or abstention.
```

To become deployable, diagnostic safety must be trained or calibrated across
broader distribution shifts, not selected from one small synthetic regime.

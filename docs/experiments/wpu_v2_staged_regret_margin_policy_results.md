# WPU V2 Regret-Margin Policy Selection

Source CSV: `docs/experiments/wpu_v2_staged_regret_margin_sweep.csv`

## Purpose

This analysis tests whether the K-dependence observed in the fixed-margin sweep can be used as a more stable scheduler policy. Margin selection is evaluated with leave-one-seed-out selection: margins are chosen on four seeds and evaluated on the held-out seed.

## Overall Results

| policy | selected policies | routed loss | sparse loss | loss delta | dense compute | routed acc | oracle excess |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_global_margin | margin_0.05 | 0.972 +/- 0.013 | 0.988 +/- 0.013 | -0.016 +/- 0.007 | 0.127 +/- 0.052 | 0.487 +/- 0.014 | 0.054 +/- 0.012 |
| loso_global_margin | margin_0.02;margin_0.05 | 0.974 +/- 0.014 | 0.988 +/- 0.013 | -0.015 +/- 0.008 | 0.160 +/- 0.076 | 0.487 +/- 0.014 | 0.055 +/- 0.012 |
| loso_k_conditioned_margin | margin_0;margin_0.02;margin_0.05;margin_0.1 | 0.973 +/- 0.014 | 0.988 +/- 0.013 | -0.015 +/- 0.008 | 0.147 +/- 0.063 | 0.488 +/- 0.013 | 0.055 +/- 0.012 |
| validation_calibrated | calibrated | 0.963 +/- 0.016 | 0.988 +/- 0.013 | -0.025 +/- 0.010 | 0.237 +/- 0.075 | 0.493 +/- 0.013 | 0.045 +/- 0.007 |

## K-Specific Results

| policy | group | selected policies | routed loss | loss delta | dense compute | oracle excess |
| --- | --- | --- | --- | --- | --- | --- |
| fixed_global_margin | K=16 | margin_0.05 | 0.955 +/- 0.026 | -0.012 +/- 0.012 | 0.047 +/- 0.042 | 0.054 +/- 0.025 |
| fixed_global_margin | K=32 | margin_0.05 | 0.989 +/- 0.022 | -0.018 +/- 0.019 | 0.224 +/- 0.071 | 0.053 +/- 0.022 |
| fixed_global_margin | K=8 | margin_0.05 | 0.972 +/- 0.008 | -0.018 +/- 0.009 | 0.111 +/- 0.078 | 0.054 +/- 0.021 |
| loso_global_margin | K=16 | margin_0.02;margin_0.05 | 0.955 +/- 0.026 | -0.012 +/- 0.012 | 0.047 +/- 0.042 | 0.054 +/- 0.025 |
| loso_global_margin | K=32 | margin_0.02;margin_0.05 | 0.994 +/- 0.022 | -0.014 +/- 0.022 | 0.291 +/- 0.150 | 0.058 +/- 0.023 |
| loso_global_margin | K=8 | margin_0.02;margin_0.05 | 0.973 +/- 0.011 | -0.018 +/- 0.009 | 0.142 +/- 0.087 | 0.054 +/- 0.019 |
| loso_k_conditioned_margin | K=16 | margin_0;margin_0.02 | 0.952 +/- 0.027 | -0.015 +/- 0.016 | 0.091 +/- 0.086 | 0.051 +/- 0.025 |
| loso_k_conditioned_margin | K=32 | margin_0.05;margin_0.1 | 0.990 +/- 0.022 | -0.018 +/- 0.020 | 0.164 +/- 0.096 | 0.054 +/- 0.024 |
| loso_k_conditioned_margin | K=8 | margin_0;margin_0.02;margin_0.05 | 0.978 +/- 0.013 | -0.013 +/- 0.009 | 0.187 +/- 0.142 | 0.059 +/- 0.018 |
| validation_calibrated | K=16 | calibrated | 0.949 +/- 0.040 | -0.017 +/- 0.025 | 0.204 +/- 0.070 | 0.049 +/- 0.013 |
| validation_calibrated | K=32 | calibrated | 0.979 +/- 0.018 | -0.028 +/- 0.010 | 0.302 +/- 0.204 | 0.043 +/- 0.016 |
| validation_calibrated | K=8 | calibrated | 0.960 +/- 0.021 | -0.031 +/- 0.014 | 0.204 +/- 0.078 | 0.042 +/- 0.009 |

## Interpretation

Leave-one-seed-out K-conditioned margin selection is a stricter test than post-hoc K-specific best margin selection. If it improves over a fixed global margin, the result supports a regime-conditioned scheduler. If it does not, the margin must be learned from richer state evidence rather than K alone.

The result is negative for K-only scheduling. `loso_k_conditioned_margin` does
not improve over `fixed_global_margin` overall: routed loss is 0.973 versus
0.972, with similar oracle excess. It helps slightly at K=16, is similar at
K=32, and hurts at K=8. Therefore K is useful as a regime descriptor, but it is
not sufficient as the only margin-control variable.

The constructive conclusion is sharper:

```text
margin = f(K alone)
```

is too weak. The next scheduler must condition on richer state evidence:

```text
margin = f(K, selector_confidence, interaction_density, regret_uncertainty,
           sparse_entropy, rollout_drift, compute_budget)
```

This keeps the WPU claim falsifiable. The useful regime is not "large N implies
WPU wins"; it is "large N, bounded K, reliable retrieval, and state evidence
that can identify when local dense recompute is worth its cost."

# World-Copy Baseline Comparison Probe

This probe compares WPU local propagation against token/graph/dense baselines on the same synthetic world-copy delta task.
Baselines are comparisons, not WPU implementation paths. This is a controlled screen, not full P2 completion.
Source CSV: `docs/experiments/world_copy_baseline_comparison_probe.csv`.

## Summary

| model | mean delta MSE | mean work proxy | mean bytes proxy | mean selected K | max selected K | accuracy/kwork |
|---|---:|---:|---:|---:|---:|---:|
| wpu-hybrid | 0.020818 | 8.789551 | 316.423828 | 9.333333 | 16 | 37867.798346 |
| wpu-hybrid-context | 0.020904 | 9.789551 | 457.056641 | 9.333333 | 16 | 32480.274504 |
| wpu-region-guard | 0.002646 | 9.333333 | 336.000000 | 9.333333 | 16 | 52432.911821 |
| dense-graph | 0.003810 | 2727.406250 | 97920.000000 | 9.333333 | 16 | 711.681407 |
| serialized-token | 0.003223 | 2727.406250 | 119680.000000 | 9.333333 | 16 | 827.955281 |
| graph-transformer-proxy | 0.003810 | 17935805.573893 | 71581696.000000 | 9.333333 | 16 | 4.180562 |

## Interpretation

- `wpu-region-guard` keeps selected `K` bounded while improving both raw delta MSE and work/bytes proxy.
- Adding shallow context features alone (`wpu-hybrid-context`) is negative; its raw MSE does not improve over base `wpu-hybrid`.
- The positive signal is that a bounded local region guard can close missing-relation gaps better than trusting only relation frontier evidence.
- This holds only when bounded regions are small and reliable. If regions grow or objectification is wrong, the WPU claim weakens again.
- `graph-transformer-proxy` is a dense quadratic work proxy, not a trained attention model.
- The next step is a streaming/H>=25 world-copy task with actual token/graph models and measured latency.

# World-Copy Baseline Comparison Probe

This probe compares WPU local propagation against token/graph/dense baselines on the same synthetic world-copy delta task.
Baselines are comparisons, not WPU implementation paths. This is a controlled screen, not full P2 completion.
Source CSV: `docs/experiments/world_copy_baseline_comparison_probe.csv`.

## Summary

| model | mean delta MSE | mean work proxy | mean bytes proxy | mean selected K | max selected K | accuracy/kwork |
|---|---:|---:|---:|---:|---:|---:|
| wpu-hybrid | 0.020818 | 8.789551 | 316.423828 | 9.333333 | 16 | 37867.798346 |
| dense-graph | 0.003778 | 2727.406250 | 97920.000000 | 9.333333 | 16 | 698.058125 |
| serialized-token | 0.004533 | 2727.406250 | 119680.000000 | 9.333333 | 16 | 621.481983 |
| graph-transformer-proxy | 0.003778 | 17935805.573893 | 71581696.000000 | 9.333333 | 16 | 4.111582 |

## Interpretation

- In this screen, WPU keeps selected `K` bounded and uses lower work/bytes proxy than full-state baselines.
- Raw delta MSE is lower for dense/token baselines. This is therefore not a pure WPU accuracy victory.
- The positive WPU signal is lower touched-state cost and higher accuracy-per-work, which supports a systems-substrate direction.
- `graph-transformer-proxy` is a dense quadratic work proxy, not a trained attention model.
- The next step is a streaming/H>=25 world-copy task with actual token/graph models and measured latency.

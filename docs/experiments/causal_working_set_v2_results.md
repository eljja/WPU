# Causal Working Set v2 Results

Source CSV: `artifacts/causal_working_set_v1_cpu/n-sweep.csv`

This report tests the large-`N` claim in the stricter form:

```text
WPU should scale with causal working set K, not total state N, when K is small and identifiable.
```

Short CPU runs are pipeline checks, not paper evidence. Treat accuracy
near the majority baseline as inconclusive until multi-seed, longer-step,
GPU-scale runs are available.

## Raw Condition Summary

| model | N | K | params | accuracy | mse | selected_K | ms/sample | cuda_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| graph-transformer | 64 | 8 | 69206 | 0.578125 | 0.009739 | 8.0 | 0.87936 | 0.0 |
| serialized-token | 64 | 8 | 52626 | 0.71875 | 0.006164 | 8.0 | 0.199615 | 0.0 |
| wpu-cws-frontier | 64 | 8 | 61271 | 0.578125 | 0.003324 | 8.0 | 0.37984 | 0.0 |
| wpu-cws-learned | 64 | 8 | 61271 | 0.578125 | 0.003969 | 8.0 | 0.46747 | 0.0 |
| wpu-cws-oracle | 64 | 8 | 61271 | 0.578125 | 0.003324 | 8.0 | 0.45082 | 0.0 |
| graph-transformer | 128 | 8 | 69206 | 0.578125 | 0.008383 | 8.0 | 1.44885 | 0.0 |
| serialized-token | 128 | 8 | 52626 | 0.578125 | 0.005911 | 8.0 | 0.29418 | 0.0 |
| wpu-cws-frontier | 128 | 8 | 61271 | 0.578125 | 0.001494 | 8.0 | 0.378415 | 0.0 |
| wpu-cws-learned | 128 | 8 | 61271 | 0.578125 | 0.003096 | 8.0 | 0.503135 | 0.0 |
| wpu-cws-oracle | 128 | 8 | 61271 | 0.578125 | 0.001494 | 8.0 | 0.481915 | 0.0 |
| graph-transformer | 256 | 8 | 69206 | 0.578125 | 0.005999 | 8.0 | 1.381545 | 0.0 |
| serialized-token | 256 | 8 | 52626 | 0.578125 | 0.008792 | 8.0 | 0.50017 | 0.0 |
| wpu-cws-frontier | 256 | 8 | 61271 | 0.578125 | 0.000763 | 8.0 | 0.527015 | 0.0 |
| wpu-cws-learned | 256 | 8 | 61271 | 0.578125 | 0.001506 | 8.0 | 0.37475 | 0.0 |
| wpu-cws-oracle | 256 | 8 | 61271 | 0.578125 | 0.000763 | 8.0 | 0.543805 | 0.0 |
| graph-transformer | 512 | 8 | 69206 | 0.578125 | 0.005312 | 8.0 | 2.132585 | 0.0 |
| serialized-token | 512 | 8 | 52626 | 0.578125 | 0.008724 | 8.0 | 1.049485 | 0.0 |
| wpu-cws-frontier | 512 | 8 | 61271 | 0.578125 | 0.000424 | 8.0 | 0.5369 | 0.0 |
| wpu-cws-learned | 512 | 8 | 61271 | 0.578125 | 0.000433 | 8.0 | 0.46278 | 0.0 |
| wpu-cws-oracle | 512 | 8 | 61271 | 0.578125 | 0.000424 | 8.0 | 0.59633 | 0.0 |

## Aggregated By Model And N

| model | N | K | seeds | params | accuracy_mean | accuracy_ci95 | mse_mean | selected_K_mean | ms_per_sample_mean | ms_per_sample_ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| graph-transformer | 64 | 8 | 1 | 69206 | 0.578125 | 0.0 | 0.009739 | 8.0 | 0.87936 | 0.0 |
| serialized-token | 64 | 8 | 1 | 52626 | 0.71875 | 0.0 | 0.006164 | 8.0 | 0.199615 | 0.0 |
| wpu-cws-frontier | 64 | 8 | 1 | 61271 | 0.578125 | 0.0 | 0.003324 | 8.0 | 0.37984 | 0.0 |
| wpu-cws-learned | 64 | 8 | 1 | 61271 | 0.578125 | 0.0 | 0.003969 | 8.0 | 0.46747 | 0.0 |
| wpu-cws-oracle | 64 | 8 | 1 | 61271 | 0.578125 | 0.0 | 0.003324 | 8.0 | 0.45082 | 0.0 |
| graph-transformer | 128 | 8 | 1 | 69206 | 0.578125 | 0.0 | 0.008383 | 8.0 | 1.44885 | 0.0 |
| serialized-token | 128 | 8 | 1 | 52626 | 0.578125 | 0.0 | 0.005911 | 8.0 | 0.29418 | 0.0 |
| wpu-cws-frontier | 128 | 8 | 1 | 61271 | 0.578125 | 0.0 | 0.001494 | 8.0 | 0.378415 | 0.0 |
| wpu-cws-learned | 128 | 8 | 1 | 61271 | 0.578125 | 0.0 | 0.003096 | 8.0 | 0.503135 | 0.0 |
| wpu-cws-oracle | 128 | 8 | 1 | 61271 | 0.578125 | 0.0 | 0.001494 | 8.0 | 0.481915 | 0.0 |
| graph-transformer | 256 | 8 | 1 | 69206 | 0.578125 | 0.0 | 0.005999 | 8.0 | 1.381545 | 0.0 |
| serialized-token | 256 | 8 | 1 | 52626 | 0.578125 | 0.0 | 0.008792 | 8.0 | 0.50017 | 0.0 |
| wpu-cws-frontier | 256 | 8 | 1 | 61271 | 0.578125 | 0.0 | 0.000763 | 8.0 | 0.527015 | 0.0 |
| wpu-cws-learned | 256 | 8 | 1 | 61271 | 0.578125 | 0.0 | 0.001506 | 8.0 | 0.37475 | 0.0 |
| wpu-cws-oracle | 256 | 8 | 1 | 61271 | 0.578125 | 0.0 | 0.000763 | 8.0 | 0.543805 | 0.0 |
| graph-transformer | 512 | 8 | 1 | 69206 | 0.578125 | 0.0 | 0.005312 | 8.0 | 2.132585 | 0.0 |
| serialized-token | 512 | 8 | 1 | 52626 | 0.578125 | 0.0 | 0.008724 | 8.0 | 1.049485 | 0.0 |
| wpu-cws-frontier | 512 | 8 | 1 | 61271 | 0.578125 | 0.0 | 0.000424 | 8.0 | 0.5369 | 0.0 |
| wpu-cws-learned | 512 | 8 | 1 | 61271 | 0.578125 | 0.0 | 0.000433 | 8.0 | 0.46278 | 0.0 |
| wpu-cws-oracle | 512 | 8 | 1 | 61271 | 0.578125 | 0.0 | 0.000424 | 8.0 | 0.59633 | 0.0 |

## Best Accuracy By N

| N | model | accuracy_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 64 | serialized-token | 0.71875 | 0.71875 | 0.199615 | 52626 |
| 128 | graph-transformer | 0.578125 | 0.578125 | 1.44885 | 69206 |
| 256 | graph-transformer | 0.578125 | 0.578125 | 1.381545 | 69206 |
| 512 | graph-transformer | 0.578125 | 0.578125 | 2.132585 | 69206 |

## Fastest Forward Latency By N

| N | model | ms_per_sample_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 64 | serialized-token | 0.199615 | 0.71875 | 0.199615 | 52626 |
| 128 | serialized-token | 0.29418 | 0.578125 | 0.29418 | 52626 |
| 256 | wpu-cws-learned | 0.37475 | 0.578125 | 0.37475 | 61271 |
| 512 | wpu-cws-learned | 0.46278 | 0.578125 | 0.46278 | 61271 |

## Interpretation Checklist

- If `wpu-cws-oracle` remains accurate at large `N`, the WPU core is plausible and selector quality is the bottleneck.
- If `wpu-cws-oracle` fails, the current working-set core lacks capacity even with correct causal access.
- If token/graph baselines keep accuracy and remain faster, WPU has no demonstrated advantage in this regime.
- If WPU keeps accuracy while latency grows slower with `N`, the large-`N` hypothesis has preliminary support.

# Causal Working Set v2 Results

Source CSV: `docs/experiments/wpu_v2_distractor_sweep_pilot.csv`

This report tests the large-`N` claim in the stricter form:

```text
WPU should scale with causal working set K, not total state N, when K is small and identifiable.
```

Short CPU runs are pipeline checks, not paper evidence. Treat accuracy
near the majority baseline as inconclusive until multi-seed, longer-step,
GPU-scale runs are available.

## Raw Condition Summary

| model | N | K | distractors | params | accuracy | majority | acc-majority | mse | selected_K | causal_recall | ms/sample | cuda_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wpu-cws-indexed | 4096 | 8 | 0 | 6853655 | 0.33 | 0.335 | -0.005 | 2e-05 | 8.0 | 1.0 | 3.681535 | 514.410645 |
| wpu-cws-indexed | 4096 | 8 | 0 | 6853655 | 0.335 | 0.335 | 0.0 | 1.4e-05 | 8.0 | 1.0 | 1.920375 | 514.410645 |
| wpu-cws-indexed | 4096 | 8 | 32 | 6853655 | 0.33 | 0.335 | -0.005 | 2.1e-05 | 8.0 | 1.0 | 7.400645 | 515.560059 |
| wpu-cws-indexed | 4096 | 8 | 32 | 6853655 | 0.335 | 0.335 | 0.0 | 1.1e-05 | 8.0 | 1.0 | 4.902225 | 515.560059 |
| wpu-cws-indexed | 4096 | 8 | 128 | 6853655 | 0.335 | 0.335 | 0.0 | 3.2e-05 | 8.0 | 1.0 | 13.859398 | 519.01123 |
| wpu-cws-indexed | 4096 | 8 | 128 | 6853655 | 0.63 | 0.335 | 0.295 | 7e-06 | 8.0 | 1.0 | 14.01979 | 519.01123 |
| wpu-cws-indexed | 4096 | 8 | 256 | 6853655 | 0.335 | 0.335 | 0.0 | 3.2e-05 | 8.0 | 1.0 | 27.996497 | 522.611816 |
| wpu-cws-indexed | 4096 | 8 | 256 | 6853655 | 0.335 | 0.335 | 0.0 | 5e-06 | 8.0 | 1.0 | 27.68283 | 522.611816 |
| wpu-cws-learned | 4096 | 8 | 0 | 6853655 | 0.335 | 0.335 | 0.0 | 1e-06 | 16.0 | 1.0 | 3.6004 | 514.410645 |
| wpu-cws-learned | 4096 | 8 | 0 | 6853655 | 0.335 | 0.335 | 0.0 | 8e-06 | 16.0 | 1.0 | 1.660712 | 514.410645 |
| wpu-cws-learned | 4096 | 8 | 32 | 6853655 | 0.335 | 0.335 | 0.0 | 1e-06 | 16.0 | 1.0 | 4.410263 | 515.560059 |
| wpu-cws-learned | 4096 | 8 | 32 | 6853655 | 0.33 | 0.335 | -0.005 | 5e-06 | 16.0 | 1.0 | 5.06401 | 515.560059 |
| wpu-cws-learned | 4096 | 8 | 128 | 6853655 | 0.33 | 0.335 | -0.005 | 8.6e-05 | 16.0 | 1.0 | 12.999805 | 519.01123 |
| wpu-cws-learned | 4096 | 8 | 128 | 6853655 | 0.725 | 0.335 | 0.39 | 2.5e-05 | 16.0 | 1.0 | 13.559022 | 519.01123 |
| wpu-cws-learned | 4096 | 8 | 256 | 6853655 | 0.335 | 0.335 | 0.0 | 3e-06 | 16.0 | 1.0 | 24.315672 | 523.611816 |
| wpu-cws-learned | 4096 | 8 | 256 | 6853655 | 0.715 | 0.335 | 0.38 | 4.4e-05 | 16.0 | 1.0 | 24.59189 | 523.611816 |
| wpu-cws-oracle | 4096 | 8 | 0 | 6853655 | 0.33 | 0.335 | -0.005 | 2e-05 | 8.0 | 1.0 | 4.01248 | 513.410645 |
| wpu-cws-oracle | 4096 | 8 | 0 | 6853655 | 0.335 | 0.335 | 0.0 | 1.4e-05 | 8.0 | 1.0 | 2.018485 | 514.410645 |
| wpu-cws-oracle | 4096 | 8 | 32 | 6853655 | 0.33 | 0.335 | -0.005 | 2.1e-05 | 8.0 | 1.0 | 5.49926 | 515.560059 |
| wpu-cws-oracle | 4096 | 8 | 32 | 6853655 | 0.335 | 0.335 | 0.0 | 1.1e-05 | 8.0 | 1.0 | 5.106985 | 515.560059 |
| wpu-cws-oracle | 4096 | 8 | 128 | 6853655 | 0.335 | 0.335 | 0.0 | 3.2e-05 | 8.0 | 1.0 | 13.145272 | 519.01123 |
| wpu-cws-oracle | 4096 | 8 | 128 | 6853655 | 0.63 | 0.335 | 0.295 | 7e-06 | 8.0 | 1.0 | 13.270875 | 519.01123 |
| wpu-cws-oracle | 4096 | 8 | 256 | 6853655 | 0.335 | 0.335 | 0.0 | 3.2e-05 | 8.0 | 1.0 | 24.307903 | 523.611816 |
| wpu-cws-oracle | 4096 | 8 | 256 | 6853655 | 0.335 | 0.335 | 0.0 | 5e-06 | 8.0 | 1.0 | 24.781958 | 523.611816 |

## Aggregated By Model And N

| model | N | K | distractors | seeds | params | accuracy_mean | accuracy_ci95 | majority_mean | acc_minus_majority | mse_mean | selected_K_mean | causal_recall_mean | ms_per_sample_mean | ms_per_sample_ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wpu-cws-indexed | 4096 | 8 | 0 | 2 | 6853655 | 0.3325 | 0.0049 | 0.335 | -0.0025 | 1.7e-05 | 8.0 | 1.0 | 2.800955 | 1.725937 |
| wpu-cws-learned | 4096 | 8 | 0 | 2 | 6853655 | 0.335 | 0.0 | 0.335 | 0.0 | 5e-06 | 16.0 | 1.0 | 2.630556 | 1.900894 |
| wpu-cws-oracle | 4096 | 8 | 0 | 2 | 6853655 | 0.3325 | 0.0049 | 0.335 | -0.0025 | 1.7e-05 | 8.0 | 1.0 | 3.015483 | 1.954115 |
| wpu-cws-indexed | 4096 | 8 | 32 | 2 | 6853655 | 0.3325 | 0.0049 | 0.335 | -0.0025 | 1.6e-05 | 8.0 | 1.0 | 6.151435 | 2.448452 |
| wpu-cws-learned | 4096 | 8 | 32 | 2 | 6853655 | 0.3325 | 0.0049 | 0.335 | -0.0025 | 3e-06 | 16.0 | 1.0 | 4.737137 | 0.640672 |
| wpu-cws-oracle | 4096 | 8 | 32 | 2 | 6853655 | 0.3325 | 0.0049 | 0.335 | -0.0025 | 1.6e-05 | 8.0 | 1.0 | 5.303122 | 0.384429 |
| wpu-cws-indexed | 4096 | 8 | 128 | 2 | 6853655 | 0.4825 | 0.2891 | 0.335 | 0.1475 | 1.9e-05 | 8.0 | 1.0 | 13.939594 | 0.157184 |
| wpu-cws-learned | 4096 | 8 | 128 | 2 | 6853655 | 0.5275 | 0.3871 | 0.335 | 0.1925 | 5.6e-05 | 16.0 | 1.0 | 13.279414 | 0.548033 |
| wpu-cws-oracle | 4096 | 8 | 128 | 2 | 6853655 | 0.4825 | 0.2891 | 0.335 | 0.1475 | 1.9e-05 | 8.0 | 1.0 | 13.208074 | 0.123091 |
| wpu-cws-indexed | 4096 | 8 | 256 | 2 | 6853655 | 0.335 | 0.0 | 0.335 | 0.0 | 1.8e-05 | 8.0 | 1.0 | 27.839664 | 0.307394 |
| wpu-cws-learned | 4096 | 8 | 256 | 2 | 6853655 | 0.525 | 0.3724 | 0.335 | 0.19 | 2.3e-05 | 16.0 | 1.0 | 24.453781 | 0.270694 |
| wpu-cws-oracle | 4096 | 8 | 256 | 2 | 6853655 | 0.335 | 0.0 | 0.335 | 0.0 | 1.8e-05 | 8.0 | 1.0 | 24.54493 | 0.464574 |

## Best Accuracy By N

| N | model | accuracy_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 4096 | wpu-cws-learned | 0.5275 | 0.5275 | 13.279414 | 6853655 |

## Fastest Forward Latency By N

| N | model | ms_per_sample_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 4096 | wpu-cws-learned | 2.630556 | 0.335 | 2.630556 | 6853655 |

## Interpretation Checklist

- If `wpu-cws-oracle` remains accurate at large `N`, the WPU core is plausible and selector quality is the bottleneck.
- If `wpu-cws-oracle` fails, the current working-set core lacks capacity even with correct causal access.
- If token/graph baselines keep accuracy and remain faster, WPU has no demonstrated advantage in this regime.
- If WPU keeps accuracy while latency grows slower with `N`, the large-`N` hypothesis has preliminary support.

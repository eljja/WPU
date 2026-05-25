# Causal Working Set v2 Results

Source CSV: `docs/experiments/wpu_v2_distilled_selective_pilot.csv`

This report tests the large-`N` claim in the stricter form:

```text
WPU should scale with causal working set K, not total state N, when K is small and identifiable.
```

Short CPU runs are pipeline checks, not paper evidence. Treat accuracy
near the majority baseline as inconclusive until multi-seed, longer-step,
GPU-scale runs are available.

## Raw Condition Summary

| model | N | K | distractors | interaction | threshold | params | accuracy | majority | acc-majority | mse | selected_K | causal_recall | sparse_ratio | local_dense_ratio | dense_compute_ratio | ms/sample | cuda_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wpu-cws-indexed-learned-selective-hybrid | 2048 | 8 | 0 | pairwise | 0.15 | 257141 | 0.4 | 0.333333 | 0.066667 | 0.023468 | 8.0 | 1.0 | 1.0 | 0.0 | 0.0 | 1.864683 | 22.211426 |
| wpu-cws-indexed-learned-selective-hybrid | 2048 | 8 | 0 | pairwise | 0.15 | 257141 | 0.455556 | 0.333333 | 0.122223 | 0.017535 | 8.0 | 1.0 | 1.0 | 0.0 | 0.0 | 1.761427 | 22.211426 |
| wpu-cws-indexed-learned-selective-hybrid | 2048 | 16 | 0 | pairwise | 0.15 | 257141 | 0.377778 | 0.333333 | 0.044445 | 0.041814 | 16.0 | 1.0 | 0.949841 | 0.050159 | 0.322222 | 4.47341 | 22.382812 |
| wpu-cws-indexed-learned-selective-hybrid | 2048 | 16 | 0 | pairwise | 0.15 | 257141 | 0.555556 | 0.333333 | 0.222223 | 0.019632 | 16.0 | 1.0 | 1.0 | 0.0 | 0.0 | 3.843947 | 22.382812 |
| wpu-cws-indexed-learned-selective-hybrid | 2048 | 32 | 0 | pairwise | 0.15 | 257141 | 0.366667 | 0.333333 | 0.033334 | 0.05945 | 32.0 | 1.0 | 0.998331 | 0.001669 | 0.011111 | 6.784367 | 22.731934 |
| wpu-cws-indexed-learned-selective-hybrid | 2048 | 32 | 0 | pairwise | 0.15 | 257141 | 0.444444 | 0.333333 | 0.111111 | 0.010557 | 32.0 | 1.0 | 0.9827 | 0.0173 | 0.111111 | 6.4226 | 22.731934 |

## Aggregated By Model And N

| model | N | K | distractors | interaction | threshold | seeds | params | accuracy_mean | accuracy_ci95 | majority_mean | acc_minus_majority | mse_mean | selected_K_mean | causal_recall_mean | sparse_ratio_mean | local_dense_ratio_mean | dense_compute_ratio_mean | selector_confidence_mean | ms_per_sample_mean | ms_per_sample_ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wpu-cws-indexed-learned-selective-hybrid | 2048 | 8 | 0 | pairwise | 0.15 | 2 | 257141 | 0.427778 | 0.054445 | 0.333333 | 0.094445 | 0.020501 | 8.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.350585 | 1.813055 | 0.101191 |
| wpu-cws-indexed-learned-selective-hybrid | 2048 | 16 | 0 | pairwise | 0.15 | 2 | 257141 | 0.466667 | 0.174222 | 0.333333 | 0.133334 | 0.030723 | 16.0 | 1.0 | 0.974921 | 0.02508 | 0.161111 | 0.486829 | 4.158679 | 0.616874 |
| wpu-cws-indexed-learned-selective-hybrid | 2048 | 32 | 0 | pairwise | 0.15 | 2 | 257141 | 0.405556 | 0.076221 | 0.333333 | 0.072223 | 0.035003 | 32.0 | 1.0 | 0.990515 | 0.009484 | 0.061111 | 0.470967 | 6.603483 | 0.354532 |

## Best Accuracy By N

| N | model | accuracy_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 2048 | wpu-cws-indexed-learned-selective-hybrid | 0.466667 | 0.466667 | 4.158679 | 257141 |

## Best Accuracy By K

| K | model | accuracy_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 8 | wpu-cws-indexed-learned-selective-hybrid | 0.427778 | 0.427778 | 1.813055 | 257141 |
| 16 | wpu-cws-indexed-learned-selective-hybrid | 0.466667 | 0.466667 | 4.158679 | 257141 |
| 32 | wpu-cws-indexed-learned-selective-hybrid | 0.405556 | 0.405556 | 6.603483 | 257141 |

## Fastest Forward Latency By N

| N | model | ms_per_sample_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 2048 | wpu-cws-indexed-learned-selective-hybrid | 1.813055 | 0.427778 | 1.813055 | 257141 |

## Fastest Forward Latency By K

| K | model | ms_per_sample_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 8 | wpu-cws-indexed-learned-selective-hybrid | 1.813055 | 0.427778 | 1.813055 | 257141 |
| 16 | wpu-cws-indexed-learned-selective-hybrid | 4.158679 | 0.466667 | 4.158679 | 257141 |
| 32 | wpu-cws-indexed-learned-selective-hybrid | 6.603483 | 0.405556 | 6.603483 | 257141 |

## Interpretation Checklist

- If `wpu-cws-oracle` remains accurate at large `N`, the WPU core is plausible and selector quality is the bottleneck.
- If `wpu-cws-oracle` fails, the current working-set core lacks capacity even with correct causal access.
- If token/graph baselines keep accuracy and remain faster, WPU has no demonstrated advantage in this regime.
- If WPU keeps accuracy while latency grows slower with `N`, the large-`N` hypothesis has preliminary support.

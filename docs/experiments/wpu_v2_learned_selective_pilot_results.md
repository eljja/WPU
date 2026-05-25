# Causal Working Set v2 Results

Source CSV: `docs/experiments/wpu_v2_learned_selective_pilot.csv`

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
| wpu-cws-indexed-learned-selective-hybrid | 2048 | 8 | 0 | pairwise | 0.5 | 257141 | 0.411111 | 0.333333 | 0.077778 | 0.030942 | 8.0 | 1.0 | 1.0 | 0.0 | 0.0 | 2.077063 | 22.210938 |
| wpu-cws-indexed-learned-selective-hybrid | 2048 | 8 | 0 | pairwise | 0.5 | 257141 | 0.455556 | 0.333333 | 0.122223 | 0.025832 | 8.0 | 1.0 | 1.0 | 0.0 | 0.0 | 1.998697 | 22.210938 |
| wpu-cws-indexed-learned-selective-hybrid | 2048 | 16 | 0 | pairwise | 0.5 | 257141 | 0.355556 | 0.333333 | 0.022223 | 0.039031 | 16.0 | 1.0 | 1.0 | 0.0 | 0.0 | 3.08643 | 22.382324 |
| wpu-cws-indexed-learned-selective-hybrid | 2048 | 16 | 0 | pairwise | 0.5 | 257141 | 0.555556 | 0.333333 | 0.222223 | 0.012599 | 16.0 | 1.0 | 1.0 | 0.0 | 0.0 | 3.14715 | 22.382324 |
| wpu-cws-indexed-learned-selective-hybrid | 2048 | 32 | 0 | pairwise | 0.5 | 257141 | 0.333333 | 0.333333 | 0.0 | 0.08091 | 32.0 | 1.0 | 1.0 | 0.0 | 0.0 | 6.64283 | 22.731445 |
| wpu-cws-indexed-learned-selective-hybrid | 2048 | 32 | 0 | pairwise | 0.5 | 257141 | 0.5 | 0.333333 | 0.166667 | 0.019751 | 32.0 | 1.0 | 1.0 | 0.0 | 0.0 | 4.90467 | 22.731445 |

## Aggregated By Model And N

| model | N | K | distractors | interaction | threshold | seeds | params | accuracy_mean | accuracy_ci95 | majority_mean | acc_minus_majority | mse_mean | selected_K_mean | causal_recall_mean | sparse_ratio_mean | local_dense_ratio_mean | dense_compute_ratio_mean | selector_confidence_mean | ms_per_sample_mean | ms_per_sample_ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wpu-cws-indexed-learned-selective-hybrid | 2048 | 8 | 0 | pairwise | 0.5 | 2 | 257141 | 0.433334 | 0.043556 | 0.333333 | 0.100001 | 0.028387 | 8.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.237154 | 2.03788 | 0.076799 |
| wpu-cws-indexed-learned-selective-hybrid | 2048 | 16 | 0 | pairwise | 0.5 | 2 | 257141 | 0.455556 | 0.196 | 0.333333 | 0.122223 | 0.025815 | 16.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.142065 | 3.11679 | 0.059506 |
| wpu-cws-indexed-learned-selective-hybrid | 2048 | 32 | 0 | pairwise | 0.5 | 2 | 257141 | 0.416666 | 0.163334 | 0.333333 | 0.083334 | 0.050331 | 32.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.394243 | 5.77375 | 1.703397 |

## Best Accuracy By N

| N | model | accuracy_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 2048 | wpu-cws-indexed-learned-selective-hybrid | 0.455556 | 0.455556 | 3.11679 | 257141 |

## Best Accuracy By K

| K | model | accuracy_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 8 | wpu-cws-indexed-learned-selective-hybrid | 0.433334 | 0.433334 | 2.03788 | 257141 |
| 16 | wpu-cws-indexed-learned-selective-hybrid | 0.455556 | 0.455556 | 3.11679 | 257141 |
| 32 | wpu-cws-indexed-learned-selective-hybrid | 0.416666 | 0.416666 | 5.77375 | 257141 |

## Fastest Forward Latency By N

| N | model | ms_per_sample_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 2048 | wpu-cws-indexed-learned-selective-hybrid | 2.03788 | 0.433334 | 2.03788 | 257141 |

## Fastest Forward Latency By K

| K | model | ms_per_sample_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 8 | wpu-cws-indexed-learned-selective-hybrid | 2.03788 | 0.433334 | 2.03788 | 257141 |
| 16 | wpu-cws-indexed-learned-selective-hybrid | 3.11679 | 0.455556 | 3.11679 | 257141 |
| 32 | wpu-cws-indexed-learned-selective-hybrid | 5.77375 | 0.416666 | 5.77375 | 257141 |

## Interpretation Checklist

- If `wpu-cws-oracle` remains accurate at large `N`, the WPU core is plausible and selector quality is the bottleneck.
- If `wpu-cws-oracle` fails, the current working-set core lacks capacity even with correct causal access.
- If token/graph baselines keep accuracy and remain faster, WPU has no demonstrated advantage in this regime.
- If WPU keeps accuracy while latency grows slower with `N`, the large-`N` hypothesis has preliminary support.

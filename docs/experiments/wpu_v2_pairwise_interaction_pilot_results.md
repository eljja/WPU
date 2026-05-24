# Causal Working Set v2 Results

Source CSV: `docs/experiments/wpu_v2_pairwise_interaction_pilot.csv`

This report tests the large-`N` claim in the stricter form:

```text
WPU should scale with causal working set K, not total state N, when K is small and identifiable.
```

Short CPU runs are pipeline checks, not paper evidence. Treat accuracy
near the majority baseline as inconclusive until multi-seed, longer-step,
GPU-scale runs are available.

## Raw Condition Summary

| model | N | K | distractors | interaction | params | accuracy | majority | acc-majority | mse | selected_K | causal_recall | sparse_ratio | local_dense_ratio | ms/sample | cuda_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wpu-cws-indexed-learned-hybrid | 2048 | 8 | 0 | pairwise | 248108 | 0.466667 | 0.333333 | 0.133334 | 0.010809 | 8.0 | 1.0 | 0.930678 | 0.069322 | 1.88628 | 22.159668 |
| wpu-cws-indexed-learned-hybrid | 2048 | 8 | 0 | pairwise | 248108 | 0.5 | 0.333333 | 0.166667 | 0.017008 | 8.0 | 1.0 | 0.924879 | 0.075121 | 1.125727 | 22.159668 |
| wpu-cws-indexed-learned-hybrid | 2048 | 16 | 0 | pairwise | 248108 | 0.444444 | 0.333333 | 0.111111 | 0.016698 | 16.0 | 1.0 | 0.924574 | 0.075426 | 3.138077 | 22.331055 |
| wpu-cws-indexed-learned-hybrid | 2048 | 16 | 0 | pairwise | 248108 | 0.6 | 0.333333 | 0.266667 | 0.020318 | 16.0 | 1.0 | 0.918324 | 0.081676 | 3.094227 | 22.331055 |
| wpu-cws-indexed-learned-hybrid | 2048 | 32 | 0 | pairwise | 248108 | 0.488889 | 0.333333 | 0.155556 | 0.012297 | 32.0 | 1.0 | 0.945728 | 0.054272 | 4.82663 | 23.144531 |
| wpu-cws-indexed-learned-hybrid | 2048 | 32 | 0 | pairwise | 248108 | 0.466667 | 0.333333 | 0.133334 | 0.026958 | 32.0 | 1.0 | 0.902465 | 0.097535 | 5.594877 | 23.144531 |
| wpu-cws-indexed-local-dense | 2048 | 8 | 0 | pairwise | 248108 | 0.533333 | 0.333333 | 0.2 | 0.013131 | 8.0 | 1.0 | 0.0 | 1.0 | 1.71581 | 21.92041 |
| wpu-cws-indexed-local-dense | 2048 | 8 | 0 | pairwise | 248108 | 0.444444 | 0.333333 | 0.111111 | 0.025671 | 8.0 | 1.0 | 0.0 | 1.0 | 2.035457 | 21.92041 |
| wpu-cws-indexed-local-dense | 2048 | 16 | 0 | pairwise | 248108 | 0.522222 | 0.333333 | 0.188889 | 0.06854 | 16.0 | 1.0 | 0.0 | 1.0 | 3.714377 | 22.148926 |
| wpu-cws-indexed-local-dense | 2048 | 16 | 0 | pairwise | 248108 | 0.566667 | 0.333333 | 0.233334 | 0.022575 | 16.0 | 1.0 | 0.0 | 1.0 | 2.575427 | 22.148926 |
| wpu-cws-indexed-local-dense | 2048 | 32 | 0 | pairwise | 248108 | 0.488889 | 0.333333 | 0.155556 | 0.04703 | 32.0 | 1.0 | 0.0 | 1.0 | 5.516307 | 23.039062 |
| wpu-cws-indexed-local-dense | 2048 | 32 | 0 | pairwise | 248108 | 0.544444 | 0.333333 | 0.211111 | 0.020318 | 32.0 | 1.0 | 0.0 | 1.0 | 5.697843 | 23.039062 |
| wpu-cws-indexed-sparse | 2048 | 8 | 0 | pairwise | 49836 | 0.488889 | 0.333333 | 0.155556 | 0.009067 | 8.0 | 1.0 | 1.0 | 0.0 | 1.760497 | 18.73877 |
| wpu-cws-indexed-sparse | 2048 | 8 | 0 | pairwise | 49836 | 0.411111 | 0.333333 | 0.077778 | 0.018853 | 8.0 | 1.0 | 1.0 | 0.0 | 1.644457 | 18.73877 |
| wpu-cws-indexed-sparse | 2048 | 16 | 0 | pairwise | 49836 | 0.433333 | 0.333333 | 0.1 | 0.015519 | 16.0 | 1.0 | 1.0 | 0.0 | 3.518963 | 18.913086 |
| wpu-cws-indexed-sparse | 2048 | 16 | 0 | pairwise | 49836 | 0.577778 | 0.333333 | 0.244445 | 0.026471 | 16.0 | 1.0 | 1.0 | 0.0 | 2.74129 | 18.913086 |
| wpu-cws-indexed-sparse | 2048 | 32 | 0 | pairwise | 49836 | 0.5 | 0.333333 | 0.166667 | 0.011411 | 32.0 | 1.0 | 1.0 | 0.0 | 4.422883 | 19.561035 |
| wpu-cws-indexed-sparse | 2048 | 32 | 0 | pairwise | 49836 | 0.6 | 0.333333 | 0.266667 | 0.012368 | 32.0 | 1.0 | 1.0 | 0.0 | 5.8093 | 19.561035 |

## Aggregated By Model And N

| model | N | K | distractors | interaction | seeds | params | accuracy_mean | accuracy_ci95 | majority_mean | acc_minus_majority | mse_mean | selected_K_mean | causal_recall_mean | sparse_ratio_mean | local_dense_ratio_mean | selector_confidence_mean | ms_per_sample_mean | ms_per_sample_ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wpu-cws-indexed-learned-hybrid | 2048 | 8 | 0 | pairwise | 2 | 248108 | 0.483333 | 0.032666 | 0.333333 | 0.15 | 0.013908 | 8.0 | 1.0 | 0.927779 | 0.072221 | 0.465901 | 1.506003 | 0.745342 |
| wpu-cws-indexed-learned-hybrid | 2048 | 16 | 0 | pairwise | 2 | 248108 | 0.522222 | 0.152445 | 0.333333 | 0.188889 | 0.018508 | 16.0 | 1.0 | 0.921449 | 0.078551 | 0.458937 | 3.116152 | 0.042973 |
| wpu-cws-indexed-learned-hybrid | 2048 | 32 | 0 | pairwise | 2 | 248108 | 0.477778 | 0.021778 | 0.333333 | 0.144445 | 0.019627 | 32.0 | 1.0 | 0.924096 | 0.075903 | 0.489263 | 5.210754 | 0.752882 |
| wpu-cws-indexed-local-dense | 2048 | 8 | 0 | pairwise | 2 | 248108 | 0.488888 | 0.087111 | 0.333333 | 0.155555 | 0.019401 | 8.0 | 1.0 | 0.0 | 1.0 | 0.455036 | 1.875634 | 0.313254 |
| wpu-cws-indexed-local-dense | 2048 | 16 | 0 | pairwise | 2 | 248108 | 0.544444 | 0.043556 | 0.333333 | 0.211112 | 0.045558 | 16.0 | 1.0 | 0.0 | 1.0 | 0.426783 | 3.144902 | 1.116171 |
| wpu-cws-indexed-local-dense | 2048 | 32 | 0 | pairwise | 2 | 248108 | 0.516667 | 0.054444 | 0.333333 | 0.183334 | 0.033674 | 32.0 | 1.0 | 0.0 | 1.0 | 0.565678 | 5.607075 | 0.177905 |
| wpu-cws-indexed-sparse | 2048 | 8 | 0 | pairwise | 2 | 49836 | 0.45 | 0.076222 | 0.333333 | 0.116667 | 0.01396 | 8.0 | 1.0 | 1.0 | 0.0 | 0.511638 | 1.702477 | 0.113719 |
| wpu-cws-indexed-sparse | 2048 | 16 | 0 | pairwise | 2 | 49836 | 0.505556 | 0.141556 | 0.333333 | 0.172223 | 0.020995 | 16.0 | 1.0 | 1.0 | 0.0 | 0.473027 | 3.130126 | 0.76212 |
| wpu-cws-indexed-sparse | 2048 | 32 | 0 | pairwise | 2 | 49836 | 0.55 | 0.098 | 0.333333 | 0.216667 | 0.01189 | 32.0 | 1.0 | 1.0 | 0.0 | 0.373812 | 5.116091 | 1.358689 |

## Best Accuracy By N

| N | model | accuracy_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 2048 | wpu-cws-indexed-sparse | 0.55 | 0.55 | 5.116091 | 49836 |

## Best Accuracy By K

| K | model | accuracy_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 8 | wpu-cws-indexed-local-dense | 0.488888 | 0.488888 | 1.875634 | 248108 |
| 16 | wpu-cws-indexed-local-dense | 0.544444 | 0.544444 | 3.144902 | 248108 |
| 32 | wpu-cws-indexed-sparse | 0.55 | 0.55 | 5.116091 | 49836 |

## Fastest Forward Latency By N

| N | model | ms_per_sample_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 2048 | wpu-cws-indexed-learned-hybrid | 1.506003 | 0.483333 | 1.506003 | 248108 |

## Fastest Forward Latency By K

| K | model | ms_per_sample_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 8 | wpu-cws-indexed-learned-hybrid | 1.506003 | 0.483333 | 1.506003 | 248108 |
| 16 | wpu-cws-indexed-learned-hybrid | 3.116152 | 0.522222 | 3.116152 | 248108 |
| 32 | wpu-cws-indexed-sparse | 5.116091 | 0.55 | 5.116091 | 49836 |

## Interpretation Checklist

- If `wpu-cws-oracle` remains accurate at large `N`, the WPU core is plausible and selector quality is the bottleneck.
- If `wpu-cws-oracle` fails, the current working-set core lacks capacity even with correct causal access.
- If token/graph baselines keep accuracy and remain faster, WPU has no demonstrated advantage in this regime.
- If WPU keeps accuracy while latency grows slower with `N`, the large-`N` hypothesis has preliminary support.

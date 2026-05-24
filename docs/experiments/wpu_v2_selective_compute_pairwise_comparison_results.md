# Causal Working Set v2 Results

Source CSV: `docs/experiments/wpu_v2_selective_compute_pairwise_comparison.csv`

This report tests the large-`N` claim in the stricter form:

```text
WPU should scale with causal working set K, not total state N, when K is small and identifiable.
```

Short CPU runs are pipeline checks, not paper evidence. Treat accuracy
near the majority baseline as inconclusive until multi-seed, longer-step,
GPU-scale runs are available.

## Raw Condition Summary

| model | N | K | distractors | interaction | params | accuracy | majority | acc-majority | mse | selected_K | causal_recall | sparse_ratio | local_dense_ratio | dense_compute_ratio | ms/sample | cuda_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wpu-cws-indexed-geometry-hybrid | 2048 | 8 | 0 | pairwise | 248366 | 0.488889 | 0.333333 | 0.155556 | 0.012552 | 8.0 | 1.0 | 1.0 | 0.0 | 0.0 | 1.79119 | 19.518555 |
| wpu-cws-indexed-geometry-hybrid | 2048 | 8 | 0 | pairwise | 248366 | 0.444444 | 0.333333 | 0.111111 | 0.01271 | 8.0 | 1.0 | 1.0 | 0.0 | 0.0 | 1.904287 | 19.518555 |
| wpu-cws-indexed-geometry-hybrid | 2048 | 16 | 0 | pairwise | 248366 | 0.455556 | 0.333333 | 0.122223 | 0.009926 | 16.0 | 1.0 | 1.0 | 0.0 | 0.0 | 3.92291 | 19.692871 |
| wpu-cws-indexed-geometry-hybrid | 2048 | 16 | 0 | pairwise | 248366 | 0.588889 | 0.333333 | 0.255556 | 0.013888 | 16.0 | 1.0 | 1.0 | 0.0 | 0.0 | 3.462303 | 19.692871 |
| wpu-cws-indexed-geometry-hybrid | 2048 | 32 | 0 | pairwise | 248366 | 0.466667 | 0.333333 | 0.133334 | 0.010713 | 32.0 | 1.0 | 1.0 | 0.0 | 0.0 | 6.508123 | 20.325195 |
| wpu-cws-indexed-geometry-hybrid | 2048 | 32 | 0 | pairwise | 248366 | 0.5 | 0.333333 | 0.166667 | 0.015294 | 32.0 | 1.0 | 1.0 | 0.0 | 0.0 | 6.795653 | 20.325195 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 8 | 0 | pairwise | 248366 | 0.566667 | 0.333333 | 0.233334 | 0.027641 | 8.0 | 1.0 | 0.852851 | 0.147149 | 1.0 | 2.027757 | 22.066895 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 8 | 0 | pairwise | 248366 | 0.555556 | 0.333333 | 0.222223 | 0.023883 | 8.0 | 1.0 | 0.851113 | 0.148887 | 1.0 | 1.947697 | 22.066895 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 16 | 0 | pairwise | 248366 | 0.6 | 0.333333 | 0.266667 | 0.018769 | 16.0 | 1.0 | 0.836898 | 0.163102 | 1.0 | 2.416403 | 22.238281 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 16 | 0 | pairwise | 248366 | 0.588889 | 0.333333 | 0.255556 | 0.021311 | 16.0 | 1.0 | 0.835124 | 0.164876 | 1.0 | 2.320587 | 22.238281 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 32 | 0 | pairwise | 248366 | 0.755556 | 0.333333 | 0.422223 | 0.017097 | 32.0 | 1.0 | 0.823278 | 0.176722 | 1.0 | 8.276697 | 23.05127 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 32 | 0 | pairwise | 248366 | 0.688889 | 0.333333 | 0.355556 | 0.009402 | 32.0 | 1.0 | 0.824409 | 0.175591 | 1.0 | 7.195993 | 23.05127 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 248366 | 0.577778 | 0.333333 | 0.244445 | 0.017521 | 8.0 | 1.0 | 0.926188 | 0.073812 | 0.344444 | 1.389147 | 22.067871 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 248366 | 0.533333 | 0.333333 | 0.2 | 0.019614 | 8.0 | 1.0 | 0.916379 | 0.083621 | 0.388889 | 2.18414 | 22.067871 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 248366 | 0.611111 | 0.333333 | 0.277778 | 0.01366 | 16.0 | 1.0 | 0.890991 | 0.109009 | 0.588889 | 3.66942 | 22.239258 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 248366 | 0.611111 | 0.333333 | 0.277778 | 0.026667 | 16.0 | 1.0 | 0.893163 | 0.106837 | 0.555556 | 2.538557 | 22.239258 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 248366 | 0.711111 | 0.333333 | 0.377778 | 0.023236 | 32.0 | 1.0 | 0.850248 | 0.149752 | 0.811111 | 5.39005 | 22.989746 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 248366 | 0.711111 | 0.333333 | 0.377778 | 0.033617 | 32.0 | 1.0 | 0.848118 | 0.151882 | 0.833333 | 4.76334 | 23.114746 |

## Aggregated By Model And N

| model | N | K | distractors | interaction | seeds | params | accuracy_mean | accuracy_ci95 | majority_mean | acc_minus_majority | mse_mean | selected_K_mean | causal_recall_mean | sparse_ratio_mean | local_dense_ratio_mean | dense_compute_ratio_mean | selector_confidence_mean | ms_per_sample_mean | ms_per_sample_ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wpu-cws-indexed-geometry-hybrid | 2048 | 8 | 0 | pairwise | 2 | 248366 | 0.466666 | 0.043556 | 0.333333 | 0.133333 | 0.012631 | 8.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.447842 | 1.847739 | 0.110835 |
| wpu-cws-indexed-geometry-hybrid | 2048 | 16 | 0 | pairwise | 2 | 248366 | 0.522223 | 0.130666 | 0.333333 | 0.18889 | 0.011907 | 16.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.287571 | 3.692607 | 0.451395 |
| wpu-cws-indexed-geometry-hybrid | 2048 | 32 | 0 | pairwise | 2 | 248366 | 0.483333 | 0.032666 | 0.333333 | 0.15 | 0.013004 | 32.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.227893 | 6.651888 | 0.281779 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 8 | 0 | pairwise | 2 | 248366 | 0.561111 | 0.010889 | 0.333333 | 0.227778 | 0.025762 | 8.0 | 1.0 | 0.851982 | 0.148018 | 1.0 | 0.459996 | 1.987727 | 0.078459 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 16 | 0 | pairwise | 2 | 248366 | 0.594445 | 0.010889 | 0.333333 | 0.261112 | 0.02004 | 16.0 | 1.0 | 0.836011 | 0.163989 | 1.0 | 0.429288 | 2.368495 | 0.0939 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 32 | 0 | pairwise | 2 | 248366 | 0.722222 | 0.065334 | 0.333333 | 0.388889 | 0.01325 | 32.0 | 1.0 | 0.823843 | 0.176156 | 1.0 | 0.376024 | 7.736345 | 1.05909 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 2 | 248366 | 0.555555 | 0.043556 | 0.333333 | 0.222222 | 0.018568 | 8.0 | 1.0 | 0.921284 | 0.078716 | 0.366667 | 0.503778 | 1.786644 | 0.779093 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 2 | 248366 | 0.611111 | 0.0 | 0.333333 | 0.277778 | 0.020164 | 16.0 | 1.0 | 0.892077 | 0.107923 | 0.572223 | 0.379235 | 3.103988 | 1.108246 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 2 | 248366 | 0.711111 | 0.0 | 0.333333 | 0.377778 | 0.028427 | 32.0 | 1.0 | 0.849183 | 0.150817 | 0.822222 | 0.315639 | 5.076695 | 0.614176 |

## Best Accuracy By N

| N | model | accuracy_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 2048 | wpu-cws-indexed-interaction-hybrid | 0.722222 | 0.722222 | 7.736345 | 248366 |

## Best Accuracy By K

| K | model | accuracy_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 8 | wpu-cws-indexed-interaction-hybrid | 0.561111 | 0.561111 | 1.987727 | 248366 |
| 16 | wpu-cws-indexed-selective-interaction-hybrid | 0.611111 | 0.611111 | 3.103988 | 248366 |
| 32 | wpu-cws-indexed-interaction-hybrid | 0.722222 | 0.722222 | 7.736345 | 248366 |

## Fastest Forward Latency By N

| N | model | ms_per_sample_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 2048 | wpu-cws-indexed-selective-interaction-hybrid | 1.786644 | 0.555555 | 1.786644 | 248366 |

## Fastest Forward Latency By K

| K | model | ms_per_sample_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 8 | wpu-cws-indexed-selective-interaction-hybrid | 1.786644 | 0.555555 | 1.786644 | 248366 |
| 16 | wpu-cws-indexed-interaction-hybrid | 2.368495 | 0.594445 | 2.368495 | 248366 |
| 32 | wpu-cws-indexed-selective-interaction-hybrid | 5.076695 | 0.711111 | 5.076695 | 248366 |

## Interpretation Checklist

- If `wpu-cws-oracle` remains accurate at large `N`, the WPU core is plausible and selector quality is the bottleneck.
- If `wpu-cws-oracle` fails, the current working-set core lacks capacity even with correct causal access.
- If token/graph baselines keep accuracy and remain faster, WPU has no demonstrated advantage in this regime.
- If WPU keeps accuracy while latency grows slower with `N`, the large-`N` hypothesis has preliminary support.

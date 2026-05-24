# Causal Working Set v2 Results

Source CSV: `docs/experiments/wpu_v2_selective_5seed_validation.csv`

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
| wpu-cws-indexed-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.15 | 248366 | 0.566667 | 0.333333 | 0.233334 | 0.027641 | 8.0 | 1.0 | 0.852851 | 0.147149 | 1.0 | 2.012823 | 22.066895 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.15 | 248366 | 0.555556 | 0.333333 | 0.222223 | 0.023883 | 8.0 | 1.0 | 0.851113 | 0.148887 | 1.0 | 1.576033 | 22.066895 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.15 | 248366 | 0.555556 | 0.333333 | 0.222223 | 0.020055 | 8.0 | 1.0 | 0.846243 | 0.153757 | 1.0 | 2.03509 | 22.066895 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.15 | 248366 | 0.633333 | 0.333333 | 0.3 | 0.012873 | 8.0 | 1.0 | 0.840519 | 0.159481 | 1.0 | 1.881377 | 22.066895 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.15 | 248366 | 0.588889 | 0.333333 | 0.255556 | 0.018932 | 8.0 | 1.0 | 0.859899 | 0.140101 | 1.0 | 1.715987 | 22.066895 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.15 | 248366 | 0.6 | 0.333333 | 0.266667 | 0.018769 | 16.0 | 1.0 | 0.836898 | 0.163102 | 1.0 | 2.998433 | 22.238281 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.15 | 248366 | 0.588889 | 0.333333 | 0.255556 | 0.021311 | 16.0 | 1.0 | 0.835124 | 0.164876 | 1.0 | 3.020647 | 22.238281 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.15 | 248366 | 0.688889 | 0.333333 | 0.355556 | 0.016649 | 16.0 | 1.0 | 0.830805 | 0.169195 | 1.0 | 3.210117 | 22.238281 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.15 | 248366 | 0.544444 | 0.333333 | 0.211111 | 0.013034 | 16.0 | 1.0 | 0.83415 | 0.16585 | 1.0 | 2.99236 | 22.238281 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.15 | 248366 | 0.522222 | 0.333333 | 0.188889 | 0.015609 | 16.0 | 1.0 | 0.834133 | 0.165867 | 1.0 | 3.16609 | 22.238281 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.15 | 248366 | 0.755556 | 0.333333 | 0.422223 | 0.017097 | 32.0 | 1.0 | 0.823278 | 0.176722 | 1.0 | 5.283517 | 23.05127 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.15 | 248366 | 0.688889 | 0.333333 | 0.355556 | 0.009402 | 32.0 | 1.0 | 0.824409 | 0.175591 | 1.0 | 4.94572 | 23.05127 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.15 | 248366 | 0.688889 | 0.333333 | 0.355556 | 0.009591 | 32.0 | 1.0 | 0.820545 | 0.179455 | 1.0 | 3.853433 | 23.05127 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.15 | 248366 | 0.666667 | 0.333333 | 0.333334 | 0.024255 | 32.0 | 1.0 | 0.822228 | 0.177772 | 1.0 | 5.244653 | 23.05127 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.15 | 248366 | 0.655556 | 0.333333 | 0.322223 | 0.012588 | 32.0 | 1.0 | 0.820506 | 0.179494 | 1.0 | 5.421617 | 23.05127 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.15 | 248366 | 0.577778 | 0.333333 | 0.244445 | 0.017521 | 8.0 | 1.0 | 0.926188 | 0.073812 | 0.344444 | 1.599207 | 22.067871 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.15 | 248366 | 0.533333 | 0.333333 | 0.2 | 0.019614 | 8.0 | 1.0 | 0.916379 | 0.083621 | 0.388889 | 1.91048 | 22.067871 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.15 | 248366 | 0.511111 | 0.333333 | 0.177778 | 0.014302 | 8.0 | 1.0 | 0.910791 | 0.089209 | 0.411111 | 1.89625 | 22.067871 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.15 | 248366 | 0.566667 | 0.333333 | 0.233334 | 0.010163 | 8.0 | 1.0 | 0.89814 | 0.10186 | 0.477778 | 1.538457 | 22.067871 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.15 | 248366 | 0.6 | 0.333333 | 0.266667 | 0.014024 | 8.0 | 1.0 | 0.924198 | 0.075802 | 0.388889 | 1.97834 | 22.067871 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.15 | 248366 | 0.611111 | 0.333333 | 0.277778 | 0.01366 | 16.0 | 1.0 | 0.890991 | 0.109009 | 0.588889 | 2.7076 | 22.239258 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.15 | 248366 | 0.611111 | 0.333333 | 0.277778 | 0.026667 | 16.0 | 1.0 | 0.893163 | 0.106837 | 0.555556 | 3.191373 | 22.239258 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.15 | 248366 | 0.6 | 0.333333 | 0.266667 | 0.021056 | 16.0 | 1.0 | 0.883263 | 0.116737 | 0.588889 | 3.04414 | 22.263672 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.15 | 248366 | 0.577778 | 0.333333 | 0.244445 | 0.015567 | 16.0 | 1.0 | 0.883906 | 0.116094 | 0.611111 | 3.105607 | 22.404785 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.15 | 248366 | 0.622222 | 0.333333 | 0.288889 | 0.0214 | 16.0 | 1.0 | 0.896102 | 0.103899 | 0.533333 | 3.126233 | 22.333984 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.15 | 248366 | 0.711111 | 0.333333 | 0.377778 | 0.023236 | 32.0 | 1.0 | 0.850248 | 0.149752 | 0.811111 | 5.104307 | 22.989746 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.15 | 248366 | 0.711111 | 0.333333 | 0.377778 | 0.033617 | 32.0 | 1.0 | 0.848118 | 0.151882 | 0.833333 | 5.355793 | 23.114746 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.15 | 248366 | 0.611111 | 0.333333 | 0.277778 | 0.023624 | 32.0 | 1.0 | 0.839179 | 0.160821 | 0.866667 | 5.423843 | 23.240234 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.15 | 248366 | 0.688889 | 0.333333 | 0.355556 | 0.019512 | 32.0 | 1.0 | 0.847447 | 0.152553 | 0.822222 | 5.379577 | 23.240234 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.15 | 248366 | 0.666667 | 0.333333 | 0.333334 | 0.02283 | 32.0 | 1.0 | 0.844467 | 0.155533 | 0.833333 | 5.392017 | 23.114746 |

## Aggregated By Model And N

| model | N | K | distractors | interaction | threshold | seeds | params | accuracy_mean | accuracy_ci95 | majority_mean | acc_minus_majority | mse_mean | selected_K_mean | causal_recall_mean | sparse_ratio_mean | local_dense_ratio_mean | dense_compute_ratio_mean | selector_confidence_mean | ms_per_sample_mean | ms_per_sample_ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wpu-cws-indexed-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.15 | 5 | 248366 | 0.58 | 0.028727 | 0.333333 | 0.246667 | 0.020677 | 8.0 | 1.0 | 0.850125 | 0.149875 | 1.0 | 0.482388 | 1.844262 | 0.172328 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.15 | 5 | 248366 | 0.557778 | 0.031105 | 0.333333 | 0.224445 | 0.015125 | 8.0 | 1.0 | 0.915139 | 0.084861 | 0.402222 | 0.508301 | 1.784547 | 0.175747 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.15 | 5 | 248366 | 0.588889 | 0.056371 | 0.333333 | 0.255556 | 0.017074 | 16.0 | 1.0 | 0.834222 | 0.165778 | 1.0 | 0.440196 | 3.077529 | 0.089998 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.15 | 5 | 248366 | 0.604444 | 0.01477 | 0.333333 | 0.271111 | 0.01967 | 16.0 | 1.0 | 0.889485 | 0.110515 | 0.575556 | 0.389199 | 3.034991 | 0.166908 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.15 | 5 | 248366 | 0.691111 | 0.034018 | 0.333333 | 0.357778 | 0.014587 | 32.0 | 1.0 | 0.822193 | 0.177807 | 1.0 | 0.441848 | 4.949788 | 0.558334 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.15 | 5 | 248366 | 0.677778 | 0.036441 | 0.333333 | 0.344445 | 0.024564 | 32.0 | 1.0 | 0.845892 | 0.154108 | 0.833333 | 0.414218 | 5.331107 | 0.113195 |

## Best Accuracy By N

| N | model | accuracy_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 2048 | wpu-cws-indexed-interaction-hybrid | 0.691111 | 0.691111 | 4.949788 | 248366 |

## Best Accuracy By K

| K | model | accuracy_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 8 | wpu-cws-indexed-interaction-hybrid | 0.58 | 0.58 | 1.844262 | 248366 |
| 16 | wpu-cws-indexed-selective-interaction-hybrid | 0.604444 | 0.604444 | 3.034991 | 248366 |
| 32 | wpu-cws-indexed-interaction-hybrid | 0.691111 | 0.691111 | 4.949788 | 248366 |

## Fastest Forward Latency By N

| N | model | ms_per_sample_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 2048 | wpu-cws-indexed-selective-interaction-hybrid | 1.784547 | 0.557778 | 1.784547 | 248366 |

## Fastest Forward Latency By K

| K | model | ms_per_sample_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 8 | wpu-cws-indexed-selective-interaction-hybrid | 1.784547 | 0.557778 | 1.784547 | 248366 |
| 16 | wpu-cws-indexed-selective-interaction-hybrid | 3.034991 | 0.604444 | 3.034991 | 248366 |
| 32 | wpu-cws-indexed-interaction-hybrid | 4.949788 | 0.691111 | 4.949788 | 248366 |

## Interpretation Checklist

- If `wpu-cws-oracle` remains accurate at large `N`, the WPU core is plausible and selector quality is the bottleneck.
- If `wpu-cws-oracle` fails, the current working-set core lacks capacity even with correct causal access.
- If token/graph baselines keep accuracy and remain faster, WPU has no demonstrated advantage in this regime.
- If WPU keeps accuracy while latency grows slower with `N`, the large-`N` hypothesis has preliminary support.

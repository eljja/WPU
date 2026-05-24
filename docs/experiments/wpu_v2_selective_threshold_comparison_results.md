# Causal Working Set v2 Results

Source CSV: `docs/experiments/wpu_v2_selective_threshold_comparison.csv`

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
| wpu-cws-indexed-geometry-hybrid | 2048 | 8 | 0 | pairwise |  | 248366 | 0.488889 | 0.333333 | 0.155556 | 0.012552 | 8.0 | 1.0 | 1.0 | 0.0 | 0.0 | 1.79119 | 19.518555 |
| wpu-cws-indexed-geometry-hybrid | 2048 | 8 | 0 | pairwise |  | 248366 | 0.444444 | 0.333333 | 0.111111 | 0.01271 | 8.0 | 1.0 | 1.0 | 0.0 | 0.0 | 1.904287 | 19.518555 |
| wpu-cws-indexed-geometry-hybrid | 2048 | 16 | 0 | pairwise |  | 248366 | 0.455556 | 0.333333 | 0.122223 | 0.009926 | 16.0 | 1.0 | 1.0 | 0.0 | 0.0 | 3.92291 | 19.692871 |
| wpu-cws-indexed-geometry-hybrid | 2048 | 16 | 0 | pairwise |  | 248366 | 0.588889 | 0.333333 | 0.255556 | 0.013888 | 16.0 | 1.0 | 1.0 | 0.0 | 0.0 | 3.462303 | 19.692871 |
| wpu-cws-indexed-geometry-hybrid | 2048 | 32 | 0 | pairwise |  | 248366 | 0.466667 | 0.333333 | 0.133334 | 0.010713 | 32.0 | 1.0 | 1.0 | 0.0 | 0.0 | 6.508123 | 20.325195 |
| wpu-cws-indexed-geometry-hybrid | 2048 | 32 | 0 | pairwise |  | 248366 | 0.5 | 0.333333 | 0.166667 | 0.015294 | 32.0 | 1.0 | 1.0 | 0.0 | 0.0 | 6.795653 | 20.325195 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 8 | 0 | pairwise |  | 248366 | 0.566667 | 0.333333 | 0.233334 | 0.027641 | 8.0 | 1.0 | 0.852851 | 0.147149 | 1.0 | 2.027757 | 22.066895 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 8 | 0 | pairwise |  | 248366 | 0.555556 | 0.333333 | 0.222223 | 0.023883 | 8.0 | 1.0 | 0.851113 | 0.148887 | 1.0 | 1.947697 | 22.066895 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 16 | 0 | pairwise |  | 248366 | 0.6 | 0.333333 | 0.266667 | 0.018769 | 16.0 | 1.0 | 0.836898 | 0.163102 | 1.0 | 2.416403 | 22.238281 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 16 | 0 | pairwise |  | 248366 | 0.588889 | 0.333333 | 0.255556 | 0.021311 | 16.0 | 1.0 | 0.835124 | 0.164876 | 1.0 | 2.320587 | 22.238281 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 32 | 0 | pairwise |  | 248366 | 0.755556 | 0.333333 | 0.422223 | 0.017097 | 32.0 | 1.0 | 0.823278 | 0.176722 | 1.0 | 8.276697 | 23.05127 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 32 | 0 | pairwise |  | 248366 | 0.688889 | 0.333333 | 0.355556 | 0.009402 | 32.0 | 1.0 | 0.824409 | 0.175591 | 1.0 | 7.195993 | 23.05127 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.05 | 248366 | 0.566667 | 0.333333 | 0.233334 | 0.027641 | 8.0 | 1.0 | 0.852851 | 0.147149 | 1.0 | 2.318373 | 22.067871 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.05 | 248366 | 0.555556 | 0.333333 | 0.222223 | 0.023883 | 8.0 | 1.0 | 0.851113 | 0.148887 | 1.0 | 2.406283 | 22.067871 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.05 | 248366 | 0.6 | 0.333333 | 0.266667 | 0.018769 | 16.0 | 1.0 | 0.836898 | 0.163102 | 1.0 | 3.07265 | 22.475098 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.05 | 248366 | 0.588889 | 0.333333 | 0.255556 | 0.021311 | 16.0 | 1.0 | 0.835124 | 0.164876 | 1.0 | 2.214503 | 22.475098 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.05 | 248366 | 0.755556 | 0.333333 | 0.422223 | 0.017097 | 32.0 | 1.0 | 0.823278 | 0.176722 | 1.0 | 5.559453 | 23.365234 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.05 | 248366 | 0.688889 | 0.333333 | 0.355556 | 0.009402 | 32.0 | 1.0 | 0.824409 | 0.175591 | 1.0 | 3.894497 | 23.365234 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.1 | 248366 | 0.577778 | 0.333333 | 0.244445 | 0.029825 | 8.0 | 1.0 | 0.873244 | 0.126756 | 0.766667 | 1.773377 | 22.067871 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.1 | 248366 | 0.533333 | 0.333333 | 0.2 | 0.019057 | 8.0 | 1.0 | 0.873621 | 0.126379 | 0.744444 | 1.912627 | 22.067871 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.1 | 248366 | 0.633333 | 0.333333 | 0.3 | 0.031572 | 16.0 | 1.0 | 0.836898 | 0.163102 | 1.0 | 2.190423 | 22.475098 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.1 | 248366 | 0.6 | 0.333333 | 0.266667 | 0.021476 | 16.0 | 1.0 | 0.837259 | 0.162741 | 0.977778 | 2.91787 | 22.475098 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.1 | 248366 | 0.755556 | 0.333333 | 0.422223 | 0.017097 | 32.0 | 1.0 | 0.823278 | 0.176722 | 1.0 | 3.889557 | 23.365234 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.1 | 248366 | 0.688889 | 0.333333 | 0.355556 | 0.009402 | 32.0 | 1.0 | 0.824409 | 0.175591 | 1.0 | 4.60713 | 23.365234 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.15 | 248366 | 0.577778 | 0.333333 | 0.244445 | 0.017521 | 8.0 | 1.0 | 0.926188 | 0.073812 | 0.344444 | 1.27918 | 22.067871 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.15 | 248366 | 0.533333 | 0.333333 | 0.2 | 0.019614 | 8.0 | 1.0 | 0.916379 | 0.083621 | 0.388889 | 2.253137 | 22.067871 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.15 | 248366 | 0.611111 | 0.333333 | 0.277778 | 0.01366 | 16.0 | 1.0 | 0.890991 | 0.109009 | 0.588889 | 3.535167 | 22.239258 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.15 | 248366 | 0.611111 | 0.333333 | 0.277778 | 0.026667 | 16.0 | 1.0 | 0.893163 | 0.106837 | 0.555556 | 3.2383 | 22.239258 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.15 | 248366 | 0.711111 | 0.333333 | 0.377778 | 0.023236 | 32.0 | 1.0 | 0.850248 | 0.149752 | 0.811111 | 3.867613 | 22.989746 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.15 | 248366 | 0.711111 | 0.333333 | 0.377778 | 0.033617 | 32.0 | 1.0 | 0.848118 | 0.151882 | 0.833333 | 5.983023 | 23.114746 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.2 | 248366 | 0.522222 | 0.333333 | 0.188889 | 0.023706 | 8.0 | 1.0 | 0.962613 | 0.037387 | 0.133333 | 2.008877 | 22.067871 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.2 | 248366 | 0.466667 | 0.333333 | 0.133334 | 0.023822 | 8.0 | 1.0 | 0.949125 | 0.050875 | 0.2 | 1.988327 | 22.067871 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.2 | 248366 | 0.522222 | 0.333333 | 0.188889 | 0.016724 | 16.0 | 1.0 | 0.967107 | 0.032893 | 0.144444 | 3.735173 | 22.239258 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.2 | 248366 | 0.611111 | 0.333333 | 0.277778 | 0.022974 | 16.0 | 1.0 | 0.956076 | 0.043924 | 0.188889 | 3.086497 | 22.239258 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.2 | 248366 | 0.566667 | 0.333333 | 0.233334 | 0.01606 | 32.0 | 1.0 | 0.954493 | 0.045507 | 0.211111 | 5.21392 | 22.588379 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.2 | 248366 | 0.588889 | 0.333333 | 0.255556 | 0.017502 | 32.0 | 1.0 | 0.957077 | 0.042923 | 0.2 | 5.399627 | 22.614258 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.3 | 248366 | 0.488889 | 0.333333 | 0.155556 | 0.014071 | 8.0 | 1.0 | 0.983645 | 0.016355 | 0.044444 | 1.744093 | 21.311523 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.3 | 248366 | 0.444444 | 0.333333 | 0.111111 | 0.017865 | 8.0 | 1.0 | 0.988035 | 0.011965 | 0.033333 | 1.673087 | 21.311523 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.3 | 248366 | 0.466667 | 0.333333 | 0.133334 | 0.011755 | 16.0 | 1.0 | 1.0 | 0.0 | 0.0 | 2.958573 | 21.48291 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.3 | 248366 | 0.566667 | 0.333333 | 0.233334 | 0.034923 | 16.0 | 1.0 | 0.996465 | 0.003535 | 0.011111 | 2.868543 | 21.48291 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.3 | 248366 | 0.466667 | 0.333333 | 0.133334 | 0.008549 | 32.0 | 1.0 | 1.0 | 0.0 | 0.0 | 5.583453 | 20.319336 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.3 | 248366 | 0.477778 | 0.333333 | 0.144445 | 0.014171 | 32.0 | 1.0 | 1.0 | 0.0 | 0.0 | 3.676783 | 20.319336 |

## Aggregated By Model And N

| model | N | K | distractors | interaction | threshold | seeds | params | accuracy_mean | accuracy_ci95 | majority_mean | acc_minus_majority | mse_mean | selected_K_mean | causal_recall_mean | sparse_ratio_mean | local_dense_ratio_mean | dense_compute_ratio_mean | selector_confidence_mean | ms_per_sample_mean | ms_per_sample_ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wpu-cws-indexed-geometry-hybrid | 2048 | 8 | 0 | pairwise |  | 2 | 248366 | 0.466666 | 0.043556 | 0.333333 | 0.133333 | 0.012631 | 8.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.447842 | 1.847739 | 0.110835 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 8 | 0 | pairwise |  | 2 | 248366 | 0.561111 | 0.010889 | 0.333333 | 0.227778 | 0.025762 | 8.0 | 1.0 | 0.851982 | 0.148018 | 1.0 | 0.459996 | 1.987727 | 0.078459 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.05 | 2 | 248366 | 0.561111 | 0.010889 | 0.333333 | 0.227778 | 0.025762 | 8.0 | 1.0 | 0.851982 | 0.148018 | 1.0 | 0.459911 | 2.362328 | 0.086152 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.1 | 2 | 248366 | 0.555555 | 0.043556 | 0.333333 | 0.222222 | 0.024441 | 8.0 | 1.0 | 0.873433 | 0.126567 | 0.755556 | 0.530701 | 1.843002 | 0.136465 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.15 | 2 | 248366 | 0.555555 | 0.043556 | 0.333333 | 0.222222 | 0.018568 | 8.0 | 1.0 | 0.921284 | 0.078716 | 0.366667 | 0.503778 | 1.766158 | 0.954478 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.2 | 2 | 248366 | 0.494444 | 0.054444 | 0.333333 | 0.161111 | 0.023764 | 8.0 | 1.0 | 0.955869 | 0.044131 | 0.166666 | 0.576944 | 1.998602 | 0.020139 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 8 | 0 | pairwise | 0.3 | 2 | 248366 | 0.466666 | 0.043556 | 0.333333 | 0.133333 | 0.015968 | 8.0 | 1.0 | 0.98584 | 0.01416 | 0.038888 | 0.491978 | 1.70859 | 0.069586 |
| wpu-cws-indexed-geometry-hybrid | 2048 | 16 | 0 | pairwise |  | 2 | 248366 | 0.522223 | 0.130666 | 0.333333 | 0.18889 | 0.011907 | 16.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.287571 | 3.692607 | 0.451395 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 16 | 0 | pairwise |  | 2 | 248366 | 0.594445 | 0.010889 | 0.333333 | 0.261112 | 0.02004 | 16.0 | 1.0 | 0.836011 | 0.163989 | 1.0 | 0.429288 | 2.368495 | 0.0939 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.05 | 2 | 248366 | 0.594445 | 0.010889 | 0.333333 | 0.261112 | 0.02004 | 16.0 | 1.0 | 0.836011 | 0.163989 | 1.0 | 0.430071 | 2.643576 | 0.840984 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.1 | 2 | 248366 | 0.616667 | 0.032666 | 0.333333 | 0.283334 | 0.026524 | 16.0 | 1.0 | 0.837079 | 0.162921 | 0.988889 | 0.347274 | 2.554146 | 0.712898 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.15 | 2 | 248366 | 0.611111 | 0.0 | 0.333333 | 0.277778 | 0.020164 | 16.0 | 1.0 | 0.892077 | 0.107923 | 0.572223 | 0.379235 | 3.386734 | 0.29093 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.2 | 2 | 248366 | 0.566666 | 0.087111 | 0.333333 | 0.233333 | 0.019849 | 16.0 | 1.0 | 0.961592 | 0.038408 | 0.166666 | 0.307859 | 3.410835 | 0.635702 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 16 | 0 | pairwise | 0.3 | 2 | 248366 | 0.516667 | 0.098 | 0.333333 | 0.183334 | 0.023339 | 16.0 | 1.0 | 0.998233 | 0.001767 | 0.005555 | 0.344139 | 2.913558 | 0.088229 |
| wpu-cws-indexed-geometry-hybrid | 2048 | 32 | 0 | pairwise |  | 2 | 248366 | 0.483333 | 0.032666 | 0.333333 | 0.15 | 0.013004 | 32.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.227893 | 6.651888 | 0.281779 |
| wpu-cws-indexed-interaction-hybrid | 2048 | 32 | 0 | pairwise |  | 2 | 248366 | 0.722222 | 0.065334 | 0.333333 | 0.388889 | 0.01325 | 32.0 | 1.0 | 0.823843 | 0.176156 | 1.0 | 0.376024 | 7.736345 | 1.05909 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.05 | 2 | 248366 | 0.722222 | 0.065334 | 0.333333 | 0.388889 | 0.01325 | 32.0 | 1.0 | 0.823843 | 0.176156 | 1.0 | 0.376123 | 4.726975 | 1.631657 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.1 | 2 | 248366 | 0.722222 | 0.065334 | 0.333333 | 0.388889 | 0.01325 | 32.0 | 1.0 | 0.823843 | 0.176156 | 1.0 | 0.376123 | 4.248343 | 0.703222 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.15 | 2 | 248366 | 0.711111 | 0.0 | 0.333333 | 0.377778 | 0.028427 | 32.0 | 1.0 | 0.849183 | 0.150817 | 0.822222 | 0.315639 | 4.925318 | 2.073102 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.2 | 2 | 248366 | 0.577778 | 0.021778 | 0.333333 | 0.244445 | 0.016781 | 32.0 | 1.0 | 0.955785 | 0.044215 | 0.205556 | 0.35307 | 5.306774 | 0.181993 |
| wpu-cws-indexed-selective-interaction-hybrid | 2048 | 32 | 0 | pairwise | 0.3 | 2 | 248366 | 0.472222 | 0.010889 | 0.333333 | 0.138889 | 0.01136 | 32.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.238539 | 4.630118 | 1.868537 |

## Best Accuracy By N

| N | model | accuracy_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 2048 | wpu-cws-indexed-interaction-hybrid | 0.722222 | 0.722222 | 7.736345 | 248366 |

## Best Accuracy By K

| K | model | accuracy_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 8 | wpu-cws-indexed-interaction-hybrid | 0.561111 | 0.561111 | 1.987727 | 248366 |
| 16 | wpu-cws-indexed-selective-interaction-hybrid | 0.616667 | 0.616667 | 2.554146 | 248366 |
| 32 | wpu-cws-indexed-interaction-hybrid | 0.722222 | 0.722222 | 7.736345 | 248366 |

## Fastest Forward Latency By N

| N | model | ms_per_sample_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 2048 | wpu-cws-indexed-selective-interaction-hybrid | 1.70859 | 0.466666 | 1.70859 | 248366 |

## Fastest Forward Latency By K

| K | model | ms_per_sample_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 8 | wpu-cws-indexed-selective-interaction-hybrid | 1.70859 | 0.466666 | 1.70859 | 248366 |
| 16 | wpu-cws-indexed-interaction-hybrid | 2.368495 | 0.594445 | 2.368495 | 248366 |
| 32 | wpu-cws-indexed-selective-interaction-hybrid | 4.248343 | 0.722222 | 4.248343 | 248366 |

## Interpretation Checklist

- If `wpu-cws-oracle` remains accurate at large `N`, the WPU core is plausible and selector quality is the bottleneck.
- If `wpu-cws-oracle` fails, the current working-set core lacks capacity even with correct causal access.
- If token/graph baselines keep accuracy and remain faster, WPU has no demonstrated advantage in this regime.
- If WPU keeps accuracy while latency grows slower with `N`, the large-`N` hypothesis has preliminary support.

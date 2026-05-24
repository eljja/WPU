# Causal Working Set v2 Results

Source CSV: `docs/experiments/wpu_v2_adaptive_hybrid_pilot.csv`

This report tests the large-`N` claim in the stricter form:

```text
WPU should scale with causal working set K, not total state N, when K is small and identifiable.
```

Short CPU runs are pipeline checks, not paper evidence. Treat accuracy
near the majority baseline as inconclusive until multi-seed, longer-step,
GPU-scale runs are available.

## Raw Condition Summary

| model | N | K | distractors | params | accuracy | majority | acc-majority | mse | selected_K | causal_recall | sparse_ratio | local_dense_ratio | ms/sample | cuda_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wpu-cws-indexed-adaptive-hybrid | 4096 | 4 | 0 | 937511 | 0.641667 | 0.333333 | 0.308334 | 0.047154 | 4.0 | 1.0 | 0.0 | 1.0 | 0.789513 | 34.197754 |
| wpu-cws-indexed-adaptive-hybrid | 4096 | 4 | 0 | 937511 | 0.741667 | 0.333333 | 0.408334 | 0.06057 | 4.0 | 1.0 | 0.441667 | 0.558333 | 0.805438 | 34.197754 |
| wpu-cws-indexed-adaptive-hybrid | 4096 | 8 | 0 | 937511 | 0.591667 | 0.333333 | 0.258334 | 0.032757 | 8.0 | 1.0 | 0.0 | 1.0 | 1.353873 | 34.313965 |
| wpu-cws-indexed-adaptive-hybrid | 4096 | 8 | 0 | 937511 | 0.825 | 0.333333 | 0.491667 | 0.039661 | 8.0 | 1.0 | 0.575 | 0.425 | 1.840742 | 34.313965 |
| wpu-cws-indexed-adaptive-hybrid | 4096 | 16 | 0 | 937511 | 0.725 | 0.333333 | 0.391667 | 0.048167 | 16.0 | 1.0 | 0.866667 | 0.133333 | 2.074708 | 34.546387 |
| wpu-cws-indexed-adaptive-hybrid | 4096 | 16 | 0 | 937511 | 0.691667 | 0.333333 | 0.358334 | 0.022932 | 16.0 | 1.0 | 0.716667 | 0.283333 | 2.3756 | 34.546387 |
| wpu-cws-indexed-adaptive-hybrid | 4096 | 32 | 0 | 937511 | 0.825 | 0.333333 | 0.491667 | 0.062498 | 32.0 | 1.0 | 0.45 | 0.55 | 4.056215 | 35.227051 |
| wpu-cws-indexed-adaptive-hybrid | 4096 | 32 | 0 | 937511 | 0.766667 | 0.333333 | 0.433334 | 0.069478 | 32.0 | 1.0 | 0.0 | 1.0 | 4.26555 | 35.227051 |
| wpu-cws-indexed-adaptive-hybrid | 4096 | 64 | 0 | 937511 | 0.808333 | 0.333333 | 0.475 | 0.035446 | 64.0 | 1.0 | 0.0 | 1.0 | 8.72837 | 37.901855 |
| wpu-cws-indexed-adaptive-hybrid | 4096 | 64 | 0 | 937511 | 0.733333 | 0.333333 | 0.4 | 0.033056 | 64.0 | 1.0 | 0.0 | 1.0 | 11.31471 | 37.901855 |
| wpu-cws-indexed-local-dense | 4096 | 4 | 0 | 937511 | 0.658333 | 0.333333 | 0.325 | 0.058341 | 4.0 | 1.0 | 0.0 | 1.0 | 0.81721 | 34.197754 |
| wpu-cws-indexed-local-dense | 4096 | 4 | 0 | 937511 | 0.766667 | 0.333333 | 0.433334 | 0.112326 | 4.0 | 1.0 | 0.0 | 1.0 | 0.818115 | 34.197754 |
| wpu-cws-indexed-local-dense | 4096 | 8 | 0 | 937511 | 0.708333 | 0.333333 | 0.375 | 0.056999 | 8.0 | 1.0 | 0.0 | 1.0 | 1.21728 | 34.313965 |
| wpu-cws-indexed-local-dense | 4096 | 8 | 0 | 937511 | 0.841667 | 0.333333 | 0.508334 | 0.0225 | 8.0 | 1.0 | 0.0 | 1.0 | 1.304153 | 34.313965 |
| wpu-cws-indexed-local-dense | 4096 | 16 | 0 | 937511 | 0.816667 | 0.333333 | 0.483334 | 0.057363 | 16.0 | 1.0 | 0.0 | 1.0 | 2.130893 | 34.546387 |
| wpu-cws-indexed-local-dense | 4096 | 16 | 0 | 937511 | 0.758333 | 0.333333 | 0.425 | 0.084841 | 16.0 | 1.0 | 0.0 | 1.0 | 2.038915 | 34.546387 |
| wpu-cws-indexed-local-dense | 4096 | 32 | 0 | 937511 | 0.808333 | 0.333333 | 0.475 | 0.039782 | 32.0 | 1.0 | 0.0 | 1.0 | 3.471087 | 35.227051 |
| wpu-cws-indexed-local-dense | 4096 | 32 | 0 | 937511 | 0.791667 | 0.333333 | 0.458334 | 0.029115 | 32.0 | 1.0 | 0.0 | 1.0 | 3.552913 | 35.227051 |
| wpu-cws-indexed-local-dense | 4096 | 64 | 0 | 937511 | 0.808333 | 0.333333 | 0.475 | 0.035446 | 64.0 | 1.0 | 0.0 | 1.0 | 6.643262 | 37.901855 |
| wpu-cws-indexed-local-dense | 4096 | 64 | 0 | 937511 | 0.733333 | 0.333333 | 0.4 | 0.033056 | 64.0 | 1.0 | 0.0 | 1.0 | 6.624503 | 37.901855 |
| wpu-cws-indexed-sparse | 4096 | 4 | 0 | 147751 | 0.716667 | 0.333333 | 0.383334 | 0.023356 | 4.0 | 1.0 | 1.0 | 0.0 | 0.996528 | 21.646973 |
| wpu-cws-indexed-sparse | 4096 | 4 | 0 | 147751 | 0.733333 | 0.333333 | 0.4 | 0.027076 | 4.0 | 1.0 | 1.0 | 0.0 | 1.003598 | 21.646973 |
| wpu-cws-indexed-sparse | 4096 | 8 | 0 | 147751 | 0.775 | 0.333333 | 0.441667 | 0.018413 | 8.0 | 1.0 | 1.0 | 0.0 | 1.197295 | 21.763184 |
| wpu-cws-indexed-sparse | 4096 | 8 | 0 | 147751 | 0.741667 | 0.333333 | 0.408334 | 0.041393 | 8.0 | 1.0 | 1.0 | 0.0 | 1.221932 | 21.763184 |
| wpu-cws-indexed-sparse | 4096 | 16 | 0 | 147751 | 0.783333 | 0.333333 | 0.45 | 0.036373 | 16.0 | 1.0 | 1.0 | 0.0 | 2.022647 | 21.995605 |
| wpu-cws-indexed-sparse | 4096 | 16 | 0 | 147751 | 0.65 | 0.333333 | 0.316667 | 0.03911 | 16.0 | 1.0 | 1.0 | 0.0 | 1.86374 | 21.995605 |
| wpu-cws-indexed-sparse | 4096 | 32 | 0 | 147751 | 0.808333 | 0.333333 | 0.475 | 0.036542 | 32.0 | 1.0 | 1.0 | 0.0 | 3.621635 | 22.462402 |
| wpu-cws-indexed-sparse | 4096 | 32 | 0 | 147751 | 0.791667 | 0.333333 | 0.458334 | 0.039906 | 32.0 | 1.0 | 1.0 | 0.0 | 3.580858 | 22.462402 |
| wpu-cws-indexed-sparse | 4096 | 64 | 0 | 147751 | 0.775 | 0.333333 | 0.441667 | 0.03596 | 64.0 | 1.0 | 1.0 | 0.0 | 6.555268 | 24.376953 |
| wpu-cws-indexed-sparse | 4096 | 64 | 0 | 147751 | 0.791667 | 0.333333 | 0.458334 | 0.046337 | 64.0 | 1.0 | 1.0 | 0.0 | 6.618178 | 24.376953 |

## Aggregated By Model And N

| model | N | K | distractors | seeds | params | accuracy_mean | accuracy_ci95 | majority_mean | acc_minus_majority | mse_mean | selected_K_mean | causal_recall_mean | sparse_ratio_mean | local_dense_ratio_mean | selector_confidence_mean | ms_per_sample_mean | ms_per_sample_ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wpu-cws-indexed-adaptive-hybrid | 4096 | 4 | 0 | 2 | 937511 | 0.691667 | 0.098 | 0.333333 | 0.358334 | 0.053862 | 4.0 | 1.0 | 0.220833 | 0.779166 | 0.337831 | 0.797476 | 0.015606 |
| wpu-cws-indexed-adaptive-hybrid | 4096 | 8 | 0 | 2 | 937511 | 0.708333 | 0.228666 | 0.333333 | 0.375 | 0.036209 | 8.0 | 1.0 | 0.2875 | 0.7125 | 0.303007 | 1.597308 | 0.477132 |
| wpu-cws-indexed-adaptive-hybrid | 4096 | 16 | 0 | 2 | 937511 | 0.708333 | 0.032666 | 0.333333 | 0.375 | 0.035549 | 16.0 | 1.0 | 0.791667 | 0.208333 | 0.524485 | 2.225154 | 0.294874 |
| wpu-cws-indexed-adaptive-hybrid | 4096 | 32 | 0 | 2 | 937511 | 0.795833 | 0.057166 | 0.333333 | 0.4625 | 0.065988 | 32.0 | 1.0 | 0.225 | 0.775 | 0.281406 | 4.160882 | 0.205148 |
| wpu-cws-indexed-adaptive-hybrid | 4096 | 64 | 0 | 2 | 937511 | 0.770833 | 0.0735 | 0.333333 | 0.4375 | 0.034251 | 64.0 | 1.0 | 0.0 | 1.0 | 0.43522 | 10.02154 | 2.534613 |
| wpu-cws-indexed-local-dense | 4096 | 4 | 0 | 2 | 937511 | 0.7125 | 0.106167 | 0.333333 | 0.379167 | 0.085333 | 4.0 | 1.0 | 0.0 | 1.0 | 0.378221 | 0.817662 | 0.000887 |
| wpu-cws-indexed-local-dense | 4096 | 8 | 0 | 2 | 937511 | 0.775 | 0.130667 | 0.333333 | 0.441667 | 0.03975 | 8.0 | 1.0 | 0.0 | 1.0 | 0.160501 | 1.260717 | 0.085136 |
| wpu-cws-indexed-local-dense | 4096 | 16 | 0 | 2 | 937511 | 0.7875 | 0.057167 | 0.333333 | 0.454167 | 0.071102 | 16.0 | 1.0 | 0.0 | 1.0 | 0.358634 | 2.084904 | 0.090138 |
| wpu-cws-indexed-local-dense | 4096 | 32 | 0 | 2 | 937511 | 0.8 | 0.016333 | 0.333333 | 0.466667 | 0.034448 | 32.0 | 1.0 | 0.0 | 1.0 | 0.357025 | 3.512 | 0.080189 |
| wpu-cws-indexed-local-dense | 4096 | 64 | 0 | 2 | 937511 | 0.770833 | 0.0735 | 0.333333 | 0.4375 | 0.034251 | 64.0 | 1.0 | 0.0 | 1.0 | 0.43522 | 6.633883 | 0.018384 |
| wpu-cws-indexed-sparse | 4096 | 4 | 0 | 2 | 147751 | 0.725 | 0.016333 | 0.333333 | 0.391667 | 0.025216 | 4.0 | 1.0 | 1.0 | 0.0 | 0.589958 | 1.000063 | 0.006929 |
| wpu-cws-indexed-sparse | 4096 | 8 | 0 | 2 | 147751 | 0.758333 | 0.032666 | 0.333333 | 0.425001 | 0.029903 | 8.0 | 1.0 | 1.0 | 0.0 | 0.20208 | 1.209614 | 0.024144 |
| wpu-cws-indexed-sparse | 4096 | 16 | 0 | 2 | 147751 | 0.716666 | 0.130666 | 0.333333 | 0.383333 | 0.037741 | 16.0 | 1.0 | 1.0 | 0.0 | 0.417689 | 1.943194 | 0.155729 |
| wpu-cws-indexed-sparse | 4096 | 32 | 0 | 2 | 147751 | 0.8 | 0.016333 | 0.333333 | 0.466667 | 0.038224 | 32.0 | 1.0 | 1.0 | 0.0 | 0.313545 | 3.601247 | 0.039961 |
| wpu-cws-indexed-sparse | 4096 | 64 | 0 | 2 | 147751 | 0.783334 | 0.016334 | 0.333333 | 0.450001 | 0.041149 | 64.0 | 1.0 | 1.0 | 0.0 | 0.356106 | 6.586723 | 0.061652 |

## Best Accuracy By N

| N | model | accuracy_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 4096 | wpu-cws-indexed-local-dense | 0.8 | 0.8 | 3.512 | 937511 |

## Fastest Forward Latency By N

| N | model | ms_per_sample_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 4096 | wpu-cws-indexed-adaptive-hybrid | 0.797476 | 0.691667 | 0.797476 | 937511 |

## Interpretation Checklist

- If `wpu-cws-oracle` remains accurate at large `N`, the WPU core is plausible and selector quality is the bottleneck.
- If `wpu-cws-oracle` fails, the current working-set core lacks capacity even with correct causal access.
- If token/graph baselines keep accuracy and remain faster, WPU has no demonstrated advantage in this regime.
- If WPU keeps accuracy while latency grows slower with `N`, the large-`N` hypothesis has preliminary support.

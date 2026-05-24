# Causal Working Set v2 Results

Source CSV: `docs/experiments/wpu_v2_learned_hybrid_pilot.csv`

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
| wpu-cws-indexed-learned-hybrid | 4096 | 4 | 0 | 971308 | 0.7 | 0.333333 | 0.366667 | 0.02221 | 4.0 | 1.0 | 0.996233 | 0.003767 | 0.819455 | 35.188477 |
| wpu-cws-indexed-learned-hybrid | 4096 | 4 | 0 | 971308 | 0.75 | 0.333333 | 0.416667 | 0.025824 | 4.0 | 1.0 | 0.978573 | 0.021427 | 1.148467 | 35.188477 |
| wpu-cws-indexed-learned-hybrid | 4096 | 8 | 0 | 971308 | 0.7 | 0.333333 | 0.366667 | 0.028068 | 8.0 | 1.0 | 0.997137 | 0.002863 | 1.962353 | 35.303711 |
| wpu-cws-indexed-learned-hybrid | 4096 | 8 | 0 | 971308 | 0.8 | 0.333333 | 0.466667 | 0.020151 | 8.0 | 1.0 | 0.972624 | 0.027376 | 1.36615 | 35.303711 |
| wpu-cws-indexed-learned-hybrid | 4096 | 16 | 0 | 971308 | 0.783333 | 0.333333 | 0.45 | 0.042417 | 16.0 | 1.0 | 0.997232 | 0.002768 | 2.293737 | 35.53418 |
| wpu-cws-indexed-learned-hybrid | 4096 | 16 | 0 | 971308 | 0.758333 | 0.333333 | 0.425 | 0.043509 | 16.0 | 1.0 | 0.946665 | 0.053335 | 2.133753 | 35.53418 |
| wpu-cws-indexed-learned-hybrid | 4096 | 32 | 0 | 971308 | 0.816667 | 0.333333 | 0.483334 | 0.041164 | 32.0 | 1.0 | 0.997196 | 0.002804 | 3.701463 | 35.996582 |
| wpu-cws-indexed-learned-hybrid | 4096 | 32 | 0 | 971308 | 0.85 | 0.333333 | 0.516667 | 0.041301 | 32.0 | 1.0 | 0.997969 | 0.002031 | 3.87132 | 35.996582 |
| wpu-cws-indexed-learned-hybrid | 4096 | 64 | 0 | 971308 | 0.808333 | 0.333333 | 0.475 | 0.056999 | 64.0 | 1.0 | 0.996612 | 0.003388 | 6.823653 | 38.42334 |
| wpu-cws-indexed-learned-hybrid | 4096 | 64 | 0 | 971308 | 0.858333 | 0.333333 | 0.525 | 0.044996 | 64.0 | 1.0 | 0.989847 | 0.010153 | 7.911458 | 38.42334 |
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
| wpu-cws-indexed-learned-hybrid | 4096 | 4 | 0 | 2 | 971308 | 0.725 | 0.049 | 0.333333 | 0.391667 | 0.024017 | 4.0 | 1.0 | 0.987403 | 0.012597 | 0.759452 | 0.983961 | 0.322432 |
| wpu-cws-indexed-learned-hybrid | 4096 | 8 | 0 | 2 | 971308 | 0.75 | 0.098 | 0.333333 | 0.416667 | 0.024109 | 8.0 | 1.0 | 0.984881 | 0.01512 | 0.616529 | 1.664251 | 0.584279 |
| wpu-cws-indexed-learned-hybrid | 4096 | 16 | 0 | 2 | 971308 | 0.770833 | 0.0245 | 0.333333 | 0.4375 | 0.042963 | 16.0 | 1.0 | 0.971948 | 0.028052 | 0.32941 | 2.213745 | 0.156784 |
| wpu-cws-indexed-learned-hybrid | 4096 | 32 | 0 | 2 | 971308 | 0.833333 | 0.032666 | 0.333333 | 0.5 | 0.041232 | 32.0 | 1.0 | 0.997583 | 0.002417 | 0.817948 | 3.786391 | 0.16646 |
| wpu-cws-indexed-learned-hybrid | 4096 | 64 | 0 | 2 | 971308 | 0.833333 | 0.049 | 0.333333 | 0.5 | 0.050998 | 64.0 | 1.0 | 0.993229 | 0.006771 | 0.963839 | 7.367555 | 1.066049 |
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
| 4096 | wpu-cws-indexed-learned-hybrid | 0.833333 | 0.833333 | 3.786391 | 971308 |

## Best Accuracy By K

| K | model | accuracy_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 4 | wpu-cws-indexed-learned-hybrid | 0.725 | 0.725 | 0.983961 | 971308 |
| 8 | wpu-cws-indexed-local-dense | 0.775 | 0.775 | 1.260717 | 937511 |
| 16 | wpu-cws-indexed-local-dense | 0.7875 | 0.7875 | 2.084904 | 937511 |
| 32 | wpu-cws-indexed-learned-hybrid | 0.833333 | 0.833333 | 3.786391 | 971308 |
| 64 | wpu-cws-indexed-learned-hybrid | 0.833333 | 0.833333 | 7.367555 | 971308 |

## Fastest Forward Latency By N

| N | model | ms_per_sample_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 4096 | wpu-cws-indexed-adaptive-hybrid | 0.797476 | 0.691667 | 0.797476 | 937511 |

## Fastest Forward Latency By K

| K | model | ms_per_sample_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 4 | wpu-cws-indexed-adaptive-hybrid | 0.797476 | 0.691667 | 0.797476 | 937511 |
| 8 | wpu-cws-indexed-sparse | 1.209614 | 0.758333 | 1.209614 | 147751 |
| 16 | wpu-cws-indexed-sparse | 1.943194 | 0.716666 | 1.943194 | 147751 |
| 32 | wpu-cws-indexed-local-dense | 3.512 | 0.8 | 3.512 | 937511 |
| 64 | wpu-cws-indexed-sparse | 6.586723 | 0.783334 | 6.586723 | 147751 |

## Interpretation Checklist

- If `wpu-cws-oracle` remains accurate at large `N`, the WPU core is plausible and selector quality is the bottleneck.
- If `wpu-cws-oracle` fails, the current working-set core lacks capacity even with correct causal access.
- If token/graph baselines keep accuracy and remain faster, WPU has no demonstrated advantage in this regime.
- If WPU keeps accuracy while latency grows slower with `N`, the large-`N` hypothesis has preliminary support.

# Event-Conditioned CWS 8M GPU Results

Source CSV: `artifacts/cws_balanced_branch_8m_gpu_event_conditioned_final/n-sweep.csv`

This report tests the large-`N` claim in the stricter form:

```text
WPU should scale with causal working set K, not total state N, when K is small and identifiable.
```

Setup: RTX 5070 Ti, PyTorch 2.11.0+cu128, hidden_dim=512, layers=2,
balanced branch labels, fixed causal working set K=8, and event-conditioned
WPU propagation. Models are parameter-matched at roughly 6.3M-7.4M trainable
parameters.

Main result: at N=4096 over five seeds, `wpu-cws-oracle` reaches 0.6947
branch accuracy at 2.15 ms/sample, while `serialized-token` remains at 0.3333
at 7.73 ms/sample and `graph-transformer` reaches 0.3567 at 9.96 ms/sample.
This supports a regime-specific claim, not universal superiority: the WPU
advantage appears when total world state is large and the causal working set is
small, identifiable, and sufficient.

## Raw Condition Summary

| model | N | K | distractors | params | accuracy | majority | acc-majority | mse | selected_K | causal_recall | ms/sample | cuda_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| graph-transformer | 64 | 8 | 0 | 7375894 | 0.583333 | 0.333333 | 0.25 | 0.001973 | 8.0 | 1.0 | 2.609065 | 143.984375 |
| graph-transformer | 64 | 8 | 0 | 7375894 | 0.333333 | 0.333333 | 0.0 | 0.002534 | 8.0 | 1.0 | 2.670976 | 143.984375 |
| serialized-token | 64 | 8 | 0 | 6325778 | 0.333333 | 0.333333 | 0.0 | 0.000701 | 8.0 | 1.0 | 0.191546 | 130.706543 |
| serialized-token | 64 | 8 | 0 | 6325778 | 0.333333 | 0.333333 | 0.0 | 0.000365 | 8.0 | 1.0 | 0.187364 | 130.706543 |
| wpu-cws-learned | 64 | 8 | 0 | 6853655 | 0.333333 | 0.333333 | 0.0 | 6e-05 | 16.0 | 1.0 | 0.94886 | 132.140625 |
| wpu-cws-learned | 64 | 8 | 0 | 6853655 | 0.72 | 0.333333 | 0.386667 | 0.00163 | 16.0 | 1.0 | 0.953479 | 132.140625 |
| wpu-cws-oracle | 64 | 8 | 0 | 6853655 | 0.333333 | 0.333333 | 0.0 | 9.8e-05 | 8.0 | 1.0 | 1.04061 | 132.265625 |
| wpu-cws-oracle | 64 | 8 | 0 | 6853655 | 0.333333 | 0.333333 | 0.0 | 3.8e-05 | 8.0 | 1.0 | 1.054189 | 132.140625 |
| graph-transformer | 1024 | 8 | 0 | 7375894 | 0.646667 | 0.333333 | 0.313334 | 0.00381 | 8.0 | 1.0 | 3.631179 | 278.002441 |
| graph-transformer | 1024 | 8 | 0 | 7375894 | 0.5 | 0.333333 | 0.166667 | 0.014989 | 8.0 | 1.0 | 3.708914 | 278.002441 |
| serialized-token | 1024 | 8 | 0 | 6325778 | 0.333333 | 0.333333 | 0.0 | 0.003357 | 8.0 | 1.0 | 1.894275 | 279.731934 |
| serialized-token | 1024 | 8 | 0 | 6325778 | 0.333333 | 0.333333 | 0.0 | 6.7e-05 | 8.0 | 1.0 | 1.165489 | 279.731934 |
| wpu-cws-learned | 1024 | 8 | 0 | 6853655 | 0.333333 | 0.333333 | 0.0 | 7e-06 | 16.0 | 1.0 | 2.113709 | 222.257324 |
| wpu-cws-learned | 1024 | 8 | 0 | 6853655 | 0.333333 | 0.333333 | 0.0 | 5e-06 | 16.0 | 1.0 | 2.655875 | 222.257324 |
| wpu-cws-oracle | 1024 | 8 | 0 | 6853655 | 0.333333 | 0.333333 | 0.0 | 5e-06 | 8.0 | 1.0 | 2.527919 | 222.257324 |
| wpu-cws-oracle | 1024 | 8 | 0 | 6853655 | 0.333333 | 0.333333 | 0.0 | 6e-06 | 8.0 | 1.0 | 2.68254 | 222.257324 |
| graph-transformer | 4096 | 8 | 0 | 7375894 | 0.333333 | 0.333333 | 0.0 | 0.000226 | 8.0 | 1.0 | 9.952025 | 710.510254 |
| graph-transformer | 4096 | 8 | 0 | 7375894 | 0.333333 | 0.333333 | 0.0 | 2.2e-05 | 8.0 | 1.0 | 9.959394 | 713.385254 |
| graph-transformer | 4096 | 8 | 0 | 7375894 | 0.333333 | 0.333333 | 0.0 | 0.0001 | 8.0 | 1.0 | 9.957336 | 710.510254 |
| graph-transformer | 4096 | 8 | 0 | 7375894 | 0.45 | 0.333333 | 0.116667 | 0.000808 | 8.0 | 1.0 | 9.971081 | 713.385254 |
| graph-transformer | 4096 | 8 | 0 | 7375894 | 0.333333 | 0.333333 | 0.0 | 4.5e-05 | 8.0 | 1.0 | 9.951298 | 713.385254 |
| serialized-token | 4096 | 8 | 0 | 6325778 | 0.333333 | 0.333333 | 0.0 | 0.012959 | 8.0 | 1.0 | 7.732398 | 763.138184 |
| serialized-token | 4096 | 8 | 0 | 6325778 | 0.333333 | 0.333333 | 0.0 | 0.000968 | 8.0 | 1.0 | 7.726594 | 763.138184 |
| serialized-token | 4096 | 8 | 0 | 6325778 | 0.333333 | 0.333333 | 0.0 | 0.000203 | 8.0 | 1.0 | 7.711526 | 763.138184 |
| serialized-token | 4096 | 8 | 0 | 6325778 | 0.333333 | 0.333333 | 0.0 | 0.00022 | 8.0 | 1.0 | 7.714317 | 763.075684 |
| serialized-token | 4096 | 8 | 0 | 6325778 | 0.333333 | 0.333333 | 0.0 | 0.000232 | 8.0 | 1.0 | 7.753065 | 761.263184 |
| wpu-cws-learned | 4096 | 8 | 0 | 6853655 | 0.55 | 0.333333 | 0.216667 | 2e-06 | 16.0 | 1.0 | 1.872481 | 513.831543 |
| wpu-cws-learned | 4096 | 8 | 0 | 6853655 | 0.713333 | 0.333333 | 0.38 | 3e-06 | 16.0 | 1.0 | 1.598293 | 513.831543 |
| wpu-cws-learned | 4096 | 8 | 0 | 6853655 | 0.773333 | 0.333333 | 0.44 | 2e-06 | 16.0 | 1.0 | 1.670203 | 513.831543 |
| wpu-cws-learned | 4096 | 8 | 0 | 6853655 | 0.333333 | 0.333333 | 0.0 | 1e-06 | 16.0 | 1.0 | 1.63644 | 513.831543 |
| wpu-cws-learned | 4096 | 8 | 0 | 6853655 | 0.333333 | 0.333333 | 0.0 | 2e-06 | 16.0 | 1.0 | 1.644555 | 513.831543 |
| wpu-cws-oracle | 4096 | 8 | 0 | 6853655 | 0.633333 | 0.333333 | 0.3 | 8e-06 | 8.0 | 1.0 | 2.106484 | 513.831543 |
| wpu-cws-oracle | 4096 | 8 | 0 | 6853655 | 0.71 | 0.333333 | 0.376667 | 7e-06 | 8.0 | 1.0 | 2.138651 | 513.831543 |
| wpu-cws-oracle | 4096 | 8 | 0 | 6853655 | 0.666667 | 0.333333 | 0.333334 | 5e-06 | 8.0 | 1.0 | 2.034085 | 512.831543 |
| wpu-cws-oracle | 4096 | 8 | 0 | 6853655 | 0.693333 | 0.333333 | 0.36 | 3e-06 | 8.0 | 1.0 | 2.414873 | 513.831543 |
| wpu-cws-oracle | 4096 | 8 | 0 | 6853655 | 0.77 | 0.333333 | 0.436667 | 2e-06 | 8.0 | 1.0 | 2.064353 | 513.831543 |

## Aggregated By Model And N

| model | N | K | distractors | seeds | params | accuracy_mean | accuracy_ci95 | majority_mean | acc_minus_majority | mse_mean | selected_K_mean | causal_recall_mean | ms_per_sample_mean | ms_per_sample_ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| graph-transformer | 64 | 8 | 0 | 2 | 7375894 | 0.458333 | 0.245 | 0.333333 | 0.125 | 0.002253 | 8.0 | 1.0 | 2.640021 | 0.060673 |
| serialized-token | 64 | 8 | 0 | 2 | 6325778 | 0.333333 | 0.0 | 0.333333 | 0.0 | 0.000533 | 8.0 | 1.0 | 0.189455 | 0.004098 |
| wpu-cws-learned | 64 | 8 | 0 | 2 | 6853655 | 0.526666 | 0.378934 | 0.333333 | 0.193333 | 0.000845 | 16.0 | 1.0 | 0.95117 | 0.004527 |
| wpu-cws-oracle | 64 | 8 | 0 | 2 | 6853655 | 0.333333 | 0.0 | 0.333333 | 0.0 | 6.8e-05 | 8.0 | 1.0 | 1.0474 | 0.013307 |
| graph-transformer | 1024 | 8 | 0 | 2 | 7375894 | 0.573333 | 0.143734 | 0.333333 | 0.24 | 0.009399 | 8.0 | 1.0 | 3.670046 | 0.07618 |
| serialized-token | 1024 | 8 | 0 | 2 | 6325778 | 0.333333 | 0.0 | 0.333333 | 0.0 | 0.001712 | 8.0 | 1.0 | 1.529882 | 0.71421 |
| wpu-cws-learned | 1024 | 8 | 0 | 2 | 6853655 | 0.333333 | 0.0 | 0.333333 | 0.0 | 6e-06 | 16.0 | 1.0 | 2.384792 | 0.531323 |
| wpu-cws-oracle | 1024 | 8 | 0 | 2 | 6853655 | 0.333333 | 0.0 | 0.333333 | 0.0 | 5e-06 | 8.0 | 1.0 | 2.60523 | 0.151529 |
| graph-transformer | 4096 | 8 | 0 | 5 | 7375894 | 0.356666 | 0.045733 | 0.333333 | 0.023333 | 0.00024 | 8.0 | 1.0 | 9.958227 | 0.006983 |
| serialized-token | 4096 | 8 | 0 | 5 | 6325778 | 0.333333 | 0.0 | 0.333333 | 0.0 | 0.002916 | 8.0 | 1.0 | 7.72758 | 0.014584 |
| wpu-cws-learned | 4096 | 8 | 0 | 5 | 6853655 | 0.540666 | 0.180708 | 0.333333 | 0.207333 | 2e-06 | 16.0 | 1.0 | 1.684394 | 0.094891 |
| wpu-cws-oracle | 4096 | 8 | 0 | 5 | 6853655 | 0.694667 | 0.044819 | 0.333333 | 0.361334 | 5e-06 | 8.0 | 1.0 | 2.151689 | 0.133609 |

## Best Accuracy By N

| N | model | accuracy_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 64 | wpu-cws-learned | 0.526666 | 0.526666 | 0.95117 | 6853655 |
| 1024 | graph-transformer | 0.573333 | 0.573333 | 3.670046 | 7375894 |
| 4096 | wpu-cws-oracle | 0.694667 | 0.694667 | 2.151689 | 6853655 |

## Fastest Forward Latency By N

| N | model | ms_per_sample_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 64 | serialized-token | 0.189455 | 0.333333 | 0.189455 | 6325778 |
| 1024 | serialized-token | 1.529882 | 0.333333 | 1.529882 | 6325778 |
| 4096 | wpu-cws-learned | 1.684394 | 0.540666 | 1.684394 | 6853655 |

## Interpretation

- The strongest positive evidence is at N=4096: oracle WPU is both more accurate and faster than token and dense graph baselines.
- The learned selector is faster than oracle WPU at N=4096 but less stable, which makes selector reliability the next bottleneck.
- The N=1024 result does not support a general WPU advantage: graph-transformer learns the branch task there, while WPU remains near random.
- The claim should therefore be stated as a regime diagram: WPU is favored when N is large, K is small, and causal access is reliable; dense graph processing can still dominate at intermediate N.

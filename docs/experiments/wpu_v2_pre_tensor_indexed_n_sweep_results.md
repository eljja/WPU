# WPU V2 Pre-Tensor Indexed N-Sweep Results

Source CSV: `docs/experiments/wpu_v2_pre_tensor_indexed_n_sweep.csv`

This report tests the large-`N` claim in the stricter form:

```text
WPU should scale with causal working set K, not total state N, when K is small and identifiable.
```

This is a v2-specific experiment. Unlike the earlier full-tensor CWS path, it
projects each `WorldState` to the event-local indexed subgraph before
tensorization. Total N still grows from 64 to 8192, but the neural model receives
only the selected K=8 subgraph.

Key result: branch accuracy stays at 0.6775 across the sweep, while latency
stays in a narrow 1.08-1.84 ms/sample range. This is the first direct evidence
that WPU's intended architecture should index world state before tensorization.

## Raw Condition Summary

| model | N | K | distractors | params | accuracy | majority | acc-majority | mse | selected_K | causal_recall | ms/sample | cuda_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wpu-cws-indexed | 64 | 8 | 0 | 6853655 | 0.69 | 0.335 | 0.355 | 0.00512 | 8.0 | 1.0 | 1.07528 | 124.199219 |
| wpu-cws-indexed | 64 | 8 | 0 | 6853655 | 0.665 | 0.335 | 0.33 | 0.010733 | 8.0 | 1.0 | 1.078191 | 126.074219 |
| wpu-cws-indexed | 128 | 8 | 0 | 6853655 | 0.69 | 0.335 | 0.355 | 0.00512 | 8.0 | 1.0 | 1.084844 | 126.074219 |
| wpu-cws-indexed | 128 | 8 | 0 | 6853655 | 0.665 | 0.335 | 0.33 | 0.010733 | 8.0 | 1.0 | 1.138523 | 126.074219 |
| wpu-cws-indexed | 256 | 8 | 0 | 6853655 | 0.69 | 0.335 | 0.355 | 0.00512 | 8.0 | 1.0 | 1.254916 | 126.074219 |
| wpu-cws-indexed | 256 | 8 | 0 | 6853655 | 0.665 | 0.335 | 0.33 | 0.010733 | 8.0 | 1.0 | 1.346073 | 126.074219 |
| wpu-cws-indexed | 512 | 8 | 0 | 6853655 | 0.69 | 0.335 | 0.355 | 0.00512 | 8.0 | 1.0 | 1.202487 | 126.074219 |
| wpu-cws-indexed | 512 | 8 | 0 | 6853655 | 0.665 | 0.335 | 0.33 | 0.010733 | 8.0 | 1.0 | 1.217308 | 126.074219 |
| wpu-cws-indexed | 1024 | 8 | 0 | 6853655 | 0.69 | 0.335 | 0.355 | 0.00512 | 8.0 | 1.0 | 1.462585 | 126.074219 |
| wpu-cws-indexed | 1024 | 8 | 0 | 6853655 | 0.665 | 0.335 | 0.33 | 0.010733 | 8.0 | 1.0 | 1.530639 | 126.074219 |
| wpu-cws-indexed | 2048 | 8 | 0 | 6853655 | 0.69 | 0.335 | 0.355 | 0.00512 | 8.0 | 1.0 | 1.517382 | 126.074219 |
| wpu-cws-indexed | 2048 | 8 | 0 | 6853655 | 0.665 | 0.335 | 0.33 | 0.010733 | 8.0 | 1.0 | 1.58879 | 126.074219 |
| wpu-cws-indexed | 4096 | 8 | 0 | 6853655 | 0.69 | 0.335 | 0.355 | 0.00512 | 8.0 | 1.0 | 1.503252 | 126.074219 |
| wpu-cws-indexed | 4096 | 8 | 0 | 6853655 | 0.665 | 0.335 | 0.33 | 0.010733 | 8.0 | 1.0 | 1.522979 | 126.074219 |
| wpu-cws-indexed | 8192 | 8 | 0 | 6853655 | 0.69 | 0.335 | 0.355 | 0.00512 | 8.0 | 1.0 | 1.741064 | 126.074219 |
| wpu-cws-indexed | 8192 | 8 | 0 | 6853655 | 0.665 | 0.335 | 0.33 | 0.010733 | 8.0 | 1.0 | 1.938904 | 126.074219 |

## Aggregated By Model And N

| model | N | K | distractors | seeds | params | accuracy_mean | accuracy_ci95 | majority_mean | acc_minus_majority | mse_mean | selected_K_mean | causal_recall_mean | ms_per_sample_mean | ms_per_sample_ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wpu-cws-indexed | 64 | 8 | 0 | 2 | 6853655 | 0.6775 | 0.0245 | 0.335 | 0.3425 | 0.007926 | 8.0 | 1.0 | 1.076735 | 0.002853 |
| wpu-cws-indexed | 128 | 8 | 0 | 2 | 6853655 | 0.6775 | 0.0245 | 0.335 | 0.3425 | 0.007926 | 8.0 | 1.0 | 1.111683 | 0.052605 |
| wpu-cws-indexed | 256 | 8 | 0 | 2 | 6853655 | 0.6775 | 0.0245 | 0.335 | 0.3425 | 0.007926 | 8.0 | 1.0 | 1.300495 | 0.089334 |
| wpu-cws-indexed | 512 | 8 | 0 | 2 | 6853655 | 0.6775 | 0.0245 | 0.335 | 0.3425 | 0.007926 | 8.0 | 1.0 | 1.209898 | 0.014525 |
| wpu-cws-indexed | 1024 | 8 | 0 | 2 | 6853655 | 0.6775 | 0.0245 | 0.335 | 0.3425 | 0.007926 | 8.0 | 1.0 | 1.496612 | 0.066693 |
| wpu-cws-indexed | 2048 | 8 | 0 | 2 | 6853655 | 0.6775 | 0.0245 | 0.335 | 0.3425 | 0.007926 | 8.0 | 1.0 | 1.553086 | 0.06998 |
| wpu-cws-indexed | 4096 | 8 | 0 | 2 | 6853655 | 0.6775 | 0.0245 | 0.335 | 0.3425 | 0.007926 | 8.0 | 1.0 | 1.513116 | 0.019332 |
| wpu-cws-indexed | 8192 | 8 | 0 | 2 | 6853655 | 0.6775 | 0.0245 | 0.335 | 0.3425 | 0.007926 | 8.0 | 1.0 | 1.839984 | 0.193883 |

## Best Accuracy By N

| N | model | accuracy_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 64 | wpu-cws-indexed | 0.6775 | 0.6775 | 1.076735 | 6853655 |
| 128 | wpu-cws-indexed | 0.6775 | 0.6775 | 1.111683 | 6853655 |
| 256 | wpu-cws-indexed | 0.6775 | 0.6775 | 1.300495 | 6853655 |
| 512 | wpu-cws-indexed | 0.6775 | 0.6775 | 1.209898 | 6853655 |
| 1024 | wpu-cws-indexed | 0.6775 | 0.6775 | 1.496612 | 6853655 |
| 2048 | wpu-cws-indexed | 0.6775 | 0.6775 | 1.553086 | 6853655 |
| 4096 | wpu-cws-indexed | 0.6775 | 0.6775 | 1.513116 | 6853655 |
| 8192 | wpu-cws-indexed | 0.6775 | 0.6775 | 1.839984 | 6853655 |

## Fastest Forward Latency By N

| N | model | ms_per_sample_mean | accuracy | ms/sample | params |
| --- | --- | --- | --- | --- | --- |
| 64 | wpu-cws-indexed | 1.076735 | 0.6775 | 1.076735 | 6853655 |
| 128 | wpu-cws-indexed | 1.111683 | 0.6775 | 1.111683 | 6853655 |
| 256 | wpu-cws-indexed | 1.300495 | 0.6775 | 1.300495 | 6853655 |
| 512 | wpu-cws-indexed | 1.209898 | 0.6775 | 1.209898 | 6853655 |
| 1024 | wpu-cws-indexed | 1.496612 | 0.6775 | 1.496612 | 6853655 |
| 2048 | wpu-cws-indexed | 1.553086 | 0.6775 | 1.553086 | 6853655 |
| 4096 | wpu-cws-indexed | 1.513116 | 0.6775 | 1.513116 | 6853655 |
| 8192 | wpu-cws-indexed | 1.839984 | 0.6775 | 1.839984 | 6853655 |

## Interpretation Checklist

- If `wpu-cws-oracle` remains accurate at large `N`, the WPU core is plausible and selector quality is the bottleneck.
- If `wpu-cws-oracle` fails, the current working-set core lacks capacity even with correct causal access.
- If token/graph baselines keep accuracy and remain faster, WPU has no demonstrated advantage in this regime.
- If WPU keeps accuracy while latency grows slower with `N`, the large-`N` hypothesis has preliminary support.

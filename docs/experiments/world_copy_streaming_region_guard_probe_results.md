# World-Copy Streaming Region Guard Probe

This probe tests whether bounded region guard preserves state integrity and correction cost over H>=25 streaming world-copy rollouts.
It includes object churn and region migration, but is not yet a real simulator or learned-transition benchmark.
Source CSV: `docs/experiments/world_copy_streaming_region_guard_probe.csv`.

## Summary

| mode | N | H | mean K | max K | trajectory MSE | integrity | correction rate | correction cost | work proxy | bytes proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wpu-relation-frontier | 128 | 25 | 8.000000 | 8 | 0.379058 | 0.727399 | 0.423333 | 0.191875 | 4.583333 | 165.000000 |
| wpu-region-guard | 128 | 25 | 8.000000 | 8 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 8.000000 | 288.000000 |
| dense-state-copy | 128 | 25 | 8.000000 | 8 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 131.636667 | 4623.420000 |
| wpu-relation-frontier | 512 | 25 | 8.000000 | 8 | 0.372720 | 0.731463 | 0.413333 | 0.187292 | 4.625000 | 166.500000 |
| wpu-region-guard | 512 | 25 | 8.000000 | 8 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 8.000000 | 288.000000 |
| dense-state-copy | 512 | 25 | 8.000000 | 8 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 515.766667 | 18447.600000 |
| wpu-relation-frontier | 2048 | 25 | 8.000000 | 8 | 0.354494 | 0.742925 | 0.406667 | 0.182083 | 4.750000 | 171.000000 |
| wpu-region-guard | 2048 | 25 | 8.000000 | 8 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 8.000000 | 288.000000 |
| dense-state-copy | 2048 | 25 | 8.000000 | 8 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 2051.780000 | 73748.580000 |
| wpu-relation-frontier | 8192 | 25 | 8.000000 | 8 | 0.370395 | 0.732183 | 0.423333 | 0.194583 | 4.541667 | 163.500000 |
| wpu-region-guard | 8192 | 25 | 8.000000 | 8 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 8.000000 | 288.000000 |
| dense-state-copy | 8192 | 25 | 8.000000 | 8 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 8195.980000 | 294929.280000 |

## Interpretation

- `wpu-region-guard` maintains low trajectory error over H=25 streams when the bounded active region is reliable.
- `wpu-relation-frontier` misses active causal objects under missing relations and needs frequent correction.
- `dense-state-copy` is close to a reference upper bound, but uses full-state work/bytes proxy.
- The next failure boundary is how guard cost and false updates grow when regions become large or mis-objectified.

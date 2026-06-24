# World-Copy Streaming Region Guard Probe

이 probe는 H>=25 streaming world-copy에서 bounded region guard가 state integrity와 correction cost를 유지하는지 검사한다.
Object churn과 region migration을 포함하지만, 아직 실제 simulator나 learned transition benchmark는 아니다.
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

- `wpu-region-guard`는 bounded active region이 신뢰 가능할 때 H=25 stream에서도 low trajectory error를 유지한다.
- `wpu-relation-frontier`는 missing relation 때문에 active causal objects를 놓치고 correction이 자주 필요하다.
- `dense-state-copy`는 reference upper bound에 가깝지만 full-state work/bytes proxy를 사용한다.
- 다음 실패 경계는 region이 커지거나 잘못 objectified될 때 guard 비용과 false update가 어떻게 증가하는지다.

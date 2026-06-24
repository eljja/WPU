# World-Copy Baseline Comparison Probe

이 probe는 같은 synthetic world-copy delta task에서 WPU local propagation과 token/graph/dense baseline을 비교한다.
Baseline은 비교용이며 WPU 구현 경로가 아니다. 이 결과는 controlled screen이지 최종 P2 완료가 아니다.
Source CSV: `docs/experiments/world_copy_baseline_comparison_probe.csv`.

## Summary

| model | mean delta MSE | mean work proxy | mean bytes proxy | mean selected K | max selected K | accuracy/kwork |
|---|---:|---:|---:|---:|---:|---:|
| wpu-hybrid | 0.020818 | 8.789551 | 316.423828 | 9.333333 | 16 | 37867.798346 |
| dense-graph | 0.003778 | 2727.406250 | 97920.000000 | 9.333333 | 16 | 698.058125 |
| serialized-token | 0.004533 | 2727.406250 | 119680.000000 | 9.333333 | 16 | 621.481983 |
| graph-transformer-proxy | 0.003778 | 17935805.573893 | 71581696.000000 | 9.333333 | 16 | 4.111582 |

## Interpretation

- 이 screen에서는 WPU가 bounded selected `K`를 유지하면서 full-state baseline보다 낮은 work/bytes proxy를 사용한다.
- Raw delta MSE는 dense/token baseline이 더 낮다. 따라서 이 결과는 WPU의 순수 accuracy victory가 아니다.
- WPU의 positive signal은 낮은 touched-state 비용과 높은 accuracy-per-work이며, 이는 energy/latency 방향의 substrate claim이다.
- `graph-transformer-proxy`는 실제 attention training이 아니라 dense quadratic work proxy다.
- 다음 단계는 streaming/H>=25 world-copy task에서 실제 token/graph model과 latency를 측정하는 것이다.

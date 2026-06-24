# World-Copy Baseline Comparison Probe

이 probe는 같은 synthetic world-copy delta task에서 WPU local propagation과 token/graph/dense baseline을 비교한다.
Baseline은 비교용이며 WPU 구현 경로가 아니다. 이 결과는 controlled screen이지 최종 P2 완료가 아니다.
Source CSV: `docs/experiments/world_copy_baseline_comparison_probe.csv`.

## Summary

| model | mean delta MSE | mean work proxy | mean bytes proxy | mean selected K | max selected K | accuracy/kwork |
|---|---:|---:|---:|---:|---:|---:|
| wpu-hybrid | 0.020818 | 8.789551 | 316.423828 | 9.333333 | 16 | 37867.798346 |
| wpu-hybrid-context | 0.020904 | 9.789551 | 457.056641 | 9.333333 | 16 | 32480.274504 |
| wpu-region-guard | 0.002646 | 9.333333 | 336.000000 | 9.333333 | 16 | 52432.911821 |
| dense-graph | 0.003810 | 2727.406250 | 97920.000000 | 9.333333 | 16 | 711.681407 |
| serialized-token | 0.003223 | 2727.406250 | 119680.000000 | 9.333333 | 16 | 827.955281 |
| graph-transformer-proxy | 0.003810 | 17935805.573893 | 71581696.000000 | 9.333333 | 16 | 4.180562 |

## Interpretation

- `wpu-region-guard`는 bounded selected `K`를 유지하면서 raw delta MSE와 work/bytes proxy를 동시에 개선한다.
- 단순 context feature 추가(`wpu-hybrid-context`)는 negative다. Raw MSE가 기본 `wpu-hybrid`보다 좋아지지 않는다.
- Positive signal은 relation frontier만 신뢰하는 것이 아니라 bounded local region을 guard로 쓰면 missing-relation gap을 줄일 수 있다는 점이다.
- 이 결과는 bounded region이 작고 신뢰 가능할 때만 성립한다. Region이 커지거나 objectification이 틀리면 WPU claim은 다시 약해진다.
- `graph-transformer-proxy`는 실제 attention training이 아니라 dense quadratic work proxy다.
- 다음 단계는 streaming/H>=25 world-copy task에서 실제 token/graph model과 latency를 측정하는 것이다.

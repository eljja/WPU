# World-Copy Index Probe

이 probe는 실세계 copy형 WPU에서 전체 `N`이 커져도 event-local causal working set `K`를 작게 유지할 수 있는지 확인한다.
현재 probe는 학습 성능 실험이 아니라 v3 state/index substrate 검증이다.
Source CSV: `docs/experiments/world_copy_index_probe.csv`.

| total N | selected K | affected fraction | non-causal selected | selected objects |
|---:|---:|---:|---:|---|
| 104 | 4 | 0.03846154 | 0 | `cup table hand edge` |
| 1004 | 4 | 0.00398406 | 0 | `cup table hand edge` |
| 5004 | 4 | 0.00079936 | 0 | `cup table hand edge` |
| 10004 | 4 | 0.00039984 | 0 | `cup table hand edge` |

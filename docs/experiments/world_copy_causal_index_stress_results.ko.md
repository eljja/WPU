# World-Copy Causal Index Stress

이 benchmark는 WPU v3 causal index가 large `N`과 relation noise 아래에서 event-local causal slice를 얼마나 안정적으로 검색하는지 측정한다.
이는 학습된 world-model accuracy가 아니라 causal retrieval substrate 검증이다.
Source CSV: `docs/experiments/world_copy_causal_index_stress.csv`.

## Summary

| missing rate | false-positive rate | mean recall | mean precision | max selected K | max touch ratio |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0 | 1.000000 | 1.000000 | 16 | 0.22377622 |
| 0.0 | 0.1 | 1.000000 | 0.925926 | 18 | 0.23448276 |
| 0.0 | 0.25 | 1.000000 | 0.800000 | 20 | 0.24489796 |
| 0.25 | 0.0 | 1.000000 | 1.000000 | 16 | 0.20714286 |
| 0.25 | 0.1 | 1.000000 | 0.925926 | 18 | 0.21830986 |
| 0.25 | 0.25 | 1.000000 | 0.800000 | 20 | 0.22916667 |
| 0.5 | 0.0 | 1.000000 | 1.000000 | 16 | 0.18382353 |
| 0.5 | 0.1 | 1.000000 | 0.925926 | 18 | 0.20143885 |
| 0.5 | 0.25 | 1.000000 | 0.800000 | 20 | 0.20714286 |

## Interpretation

- Region-scoped retrieval은 `N`이 커져도 touched units를 full-state scan보다 훨씬 낮게 유지한다.
- 누락된 true relation은 active region이 causal scope로 작동하기 때문에 일부 복구된다. 단, 이는 objectification과 region assignment가 맞다는 가정에 의존한다.
- False-positive relation은 non-causal object를 추가해 precision을 낮춘다. 이것이 현재 index의 failure boundary다.

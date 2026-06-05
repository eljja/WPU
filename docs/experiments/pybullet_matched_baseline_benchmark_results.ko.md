# PyBullet Matched-Baseline Benchmark

이 실험은 simulator-grounded PyBullet 컵 benchmark를 parameter budget에 맞춰 다시
실행한 것이다. Benchmark script는 이제 `--target-params`를 지원하며, 각 model의
trainable parameter count가 요청한 budget에 가장 가깝도록 hidden dimension을
선택한다.

Source CSV:

- `docs/experiments/pybullet_matched_baseline_benchmark.csv`

## 프로토콜

- Simulator: PyBullet `DIRECT` rigid-body rollout.
- Base task: balanced cup impulse branch prediction.
- Target parameter budget: `50,000`.
- Model: `wpu-cws-indexed-sparse`, `wpu-cws-indexed-local-dense`,
  `graph-transformer`, `serialized-token`.
- Seed: `11, 13`.
- Background object: `0, 128`.
- Training: 20 steps, batch 8.
- Evaluation: 조건별 36 samples.
- WPU input: pre-tensor indexed event-local subgraph.
- Baseline input: full simulator-derived state graph/token sequence.

## 요약

| background objects | model | params | hidden dim | accuracy | latency ms/sample | pre-tensor indexed |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | graph-transformer | 47,622 | 40 | 0.528 | 2.199 | false |
| 0 | serialized-token | 58,530 | 48 | 0.569 | 0.145 | false |
| 0 | wpu-cws-indexed-local-dense | 48,300 | 40 | 0.500 | 1.430 | true |
| 0 | wpu-cws-indexed-sparse | 52,924 | 112 | 0.569 | 1.299 | true |
| 128 | graph-transformer | 47,622 | 40 | 0.472 | 41.766 | false |
| 128 | serialized-token | 58,530 | 48 | 0.472 | 0.294 | false |
| 128 | wpu-cws-indexed-local-dense | 48,300 | 40 | 0.500 | 1.438 | true |
| 128 | wpu-cws-indexed-sparse | 52,924 | 112 | 0.569 | 2.177 | true |

## 해석

Parameter matching은 simulator benchmark를 WPU regime claim에 더 유리하게
만든다. 하지만 여전히 universal win은 아니다. 같은 근사 parameter budget에서
`wpu-cws-indexed-sparse`는 irrelevant background state가 0에서 128개로 증가해도
accuracy를 유지했다. 반면 full-state graph와 serialized-token baseline은 이 작은
run에서 하락했다.

Systems 결과는 혼합적이다. WPU는 `N=128`에서 full-state graph transformer보다 훨씬
빠르다. WPU path는 event-local subgraph만 tensorize하기 때문이다. 하지만 현재
serialized-token 구현은 이 규모에서 여전히 WPU보다 빠르다. 따라서 방어 가능한
주장은 “WPU가 모든 token baseline보다 빠르다”가 아니다. 더 좁은 주장은 다음이다.

```text
Objectified state와 식별 가능한 local K가 있을 때, pre-tensor WPU retrieval은
full-state graph processing cost를 피하면서 accuracy를 유지할 수 있다.
```

## 발견한 문제

- Parameter count만으로 compute fairness가 완성되지는 않는다. serialized-token은
  작은 sequence에서 PyTorch 실행이 효율적이고 여전히 매우 빠르다.
- 같은 parameter budget에서 WPU sparse는 WPU local-dense보다 큰 hidden dimension을
  받았다. local-dense는 Transformer encoder를 포함하기 때문이다. 이는 parameter
  count 기준으로는 공정하지만 operator type 기준으로는 동일하지 않다.
- 이 run은 two seed와 짧은 training이다. 최종 paper evidence가 아니라
  matched-baseline correction pilot으로 봐야 한다.

## 다음 단계

- five seeds, 더 큰 training budget, `N=0,32,128,512`로 반복한다.
- parameter matching뿐 아니라 compute-normalized metric을 추가한다.
- memory traffic과 tensorized-object count를 first-class metric으로 기록한다.
- long-horizon rollout을 추가해 one-step WPU accuracy가 반복 delta 적용에서도
  유지되는지 검증한다.

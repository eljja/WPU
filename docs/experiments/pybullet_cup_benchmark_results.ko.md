# PyBullet 컵 벤치마크

이 실험은 저장소의 첫 simulator-grounded WPU 벤치마크다. 샘플은
PyBullet rigid-body rollout에서 생성하고, 이를 명시적 `WorldState` 객체로
변환한 뒤 기존 synthetic WPU 실험과 동일한 `StateGraphBatch` 인터페이스로
평가한다.

목표는 “현실 물리를 이해했다”는 주장이 아니다. 더 좁은 목표는 독립
simulator가 내보낸 객체화 상태를 WPU가 처리할 수 있는지 확인하고,
pre-tensor indexed retrieval이 무관한 background state 증가로부터 WPU
latency를 분리하는지 검증하는 것이다.

## 프로토콜

- Simulator: PyBullet `DIRECT` rigid-body rollout.
- Scenario: 테이블 가장자리 근처 컵에 robot hand impulse가 가해지는 상황.
- Branch label: `stable`, `falls`, `caught/recovered`.
- Label: rejection sampling으로 균형화.
- WPU 입력: pre-tensor indexed event-local subgraph.
- Baseline 입력: simulator에서 나온 full state graph.
- Model: `wpu-cws-indexed-sparse`, `wpu-cws-indexed-local-dense`,
  `graph-transformer`, `serialized-token`.
- Seed: `11, 13`.
- Background object: `0, 32, 128`.
- Training: 30 steps, batch 8, hidden 64.
- Evaluation: 조건별 48 samples.

Raw result:

- `docs/experiments/pybullet_cup_benchmark.csv`

## 요약

| background N | model | branch accuracy | majority | latency ms/sample | selected K | causal recall | dense compute |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | graph-transformer | 0.583 | 0.333 | 2.211 | 4.44 | 1.000 | 1.000 |
| 0 | serialized-token | 0.562 | 0.333 | 0.139 | 4.44 | 1.000 | 1.000 |
| 0 | wpu-cws-indexed-local-dense | 0.573 | 0.333 | 1.432 | 4.44 | 1.000 | 1.000 |
| 0 | wpu-cws-indexed-sparse | 0.552 | 0.333 | 1.471 | 4.44 | 1.000 | 0.000 |
| 32 | graph-transformer | 0.531 | 0.333 | 11.840 | 4.44 | 1.000 | 1.000 |
| 32 | serialized-token | 0.573 | 0.333 | 0.136 | 4.44 | 1.000 | 1.000 |
| 32 | wpu-cws-indexed-local-dense | 0.573 | 0.333 | 1.574 | 4.44 | 1.000 | 1.000 |
| 32 | wpu-cws-indexed-sparse | 0.552 | 0.333 | 1.288 | 4.44 | 1.000 | 0.000 |
| 128 | graph-transformer | 0.521 | 0.333 | 42.716 | 4.44 | 1.000 | 1.000 |
| 128 | serialized-token | 0.562 | 0.333 | 0.303 | 4.44 | 1.000 | 1.000 |
| 128 | wpu-cws-indexed-local-dense | 0.573 | 0.333 | 1.550 | 4.44 | 1.000 | 1.000 |
| 128 | wpu-cws-indexed-sparse | 0.552 | 0.333 | 1.920 | 4.44 | 1.000 | 0.000 |

## 해석

이 결과는 제한된 systems claim을 지지한다. indexed WPU path는 full
simulator state가 background object로 증가해도 neural working set을
`K ~= 4-5` 근처로 유지한다. 반면 graph-transformer baseline은 full state
graph를 처리하기 때문에 latency가 크게 증가한다. 이는 WPU가 유리한
regime, 즉 큰 `N`, 작고 식별 가능한 `K`, event-local propagation 조건과
일치한다.

하지만 이 결과는 accuracy dominance를 지지하지 않는다. 이 작은
benchmark에서 WPU 정확도는 token/graph baseline과 비슷하지만 명확히 더
좋지는 않다. 또한 현재 PyTorch 구현에서 serialized-token baseline은 이
작은 규모에서 매우 빠르므로, WPU가 보편적으로 latency 우위라는 주장도
아직 할 수 없다.

## 발견한 문제와 수정

- 첫 benchmark는 label imbalance가 심했고 majority accuracy가 약 0.83에
  가까웠다. balanced rejection sampling으로 교체했다.
- 첫 benchmark는 모든 model에 state projection을 적용해 baseline에
  불공정하게 유리했다. 이제 pre-tensor indexed retrieval은 WPU indexed
  model에만 적용하고 baseline은 full simulator state를 받는다.
- 원래 final-state rule에서는 `caught` label이 거의 발생하지 않았다.
  catcher가 활성화되어 있고 큰 변위가 있었지만 fall하지 않은 경우를
  `caught/recovered`로 정의했다.

## 다음 단계

- seed, training step, simulator sample 수를 늘린다.
- perception-like error를 모사하기 위해 relation/objectification corruption을 추가한다.
- PyBullet이 모든 background body를 동역학적으로 계산하지 않아도 되도록
  frozen simulator-exported background state를 추가한다.
- 동일 parameter/compute 조건의 token 및 graph baseline을 더 강하게 맞춘다.
- model-predicted delta를 다시 `WorldState`에 적용하는 long-horizon
  closed-loop PyBullet rollout을 추가한다.

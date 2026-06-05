# PyBullet 객체화 Stress

이 실험은 PyBullet 컵 benchmark 위에 만든 첫 perception-like robustness
test다. 모델은 clean simulator-objectified `WorldState`로 학습하고, 평가는
corrupted objectified state에서 수행한다. Label과 target delta는 clean
simulator rollout에 고정되어 있으므로, 성능 변화는 물리 scenario 변화가 아니라
state interface 품질 문제로 해석할 수 있다.

Source CSV:

- `docs/experiments/pybullet_objectification_stress.csv`

## 프로토콜

- Simulator: PyBullet `DIRECT` rigid-body rollout.
- Base task: balanced cup impulse branch prediction.
- Training state: clean simulator-derived `WorldState`.
- Evaluation state: corrupted `WorldState`.
- Model: `wpu-cws-indexed-sparse`, `wpu-cws-indexed-local-dense`,
  `graph-transformer`.
- Seed: `11, 13`.
- Background object: `32`.
- Training: 20 steps, batch 8, hidden 64.
- Evaluation: 조건별 36 samples.

Corruption preset:

- `drop_relations_light`: event-target relation의 25% 제거.
- `drop_relations_heavy`: event-target relation의 60% 제거.
- `position_noise`: position에 std `0.08` Gaussian noise 추가.
- `low_confidence`: object/relation confidence에 `0.55` 곱함.
- `identity_swap`: non-target causal object의 attributes/type swap.
- `combined`: relation drop, object drop, position noise, low confidence,
  partial identity swap을 함께 적용.

## 요약

| corruption | model | accuracy | objectification score | frontier recall | selected K |
| --- | --- | ---: | ---: | ---: | ---: |
| clean | graph-transformer | 0.500 | 0.944 | 1.000 | 4.463 |
| clean | wpu-cws-indexed-local-dense | 0.542 | 0.944 | 1.000 | 4.463 |
| clean | wpu-cws-indexed-sparse | 0.528 | 0.944 | 1.000 | 4.463 |
| drop_relations_light | graph-transformer | 0.500 | 0.943 | 0.848 | 4.463 |
| drop_relations_light | wpu-cws-indexed-local-dense | 0.542 | 0.943 | 0.848 | 3.775 |
| drop_relations_light | wpu-cws-indexed-sparse | 0.528 | 0.943 | 0.848 | 3.775 |
| drop_relations_heavy | graph-transformer | 0.500 | 0.941 | 0.602 | 4.463 |
| drop_relations_heavy | wpu-cws-indexed-local-dense | 0.514 | 0.941 | 0.602 | 2.700 |
| drop_relations_heavy | wpu-cws-indexed-sparse | 0.528 | 0.941 | 0.602 | 2.700 |
| position_noise | graph-transformer | 0.500 | 0.944 | 1.000 | 4.463 |
| position_noise | wpu-cws-indexed-local-dense | 0.542 | 0.944 | 1.000 | 4.463 |
| position_noise | wpu-cws-indexed-sparse | 0.542 | 0.944 | 1.000 | 4.463 |
| low_confidence | graph-transformer | 0.486 | 0.789 | 1.000 | 4.463 |
| low_confidence | wpu-cws-indexed-local-dense | 0.514 | 0.789 | 1.000 | 4.463 |
| low_confidence | wpu-cws-indexed-sparse | 0.500 | 0.789 | 1.000 | 4.463 |
| identity_swap | graph-transformer | 0.500 | 0.944 | 1.000 | 4.463 |
| identity_swap | wpu-cws-indexed-local-dense | 0.542 | 0.944 | 1.000 | 4.463 |
| identity_swap | wpu-cws-indexed-sparse | 0.528 | 0.944 | 1.000 | 4.463 |
| combined | graph-transformer | 0.500 | 0.840 | 0.751 | 4.175 |
| combined | wpu-cws-indexed-local-dense | 0.514 | 0.840 | 0.751 | 3.362 |
| combined | wpu-cws-indexed-sparse | 0.542 | 0.840 | 0.751 | 3.362 |

## 해석

이 stress test는 base PyBullet benchmark보다 객체화 경계를 더 명확하게 보여준다.
Relation corruption은 event-local frontier recall과 WPU selected K를 낮춘다.
반면 full-state graph baseline은 여전히 모든 object를 본다. 이는 WPU의 예상된
실패 모드다. Relation이 event와 causal working set을 연결하지 못하면, indexed
sparse processing은 propagation 이전에 필요한 state를 잃는다.

다만 현재 task는 아직 작아서 decisive accuracy 결과는 아니다. Branch accuracy는
corruption 아래에서도 좁은 0.49-0.54 구간에 머문다. 따라서 이 결과는 dominance
claim이 아니라 diagnostic evidence로 사용해야 한다.

## 발견한 문제

- `ObjectificationReport.contract_score`는 relation-drop corruption에서 거의
  변하지 않는다. 현재 relation validity는 잘못된 endpoint를 확인할 뿐, 기대되는
  causal edge 누락을 측정하지 않는다. 이를 보완하기 위해
  `frontier_causal_recall_mean`을 추가했다.
- `identity_swap`은 현재 objectification score로 감지되지 않는다. Object ID는
  syntactically valid하기 때문이다. Semantic identity consistency는 history,
  tracking, role/affordance check가 필요하다.
- pre-tensor projection 이후 계산되는 model-level `causal_recall_mean`은 stress
  test에 충분하지 않다. 이미 선택된 subgraph 내부 recall만 보므로 1.0으로 남을 수
  있다. 따라서 benchmark는 projection 이전 `frontier_causal_recall_mean`을 별도로
  기록한다.

## 다음 단계

- `ObjectificationReport`에 expected-frontier completeness와 semantic identity
  consistency check를 추가한다.
- Evaluation 중 relation repair를 적용해 recovered edge가 WPU selected K와 accuracy를
  복구하는지 측정한다.
- Long-horizon closed-loop rollout을 추가해 missing relation이 one-step branch noise가
  아니라 누적 state error로 나타나는지 본다.
- event target 자체가 누락되거나 alias되는 강한 corruption을 추가하고, WPU가
  confident sparse update 대신 abstain/escalate하도록 만든다.

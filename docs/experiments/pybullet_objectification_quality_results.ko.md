# PyBullet Objectification Quality Benchmark

이 benchmark는 objectification quality를 downstream model accuracy와 분리해서
측정한다. Corrupted PyBullet `WorldState`를 clean simulator-derived state와 비교해
identity, semantic consistency, relation precision/recall, event-frontier recall,
selected `K`, 기존 `ObjectificationReport`를 기록한다.

Source CSV:

- `docs/experiments/pybullet_objectification_quality.csv`

## 프로토콜

- Simulator: PyBullet `DIRECT` cup scene.
- Sample: seed/background setting마다 `12`.
- Seed: `11, 13`.
- Background objects: `32, 128, 512`.
- Corruption: `clean`, `drop_relations_heavy`, `drop_objects_light`,
  `position_noise`, `low_confidence`, `identity_swap`, `combined`.
- Indexed frontier: event target과 relation frontier, `max_nodes=12`,
  `max_depth=1`.

## 대표 요약: Background N=128

| corruption | contract score | identity recall | semantic consistency | relation recall | frontier recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 0.939715 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| combined | 0.837442 | 0.844246 | 0.814672 | 0.834180 | 0.716667 |
| drop_objects_light | 0.939794 | 0.796720 | 0.796720 | 0.795459 | 0.866667 |
| drop_relations_heavy | 0.938665 | 1.000000 | 1.000000 | 0.979556 | 0.585417 |
| identity_swap | 0.939715 | 1.000000 | 0.984895 | 1.000000 | 1.000000 |
| low_confidence | 0.786843 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| position_noise | 0.939715 | 1.000000 | 0.675541 | 1.000000 | 1.000000 |

## 해석

결과는 objectification을 하나의 scalar가 아니라 multi-part contract로 평가해야
한다는 점을 확인한다.

`low_confidence`는 기존 contract score에 잘 반영된다. Background `N=128`에서 score가
`0.939715`에서 `0.786843`으로 내려간다.

반대로 `drop_relations_heavy`는 다른 실패 양상을 보여준다. Contract score는
`0.938665`로 거의 변하지 않고, global relation recall도 대부분의 background relation이
남아 있어 `0.979556`으로 높다. 하지만 sparse WPU propagation에 직접 중요한
event-frontier recall은 `0.585417`까지 떨어진다.

`position_noise`도 semantic consistency를 syntactic identity validity와 분리해야
함을 보여준다. Object ID와 relation은 유효하므로 contract score는 높게 유지되지만,
semantic consistency는 `0.675541`로 떨어진다.

`identity_swap`은 이 PyBullet scene에서는 swap 가능한 role-bearing non-protected
object가 적어 약하게 나타난다. 그래도 relation validity나 identity coverage가 아니라
semantic consistency가 감지한다.

## WPU 함의

WPU에는 단순히 “객체”가 필요한 것이 아니다. 최소한 다음 네 가지가 측정되는
objectified state가 필요하다.

- stable identity coverage;
- role, type, geometry에 대한 semantic identity consistency;
- relation precision/recall;
- causal working set에 대한 event-frontier completeness.

현재 `ObjectificationReport`는 유용하지만 불완전하다. Perception-like corruption
조건에서 sparse WPU claim을 하려면 report를 확장하거나 task-specific quality metric과
함께 사용해야 한다.

## 다음 단계

- Frontier completeness와 semantic identity consistency를 public objectification
  report에 추가한다.
- 각 objectification metric이 downstream branch loss와 어떤 관계인지 평가한다.
- Clean-state relation comparison뿐 아니라 가능한 경우 simulator relation ground truth를
  사용한다.
- 더 많은 role-bearing object가 있는 scene에서 identity swap stress를 강화한다.

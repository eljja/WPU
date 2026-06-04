# WPU에서의 객체화

이 문서는 WPU에서 말하는 객체화(objectification)를 정의하고, 왜 이것이
state-native world processing의 중심인지 설명한다. 또한 장기 연구 목표를
명확히 한다. 객체화된 state entity들 사이의 관계는 먼저 알려진 국소 물리 이론의
근사에 도달해야 하며, 궁극적으로는 아직 명시적으로 이해하지 못한 규칙성까지
학습 가능한 구조로 드러내야 한다.

## 정의

WPU에서 객체화란 관측되었거나 시뮬레이션된 세계를 지속적이고 주소 지정 가능한
state entity로 변환하는 것이다.

```text
Objectification(x) =
  identity
  + typed attributes
  + role/affordance state
  + typed relations
  + time/history
  + uncertainty
  + admissible deltas
  + branch-local overlays
```

객체화된 entity는 단순한 detection region, token span, table row, type label,
embedding이 아니다. 그것은 query되고, patch되고, relation을 따라 propagation되고,
여러 미래로 branch되며, 시간에 따라 consistency를 검사할 수 있는 state-bearing
unit이다. Type label은 유용한 metadata지만 그 자체로 충분한 객체화가 아니다. WPU에는
dynamic role, manipulator role, support role, boundary role, context role,
geometry, confidence, history처럼 relation을 지탱하는 state variable이 필요하다.

## 최소 객체 계약

WPU 객체는 다음 계약을 만족해야 한다.

- `identity`: 사건과 시간 step이 바뀌어도 같은 객체로 참조될 수 있다.
- `attributes`: 위치, 속도, 역할, 상태, 소유권, 온도, confidence 같은 typed state를 가진다.
- `role/affordance state`: nominal type name에만 의존하지 않고, 어떤 relation family가
  변화 propagation을 매개할 수 있는지 판단할 수 있는 변수를 제공한다.
- `relations`: `on`, `near`, `touching`, `supports`, `connected_to`, `causes`,
  `depends_on` 같은 typed edge에 참여한다.
- `uncertainty`: 객체와 속성은 불확실할 수 있다.
- `delta semantics`: event는 전체 world를 다시 쓰지 않고 explicit patch로 객체를 갱신한다.
- `branch semantics`: 여러 미래는 같은 base object 위에 서로 다른 delta를 overlay할 수 있다.

현재 코드는 이 계약을 `WorldObject`, `Relation`, `Event`, `DeltaState`,
`Branch`, `WorldState`로 표현한다.

Public API는 `evaluate_objectification(state, ...)`도 제공한다. 이 함수는
`ObjectificationReport`를 반환하며, propagation 전에 supplied state가 operational
contract를 만족하는지 측정한다. 측정 항목은 identity coverage, relation endpoint
validity, object/relation confidence, delta validity, expected causal working
set 대비 optional delta locality다.

Public API는 `infer_missing_relations`와 `repair_objectification_relations`도
제공한다. 이 함수들은 object identity는 있지만 relation extraction이 local
connectivity를 놓친 경우, 약한 `near`, `touching` 같은 geometry-derived relation
patch를 보수적으로 추가한다. 즉 relation repair를 token fallback이나 silent dense
fallback이 아니라 명시적이고 audit 가능한 hypothesis로 만든다. Repair는 type-gated
될 수 있다. Geometry-only repair는 recall은 복구하면서 spurious distractor edge를
많이 만들 수 있으므로, typed object identity는 repair contract의 일부다. 현재 probe는
작은 learned relation scorer도 포함한다. 이 scorer는 controlled distractor
distribution에서 hand-written type gate와 같은 결과를 내고, 더 조밀한 distractor
조건에서도 precision을 유지하며, role/affordance state variable이 보존되면 aliased type
name을 넘어 transfer한다. 반대로 type label과 role variable이 모두 제거되면 실패한다.
이것이 operational boundary다. 객체화는 이름 붙이기가 아니라, propagation을 지탱할
충분한 relational variable을 가진 persistent state다. 같은 probe는 toy downstream
branch diagnostic도 포함한다. Role-aware learned repair는 aliased-type 조건에서 branch
accuracy를 `0.343750`에서 `0.671875`로 올리고 loss를 `1.319667`에서 `0.885275`로
낮춘다. 반대로 ungated dense-distractor repair는 frontier recall을 복구해도 loss를
악화시킨다. 이것은 여전히 diagnostic이지 물리 법칙 발견 증거가 아니다.

## 객체화가 아닌 것

객체화는 WPU가 이미 perception을 해결했다는 뜻이 아니다. 현재 WPU core는 simulator,
database, supervised state extractor, tracker, 또는 미래의 perception adapter가
object state를 제공한다고 가정한다.

객체화는 token이 객체 정보를 설명할 수 없다는 뜻도 아니다. Token은 객체 정보를
encode할 수 있다. WPU의 주장은 operational claim이다. Explicit object는 identity,
update, relation traversal, delta overlay, branch-local state를 1차 연산으로 만든다.

## 성능에서 왜 중요한가

객체화는 sparse execution을 의미 있게 만든다. 전체 world state 크기가 `N`이지만
하나의 event가 causal working set `K`만 건드린다면, WPU는 다음을 실행하려 한다.

```text
retrieve(K) + propagate(K) + patch(K)
```

즉 매번 전체 `N`을 처리하지 않는다. 이 방식은 `K << N`이고 `K`를 전체 scan 없이
식별할 수 있을 때 event latency, effective state-update throughput, branch-rollout
efficiency, memory traffic을 개선할 수 있다.

따라서 관련 metric은 raw token/sec가 아니라 다음이다.

- event/sec;
- useful causal state updates/sec;
- branch rollout/sec;
- state-patch latency;
- bytes moved per causal update;
- sparse runtime crossover에서 유지되는 accuracy.

객체화 품질은 이러한 metric의 전제 조건이다. identity가 불안정하거나 relation
endpoint가 깨져 있거나 delta가 non-causal object를 갱신하면, WPU는 계산량을 줄이면서
동시에 더 틀릴 수 있다. 따라서 성능 보고에는 execution metric과 objectification
metric을 함께 넣어야 한다.
실패 원인이 object 누락이 아니라 local relation 누락이라면, WPU는 retrieval budget을
넓히거나 dense recompute로 가기 전에 relation repair를 시도할 수 있다. 단 repair
자체는 relation precision/recall과 downstream prediction loss로 평가해야 한다.
Frontier recall만으로는 부족하다. Spurious relation은 `K`를 키우고 prediction을
악화시킬 수 있다.

## 물리적 근사와의 관계

WPU propagation은 단순화된 local-causality prior로 이해해야 한다. 이것은 물리 법칙을
해결했다는 뜻이 아니다. 많은 세계 변화가 persistent entity 사이의 relation,
즉 contact, support, containment, proximity, connectivity, ownership, dependency,
constraint를 통해 매개된다는 가정이다.

가까운 과학적 목표는 다음이다.

```text
object relations + learned propagation
  -> approximate local physical rules
  -> maintain predictive accuracy under sparse state updates
```

예시는 다음과 같다.

- 접촉과 힘 전달;
- 지지와 낙하;
- containment와 spill risk;
- 연결성과 flow;
- 충돌과 local constraint violation;
- dependency와 cascading failure.

이는 단순 물리 모델이 제한된 regime에서 유용한 근사를 제공하는 것과 비슷하다.
WPU는 이 근사를 명시적이고 측정 가능하며 반증 가능하게 만들어야 한다.

## 장기 목표: 아직 모르는 규칙성

더 강한 장기 목표는 모든 relation을 사람이 손으로 정의하는 것이 아니다. Object
history에서 유용한 latent regularity를 드러내는 object-relation structure를 학습하는
것이다.

```text
observed object histories
  -> candidate relations
  -> propagation rules
  -> falsifiable predictions
  -> revised object/relation theory
```

알려진 domain에서는 이러한 learned relation이 물리 이론을 근사할 수 있다. 아직 잘
이해하지 못한 domain에서는 인간이 이름 붙이기 전의 안정적인 interaction pattern을
드러낼 수도 있다. 이것은 현재 결과가 아니라 연구 프로그램으로 취급해야 한다.

현재 저장소에는 이 방향을 위한 toy hidden-mechanism probe가 추가되어 있다.
`contact_transfer`와 `support_transfer`의 object history로 학습한 relation scorer를
object type name이 바뀐 held-out `hidden_field`에서 평가한다. 이 scorer는 5개 seed 평균
relation precision/recall `0.987500`, downstream accuracy `0.992188`를 기록하지만,
type prior는 `0.494531`에 머문다. 이는 synthetic setting에서 history-derived relation variable이
이름을 넘어 transfer할 수 있음을 보여주는 증거다. 실제 물리 법칙 발견 증거는 아니다.

후속 local-law probe는 relation discovery와 theory discovery 사이의 중간 단계를
추가한다. History-derived relation selector와 해석 가능한 inverse-distance law를 결합해
object type name이 바뀐 held-out `hidden_inverse`에서 평가한다. 5개 seed 평균 relation
precision/recall `0.988281`, delta MSE `0.000828`를 기록하며, no-relation 또는 type
prior는 `0.445909`에 머문다. 이는 objectified history가 controlled synthetic setting에서
approximate local law fitting을 지탱할 수 있음을 보인다. 아직 unknown physical law를
발견했다는 증거는 아니다.

같은 probe의 OOD version은 distance, gain, response-form shift를 평가한다. Relation/law
stack은 여전히 유용하지만 보편적으로 안정적이지는 않다. Far-distance relation recall은
`0.658594`로 떨어지고, gain 또는 denominator shift에서는 oracle relation을 사용해도
residual MSE가 남는다. 이것이 WPU에서 approximate theory가 갖는 operational meaning이다.
객체화는 local rule을 제안하고, OOD stress는 그 rule을 신뢰할지, 재보정할지, 교체할지를
결정한다.

따라서 개발 단계는 다음과 같다.

```text
measured object contract
  -> deterministic relation repair
  -> learned relation candidates
  -> held-out-rule prediction gain
  -> interpretable local-law fit
  -> OOD stress and rule revision
  -> falsifiable revised relation theory
```

## 개선 경로

객체화에 기반해 WPU를 개선하는 구체적 경로는 다음이다.

1. Schema validation, `ObjectificationReport`, state-integrity test로 object contract를 강화한다.
2. Contact, support, containment, flow, dependency, ownership, constraint relation family를 추가한다.
3. Deterministic relation repair는 conservative fallback으로만 사용하고, 모든 repaired edge는 ground truth가 아니라 hypothesis로 log한다.
4. Domain knowledge가 있을 때 local conservation, consistency, no-spurious-delta loss로 propagation을 학습한다.
5. 반복 delta 이후에도 object identity와 relation consistency가 유지되는지 long-horizon rollout test를 추가한다.
6. Ground-truth object/relation이 있는 simulator-backed benchmark를 추가한다.
7. Object history에서 candidate relation을 학습하고 held-out regime에서 prediction을 개선하는지 평가한다.
8. Retriever/projection budget을 객체화 품질에 연결한다. 낮은 relation validity나 낮은 delta locality는 blind sparse propagation이 아니라 wider retrieval, dense recompute, state repair를 trigger해야 한다. 현재 scheduler는 낮은 objectification score를 sparse routing에서 hybrid/dense로 올리는 첫 버전을 구현한다.
9. 객체화 실패도 보고한다. Missed object, identity swap, relation hallucination, `K`가 작지 않은 global event가 포함된다.

## 주장 경계

현재 WPU evidence는 객체화를 explicit computational interface로 지지한다. 아직 다음을
증명하지는 않는다.

- end-to-end perception-to-object construction;
- broad physical understanding;
- unknown physical law discovery;
- hardware-level energy 또는 speed advantage;
- token 또는 graph model 대비 보편적 accuracy 우월성.

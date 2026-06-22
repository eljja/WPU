# WPU World-Copy Model

이 문서는 WPU의 장기 목표를 정의한다. 목표는 world를 token sequence로 반복 직렬화하는
것이 아니라, object state와 causal propagation을 통해 유지되는 persistent executable
copy로 만드는 것이다.

## 정의

WPU world copy는 persistent하고 query/update 가능한 state substrate다.

```text
WorldCopy(t) =
  Objects(t)
  + Relations(t)
  + Regions(t)
  + EventLog(<=t)
  + Uncertainty(t)
  + DeltaLog(<=t)
  + Branches(t -> t+k)
```

이는 물리 세계의 완전한 복제본이 아니다. 여기서 copy는 세계를 예측, 갱신, 수정하는 데
필요한 identity, attribute, relation, uncertainty, possible future를 시간에 따라
유지하는 executable state model이라는 뜻이다.

Operational meaning은 다음과 같다.

- 같은 object를 event와 time step을 넘어 참조할 수 있다.
- 전체 world를 다시 쓰지 않고 object attribute를 patch할 수 있다.
- relation이 어떤 object로 변화가 전파될 수 있는지 결정한다.
- event는 local delta를 만들고 alternative branch를 열 수 있다.
- 새 observation은 stored state를 correction하고 correction 이유를 남긴다.
- uncertainty는 sparse propagation, local dense recompute, global recompute, external
  observation 중 무엇을 쓸지 결정한다.

## 객체지향적 Processing Unit

WPU가 객체지향적이라는 말은 프로그래밍 언어의 class 문법이 아니라 계산 단위의 의미다.
WPU object는 다음을 가진 execution unit이다.

- persistent identity;
- mutable typed state;
- geometry, role, affordance, confidence, history, physical/action context 같은
  relation-bearing variable;
- admissible local delta;
- propagation을 위한 typed relation channel;
- branch-local overlay;
- uncertainty와 validity check.

Token processor에서 object는 대개 sequence 안의 evidence로 간접 표현된다. WPU에서
object는 addressable state다. Local event는 관련 object 집합에 대한 method call에
가깝다.

```text
event(object_i, action) -> delta(object_i)
delta(object_i) --relation_j--> delta(object_k)
BaseState + DeltaState -> updated WorldCopy
```

성능 이득은 state가 항상 token보다 정확하다는 데서 나오지 않는다. 세계는 크지만 각
event가 작은 causal working set만 필요로 할 때 이득이 나온다. Processor는 다음을
수행해야 한다.

```text
retrieve(K) + propagate(K) + patch(K)
```

반복적으로 다음을 수행하는 대신이다.

```text
serialize(N) + attend/recompute(N)
```

## 절대 흔들리면 안 되는 목표 정렬

WPU의 궁극 목표는 더 좋은 graph benchmark model을 만드는 것이 아니다. 목표는 executable
world copy를 유지할 수 있는 processing model을 만드는 것이다. 이를 위해 다음 조건은
non-negotiable이다.

- 계산의 persistent unit은 token order가 아니라 objectified state다.
- Propagation은 unrelated global averaging이 아니라 typed relation과 causal mechanism을
  통해 일어나야 한다.
- Large-`N` 주장은 non-causal background state가 event-local working set 밖에 유지된다는
  증거가 있어야 한다.
- Accuracy 개선을 위해 primary processing path를 full token serialization으로 조용히
  되돌리면 안 된다.
- Sparse execution은 causal retrieval이 신뢰 가능할 때만 허용된다. 그렇지 않으면
  escalate하고 이유를 보고해야 한다.
- World-copy quality는 시간에 따라 평가해야 한다. One-step accuracy는 이기지만 long
  horizon에서 drift하는 state model은 world copy가 아니다.

이 기준이 WPU를 LPU-style token processor와 구분한다. LPU는 긴 serialized context를
처리해 state를 흉내낼 수 있지만, WPU는 identity, relation traversal, local mutation,
branch overlay, correction을 native operation으로 만들어야 한다. 따라서 비교 대상은
token/sec만이 아니라 latency, memory traffic, correction cost당 event-local state update
quality다.

## 자연스러운 Propagation

자연스러운 propagation은 state change가 world structure를 근사하는 object relation을
따라 흐른다는 뜻이다. 이는 learned local-causality prior이지, 현재 구현이 full physics를
발견했다는 주장이 아니다.

물리 scene의 예시는 다음과 같다.

- support: table이 cup을 제약한다.
- contact: hand가 cup에 force를 전달할 수 있다.
- proximity: edge는 충분히 가까울 때만 fall risk를 바꾼다.
- containment: container 안의 object는 container motion을 따른다.
- occlusion: sensor-visible state가 uncertainty로 바뀔 수 있다.
- constraint: 불가능한 position/velocity는 correction되어야 한다.

비물리 state world에서는 dependency, ownership, connectivity, permission, health,
supply, causal influence가 유사한 propagation relation이 될 수 있다.

v3 목표는 relation-conditioned local mechanism을 학습하는 것이다.

```text
message = f_relation(source_state, target_state, relation_state, event_state)
delta_target = g(target_state, aggregate(messages), uncertainty)
```

학습된 mechanism은 auditable해야 한다. 예측이 실패하면 그 원인이 objectification,
causal retrieval, propagation dynamics, uncertainty calibration, state correction 중
무엇인지 드러나야 한다.

## 필요한 Runtime Component

World-copy runtime에는 다음 component가 필요하다.

| Component | 역할 |
|---|---|
| State store | Persistent object, relation, region, uncertainty, delta memory. |
| Causal index | Tensor projection 전에 event-local working set `K`를 검색. |
| Propagation core | `K` 위에서 learned relation-conditioned update 수행. |
| Branch manager | Alternative future를 `BaseState + DeltaState`로 유지. |
| Correction loop | Prediction과 새 observation을 reconciliation. |
| Integrity monitor | Invalid state, drift, stale object, uncertainty growth 감지. |
| Dense fallback | Sparse evidence가 안전하지 않을 때 local/global state recompute. |

## Benchmark Definition

유효한 world-copy benchmark는 다음을 제공해야 한다.

- total world size `N`;
- 가능한 경우 event-local causal working set reference `K_ref`;
- 시간에 따른 object identity;
- object attribute와 uncertainty;
- typed relation과 relation noise setting;
- event/action stream;
- ground-truth future observation 또는 simulator state;
- branching이 있으면 branch label 또는 future outcome distribution.

Benchmark는 다음을 변화시켜야 한다.

- non-causal background size;
- causal working set size;
- relation missing rate;
- relation false-positive rate;
- object creation/deletion;
- region migration;
- occlusion 또는 confidence degradation;
- mechanism shift;
- horizon length.

## Success Metrics

WPU v3는 token-like accuracy만이 아니라 state-native metric으로 평가해야 한다.

- causal slice recall과 precision;
- selected `K`와 affected fraction `K/N`;
- event latency;
- state updates/sec;
- bytes moved per update;
- next-state error;
- branch accuracy, NLL, Brier, ECE;
- identity continuity;
- relation consistency;
- state-integrity score;
- correction cost;
- long-horizon trajectory error;
- dense fallback rate.

## Success Criteria

v3 결과가 강하다고 말하려면 최소 하나의 large-`N` benchmark에서 다음을 모두 만족해야
한다.

- `N`이 최소 한 order 이상 커져도 selected `K`가 bounded 또는 sublinear로 유지된다.
- Causal slice recall이 충분히 높아 sparse propagation이 필요한 state를 떨어뜨리지
  않는다.
- Matched budget에서 WPU가 best token/graph baseline과 같거나 더 높은 state 또는 branch
  accuracy를 보인다.
- WPU가 event latency, memory traffic, dense compute 중 하나 이상에서 의미 있게 낮다.
- Long-horizon state integrity가 constant global recompute 없이 안정적이다.
- Failure case가 aggregate score에 숨지 않고 objectification, retrieval, propagation,
  uncertainty, correction 중 어디서 발생했는지 trace된다.

## Failure Criteria

다음이 관찰되면 WPU v3 주장은 약화되어야 한다.

- selected `K`가 `N`과 선형으로 증가한다.
- causal retrieval이 event마다 full world scan을 필요로 한다.
- missing/noisy relation이 causal slice recall을 무너뜨린다.
- sparse propagation은 빠르지만 같은 budget의 token baseline보다 정확도가 낮다.
- long-horizon delta가 frequent dense recompute 없이는 drift한다.
- streaming update에서 object identity가 불안정하다.
- correction cost가 full-state recompute cost에 가까워진다.

## 현재 상태

현재 repository에는 v3 substrate의 첫 조각이 있다.

- region/object membership을 위한 `HierarchicalWorldState`.
- multi-signal causal retrieval을 위한 `WorldCausalIndex`.
- controlled non-causal-background setting에서 total `N`이 `104`에서 `10004`까지 커져도
  selected `K = 4`를 유지하는 `world_copy_index_probe`.

이는 필요하지만 충분하지 않다. 저장소가 large-world causal indexing을 표현할 수 있음을
보이지만, trained world modeling, 실제 물리 이해, perception-to-state construction,
long-horizon world-copy stability를 증명하지는 않는다.

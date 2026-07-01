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

현재 repository에는 v3 world-copy substrate의 첫 조각들이 있다.

- region/object membership을 위한 `HierarchicalWorldState`.
- multi-signal causal retrieval을 위한 `WorldCausalIndex`.
- controlled non-causal-background setting에서 total `N`이 `104`에서 `10004`까지 커져도
  selected `K = 4`를 유지하는 `world_copy_index_probe`.
- missing relation과 false-positive relation이 있는 controlled noisy-index setting에서
  `N=8192`까지 retrieval behavior를 측정하는 `world_copy_causal_index_stress`.
- low-confidence relation escalation 이후 local region correction candidate가 causal
  update-set recall을 회복할 수 있음을 보이는 `world_copy_escalation_correction_probe`.
- bounded local correction candidate가 controlled synthetic local-law setting에서 learned
  delta MSE를 낮출 수 있음을 보이는 `world_copy_learned_correction_probe`.
- 같은 task에서 WPU/token/graph/dense를 비교하는 첫 screen인
  `world_copy_baseline_comparison_probe`. 업데이트된 `wpu-region-guard` path는 bounded
  selected `K`와 훨씬 낮은 work/bytes proxy를 유지하면서 controlled dense/token baseline보다
  낮은 raw delta MSE를 보인다. 이 결과는 bounded, reliable local region에 의존한다.
- `wpu-region-guard` path를 object churn과 region migration이 포함된 H=25 controlled
  stream으로 확장하는 `world_copy_streaming_region_guard_probe`. Bounded-region setup에서
  state integrity와 zero correction cost를 유지하면서 full-state work/bytes를 피한다.

이는 필요하지만 충분하지 않다. 저장소가 large-world causal indexing과 첫 learned local
correction diagnostic을 표현할 수 있음을 보이지만, full trained world modeling, 실제 물리
이해, perception-to-state construction, raw token/graph accuracy superiority,
real-simulator long-horizon world-copy stability를 증명하지는 않는다.

### Selective-region 실패 경계

N=8192, contamination=128에서 일반 region guard는 약 134--136개 객체를
선택하고 MSE 0.361--0.477을 기록한다. typed role과 confidence로 순위를 매긴
`2*K_ref` guard는 K=16을 유지하고 MSE를 0.034--0.115로 낮춘다. 따라서 region
membership은 causal truth가 아니라 noisy index이며, region과 relation evidence에서
모두 누락된 객체는 local retrieval로 복구할 수 없다.

### Dual-index omission correction 경계

`world_copy_dual_index_escalation_probe`는 이 복구 불가능 경계를 직접 테스트한다.
Causal object가 active region과 relation frontier에서 동시에 빠지면
`wpu-selective-region-guard`는 recall을 잃는다. 하지만 누락된 object가 인접
observation/correction pool에 남아 있으면 bounded `wpu-escalating-neighbor-guard`가
상당 부분을 복구할 수 있다. `N=8192`, `dual_omission=0.75`,
`escape_rate=0.0`에서 escalating guard는 selected `K`를 16에서 24로만 늘리면서
trajectory MSE를 `0.416213`에서 `0.084905`로 낮춘다. `escape_rate=0.25`에서도
MSE를 `0.377478`에서 `0.163802`로 낮추지만, dense state copy는 여전히 raw accuracy
상한이다. 따라서 v3의 다음 경계는 명확하다. Bounded correction은 objectification이
근처 correction candidate를 노출할 때만 누락 state를 복구할 수 있고, 완전히 관측되지
않은 causal object에는 external observation, broader escalation, dense recompute가
필요하다.

### Uncertainty-triggered observation 경계

`world_copy_uncertainty_observation_policy_probe`는 그 다음 단계를 테스트한다. 즉 causal
object가 adjacent correction pool에서도 빠진 경우다. WPU policy는 world 전체를
serialize하지 않는다. 대신 local support evidence가 부족할 때 bounded external
observation probe를 요청하고, 반환된 object만 causal slice에 patch한다. `N=8192`,
`escape_rate=0.75`, observation budget `8`에서 이 방식은 neighbor-only의 trajectory MSE
약 `0.323295`를 `0.098747`로 낮추며, selected `K`는 `N`과 함께 커지지 않고 `32`로
유지된다. `escape_rate=0.50`, budget `8`에서는 MSE가 `0.255797`에서 `0.083280`으로
개선된다. Dense state copy는 controlled setup에서 여전히 exact하므로 raw accuracy
superiority를 증명한 것은 아니다. 이 결과는 실제 world copy에 필요한 state-native
correction pattern을 보여준다. Local index가 부족하면 uncertainty가 bounded observation을
구매할 수 있어야 한다.

### Adaptive observation-budget 경계

`world_copy_adaptive_observation_budget_probe`는 observation을 사람이 정한 고정 budget에서
bounded WPU decision으로 옮긴다. 이 policy는 local support deficit과 cheap anomaly
signal을 함께 사용하므로, 누락 support가 harmless index noise로 설명될 때는 observation을
사지 않는다. `N=8192`, `escape_rate=0.0`에서 adaptive observation은 budget `0.0`,
selected `K=24`를 유지하고 fixed-budget objective penalty를 피한다(`0.082287` 대
`0.206153`). `escape_rate=0.75`에서는 mean budget `4.3125`만 쓰면서 selected `K=32`를
유지하고 MSE `0.079298`을 달성한다. 이는 fixed-budget MSE `0.079620`과 거의 같거나
조금 낮고, cost-aware objective는 `0.199620`에서 `0.143985`로 낮아진다. 아직 learned
policy는 아니지만, 다음 runtime contract를 고정한다. Observation은 constant full-world
fallback이 아니라 bounded budget을 가진 state-native correction decision이다.

### Learned observation-policy 경계

`world_copy_learned_observation_policy_probe`는 hand-specified budget rule을 WPU
uncertainty summary 위의 작은 learned classifier로 대체한다. Clean setting에서는 hand
adaptive policy를 근사할 수 있다. `N=8192`, `escape_rate=0.75`에서 learned observation은
mean budget `4.625`를 사용하고 hand adaptive `4.71875`와 비슷하게 bounded `K`를
유지하며 objective `0.163047`을 달성한다. Hand adaptive objective는 `0.163946`이다.
`escape_rate=0.50`에서는 learned objective `0.132664`, hand adaptive objective
`0.133466`이다. 실패 경계는 anomaly shift다. `noisy_anomaly`에서는 learned policy가
budget을 과하게 쓰고 hand rule보다 objective가 나쁘며, `weak_anomaly`에서는 learned와
hand adaptive 모두 과소 관측해 causal object를 놓친다. 따라서 다음 correction 문제는
clean rule 학습 자체가 아니라 shifted observation signal 아래의 calibration이다.

### Calibrated learned observation-policy 경계

`world_copy_calibrated_observation_policy_probe`는 learned budget decision 앞에 작은
calibration set과 neighbor-support credit을 추가한다. 이는 `noisy_anomaly`의 false high
anomaly score와 `weak_anomaly`의 suppressed causal anomaly score라는 두 shift 실패를 직접
공격한다. `N=8192`, `escape_rate=0.75`에서 `noisy_anomaly`는 learned objective
`0.258949`에서 calibrated objective `0.203604`로 개선되고, mean budget은 `6.0`에서
`4.625`로 줄며 recall은 `0.789062`에서 `0.871094`로 오른다. `weak_anomaly`에서는 같은
조건에서 objective가 `0.327296`에서 `0.184247`로 개선되고 recall은 `0.414062`에서
`0.863281`로 오른다. 아직 완벽하지는 않다. Dense state copy는 여전히 exact이며,
low-escape calibration은 neighbor correction만으로 충분했던 경우를 약간 흔들 수 있다.
다음 실패는 labeled shift calibration set에 의존하지 않는 robust calibration이다.

### Online observation-calibration 경계

`world_copy_online_calibration_policy_probe`는 labeled calibration set 가정을 제거하고,
bounded observation의 hit/miss feedback으로 anomaly calibration을 온라인 갱신한다. 현재
버전은 모든 mode를 paired event stream에서 평가하고 conservative stability gate를
추가하며, `wpu-verified-online-observation`을 추가한다. 이 mode는 estimated marginal
correction value가 observation cost보다 클 때만 최대 2개의 추가 관측을 요청하는 bounded
correction-policy verifier다. 이는 의도한 world-copy correction loop에 더 가깝다. WPU는
작은 candidate set만 관측하고, 그 관측이 누락된 causal state를 복구했는지 측정한 뒤,
full-world serialization 없이 다음 observation sensitivity를 조정한다.

`N=8192`, `escape_rate=0.75`에서 `noisy_anomaly`는 learned objective `0.266230`에서
verified online `0.193618`로 개선되고, recall은 `0.800781`에서 `0.957031`로 오른다.
`weak_anomaly`에서는 verified online이 learned objective `0.334783`을 `0.202765`로
개선하며, unverified online `0.211687`보다 좋고 labeled calibration `0.196455`에
가까워진다. Recall은 `0.390625`에서 `0.822266`으로 오른다. Clean paired stream에서는
verified online이 learned objective `0.166575`를 `0.159478`로 개선해 hand adaptive
`0.154890`에 가까워진다. 이때 mean verifier top-up은 bounded이며 value-gated다. Clean은
top-up `0.171875`, `noisy_anomaly`는 top-up `0.0`, `weak_anomaly`는 top-up `1.09375`다.
Dense state copy는 여전히 exact지만 `8192` state unit 전체를 touch한다.

남은 경계는 더 선명하지만, 첫 안전한 base-budget 결과는 positive다. Naive value trimming은
hidden causal state를 복구했는지 측정하기 전에 tail observation을 제거했기 때문에 실패했다.
Sequential online verifier는 ranked candidate를 하나씩 관측하고, hit/miss evidence가 남은
marginal value가 낮다고 보일 때만 멈추며, calibration이 불안정하면 full proposed budget으로
fallback한다. 같은 `N=8192`, `escape_rate=0.75` noisy setting에서 mean base observation
budget은 `6.796875`에서 `6.140625`로 줄고, recall은 오히려 `0.957031`에서 `0.960938`로
소폭 오르며, objective는 online/verified `0.193618`에서 `0.181400`으로 개선된다. 이는
labeled calibration set 없이 작은 labeled-calibration objective `0.180837`에 근접한 결과다.

첫 composition policy는 `wpu-composed-online-observation`으로 구현됐다. Online calibration이
under-observation을 가리키면(`offset > 0.03`) verified top-up path를 선택하고, 그 외에는
sequential base-budget stopping을 선택한다. 이 selector는 noisy sequential 결과 `0.181400`과
weak verified 결과 `0.202765`를 보존하면서 `K`를 약 `32`로 bounded하게 유지한다. 하지만
universal best policy는 아니다. Clean stream에서는 neutral(`0.166575`)이고 verified top-up
`0.159478`보다 나쁘며, weak anomaly에서는 labeled calibration `0.196455`가 여전히 가장 좋다.
따라서 다음 실패는 path composition 자체가 아니라, noisy over-observation을 다시 만들지 않으면서
clean miss까지 복구하는 no-harm composition gate를 학습하는 것이다.

Learned no-harm composition gate는 다음 positive step이다. 이 gate는 sequential path와
verified path의 paired outcome에서 학습하며, calibration offset/scale, streak state, proposed
budget, sequential trim, observed hit precision, support deficit, background anomaly pressure 같은
WPU-native feedback feature를 사용한다. Conservative clean-recovery prior는 sequential trim이
없고, observed precision이 높고, support deficit이 남아 있으며, high-anomaly background fraction이
낮을 때만 작은 top-up을 연다. `N=8192`, `escape_rate=0.75`에서 learned-composed는 clean miss를
verified 수준인 `0.159211`까지 복구한다(sequential `0.166043`). 동시에 noisy no-harm은
`0.181089`로 보존해 verified `0.193234`로 퇴행하지 않고, weak anomaly는 `0.194131`까지
개선되어 verified `0.200840`, hand-composed `0.207904`, sequential `0.215912`보다 좋다. 남은
gap은 clean recovery 자체가 아니라, hand-coded clean-recovery prior를 learned safety-calibrated
gate로 대체하고 weak labeled calibration `0.191310` 및 clean hand adaptive `0.154867`과의 차이를
줄이는 것이다.

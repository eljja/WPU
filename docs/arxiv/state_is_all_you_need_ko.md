# State Is All You Need: World-State Processing Unit

이 문서는 영문 arXiv 원고 `state_is_all_you_need_en.tex`의 한글 동반 문서다.
제출 기준 원고는 영문 LaTeX이며, 이 문서는 주장 구조와 실험 해석을 한국어로
검토하기 위한 버전이다.

## 초록

현대 sequence model은 세계를 token stream으로 처리한다. 이 논문은 다른
primitive를 제안한다: persistent world state다. World-State Processing Unit,
즉 WPU는 장면을 객체, 관계, 시간, 불확실성, 사건, 미래 branch로 표현하고,
state delta를 sparse propagation, hybrid correction, dense recomputation으로
갱신하는 state-native neural architecture다.

첫 reference implementation인 `WorldStateProcessor`는 synthetic robot-cup
object physics task에서 200 step 학습 후 next-state MSE를 `0.8111`에서
`0.0005`로 낮추고 branch accuracy를 `0.1289`에서 `0.7188`로 높였다. 이는
`0.6680` majority baseline을 넘는다. 그러나 더 강한 5-seed baseline suite는
WPU가 보편적으로 우월하지 않음을 보여준다. WPU는 medium local regime과 noisy
relation regime에서 경쟁력이 있지만, large-N regime에서는 graph/token baseline이
accuracy에서 우세하다.

따라서 현재 논문의 핵심 주장은 보편 우월성이 아니다. WPU는 persistent identity,
local causal change, uncertainty, branching이 지배적인 world-processing workload에
적합한 계산 primitive가 무엇인지 묻는 반증 가능한 regime hypothesis다. 따라서
실험 목표도 “WPU가 항상 이긴다”가 아니라 `rho`, `N`, branch pressure, noise,
affected-region size에 따라 state-native execution이 언제 유효한지를 그리는 것이다.

최근 v2 실험은 이 주장을 한 단계 좁혔다. WPU의 장점은 propagation 자체뿐 아니라
propagation 이전의 object working-set control에서도 나타난다. Regret-distilled
retriever는 15개 seed/K 조건 중 14개에서 learned interaction retriever보다 loss를
낮췄고, 최신 cross-seed 실험은 role/geometry/family descriptor와 risk-adjusted
mechanism selection을 사용해 `N=2048`, `K=8,16,32` held-out seed 조건 모두에서
static learned selector보다 평균 loss를 낮췄다.

## 1. 문제의식

Token sequence는 언어와 ordered evidence를 표현하기 위한 강력한 추상화다.
하지만 world processing은 단순히 다음 token을 예측하는 문제가 아니다. 세계는
지속되는 객체, 명시적 관계, 시간적 연속성, 불확실성, 국소적 인과 변화, 여러
가능한 미래를 가진다.

핵심 구분은 다음과 같다.

```text
Token = ordered evidence for append / attend
State = persistent substrate for patch / propagate / branch
```

중요한 점은 token이 state를 표현할 수 없다는 주장이 아니라는 것이다. 모든 유한한
state는 token sequence로 직렬화될 수 있다. 그러나 직렬화된 sequence는 object
identity, relation traversal, delta update, branch overlay를 계산 불변량으로
제공하지 않는다. WPU의 주장은 representational universality가 아니라 operational
primitive에 관한 주장이다.

이때 필요한 변환을 이 문서에서는 **객체화(objectification)** 라고 부른다.
객체화란 관측, simulator export, database state, tracker output을 persistent하고
addressable한 객체로 바꾸고, 그 객체에 typed attribute, typed relation,
uncertainty, admissible delta, branch-local overlay를 부여하는 것이다. 객체화는
perception이 해결되었다는 주장이 아니라, WPU core가 소비하는 state substrate의
계약이다. Reference implementation은 이 계약을 단순 구호가 아니라 측정 가능한
pre-propagation report로 다룬다. Identity coverage, relation endpoint validity,
confidence, delta validity, causal-working-set locality를 보고할 수 있어야 한다.
그래야 WPU 실패가 객체화 실패인지 propagation model 실패인지 분리할 수 있다.
현재 scheduler는 이 contract를 safety signal로도 사용한다. 객체화 품질이 낮으면
blind sparse propagation이 아니라 hybrid 또는 dense recomputation으로 올린다.
정식 정의는 `docs/objectification.ko.md`에 둔다.

## 2. 선행연구와 차별성

리뷰에서 지적된 것처럼 WPU는 object-centric learning, GNN, learned physics
simulator, latent world model, sparse Transformer world model과 강하게 맞닿아
있다. 따라서 WPU를 “완전히 새로운 message passing 수식”으로 주장하면 방어하기
어렵다.

WPU의 차별성은 다음 요소를 하나의 execution abstraction으로 묶는 데 있다.

```text
persistent state memory
event frontier
sparse / hybrid / dense routing
delta overlay
branch sharing
accuracy-compute-memory regime surface
```

Slot Attention이나 IODINE은 perception-to-object-state front-end가 될 수 있다.
Graph Network-based Simulator는 propagation core가 될 수 있다. Set/Graph Transformer는
dense fallback 또는 baseline이 될 수 있다. WPU가 묻는 질문은 “어떤 neural block이
가장 좋은가”가 아니라 “explicit world state를 계속 유지하고 갱신하는 workload에
어떤 실행 기질이 필요한가”이다.

## 3. Token과 State

Token model은 보통 다음 입력을 처리한다.

```text
X = (x_1, x_2, ..., x_T)
```

Transformer attention의 질문은 대략 다음이다.

```text
이 position은 어떤 다른 position을 참조해야 하는가?
```

World state는 다음처럼 persistent structured object다.

```text
S_t = {O_t, R_t, T_t, P_t}
```

여기서 `O_t`는 객체 집합, `R_t`는 typed relation graph, `T_t`는 temporal memory,
`P_t`는 uncertainty 또는 belief state다. 사건이 발생하면 state는 append되는 것이
아니라 patch된다.

```text
S_{t+1} = S_t + Delta S_t
Branch = BaseState + DeltaState
```

| 속성 | Token sequence | World state |
|---|---|---|
| 기본 단위 | position embedding | object / relation / belief |
| identity | context 안에 암묵적 | object id로 명시적 |
| memory update | append 또는 rewrite | existing state patch |
| relation | attention으로 암묵 추론 | typed edge로 저장 |
| future | continuation | branch overlay |
| target | next token | next state / delta / branch |

따라서 token과 state의 차이는 file format의 차이가 아니라 계산 primitive의 차이다.

## 4. WPU가 필요한 이유

GPU, TPU, NPU는 dense tensor computation에 강하다. LPU 계열 시스템은 deterministic
token stream inference에 초점을 둔다. WPU가 대상으로 삼는 workload는 다르다.

```text
World-state maintenance and update
```

이 workload에는 네 가지 성질이 반복된다.

- 대부분의 state는 시간에 따라 유지된다.
- 대부분의 event는 전체 state가 아니라 작은 subset만 바꾼다.
- 변화는 relation neighborhood를 따라 전파된다.
- 미래 불확실성은 full copy가 아니라 공유된 base state 위의 local branch로 나타난다.

WPU는 GPU/NPU/TPU/LPU를 폐기하자는 주장이 아니다. 현재 v1은 PyTorch 위의 reference
model이다. 장기적으로는 object store, relation fetch, frontier queue, delta log,
branch overlay, sparse-dense scheduler를 시스템 또는 하드웨어 primitive로 올릴 수
있는지 묻는다. 다만 hardware나 chiplet/IP는 아직 제품 주장이 아니라 미래 가설이다.
먼저 software runtime에서 frontier queue, relation fetch, scatter/gather, sparse
kernel, delta log, branch overlay 비용을 실제로 계측해야 한다.

## 5. Sparse-Dense Regime

세계에 `N`개의 object가 있고, 한 사건이 처음 바꾸는 object 수가 `DeltaN`, 평균
fanout이 `k`, propagation depth가 `h`, branch pressure가 `B`라면 affected-state
ratio는 다음과 같다.

```text
rho = (DeltaN * k^h * B) / N
```

v1 hard scheduler는 다음 정책을 사용한다.

```text
rho < 0.05  -> sparse
rho < 0.25  -> hybrid
otherwise   -> dense
```

이는 최적 정책이 아니라 engineering default다. 중요한 점은 “항상 sparse”가
아니라, affected fraction이 작을 때 sparse를 쓰고, 국소성이 깨질 때 hybrid/dense로
전환한다는 점이다.

v2 scheduler는 단순 `rho`가 아니라 prediction risk와 update cost를 함께 최적화해야
한다. uncertainty, relation quality, fanout, branch divergence가 크다면 `rho`가 작아도
regional dense correction을 선택할 수 있어야 한다.

## 6. WPU 구조

`WorldStateProcessor`는 다음 입력을 받는다.

- object features
- relation edge indices
- relation features
- event/action features
- object/relation masks
- target indices
- time features
- scheduler metrics

모델 출력은 다음이다.

- object delta
- relation logits
- uncertainty update
- branch logits/probabilities
- selected execution path

구현된 실행 경로는 세 가지다.

- Sparse path: event frontier에서 시작하는 relation-conditioned message passing.
- Hybrid path: sparse propagation과 regional dense correction의 혼합.
- Dense path: 전체 object set에 대한 global attention 기반 recomputation.

## 7. Attention이 아니라 Propagation

Attention은 WPU에서도 유용하다. Dense fallback이나 global consistency check에서는
attention 또는 Set/Graph Transformer 계열 연산을 사용할 수 있다. 그러나 WPU의
정의적 연산은 attention이 아니라 propagation이다.

```text
Token attention:
  which token should this token attend to?

State propagation:
  which consequence should this state delta cause?
```

Propagation은 단순화된 local physics prior로 해석할 수 있다. 실제 물리 법칙을
정확히 풀겠다는 뜻이 아니라, 많은 물리적 변화가 접촉, 지지, 근접, 제약, 신호
전달 같은 국소 인과 관계를 따라 시작된다는 계산적 근사다. 뉴턴역학이 낮은
속도/작은 스케일에서 유용한 근사인 것처럼, WPU propagation은 world processing에서
유용한 낮은 차수의 인과 근사로 볼 수 있다. Dense fallback은 이 국소 근사가 깨질
때 수행하는 global correction이다.

핵심 문장은 다음이다.

```text
For world processing, propagation is to state what attention is to tokens.
```

## 8. 소규모 검증

첫 검증 task는 synthetic robot-cup object physics다.

Scene:

- cup
- table
- robot hand
- table edge
- configurable background objects

Event:

```text
hand_touched_cup(target=cup, force=f)
```

Target:

- next object delta
- branch label: `stable`, `falls`, `caught`

이 task는 일반 물리 추론 benchmark가 아니다. WPU 가설의 unit test다. 하나의
neural model 안에서 explicit state graph, local event, sparse-dense route decision,
branch probability가 함께 학습되고 관찰 가능한지를 확인한다.

Primary validation:

| Model | Next-state MSE | Branch NLL | Branch Accuracy |
|---|---:|---:|---:|
| Untrained WSP | 0.8111 | 1.2070 | 0.1289 |
| Majority baseline | n/a | n/a | 0.6680 |
| Trained WSP | 0.0005 | 0.8074 | 0.7188 |

## 9. 강화 실험 요약

리뷰 비판을 반영해 5개 seed, 강한 baseline, dense N sweep, B sweep, step sweep,
controlled stress, CPU forward latency 측정을 수행했다. 상세 수치는
`docs/experiments/`에 있고, 논문 본문에는 핵심 evidence만 남겼다.

Robust comparison:

| N | Best WPU | Accuracy | Best non-WPU | Accuracy | 해석 |
|---:|---|---:|---|---:|---|
| 4 | WPU-hybrid | 0.7242 +/- 0.0260 | Dense graph | 0.6398 +/- 0.1257 | WPU 우세, routed scheduler는 dense 선택 때문에 실패 가능. |
| 24 | WPU-hybrid | 0.7320 +/- 0.0280 | Graph Transformer | 0.6609 +/- 0.0680 | medium local regime에서 WPU 우세. |
| 84 | WPU-hybrid | 0.7508 +/- 0.0244 | Graph Transformer | 0.6953 +/- 0.0388 | synthetic regime에서 WPU가 가장 강함. |
| 204 | WPU-sparse/routed | 0.4516 +/- 0.1957 | Graph Transformer | 0.7172 +/- 0.0615 | WPU 실패, token/graph baseline이 accuracy 우세. |

Dense N sweep:

- `N = 4, 8, 12, 16, 24, 36, 52, 68, 84, 108, 132, 164, 204, 260`
- route: dense to hybrid at measured `N=16`
- route: hybrid to sparse at measured `N=68`
- accuracy crossover: WPU-family advantage disappears around `N≈120`
- runtime crossover versus serialized-token: around `N≈124`
- runtime crossover versus dense-graph: around `N≈178`

따라서 v1의 핵심 tension은 다음이다.

```text
WPU efficiency advantage appears at large N.
WPU accuracy advantage currently appears at medium N.
The unsolved problem is to make these regimes overlap.
```

Controlled stress:

- Irrelevant relation noise에서는 WPU-hybrid가 가장 견고하다. Noise edge 0에서
  128까지 branch accuracy drop은 `0.0250`이고, Graph Transformer는 `0.3438` 떨어진다.
- Affected-background delta에서는 serialized-token이 가장 큰 affected count에서
  가장 낮은 background MSE를 보인다. 즉 WPU v1은 모든 state-delta regime에서
  우월하지 않다.

## 10. V2: Risk-Adjusted Working-Set Control

v1의 실패 경계는 명확했다. Sparse routed work는 줄일 수 있지만, large `N`에서
accuracy가 충분히 유지되지 않았다. 따라서 v2의 핵심 질문은 propagation block을
무작정 키우는 것이 아니라 “어떤 object set을 propagation 대상으로 선택하고, 어떤
retrieval mechanism을 배포할 것인가”다.

기존 learned retriever는 hand-built interaction selector를 모방했다. 이는
state-native objective이지만, 실제 downstream branch loss를 직접 최적화하지 않는다.
첫 번째 개선은 validation split에서 다음 candidate들을 평가하는 regret-distilled
retrieval이었다.

- `indexed`
- `proximity`
- `interaction`
- `learned`
- `generated_0..generated_3`

각 sample마다 downstream branch cross-entropy가 가장 낮은 candidate set을
pseudo-label object set으로 삼고, 작은 state-native object scorer가 그 set을
선택하도록 학습한다.

`N=2048`, 5 seeds 평균:

| K | Static learned interaction loss | Regret-distilled loss | Accuracy before | Accuracy after |
|---:|---:|---:|---:|---:|
| 8 | 0.988727 | 0.977017 | 0.508889 | 0.542222 |
| 16 | 0.966098 | 0.955077 | 0.504444 | 0.513333 |
| 32 | 1.004100 | 0.999112 | 0.480000 | 0.513333 |

Regret-distilled retriever는 learned interaction retriever 대비 15개 seed/K 조건 중
14개에서 loss를 낮췄다. 그러나 same-seed validation-to-test 결과만으로는 충분하지
않다. 이후 실험은 cross-seed 조건에서 candidate set 자체가 아니라 mechanism을
선택하는 문제로 확장했다.

최신 실험은 candidate별 role/geometry/family descriptor를 사용한다. 여기에는 hand
포함 여부, obstacle 비율, pair density, event target 기준 obstacle 거리 통계, lateral
spread, axis alignment, hand/edge distance, candidate family flag가 포함된다. 이후
static learned selector, composition-aware selector, invariant set scorer 중 하나를
train seed evidence로 고른다. Strict no-harm seed-stable gate는 K=32에서 너무
보수적이었고, risk-adjusted selector가 가장 좋은 균형을 냈다.

`N=2048`, 5 held-out seeds 평균:

| K | Static learned loss | Risk-adjusted mechanism loss | Accuracy before | Accuracy after |
|---:|---:|---:|---:|---:|
| 8 | 0.988432 | 0.982002 | 0.506667 | 0.522222 |
| 16 | 0.966183 | 0.951243 | 0.504444 | 0.517778 |
| 32 | 1.004095 | 1.002597 | 0.475556 | 0.522222 |

이 결과의 의미는 중요하다. Explicit state는 sparse propagation을 가능하게 할 뿐
아니라, propagation 이전의 object-level working-set selection과 mechanism routing을
학습 가능한 control problem으로 노출한다. Token baseline은 같은 scene을 serialize할
수 있지만, 이 object-level intervention point를 독립적인 제어면으로 자연스럽게
제공하지 않는다.

단, 이 결과도 한계가 있다. Generated/candidate oracle과의 gap은 여전히 크다. Opaque
set evaluator, score-margin confidence gate, strict no-harm seed-stable gate는 충분하지
않았다. 따라서 v2의 다음 핵심 문제는 더 많은 candidate 생성이 아니라 invariant
candidate descriptor, risk-aware mechanism selection, retriever-propagator joint
training이다.

## 11. 현재 주장 경계

현재 지지되는 주장:

- explicit world-state processing은 학습 가능한 architecture로 구현 가능하다.
- hard scheduler는 sparse/hybrid/dense crossover를 만든다.
- WPU-family는 small-to-medium local synthetic regime에서 경쟁력이 있다.
- WPU-hybrid는 irrelevant relation noise에 강하다.
- routed sparse execution은 large `N`에서 CPU latency를 줄일 수 있다.
- regret-distilled state retrieval은 same-seed validation-to-test 조건에서
  interaction-teacher retrieval보다 downstream loss를 낮춘다.
- risk-adjusted state-native mechanism selection은 현재 `N=2048`, `K=8,16,32`
  held-out seed 조건에서 static learned selector보다 평균 loss를 낮춘다.

현재 지지되지 않는 주장:

- WPU가 token/graph baseline보다 항상 우월하다.
- WPU가 실제 물리 세계를 이해한다.
- perception에서 state를 end-to-end로 구성한다.
- GPU/NPU/TPU/LPU보다 항상 빠르다.
- fixed `rho` threshold가 최종 scheduler다.
- cross-seed candidate oracle gap은 아직 닫히지 않았다. Risk-adjusted mechanism
  selection은 positive result지만 최종 candidate scorer는 아니다.

`N=204`에서의 accuracy collapse는 숨기면 안 되는 결과다. 이 실패는 WPU 개념 자체의
반증은 아니지만, v1 propagation capacity와 hard scheduler가 large graph에서 충분한
predictive state를 유지하지 못한다는 강한 증거다.

## 12. 보충 자료와 반증 기준

논문 본문은 주장에 직접 필요한 figure와 table만 남겼다. 촘촘한 sweep, stress
figure, 세부 table은 영문 PDF의 supplementary materials와 `docs/experiments/`로
이동했다. 논문 안에 roadmap-style section을 길게 두지 않는 대신,
`docs/arxiv/README.md`가 후속 검증 계획을 관리한다. 본문에서 필요한 것은 계획의
나열이 아니라 현재 주장이 어떤 조건에서 참/거짓이 되는지 명확히 하는 것이다.

현재 WPU 주장은 다음 조건을 만족할 때 강화된다.

```text
accuracy crossover >= runtime crossover
small identifiable K inside large N
stable branch/delta rollout
matched or acceptable accuracy at lower routed work
```

반대로 다음 결과가 반복되면 WPU v2의 핵심 주장은 약해진다.

- `K`가 작고 식별 가능한데도 token/graph baseline이 동일 work에서 계속 우세하다.
- pre-tensor retrieval 비용이 실제 구현에서 `O(N)`에 가깝게 증가한다.
- risk-adjusted mechanism selection의 cross-seed gain이 더 큰 seed/model sweep에서
  사라진다.
- long-horizon rollout에서 delta overlay가 누적 오차와 state corruption을 제어하지
  못한다.
- sparse advantage가 실제 sparse kernel, memory traffic, branch overlay 비용을
  포함하면 사라진다.

따라서 제출용 논문의 태도는 “WPU가 항상 우월하다”가 아니라 “명시적 state와
propagation이 유리해지는 regime을 예측하고, 그 regime 밖에서는 실패할 수 있음을
실험적으로 드러낸다”가 되어야 한다.

## 13. 적용 가능성의 경계

상업적 방향은 chiplet/IP나 robot OS core보다 software runtime 또는 middleware로 낮춰
잡는 것이 맞다. 가까운 적용 후보는 digital twin state update, simulator backend,
game/server synchronization, robotics world-model maintenance처럼 state는 크지만
event가 바꾸는 영역은 국소적인 시스템이다.

hardware claim은 matched accuracy에서의 speedup, sparse-kernel overhead, memory
traffic, branch-overlay memory가 검증된 뒤에야 강하게 말할 수 있다.

## 14. 결론

World processing은 세계를 token으로 설명하는 문제만이 아니다. 세계는 유지되고,
수정되고, 여러 미래로 분기되는 state다. WPU는 state를 기본 계산 객체로 두고,
propagation을 중심 연산으로 둔다. Token과 attention은 여전히 중요하지만,
state-native world model의 정의적 primitive는 token attention이 아니라 state
propagation이다. 최신 v2 결과는 여기에 한 가지를 더한다. State를 명시적으로 두면
propagation 이전의 causal working set selection과 mechanism routing도 학습 가능한
object-level control 문제가 된다.

Nature/Science급 방향으로 가려면 필요한 태도는 보편 우월 주장이 아니다. 새로운
계산 원리, 즉 state-native propagation이 작동하는 regime을 명확히 제시하고,
그 regime 밖에서는 실패할 수 있음을 열어두며, 이를 반증 가능한 실험으로 만드는
것이다.

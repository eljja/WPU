# WPU 연구 Thesis

이 문서는 WPU의 학문적 기여와 모든 공개 문서가 유지해야 할 claim boundary를
요약한다.

## 한 문장 Thesis

WPU는 world processing을 **state-native computation**으로 연구해야 한다고 제안한다.
즉 persistent object와 typed relation을 token sequence로 반복 직렬화하거나 dense tensor로
매번 전체 재계산하는 대신, state 자체를 patch, propagate, branch, revise하는 계산
단위로 삼는다.

## 무엇이 새로운가

독창성은 하나의 message-passing layer에 있지 않다. WPU의 연구 기여는 다음 다섯 가지를
하나의 반증 가능한 execution model로 묶은 데 있다.

- **객체화를 contract로 정의**: raw observation, simulator state, log는 persistent하고
  addressable한 object, attribute, relation, uncertainty, admissible delta, branch
  overlay로 변환되어야 WPU의 state-native advantage를 주장할 수 있다.
- **Event-frontier execution**: 계산은 바뀐 entity에서 시작해 typed causal relation을
  따라 전파된다. 기본값은 full sequence attention이나 full graph recompute가 아니다.
- **Delta-state memory**: 미래 state는 `BaseState + DeltaState`로 표현해 branch sharing과
  copy-on-write update를 가능하게 한다.
- **Sparse-first, dense-when-needed routing**: sparse, hybrid, dense execution을 고정
  이념이 아니라 측정 가능한 regime surface로 다룬다.
- **Claim-boundary instrumentation**: 모든 장점 주장을 `N`, causal working set `K`,
  branch count, objectification quality, state integrity, calibration, latency, memory
  traffic에 연결한다.

## 핵심 과학적 주장

방어 가능한 주장은 조건부다.

```text
WPU는 전체 world state N은 크지만 causal working set K가 작고 tensorization 전에
식별 가능하며, update가 local/relation-mediated이고 branch/uncertainty state가 event
사이에서 재사용되는 경우에 유용하다.
```

이는 Transformer, Graph Transformer, GPU, TPU, NPU, LPU, dense world model보다 항상
우월하다는 주장이 아니다. `K`가 `N`과 함께 커지거나, 객체화가 causal state를 놓치거나,
retrieval이 world 전체를 scan하거나, long-horizon delta가 drift하면 WPU의 장점은 사라질
수 있다.

## 왜 중요한가

이 프로젝트의 연구적 가설은 명확하다. 일부 world-model workload의 다음 병목은 parameter
count나 matrix throughput만이 아니라, persistent world state와 local causal change 사이의
execution interface일 수 있다. Token/dense tensor system도 state를 표현할 수는 있지만,
object identity, relation traversal, branch overlay, partial state update를 native
operation으로 만들지는 않는다. WPU는 이 operation들을 native하게 만들면 별도의
accuracy/latency/memory regime이 생기는지 묻는다.

v3 목표의 정의는 `world_copy_model.ko.md`와 `versions/wpu_v3_plan.ko.md`에 고정한다.
두 문서는 추가 구현 전에 world copy, 객체지향적 processing, natural propagation,
benchmark requirement, success/failure criteria를 정의한다.

## 현재 증거

현재 저장소는 early prototype 수준의 의미 있는 주장을 지지한다.

- state model, objectification report, delta overlay, sparse/dense scheduler, branch
  rollout, PyTorch module이 구현되고 테스트된다.
- v1 synthetic experiment는 첫 accuracy-runtime tension을 보여준다. WPU는 일부 local
  regime에서 이기지만, accuracy가 runtime crossover 전에 무너지는 실패 구간도 있다.
- v2 candidate-control experiment는 explicit state가 propagation 이전 working-set
  control surface를 드러냄을 보인다. `N=2048`, `K=8`에서는 강한 positive sub-regime이
  있지만 larger-`K` gap은 남아 있다.
- Targeted large-N readout probe는 v1 `N>=204` branch collapse의 일부 원인이
  non-causal object까지 포함한 global mean readout임을 보인다. Event target/frontier
  readout으로 바꾸면 `wpu-sparse-frontier`는 total `N=404`까지 mean accuracy
  `0.781250`을 유지하고, serialized-token `0.778646`을 근소하게 넘으며 work proxy는
  `3` 대 `166464`다.
- v3 world-copy index substrate는 hierarchical region과 multi-signal causal index를
  추가한다. Controlled probe에서 total `N`이 `104`에서 `10004`까지 커져도 selected
  `K`는 `4`로 유지되고, non-causal background object는 선택되지 않는다. `N=10004`의
  affected fraction은 `0.00039984`다. 이는 scalable causal-slice retrieval 증거이지
  trained world modeling 증거는 아니다.
- 첫 v3 noisy-index stress benchmark는 이 substrate evidence를 `N=8192`,
  `K_ref=4/8/16`, missing relation, false-positive relation으로 확장한다.
  Controlled setup에서 region-scoped retrieval은 recall `1.000000`을 유지하고,
  `N=8192` touch ratio를 `0.004385` 이하로 유지한다. Relation-confidence gate가 없으면
  false-positive relation은 mean precision을 `0.800000`까지 낮추지만,
  `min_relation_confidence=0.3`에서는 recall `1.000000`을 유지하면서 precision도
  `1.000000`으로 회복된다. True causal relation confidence를 `0.2`로 낮추면 region
  scope가 recall과 precision을 회복하지만 mean escalation signal은 `0.981481`로 켜진다.
  다음 boundary는 escalation 이후 local dense/hybrid correction이 propagation accuracy를
  개선하는지다.
- v3 escalation-correction probe는 이 boundary를 substrate 수준에서 테스트한다. True
  relation confidence가 `0.2`일 때 sparse confident-relation update는 mean recall
  `0.145833`, F1 `0.246623`으로 떨어지지만, local hybrid escalation-region candidate는
  max selected `K=16`을 유지하면서 mean recall/precision/F1을 `1.000000`으로 회복한다.
  이는 escalation 이후 bounded correction candidate를 검증하는 증거이지, learned
  transition quality나 baseline superiority 증거는 아니다.
- v3 learned-correction probe는 회복된 candidate set을 작은 learned local-delta head와
  연결한다. True relation confidence가 `0.2`일 때 sparse confident-relation update는
  mean delta MSE `0.275312`를 남기지만, hybrid escalation-region candidate는 이를
  `0.006365`까지 낮추고 max selected `K=16`을 유지한다. 이는 controlled P2 substrate
  positive이지, baseline-complete world-model result는 아니다.
- 첫 v3 baseline-comparison screen은 mixed result다. WPU는 max selected `K=16`과 mean
  work proxy `8.789551`을 유지해 dense/token full-state baseline보다 훨씬 낮은 비용을
  보이지만, dense graph와 serialized-token baseline의 raw delta MSE(`0.003778`,
  `0.004533`)가 WPU(`0.020818`)보다 낮다. 이는 systems-efficiency 방향을 지지하지만
  raw accuracy dominance는 아니다.
- PyBullet experiment는 simulator-derived object state에서 mechanism shift, calibration,
  objectification quality, long-horizon rollout diagnostic을 제공한다.
- Relation-conditioned sparse propagation은 현재 가장 강한 large-state evidence다.
  N=517, N=1029, N=2053 baseline-complete non-causal distractor scaling screen은
  positive이고, N=4101은 WPU-only sparse feasibility로만 해석해야 한다.
- Systems profile은 indexed sparse execution이 `K`를 작게 유지할 때 tensor-byte와 latency
  proxy를 크게 줄일 수 있음을 보인다. 하지만 실제 power, sparse-kernel,
  allocator-level evidence는 아직 없다.

## 현재 한계

이 프로젝트는 아직 완성된 process unit, chip design, broad world model이 아니다.
남은 과학적 문제는 다음과 같다.

- larger `K`에서 safe candidate generation;
- guarded state correction이 아닌 raw learned long-horizon dynamics;
- N=2053을 넘어서는 baseline-complete large-N comparison;
- `K`가 변하거나 커지는 harder causal large-N setting;
- shift에서 calibration-safe low-cost routing;
- raw sensor input에서 perception-to-state objectification;
- streaming update 아래의 hierarchical state persistence;
- 고정 local rule이 아니라 causal slice 위에서 학습되는 mechanism module;
- noisy 또는 missing relation 아래의 causal-index recall;
- sparse kernel, memory traffic, energy에 대한 실제 systems measurement.

## 공표 태도

올바른 공표 문장은 다음이다.

```text
WPU는 작동하는 PyTorch reference implementation, 강한 negative-result discipline,
특정 small-K world-state regime에서의 positive evidence를 가진 반증 가능한
state-native execution hypothesis다.
```

틀린 문장은 다음이다.

```text
WPU가 이미 state가 token보다 항상 낫다는 것, 또는 GPU/TPU/NPU/LPU를 대체할 새 hardware
unit이 준비됐다는 것을 증명했다.
```

이 framing이 WPU의 학문적 가치를 만든다. 새로운 computational primitive를 제안하고,
구현하고, 작동하는 구간과 실패하는 구간을 함께 지도화하며, 남은 gap을 구체적 실험으로
바꾸기 때문이다.

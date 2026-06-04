# WPU: World-State Processing Unit

[![CI](https://github.com/eljja/WPU/actions/workflows/ci.yml/badge.svg)](https://github.com/eljja/WPU/actions/workflows/ci.yml)

![WPU compute architecture overview](docs/figures/wpu_compute_architectures.png)

이 저장소는 **State Is All You Need**와 **World-State Processing Unit
(WPU)** 아이디어의 첫 PyTorch 연구 prototype이다.

WPU는 chatbot memory도 아니고, Transformer의 보편 대체재도 아니며, 아직 완성된
chip design도 아니다. 현재 목표는 world processing을 token sequence가 아니라
명시적 world state의 유지, 갱신, 전파, 분기 문제로 다루는 reference implementation과
실험 scaffold를 만드는 것이다.

중심 primitive는 **객체화(objectification)** 다. 즉 세계를 persistent하고 addressable한
객체로 변환하고, 그 객체의 state, relation, uncertainty, delta, branch overlay를 직접
갱신할 수 있게 만드는 것이다. 객체화는 단순히 type label을 붙이는 일이 아니다.
Role, affordance, geometry, confidence, history처럼 relation을 지탱하는 state
variable이 필요하다. 정의는 `docs/objectification.ko.md`에 둔다.

## 핵심 주장

Token sequence는 세계를 설명할 수 있지만, world-state operation을 first-class로
만들지는 않는다. WPU는 다음 요소를 기본 계산 단위로 둔다.

- persistent object/relation state
- event frontier generation
- local causal propagation
- sparse, hybrid, dense execution routing
- full-state rewrite가 아닌 delta-state patching
- multiple futures를 위한 branch overlay
- uncertainty와 branch probability update

핵심 구분은 다음이다.

```text
Token = ordered evidence for append / attend
State = persistent substrate for patch / propagate / branch
```

현재 주장은 보편 우월성이 아니다. WPU가 유리한 조건은 `N`이 크지만 실제 event가
참조하고 갱신해야 하는 causal working set `K`가 작고 식별 가능할 때다. 즉 WPU는
large state, local causal change, persistent identity, branching이 지배적인 regime을
목표로 한다.

## Compute Context

![WPU in the AI compute architecture landscape](docs/figures/wpu_compute_context.svg)

CPU/GPU/TPU/NPU/LPU는 각자 다른 workload에 최적화되어 있다. WPU는 dense matrix
compute나 deterministic token stream이 아니라 다음 workload를 정의한다.

```text
World-state maintenance and update
```

따라서 hardware claim은 아직 미래 가설이다. 현재는 software runtime 수준에서
frontier queue, relation fetch, scatter/gather, sparse kernel, delta log, branch
overlay 비용을 계측하며 regime을 찾는 단계다.

## Hybrid Execution Architecture

![WPU hybrid sparse-dense execution architecture](docs/figures/wpu_hybrid_architecture.svg)

v1 reference model은 event-driven sparse propagation과 dense tensor recompute
fallback을 함께 사용한다. Scheduler는 affected-state ratio, fanout, propagation
depth, branch pressure를 보고 sparse/hybrid/dense path를 선택한다. Scheduler는
objectification score도 사용한다. Identity/relation/delta 품질이 낮으면 blind
sparse routing을 피하고 hybrid 또는 dense recompute로 올린다.

```text
rho = (DeltaN * fanout^depth * branches) / N
```

## Repository Layout

```text
wpu/                 PyTorch package and state/model implementation
tests/               Unit and smoke tests
demos/               End-to-end dataflow demo
scripts/             Training, evaluation, sweeps, plotting
docs/arxiv/          English LaTeX paper, Korean companion, generated PDF
docs/experiments/    Experiment reports
docs/figures/        Paper figures and README diagrams
docs/Review/         External review notes and response matrix
```

## 설치

Python 3.11+ 권장.

```bash
python -m pip install -e ".[dev]"
```

기본 설치는 standard PyTorch package를 함께 설치한다. 특정 CUDA build가 필요하면
해당 PyTorch build를 먼저 설치한 뒤 editable install을 실행한다.

Windows에서는 `python`이 Microsoft Store alias로 잡히지 않는지 먼저 확인한다.
로컬 재현에는 다음처럼 venv interpreter를 명시하는 경로가 안전하다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

## 데모 실행

```bash
python demos/robot_cup_demo.py
```

출력 trace:

- event와 initial frontier
- scheduler decision: sparse, hybrid, dense
- changed objects와 relation updates
- stable/falls/caught branch probabilities
- base state, deltas, branches memory estimate

## 최소 Public API

설치 후 핵심 state-processing 흐름은 package root에서 바로 사용할 수 있다.

```python
import wpu
from wpu.data.object_physics import create_robot_cup_state, create_touch_event

state = create_robot_cup_state()
event = create_touch_event()

event_delta = wpu.StateStore(state).apply_event(event)
sparse_delta = wpu.SparsePropagationEngine(max_depth=1).sparse_propagate(state, event).delta
dense_delta = wpu.DenseRecomputeEngine().dense_recompute(state, region=["cup_001"]).delta
objectification = wpu.evaluate_objectification(state, delta=event_delta)

print(event_delta.object_updates["cup_001"])
print(sparse_delta.object_updates["cup_001"])
print(dense_delta.object_updates["cup_001"])
print(objectification.contract_score)
```

이것이 v1의 의도된 interface다. Explicit world state는 event delta로 patch되고,
local propagation을 거친 뒤, 필요하면 제한된 dense region에서 recompute된다.
`evaluate_objectification`은 propagation 전에 supplied state가 WPU object contract를
만족하는지 검사한다. 즉 stable identity, valid relation endpoint, confidence,
valid delta, optional causal-working-set locality를 확인한다.

Object identity는 있지만 local relation extraction이 edge를 놓친 경우,
`repair_objectification_relations`는 sparse propagation 전에 보수적인
geometry-inferred relation patch를 추가할 수 있다. 이것은 state repair heuristic이지
물리 법칙을 해결했다는 주장이 아니다. Repair probe는 typed objectification이 왜
중요한지 보여준다. Geometry-only repair는 frontier recall을 복구하지만 distractor를
붙일 수 있고, type-gated repair는 controlled case에서 precision을 유지한다. 작은
learned relation scorer는 type gate와 같은 결과를 내고, 더 조밀한 distractor에서도
precision을 유지하며, role/affordance state variable이 보존되면 aliased type name을
넘어 transfer한다. 반대로 type과 role 정보가 모두 제거되면 실패하므로, 객체화의
경계가 측정 가능해진다. 같은 toy probe는 downstream branch prediction도 측정한다.
Role-aware learned repair는 aliased-type accuracy를 `0.343750`에서 `0.671875`로
올리고 loss를 `1.319667`에서 `0.885275`로 낮춘다. Ungated dense-distractor repair는
frontier recall을 복구하지만 loss를 악화시킨다.

두 번째 toy probe는 장기 객체화 방향을 테스트한다. Type name이 아니라 object
history에서 relation candidate를 학습하는 방식이다. `contact_transfer`와
`support_transfer`로 학습한 history scorer는 held-out `hidden_field`로 transfer하여
relation precision/recall `0.988281`, downstream accuracy `0.996094`를 기록한다.
No-relation 또는 type prior는 `0.496094`에 머문다. 이것은 synthetic hidden-mechanism
diagnostic이지 실제 물리 법칙 발견 증거는 아니다.

현재 v2 working-set 모델도 package root의 model factory에서 생성할 수 있다.

```python
import wpu

model = wpu.create_model(
    "wpu-cws-indexed",
    hidden_dim=64,
    working_set_size=16,
)
```

## 주요 실험 요약

현재 evidence는 “WPU가 항상 이긴다”가 아니라 regime hypothesis를 지지한다.

- WPU-family는 synthetic local regime에서 `N≈108`까지 경쟁력이 있지만,
  advantage는 `N≈120` 근처에서 사라진다.
- Routed WPU는 CPU v1 sweep에서 serialized-token 대비 `N≈124`, dense-graph 대비
  `N≈178` 근처부터 runtime advantage가 나타난다.
- WPU-hybrid는 irrelevant relation noise에 강하다. Noise edge 0에서 128까지
  accuracy drop은 `0.0250`이고, Graph Transformer는 `0.3438` 떨어진다.
- 하지만 `N=204` 같은 large-N regime에서는 현재 WPU accuracy가 무너지고
  graph/token baseline이 더 강하다.
- 객체화는 이제 public API에서 측정 가능한 contract가 되었지만, 실제
  perception-to-state adapter의 객체화 품질 benchmark는 아직 필요하다.

v1의 핵심 목표는 명확하다.

```text
Push the accuracy crossover beyond the runtime crossover.
```

## WPU v2: State-Native Working-Set Control

최근 v2에서 가장 강한 개선은 propagation block을 키운 것이 아니라 propagation
이전의 state-native control loop를 만든 것이다. 후보 causal working set을 생성하고,
명시적 role/geometry/family descriptor로 설명한 뒤, train seed evidence를
risk-adjusted 방식으로 평가해 사용할 retrieval mechanism을 고른다.

이전 v2 실험에서는 downstream branch loss를 최소화한 candidate set에서 학습한
regret-distilled retriever가 learned interaction retriever 대비 15개 seed/K 조건 중
14개에서 loss를 낮췄다. 최신 결과는 더 엄격하다. `N=2048`에서 held-out seed에 대해
mechanism selection 자체를 검증한다.

`N=2048`, 5 held-out seeds 평균:

| K | Static learned loss | Risk-adjusted mechanism loss | Accuracy gain |
|---:|---:|---:|---:|
| 8 | 0.988432 | 0.982002 | 0.506667 -> 0.522222 |
| 16 | 0.966183 | 0.951243 | 0.504444 -> 0.517778 |
| 32 | 1.004095 | 1.002597 | 0.475556 -> 0.522222 |

이는 WPU의 중요한 주장을 강화한다. Explicit state는 sparse propagation뿐 아니라,
propagation 이전의 object-level working-set control을 학습 가능하고 검증 가능한
문제로 노출한다. Token baseline은 scene을 serialize할 수 있지만, 이 object-level
intervention point를 자연스럽게 제공하지 않는다.

남은 병목은 generated/candidate oracle과의 gap이다. Opaque set evaluator,
score-margin gate, strict no-harm seed-stable gate만으로는 충분하지 않았다. 다음 v2
목표는 invariant candidate descriptor, risk-adjusted mechanism routing, 그리고
retriever-propagator joint training이다.

## 논문 및 문서

- English LaTeX: `docs/arxiv/state_is_all_you_need_en.tex`
- English PDF: `docs/arxiv/state_is_all_you_need_en.pdf`
- Korean companion: `docs/arxiv/state_is_all_you_need_ko.md`
- Compact research brief: `docs/paper/state_is_all_you_need.md`
- Claim ledger: `docs/claims.ko.md`
- Objectification definition: `docs/objectification.ko.md`
- Publication readiness / gap register: `docs/publication_readiness.ko.md`
- Reproducibility guide: `docs/reproducibility.ko.md`
- Experiment index: `docs/experiments/README.md`
- Review response: `docs/Review/review_response_and_differentiation.md`

PDF build:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs/arxiv docs/arxiv/state_is_all_you_need_en.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs/arxiv docs/arxiv/state_is_all_you_need_en.tex
```

## 테스트

```bash
python -m pytest
```

## 라이선스

이 프로젝트는 **GNU Affero General Public License v3.0 only
(AGPL-3.0-only)**로 배포된다. 네트워크 서비스 형태로 수정본을 제공하는 경우에도
동일한 license 조건으로 source code를 제공해야 한다.

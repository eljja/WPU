# WPU 주장 관리표

이 문서는 WPU 논문 수준의 주장을 현재 증거, 한계, 반증 조건과 연결한다. 목적은
과장 주장을 막고, 심사 대응 시 어떤 주장이 어떤 실험으로 뒷받침되는지 명확히
보여주는 것이다.
외부 공표 준비도와 미해결 gap은 `docs/publication_readiness.ko.md`를 함께 본다.
객체화의 정식 정의는 `docs/objectification.ko.md`를 본다.

## 주장 상태

| ID | 주장 | 현재 상태 | 주요 증거 | 경계 |
|---|---|---|---|---|
| C1 | Token, state, objectified state는 서로 다른 operational primitive다. | framing claim으로 지지됨. | `docs/objectification.ko.md`, `docs/arxiv/state_is_all_you_need_en.tex`, `docs/arxiv/state_is_all_you_need_ko.md`, `wpu/core/state.py`, `wpu/memory/state_store.py`. | token이 state를 encode할 수 없다는 뜻이 아니다. Persistent identity, role/affordance state, relation traversal, delta patching, branch overlay가 객체화 이후 native operation이 된다는 주장이다. |
| C2 | Explicit objectified world-state processing은 학습 가능한 neural model로 구현 가능하다. | synthetic object-physics prototype에서 지지됨. | `docs/objectification.ko.md`, `wpu/core/objectification.py`, `wpu/models/world_state_processor.py`, `wpu/models/causal_working_set_processor.py`, `wpu/data/object_physics.py`, `demos/robot_cup_demo.py`, 통과 중인 test suite. | 일반 물리 이해, end-to-end object construction, perception-to-state 구성을 증명하지 않는다. |
| C3 | Sparse/hybrid/dense routing은 단순 그림이 아니라 측정 가능한 execution regime이다. | v1 routing instrumentation에서 지지됨. | `wpu/engines/scheduler.py`, `docs/experiments/b_sweep_v1_results.md`, `docs/experiments/n_sweep_v1_results.md`, `docs/experiments/baseline_and_regime_results.md`. | fixed `rho` threshold는 최적 scheduler가 아니라 engineering default다. |
| C4 | v1 WPU에는 실제 accuracy-runtime tension이 있다. | 지지됨. | `docs/experiments/robust_v1_results.md`, `docs/experiments/n_sweep_v1_results.md`, `docs/versions/wpu_v1_closure.md`. | v1에서는 accuracy advantage가 runtime advantage보다 먼저 사라진다. 이는 win이 아니라 failure boundary다. |
| C5 | WPU-hybrid는 synthetic task의 irrelevant relation noise에 강하다. | 해당 stress regime에서 지지됨. | `docs/experiments/controlled_stress_v1_results.md`. | 모든 state-delta/affected-background regime에서 우월하다는 뜻은 아니다. |
| C6 | large `N`은 causal working set `K`가 작고 tensorization 전에 식별될 때만 WPU에 유리하다. | 조건부 systems hypothesis로 초기 증거가 있음. | `docs/experiments/cws_balanced_branch_8m_gpu_event_conditioned_final_results.md`, `docs/experiments/wpu_v2_pre_tensor_indexed_n_sweep_results.md`, `docs/versions/wpu_v1_closure.md`. | `N`이 큰 것만으로는 충분하지 않다. retrieval이 `O(N)`에 가깝거나 causal state를 놓치면 실패한다. |
| C7 | explicit state는 propagation 이전의 working-set control을 노출한다. | 현재 가장 강한 v2 mechanism claim으로 지지됨. | `docs/experiments/wpu_v2_retriever_regret_distillation_results.md`, `docs/experiments/wpu_v2_invariant_set_scorer_results.md`. | candidate oracle은 여전히 더 강하다. cross-seed robust scoring은 아직 해결되지 않았다. |
| C8 | WPU는 아직 hardware/chiplet/IP 결과가 아니다. | 명시적으로 지지되지 않음. | `docs/Review/review_response_and_differentiation.md`, `docs/paper/state_is_all_you_need.md`, `README.md`. | hardware claim에는 sparse kernel, memory traffic, branch overlay, matched-accuracy speedup 증거가 필요하다. |
| C9 | 객체화된 relation 위의 WPU propagation은 full physics가 아니라 단순화된 local-causality prior다. | 제한된 analogy로 지지됨. | `docs/objectification.ko.md`, `docs/arxiv/state_is_all_you_need_en.tex`, `docs/arxiv/state_is_all_you_need_ko.md`, `docs/paper/state_is_all_you_need.md`, `docs/experiments/object_history_hidden_mechanism_probe_results.md`, `docs/experiments/object_relation_law_probe_results.md`, `docs/experiments/object_relation_law_ood_probe_results.md`. | hidden-mechanism 및 local-law probe는 synthetic이다. 이들은 history-derived relation variable과 단순 inverse-distance law가 generated mechanism 아래에서 nominal type name을 넘어 transfer될 수 있고, OOD stress가 relation failure와 law mis-specification을 분리할 수 있음을 보일 뿐이다. 실제 물리 competence나 unknown-theory discovery는 simulator/robotics benchmark, long-horizon stability, learned relation이 training generator 밖에서도 일반화된다는 증거가 필요하다. |
| C10 | 단기 WPU 가치는 silicon보다 software runtime/middleware에서 더 가능성이 있다. | plausible direction이며 아직 실험적으로 증명되지 않음. | `docs/reproducibility.md`, `docs/arxiv/README.md`, current PyTorch package under `wpu/`. | digital-twin, simulation backend, game/server, robotics middleware benchmark가 필요하다. |
| C11 | 객체화 품질은 propagation 전에 contract로 측정 가능하고 국소적으로 repair 가능하다. | 구현 주장으로 지지됨. | `wpu/core/objectification.py`, `tests/test_objectification.py`, `tests/test_script_entrypoints.py`, `docs/experiments/objectification_relation_repair_probe_results.md`, `README.ko.md`, `docs/objectification.ko.md`. | Relation repair는 보수적 hypothesis를 만들 뿐 ground-truth physics가 아니다. 최신 probe는 role/affordance state가 보존될 때 learned repair가 aliased type name을 넘어 transfer하고 toy downstream branch diagnostic을 개선할 수 있음을 보인다. perception-to-object construction이나 unknown-theory discovery가 해결됐다는 증거는 아니다. |

## 반증 조건

- controlled identity/locality/branching benchmark에서 serialized-token 또는 graph
  baseline이 동일 compute로 WPU와 같거나 더 좋으면, 그 regime에서 WPU는 필수
  primitive가 아니다.
- realistic state store에서 pre-tensor retrieval 비용이 `O(N)`에 가깝게 증가하면,
  large-`N` systems claim은 무너진다.
- 더 큰 seed/model sweep에서 risk-adjusted mechanism selection의 held-out gain이
  사라지면, 현재 v2 working-set-control 주장은 diagnostic result로 낮춰야 한다.
- sparse routing이 meaningful latency/memory saving을 만들기 전에 accuracy를 해치면,
  해당 workload에서 propagation은 중심 연산으로 부적절하다.
- geometry-derived relation repair가 spurious edge를 추가해 downstream prediction이나
  working-set precision을 낮추면, repair는 gate되거나 learned candidate scoring으로
  대체되어야 한다.
- learned relation scoring이 role/affordance variable이 보존된 aliasing 조건에서도
  nominal type label에만 의존해 실패한다면, 객체화는 brittle type classification을
  벗어나지 못한 것이다.
- long-horizon rollout에서 delta overlay가 state corruption을 누적하고 회복하지
  못하면, verification/rollback 없이 persistent state는 장점이 아니라 위험이 된다.

## 제출용 태도

방어 가능한 논문 태도는 다음이다.

```text
WPU는 regime-specific state-native execution model이다.
유리한 조건은 large explicit state, small identifiable causal working set,
local relation propagation, branchable uncertainty, event 간 state reuse다.
```

논문이 주장하면 안 되는 것:

- token, graph, latent world model에 대한 보편 우월성;
- 실제 물리 세계 이해;
- end-to-end perception-to-state 구성 완료;
- hardware-level speed/energy advantage;
- candidate-oracle gap 해결.

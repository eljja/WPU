# WPU 주장 관리표

이 문서는 WPU 논문 수준의 주장을 현재 증거, 한계, 반증 조건과 연결한다. 목적은
과장 주장을 막고, 심사 대응 시 어떤 주장이 어떤 실험으로 뒷받침되는지 명확히
보여주는 것이다.
외부 공표 준비도와 미해결 gap은 `docs/publication_readiness.ko.md`를 함께 본다.
현재 v2 우선순위의 pass/partial/fail dashboard는
`docs/experiments/wpu_v2_priority_dashboard.ko.md`를 함께 본다.
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
| C7 | explicit state는 propagation 이전의 working-set control을 노출한다. | 현재 가장 강한 v2 mechanism claim으로 지지되지만 dashboard 기준으로는 아직 fail gap이 남아 있다. | `docs/experiments/wpu_v2_retriever_regret_distillation_results.md`, `docs/experiments/wpu_v2_invariant_set_scorer_results.md`, `docs/experiments/wpu_v2_candidate_oracle_gap_decomposition.ko.md`, `docs/experiments/wpu_v2_candidate_noharm_gate_results.ko.md`, `docs/experiments/wpu_v2_candidate_regret_gate_results.ko.md`, `docs/experiments/wpu_v2_candidate_regret_gate_penalty_results.ko.md`, `docs/experiments/wpu_v2_candidate_regret_gate_perturbed_results.ko.md`, `docs/experiments/wpu_v2_candidate_regret_crossfit_results.ko.md`, `docs/experiments/wpu_v2_candidate_safety_gate_results.ko.md`, `docs/experiments/wpu_v2_candidate_invariant_gate_results.ko.md`, `docs/experiments/wpu_v2_candidate_joint_gate_results.ko.md`, `docs/experiments/wpu_v2_candidate_joint_gate_regression_heavy_k16_results.ko.md`, `docs/experiments/wpu_v2_candidate_safety_frontier_results.ko.md`, `docs/experiments/pybullet_branch_prior_shift_results.ko.md`, `docs/experiments/pybullet_mechanism_prior_adaptation_results.ko.md`, `docs/experiments/pybullet_prior_strength_sweep_results.ko.md`, `docs/experiments/pybullet_selected_prior_adaptation_results.ko.md`, `docs/experiments/pybullet_fewshot_mechanism_adaptation_results.ko.md`, `docs/experiments/pybullet_mechanism_adaptive_policy_results.ko.md`, `docs/experiments/pybullet_uncertainty_gated_recompute_results.ko.md`, `docs/experiments/pybullet_learned_uncertainty_gate_results.ko.md`, `docs/experiments/pybullet_calibration_cost_frontier_results.ko.md`, `docs/experiments/wpu_v2_priority_dashboard.ko.md`. | candidate oracle은 여전히 더 강하다. Direct candidate-regret deployment는 test sweep에서 `0.329950`, train-selected deployment에서 `0.328025`까지 도달했지만 목표 `0.5`에는 못 미치고 harmful accept가 safety limit 근처에 남아 있다. Safety-frontier audit은 tradeoff를 직접 보인다. Harmful limit `0.25`에서는 direct best closure가 약 `0.327-0.330`이지만, harmful limit `0.10`에서는 direct closure `0.081898`, perturbed closure `0.154320`으로 낮아진다. Cross-fit ensemble regret gate는 train-selected overfit이 missing fix인지 검사했지만 negative result다. 최고 closure는 `0.287268`, safe best는 `0.279738`, cross-fit selected closure는 `0.270989`로 direct gate보다 낮다. Harmful-accept/ranking penalty 학습은 harmful accept를 낮추지만 closure를 `0.081253`까지 떨어뜨린다. Feature perturbation은 safe test-sweep closure를 `0.329756`까지 올리지만 train-selected deployment는 `0.312586`으로 낮아진다. 별도 safety/utility head도 negative result다. Best closure는 `0.147450`, safe best는 `0.090719`, train-selected closure는 `0.144863`에 그친다. Descriptor standardization과 group-DRO no-harm training도 단독 해결책으로는 negative result다. Best closure와 safe best는 모두 `0.110889`, train-selected closure는 `0.093863`이다. Joint object-set candidate gate도 negative result다. Best/safe closure는 `0.101454`, train-selected closure는 `0.072167`, regression-heavy K=16 ablation은 `0.034751`에 그친다. Branch-prior audit은 P4/P5의 failure boundary를 추가한다. `catch_heavy` majority prior accuracy는 `0.753968`이고 best WPU는 `0.408730`에 그친다. Mechanism-prior adaptation은 shifted WPU win-rate를 `0.666667`로 올리지만 shifted mean WPU ECE를 `0.024819` 악화시킨다. Prior-strength sweep은 accuracy-best 비영점 strength(`0.75`)도 `strength=0` 대비 ECE를 악화시킴을 확인한다. Calibration-selected prior strength는 shifted mean WPU ECE를 `-0.046204`, Brier를 `-0.105470` 개선하지만 shifted WPU-vs-baseline win-rate는 `0.333333`에 머문다. Few-shot mechanism adaptation은 shifted WPU-vs-baseline win-rate `1.000000`, mean margin change `0.050264`까지 도달하지만 mechanism-specific calibration sample을 사용하므로 zero-shot generalization은 아니다. Mechanism-aware adaptive policy는 adapted regime을 더 개선한다. Shifted WPU win-rate는 `1.000000`, mean accuracy change는 `0.198412`, mean margin change는 `0.058201`, mean ECE change는 `-0.099347`, mean Brier change는 `-0.155443`이다. 이는 detect-and-adapt evidence이지 zero-shot evidence는 아니다. Uncertainty-gated local-dense recompute는 aggregate accuracy를 `0.071428`, ECE를 `-0.016396` 개선하지만 dense recompute rate가 `0.985450`이고, low-cost gate는 ECE를 `0.005395` 악화시킨다. Learned sparse-output benefit gate는 recompute rate `0.205027`에서 source low-cost accuracy를 `0.052910` 개선하지만 ECE를 `0.010769` 악화시킨다. Calibration-cost frontier audit은 `cost_proxy <= 0.25`에서 non-reference calibration-safe policy가 `0`개이고 최저 비용 non-reference calibration-safe policy의 cost가 `0.867725`임을 보인다. Cross-seed robust scoring, calibration-safe 저비용 uncertainty gating, long-horizon model-delta stability도 남아 있다. |
| C8 | WPU는 아직 hardware/chiplet/IP 결과가 아니다. | 명시적으로 지지되지 않음. | `docs/Review/review_response_and_differentiation.md`, `docs/paper/state_is_all_you_need.md`, `docs/experiments/pybullet_system_profile_results.ko.md`, `docs/experiments/pybullet_matched_speedup_audit_results.ko.md`, `docs/experiments/pybullet_matched_speedup_tolerance_results.ko.md`, `docs/experiments/pybullet_pareto_frontier_results.ko.md`, `docs/experiments/pybullet_system_energy_proxy_results.ko.md`, `docs/experiments/pybullet_system_claim_boundary_results.ko.md`, `README.ko.md`. | CUDA random-forward profiling은 큰 latency-reduction 기회를 지지하고, screening-only energy proxy는 실제 power study를 어디서 수행할지 알려주는 보조 지표다. 수정된 matched-or-better audit은 `N=133`에서 best-accuracy non-WPU baseline 대비 양성이고, 작은 Pareto audit도 `N=133`에서 WPU를 accuracy-latency frontier에 올린다. Systems claim-boundary audit은 supported proxy 축 `4`개, partial trained 축 `2`개, real-power/sparse-kernel 미측정 축 `1`개를 분리하며, branch-overlay memory proxy reduction `0.874128`과 약한 CUDA peak-memory proxy reduction `0.304080`을 기록한다. 하지만 `N=5`에서는 WPU가 느리거나 지배되고, 이는 hardware evidence가 아니다. Hardware claim에는 실제 sparse kernel, memory traffic, branch-overlay allocation, power/energy, trained matched-or-better speedup 증거가 필요하다. |
| C9 | 객체화된 relation 위의 WPU propagation은 full physics가 아니라 단순화된 local-causality prior다. | 제한된 analogy로 지지됨. | `docs/objectification.ko.md`, `docs/arxiv/state_is_all_you_need_en.tex`, `docs/arxiv/state_is_all_you_need_ko.md`, `docs/paper/state_is_all_you_need.md`, `docs/experiments/object_history_hidden_mechanism_probe_results.md`, `docs/experiments/object_relation_law_probe_results.md`, `docs/experiments/object_relation_law_ood_probe_results.md`, `docs/experiments/object_relation_law_revision_probe_results.md`. | hidden-mechanism, local-law, OOD, revision probe는 synthetic이다. 이들은 history-derived relation variable과 단순 inverse-distance law가 generated mechanism 아래에서 nominal type name을 넘어 transfer될 수 있고, OOD stress가 relation failure와 law mis-specification을 분리하며, 작은 calibration이 일부 generated law shift를 수정할 수 있음을 보일 뿐이다. 실제 물리 competence나 unknown-theory discovery는 simulator/robotics benchmark, long-horizon stability, learned relation이 training generator 밖에서도 일반화된다는 증거가 필요하다. |
| C10 | 단기 WPU 가치는 silicon보다 software runtime/middleware에서 더 가능성이 있다. | plausible direction이며 아직 실험적으로 증명되지 않음. | `docs/reproducibility.md`, `docs/arxiv/README.md`, current PyTorch package under `wpu/`. | digital-twin, simulation backend, game/server, robotics middleware benchmark가 필요하다. |
| C11 | 객체화 품질은 propagation 전에 contract로 측정 가능하고 국소적으로 repair 가능하다. | 구현 주장으로 지지됨. | `wpu/core/objectification.py`, `tests/test_objectification.py`, `tests/test_script_entrypoints.py`, `docs/experiments/objectification_relation_repair_probe_results.md`, `docs/experiments/pybullet_objectification_quality_results.ko.md`, `docs/experiments/pybullet_objectification_loss_coupling_results.ko.md`, `README.ko.md`, `docs/objectification.ko.md`. | Relation repair와 `LocalLawHypothesis`는 보수적 hypothesis 및 revision report를 만들 뿐 ground-truth physics가 아니다. 최신 probe는 learned repair가 aliased type name을 넘어 transfer하고 toy downstream diagnostic을 개선하며 law-revision gap을 보고할 수 있음을 보인다. PyBullet loss-coupling audit은 selected-K/frontier degradation이 MSE increase와 연결됨을 보이지만 branch accuracy 변화는 아직 작다. perception-to-object construction이나 unknown-theory discovery가 해결됐다는 증거는 아니다. |

P2 correction-trigger frontier는 C7의 한계를 더 명확히 한다. 테스트한 trigger policy 중
integrity >= `0.8`과 correction rate <= `0.25`를 동시에 만족한 경우는 `0`개이며,
최고 low-correction trigger는 `selective_corrected_entropy035`로 integrity `0.653668`에
그친다. 따라서 현재 증거는 low-frequency stable sparse dynamics가 아니라 bounded
memory-safety layer다.

## 반증 조건

- controlled identity/locality/branching benchmark에서 serialized-token 또는 graph
  baseline이 동일 compute로 WPU와 같거나 더 좋으면, 그 regime에서 WPU는 필수
  primitive가 아니다.
- realistic state store에서 pre-tensor retrieval 비용이 `O(N)`에 가깝게 증가하면,
  large-`N` systems claim은 무너진다.
- 더 큰 seed/model sweep에서 risk-adjusted mechanism selection의 held-out gain이
  사라지면, 현재 v2 working-set-control 주장은 diagnostic result로 낮춰야 한다.
- Sample-level no-harm gate가 held-out seed에서 계속 음수 closure를 보이면,
  candidate selection은 deployment threshold만으로는 부족하며 calibrated
  candidate-regret 및 uncertainty target으로 학습되어야 한다.
- Direct candidate-regret gate가 평균 loss를 개선하더라도 harmful accept rate가
  높게 유지되면, selector가 K와 held-out seed 전반에서 harmful candidate를
  거부할 수 있을 때까지 P1은 해결된 것이 아니다.
- Cross-fit 또는 ensemble candidate-regret gate가 in-sample optimism은 줄이지만
  closure를 낮춘다면, P1 병목은 deployment threshold만의 문제가 아니다.
  Candidate score 자체를 retrieval/propagation과 함께 더 invariant하게 학습해야
  한다.
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
- finite-safe 또는 selective correction이 대부분의 sparse update에서 여전히 trigger되어야
  한다면, memory layer는 applied state를 보호할 수 있지만 raw transition model은 아직
  안정적이지 않다. Correction-trigger frontier는 이 경계를 확인한다. 테스트한 trigger
  policy 중 integrity >= `0.8`과 correction rate <= `0.25`를 동시에 만족한 경우는
  `0`개다. 더 강한 주장을 하려면 rollback 제거나 corrected object set 축소뿐 아니라
  learned low-frequency correction이 필요하다.

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

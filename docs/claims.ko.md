# WPU 주장 관리표

이 문서는 WPU 논문 수준의 주장을 현재 증거, 한계, 반증 조건과 연결한다. 목적은
과장 주장을 막고, 심사 대응 시 어떤 주장이 어떤 실험으로 뒷받침되는지 명확히
보여주는 것이다.
외부 공표 준비도와 미해결 gap은 `docs/publication_readiness.ko.md`를 함께 본다.
저장소 수준의 compact thesis와 신규성 경계는 `docs/research_thesis.ko.md`를 함께 본다.
Process-unit 공개 감사는 `docs/process_unit_release_audit.ko.md`를 함께 본다.
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
| C6 | large `N`은 causal working set `K`가 작고 tensorization 전에 식별될 때만 WPU에 유리하다. | 조건부 systems hypothesis로 초기 증거가 있음. | `docs/experiments/cws_balanced_branch_8m_gpu_event_conditioned_final_results.md`, `docs/experiments/wpu_v2_pre_tensor_indexed_n_sweep_results.md`, `docs/experiments/large_n_target_frontier_pooling_probe_results.ko.md`, `docs/experiments/world_copy_index_probe_results.ko.md`, `docs/experiments/world_copy_causal_index_stress_results.ko.md`, `docs/experiments/world_copy_escalation_correction_probe_results.ko.md`, `docs/experiments/pybullet_shift_generalization_n1024_mechanism_relation_results.md`, `docs/experiments/pybullet_shift_generalization_n2048_mechanism_relation_results.md`, `docs/experiments/pybullet_shift_generalization_n4096_mechanism_relation_results.md`, `docs/versions/wpu_v1_closure.md`. | `N`이 큰 것만으로는 충분하지 않다. retrieval이 `O(N)`에 가깝거나 causal state를 놓치면 실패한다. Total `N=4101` 결과는 WPU-only sparse feasibility evidence이지 baseline-complete accuracy claim이 아니다. World-copy index, noisy-index stress, escalation-correction probe는 index/substrate evidence이지 trained accuracy evidence가 아니다. |
| C7 | explicit state는 propagation 이전의 working-set control을 노출한다. | 현재 가장 강한 v2 mechanism claim으로 지지되며, 조건부 P1-positive sub-regime과 더 큰 K gap이 함께 남아 있다. | `docs/experiments/wpu_v2_retriever_regret_distillation_results.md`, `docs/experiments/wpu_v2_invariant_set_scorer_results.md`, `docs/experiments/wpu_v2_candidate_oracle_gap_decomposition.ko.md`, `docs/experiments/wpu_v2_candidate_noharm_gate_results.ko.md`, `docs/experiments/wpu_v2_candidate_regret_gate_results.ko.md`, `docs/experiments/wpu_v2_candidate_regret_gate_penalty_results.ko.md`, `docs/experiments/wpu_v2_candidate_regret_gate_perturbed_results.ko.md`, `docs/experiments/wpu_v2_candidate_regret_crossfit_results.ko.md`, `docs/experiments/wpu_v2_candidate_safety_gate_results.ko.md`, `docs/experiments/wpu_v2_candidate_invariant_gate_results.ko.md`, `docs/experiments/wpu_v2_candidate_joint_gate_results.ko.md`, `docs/experiments/wpu_v2_candidate_joint_gate_regression_heavy_k16_results.ko.md`, `docs/experiments/wpu_v2_end_to_end_candidate_selector_results.ko.md`, `docs/experiments/wpu_v2_candidate_safety_frontier_results.ko.md`, `docs/experiments/wpu_v2_joint_candidate_generator_results.ko.md`, `docs/experiments/wpu_v2_verified_candidate_controller_results.ko.md`, `docs/experiments/wpu_v2_joint_propagation_adapter_results.ko.md`, `docs/experiments/wpu_v2_joint_selector_propagator_results.ko.md`, `docs/experiments/wpu_v2_joint_selector_propagator_geometry_results.ko.md`, `docs/experiments/wpu_v2_joint_selector_propagator_budget8_results.ko.md`, `docs/experiments/wpu_v2_joint_selector_propagator_relation_results.ko.md`, `docs/experiments/wpu_v2_joint_selector_propagator_pairwise_noharm_w03_results.ko.md`, `docs/experiments/wpu_v2_joint_selector_propagator_structured_candidates_no_margin_results.ko.md`, `docs/experiments/wpu_v2_joint_selector_propagator_score_regression_results.ko.md`, `docs/experiments/pybullet_branch_prior_shift_results.ko.md`, `docs/experiments/pybullet_mechanism_prior_adaptation_results.ko.md`, `docs/experiments/pybullet_prior_strength_sweep_results.ko.md`, `docs/experiments/pybullet_selected_prior_adaptation_results.ko.md`, `docs/experiments/pybullet_fewshot_mechanism_adaptation_results.ko.md`, `docs/experiments/pybullet_mechanism_adaptive_policy_results.ko.md`, `docs/experiments/pybullet_shift_detector_policy_results.ko.md`, `docs/experiments/pybullet_shift_generalization_n512_results.ko.md`, `docs/experiments/pybullet_uncertainty_gated_recompute_results.ko.md`, `docs/experiments/pybullet_learned_uncertainty_gate_results.ko.md`, `docs/experiments/pybullet_mechanism_selective_calibration_gate_results.ko.md`, `docs/experiments/pybullet_calibration_cost_frontier_results.ko.md`, `docs/experiments/pybullet_correction_trigger_frontier_results.ko.md`, `docs/experiments/wpu_v2_priority_dashboard.ko.md`. | 이전 direct candidate-regret deployment는 train-selected closure `0.328025`에 그쳤고, post-hoc/object-set/generator/verification/adapter 계열 probe는 대부분 negative였다. Minimal joint selector-propagator objective는 첫 P1-positive 결과다. `N=2048`, 5개 held-out seed, `K=8`에서 safe closure는 `0.877854`, harmful accept는 `0.075555`다. 하지만 일반적 해결은 아니다. `K=16`에서는 unconstrained closure `0.171659`와 harmful accept `0.444444`, train-selected closure `-0.026759`이고, `K=32` train-selected closure는 `0.059919`다. Larger-K follow-up은 geometry/force descriptor concatenation과 budget=8만으로는 부족함을 보이며, relation-conditioned propagation은 `K=32` closure를 `0.266805`까지 올리지만 harmful accept `0.333333`을 남긴다. Pairwise no-harm score margin은 harmful accept를 낮춘다. 테스트한 best trade-off인 weight `0.3`은 `K=16` closure `0.175125`, harmful accept `0.195555`, `K=32` closure `0.200230`, harmful accept `0.053333`을 만든다. No-harm margin이 없는 deterministic structured candidate는 K=16 confidence-selected closure를 `0.241624`, harmful accept `0.200000`까지 올리지만 K=32는 `0.124512`에 그치고, no-harm margin을 붙이면 이 headroom이 억제된다. Score-to-loss regression도 negative다. Best closure는 K=16에서 `0.186333`이지만 harmful accept가 `0.435556`이고, conservative best는 K=32의 `0.095543`에 그친다. 따라서 explicit state는 유용한 working-set control surface를 노출하지만, deployable oracle-gap closure는 아직 작은 K 또는 쉬운 larger-K regime에 제한되며 더 넓은 K=16/32에는 learned safe candidate generation과 propagation-aware verification이 필요하다. Mechanism shift, calibration, correction trigger, long-horizon model-delta stability도 미해결이다. |
| C8 | WPU는 아직 hardware/chiplet/IP 결과가 아니다. | 명시적으로 지지되지 않음. | `docs/Review/review_response_and_differentiation.md`, `docs/paper/state_is_all_you_need.md`, `docs/experiments/pybullet_system_profile_results.ko.md`, `docs/experiments/pybullet_matched_speedup_audit_results.ko.md`, `docs/experiments/pybullet_matched_speedup_tolerance_results.ko.md`, `docs/experiments/pybullet_pareto_frontier_results.ko.md`, `docs/experiments/pybullet_system_energy_proxy_results.ko.md`, `docs/experiments/pybullet_system_claim_boundary_results.ko.md`, `README.ko.md`. | CUDA random-forward profiling은 큰 latency-reduction 기회를 지지하고, screening-only energy proxy는 실제 power study를 어디서 수행할지 알려주는 보조 지표다. 수정된 matched-or-better audit은 `N=133`에서 best-accuracy non-WPU baseline 대비 양성이고, 작은 Pareto audit도 `N=133`에서 WPU를 accuracy-latency frontier에 올린다. Systems claim-boundary audit은 supported proxy 축 `4`개, partial trained 축 `2`개, real-power/sparse-kernel 미측정 축 `1`개를 분리하며, branch-overlay memory proxy reduction `0.874128`과 약한 CUDA peak-memory proxy reduction `0.304080`을 기록한다. 하지만 `N=5`에서는 WPU가 느리거나 지배되고, 이는 hardware evidence가 아니다. Hardware claim에는 실제 sparse kernel, memory traffic, branch-overlay allocation, power/energy, trained matched-or-better speedup 증거가 필요하다. |
| C9 | 객체화된 relation 위의 WPU propagation은 full physics가 아니라 단순화된 local-causality prior다. | 제한된 analogy로 지지됨. | `docs/objectification.ko.md`, `docs/arxiv/state_is_all_you_need_en.tex`, `docs/arxiv/state_is_all_you_need_ko.md`, `docs/paper/state_is_all_you_need.md`, `docs/experiments/object_history_hidden_mechanism_probe_results.md`, `docs/experiments/object_relation_law_probe_results.md`, `docs/experiments/object_relation_law_ood_probe_results.md`, `docs/experiments/object_relation_law_revision_probe_results.md`. | hidden-mechanism, local-law, OOD, revision probe는 synthetic이다. 이들은 history-derived relation variable과 단순 inverse-distance law가 generated mechanism 아래에서 nominal type name을 넘어 transfer될 수 있고, OOD stress가 relation failure와 law mis-specification을 분리하며, 작은 calibration이 일부 generated law shift를 수정할 수 있음을 보일 뿐이다. 실제 물리 competence나 unknown-theory discovery는 simulator/robotics benchmark, long-horizon stability, learned relation이 training generator 밖에서도 일반화된다는 증거가 필요하다. |
| C10 | 단기 WPU 가치는 silicon보다 software runtime/middleware에서 더 가능성이 있다. | plausible direction이며 아직 실험적으로 증명되지 않음. | `docs/reproducibility.md`, `docs/arxiv/README.md`, current PyTorch package under `wpu/`. | digital-twin, simulation backend, game/server, robotics middleware benchmark가 필요하다. |
| C11 | 객체화 품질은 propagation 전에 contract로 측정 가능하고 국소적으로 repair 가능하다. | 구현 주장으로 지지됨. | `wpu/core/objectification.py`, `tests/test_objectification.py`, `tests/test_script_entrypoints.py`, `docs/experiments/objectification_relation_repair_probe_results.md`, `docs/experiments/pybullet_objectification_quality_results.ko.md`, `docs/experiments/pybullet_objectification_loss_coupling_results.ko.md`, `README.ko.md`, `docs/objectification.ko.md`. | Relation repair와 `LocalLawHypothesis`는 보수적 hypothesis 및 revision report를 만들 뿐 ground-truth physics가 아니다. 최신 probe는 learned repair가 aliased type name을 넘어 transfer하고 toy downstream diagnostic을 개선하며 law-revision gap을 보고할 수 있음을 보인다. PyBullet loss-coupling audit은 selected-K/frontier degradation이 MSE increase와 연결됨을 보이지만 branch accuracy 변화는 아직 작다. perception-to-object construction이나 unknown-theory discovery가 해결됐다는 증거는 아니다. |

최신 C7 large-state 보정: 원본 N_bg=512 mechanism-diversity screen은 여전히 유효한
failure-boundary evidence지만, 이제 단독으로 읽으면 안 된다. 현재 해석은
action-conditioned event state와 physical object-state scalar를 보존하면 nominal-train
large-N shift screen이 `4/0/3`, 평균 margin `+0.002976`까지 회복되지만,
multi-mechanism training은 `2/2/3`, 평균 margin `-0.032738`로 여전히
mixed/negative라는 것이다. 따라서 주장은 더 좁아진다. WPU의 large-state advantage에는
작은 identifiable `K`, 충실한 object/action state tensorization, 그리고
mechanism-aware propagation 또는 adaptation이 필요하다.

최신 C6 sparse-feasibility 경계: N_bg=4096 mechanism-relation run은 large-state
stress를 total `N=4101`까지 확장하지만, 이는 WPU sparse feasibility 결과일 뿐이다.
선택된 working set이 작게 유지될 때 indexed state가 full recompute를 피할 수 있음을
지지한다. 같은 protocol에서 matched dense/token/graph baseline이 완료되지 않았으므로
accuracy superiority를 의미하지 않는다.

최신 C6/C7 large-N readout 업데이트:
`docs/experiments/large_n_target_frontier_pooling_probe_results.ko.md`는 v1
`N>=204` branch collapse의 한 원인을 분리한다. Global mean branch readout이
event-causal state를 non-causal object와 평균내어 희석한다. 같은 3-seed, 80-step
training에서 total `N=404` 기준 기존 `wpu-sparse` mean branch accuracy는
`0.359375`지만, `wpu-sparse-frontier`는 `0.781250`에 도달해 `serialized-token`
`0.778646`을 근소하게 넘고 work proxy는 `3` 대 `166464`다. 이는 non-causal
background scaling에서 event target/frontier readout이 WPU-native fix임을 지지한다.
하지만 causal `K`가 커지거나 relation이 틀리거나 mechanism variable이 누락되는 경우를
해결한 것은 아니다.

최신 C6 v3 world-copy index 업데이트:
`docs/experiments/world_copy_index_probe_results.ko.md`는 hierarchical world-copy
substrate와 `WorldCausalIndex`를 추가한다. Controlled probe에서 total `N`이 `104`에서
`10004`까지 커져도 selected `K`는 `4`로 유지되고 non-causal background object는
선택되지 않는다. `N=10004`의 affected fraction은 `0.00039984`다. 이는 large `N`에서
pre-tensor causal slice retrieval이 가능함을 지지한다. 하지만 trained accuracy,
noisy objectification robustness, 실제 물리 세계 이해를 증명하지는 않는다.

최신 C6 v3 noisy-index 업데이트:
`docs/experiments/world_copy_causal_index_stress_results.ko.md`는 noisy causal
retrieval을 위한 첫 P1 stress benchmark다. `N=128..8192`, `K_ref=4/8/16`, missing
relation rate `0/0.25/0.5`, false-positive relation rate `0/0.1/0.25`, relation
confidence threshold `0/0.3`, true relation confidence `0.95/0.2`를 sweep한다.
Controlled region-scoped setup에서 recall은 `1.000000`을 유지하고 `N=8192` touch ratio는 `0.004385` 이하로 유지된다.
Confidence gate가 없으면 false-positive relation은 mean precision을 `0.800000`까지
낮추지만, `min_relation_confidence=0.3`에서는 recall `1.000000`을 유지하면서 precision도
`1.000000`으로 회복된다. True causal relation confidence를 `0.2`로 낮추면 region
scope가 recall과 precision을 회복하지만 mean escalation은 `0.981481`에 도달한다. 이는
index scalability claim을 강화하면서 다음 문제는 uncertainty escalation 이후 local
dense/hybrid correction이라는 점을 보여준다.

최신 C6 v3 escalation-correction 업데이트:
`docs/experiments/world_copy_escalation_correction_probe_results.ko.md`는 그 다음
boundary를 substrate 수준에서 테스트한다. True relation confidence가 `0.2`일 때
`sparse_confident_relations`는 mean recall `0.145833`, precision `1.000000`, F1
`0.246623`에 그치지만, `hybrid_escalation_region`은 mean recall/precision/F1
`1.000000`에 도달하고 max selected `K=16`, max touch ratio `0.24489796`을 유지한다.
이는 escalation 이후 local correction candidate를 지지하지만, learned transition
accuracy나 token/graph superiority를 증명하지는 않는다.

최신 C7 P1 verification-context 업데이트:
`docs/experiments/wpu_v2_joint_selector_propagator_verification_context_results.ko.md`는
joint selector-propagator selector 입력에 label-free propagation signature를 추가한다.
이는 유용하지만 충분하지 않은 larger-K 개선이다. `K=16`에서 unconstrained closure는
`0.409420`까지 오르지만 harmful accept가 `0.342222`로 남고, confidence-selected
closure는 harmful accept `0.115555`에서 `0.269216`이다. `K=32`의 confidence-selected
closure는 `0.153386`이다. 따라서 propagation-aware verification은 방향으로 지지되지만,
safe candidate generation을 아직 해결하지 못한다.

최신 C7 P1 verification-head 업데이트:
`docs/experiments/wpu_v2_joint_selector_propagator_verification_head_results.ko.md`는
label-free verification signature 위에 explicit harmful-candidate head를 추가한다.
이는 standalone fix로는 negative다. `K=16`에서 unconstrained closure는 `0.345395`,
harmful accept는 `0.391111`로 verification-context보다 나쁘고, confidence-selected
closure는 harmful accept `0.102222`에서 `0.193197`로 떨어진다. `K=32`의
confidence-selected closure는 `0.060597`에 그친다. 따라서 다음 P1 목표는 더 강한
rejection head가 아니라 propagation-aware supervision을 가진 learned safe candidate
generation이다.

최신 C7 P1 learned-safe-candidate 업데이트:
`docs/experiments/wpu_v2_joint_selector_propagator_learned_safe_candidates_results.ko.md`는
train fold에서 interaction, proximity, density, axis teacher를 모방하는 object-level
generator를 추가한다. 이는 larger-K missing fix가 아니다. `K=16` confidence-selected
closure는 harmful accept `0.222222`에서 `0.246071`이고, `K=32` confidence-selected
closure는 `0.143398`이다. Verification-context 결과보다 약하므로 다음 candidate-generation
목표는 hand-built teacher imitation이 아니라 propagation loss와 no-harm transfer를 직접
학습하는 것이다.

최신 route-state contract 보정:
`docs/experiments/wpu_v2_route_physics_contract_smoke_results.ko.md`는 더 작은 구현
수정을 기록한다. Adaptive route-regret context는 이제 pair geometry, target physical
scalar, selected-set physical scalar, `force`, `catch_action`을 받는다. 이전에는 이
context가 pair distance, target xy, event norm으로 압축됐다. 이는 P1 해결 결과가
아니지만, 후속 route 실험이 mechanism을 정의하는 state variable을 조용히 버리는 문제를
막는다.
`docs/experiments/wpu_v2_regret_router_variant_results.ko.md`의 full 5-seed
staged-regret rerun은 이 경계를 확인한다. 확장된 physical/action context를 단순
concatenation으로 붙이는 것은 neutral하다. `physics_hidden` routed loss는 `0.962987`,
internal은 `0.962894`이고, `state_only`는 `0.982804`로 여전히 나쁘다. 따라서 structured
verification 또는 joint route training이 필요하다.
`docs/experiments/pybullet_route_regret_training_smoke_results.ko.md`는 같은 explicit
route-regret training을 PyBullet mechanism-shift path에 연결한다. 이는 infrastructure
evidence일 뿐이다. 작은 smoke run은 route metric이 출력되고, all-dense routing 붕괴를
피하려면 configurable threshold가 필요하다는 점을 보인다. 이어진 selected-threshold
smoke는 trivial all-dense/all-sparse endpoint를 피하지만, tiny run에서 shifted
accuracy를 개선하지는 않는다.
`docs/experiments/pybullet_shift_generalization_n512_route_regret_selected_results.ko.md`는
이를 3-seed N_bg=512 mechanism screen으로 확장한다. 결과는 mixed/negative다.
Selected route-regret WPU는 낮은 dense compute(`0.071429`)를 쓰고 `no_catch`에서는
도움이 되지만, best-WPU 대 best-baseline win/tie/loss는 `2/1/4`에 그치며
graph-transformer가 더 높은 macro accuracy를 유지한다.
`docs/experiments/pybullet_shift_generalization_n512_route_regret_adapted_screen_results.ko.md`는
4개 shifted mechanism에서 matched mechanism-prior adaptation을 추가한다. 이 결과도
route-regret WPU에는 negative다. Best baseline 대비 win/tie/loss는 `0/0/4`다. 따라서
다음 단계는 prior-bias adaptation이 아니라 mechanism-conditioned propagation dynamics다.
`docs/experiments/pybullet_shift_generalization_n512_mechanism_conditioned_screen_results.ko.md`는
그 방향의 첫 positive follow-up이다. Sparse propagation을 explicit physics/action
context로 직접 condition하고 dense fallback을 끈다. 3-seed N_bg=512 screen에서
mechanism-conditioned WPU는 macro accuracy `0.541667`을 달성했고, best non-WPU baseline은
`0.500000`이었다. Dense compute는 `0.000000`, win/tie/loss는 `1/2/1`이다. 이는 유망한
screen이지 해결된 주장이 아니다. `edge_shift`는 여전히 negative이며, 더 큰 seed/mechanism
sweep으로 확장해야 강한 증거가 된다.
`docs/experiments/pybullet_shift_generalization_n512_mechanism_adapter_multitrain_results.ko.md`는
그 경계를 더 선명하게 만든다. Nominal-only 5-seed/7-mechanism 확장은 negative다.
Macro accuracy는 `0.433333`이고 best baseline은 `0.476190`이다. Object-wise adapter도
nominal-only training에서는 negative다. 그러나 adapter를 primitive mechanisms로 학습하면
5-seed N_bg=512 macro accuracy `0.497143`에 도달하고, best baseline은 `0.472857`이다.
Dense compute는 `0.000000`, win/tie/loss는 `3/1/3`이다. 따라서 주장은 더 좁아진다.
WPU에는 primitive mechanism variation으로 학습된 object-wise mechanism-conditioned
propagation이 필요하며, large-N sparse state만으로 또는 nominal-only zero-shot
extrapolation만으로는 부족하다.
`docs/experiments/pybullet_shift_generalization_n512_mechanism_factorized_shuffled_results.ko.md`는
multi-mechanism training protocol을 보정한다. 이전 `ConcatDataset` training loader는
shuffle되지 않았기 때문에 작은 step budget에서 mechanism 순서에 민감할 수 있었다.
Seed-fixed training shuffle을 추가한 뒤 factorized sparse mechanism adapter는 5 seeds에서
negative다. Macro accuracy는 `0.497143`, graph-transformer는 `0.548571`이고, dense compute는
`0.000000`, win/tie/loss는 `2/1/4`다. 따라서 이전 multi-mechanism positive는 screen으로
낮춰야 하며, robust local-law composition, 특히 edge-conditioned composition에는 explicit
composition supervision이 필요하다.
`docs/experiments/pybullet_shift_generalization_n512_target_local_loss_results.ko.md`는
그 supervision의 가장 직접적인 형태인 event target object에 대한 target-local delta
MSE를 감사한다. 이 실험은 large-N loss 정렬 문제가 실제로 있음을 드러냈지만, 단독
해결책으로는 negative다. Target-local weight `1.0`에서는 WPU target MSE가 낮아지지만
macro branch accuracy는 `0.418571`로 떨어지고, 같은 run의 graph-transformer는
`0.494286`이다. Weight `0.25`, `0.5`도 branch-composition gap을 회복하지 못한다.
따라서 경계는 더 선명해진다. 다음 WPU 개선은 delta loss reweighting이 아니라
branch-conditioned 또는 mechanism-specific local propagation 같은 transition dynamics
변경이어야 한다.
`docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_results.ko.md`는
그 architecture 변경을 구현한다. 새 `wpu-cws-indexed-mechanism-branch` 모델은 indexed
sparse execution과 zero dense fallback을 유지하면서 mechanism-conditioned branch
transition head를 추가한다. 5-seed N_bg=512 shuffled multi-mechanism screen에서 macro
branch accuracy는 `0.568571`, graph-transformer는 `0.548571`이고, ECE는 `0.247101`
대 `0.254194`, dense compute는 `0.000000`, win/tie/loss는 `4/0/3`이다. 이는 corrected
factorized 및 target-local negative diagnostic 이후 첫 positive follow-up이다. 다만
보편 우월성은 아니다. `catch_heavy`, `edge_shift`, `edge_high_force`는 여전히 best
dense baseline보다 낮다.
`docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_stress_results.ko.md`는
그 positive screen을 stress-test한다. 명시적 `--train-samples-per-mechanism` control을
추가한 뒤 larger train/eval pilot은 WPU accuracy 관점에서 negative다. h32 WPU는
`0.534524`, best baseline은 `0.598810`이고, 공정한 h64 capacity check에서도 WPU는
`0.603571`, serialized-token은 `0.622619`이다. Dense compute는 계속 `0.000000`이므로
efficiency claim은 유지되지만, accuracy claim은 transition-head expressivity가 개선될
때까지 short-budget screen으로 낮춰야 한다.
`docs/experiments/pybullet_shift_generalization_n512_branch_expert_results.ko.md`는
첫 expressivity fix인 branch-specific output expert를 검증한다. 결과는 단독 해결책으로
negative다. Expert model은 h32 stress protocol에서 macro accuracy `0.505952`에 그쳐
기존 mechanism-branch head (`0.534524`)와 graph-transformer (`0.598810`)보다 낮다.
일부 edge/catch composition case는 개선하지만 일반 mechanism accuracy를 잃는다. 따라서
다음 architecture 단계는 branch logit이 아니라 relation-type-conditioned sparse
propagation message로 내려가야 한다.
`docs/experiments/pybullet_shift_generalization_n512_mechanism_relation_results.ko.md`는
그 propagation-level 변경을 구현한다. 새 `wpu-cws-indexed-mechanism-relation` route는
source/target hidden state, relation feature, route physics feature를 사용해 selected
working-set relation 위로 learned message를 scatter한다. h32 trainpool40/steps16/eval40
stress protocol의 5-seed 확장에서 WPU macro accuracy는 `0.639286`,
graph-transformer는 `0.597143`이고 dense compute는 `0.000000`이다. Best baseline
대비 win/tie/loss는 `5/0/2`다. 3-seed h64 fair-capacity check에서도 WPU는
`0.678571`, serialized-token은 `0.622619`이다. 더 큰 5-seed N=1029 distractor
screen도 positive다. WPU는 `0.639286`, graph-transformer는 `0.577143`이고 dense
compute는 `0.000000`, win/tie/loss는 `6/0/1`이다. N=2053 3-seed distractor
screen은 scaling trend를 더 강화한다. WPU는 `0.644048`, graph-transformer는
`0.516667`이고 dense compute는 `0.000000`, win/tie/loss는 `7/0/0`이다. 이는 현재 WPU v2의 가장 강한 증거지만, 아직
PyBullet synthetic, single-step, non-causal-distractor-bounded evidence다. 따라서
calibration-aware evaluation, rollout, 더 어려운 causal large-N 확장이 필요하다.
Relation-conditioned closed-loop rollout diagnostic은 이 경계를 더 선명하게 만든다.
최신 N=517 audit에서 raw relation WPU는 selected `K = 4.354167`을 유지하지만 H=25
integrity `0.250368`, trajectory MSE `6.975125`, branch accuracy `0.208333`으로
붕괴한다. Finite projection은 H=25 integrity를 `0.876760`까지 올리지만 trajectory MSE
`1.695024`, branch accuracy `0.250000`으로 예측 성능은 약하다. 이는 safety guard
결과이지 learned long-horizon dynamics가 아니다. 이전 scalar learned-stability ablation,
고정 temporal scaling, 짧은 simulator stride target, explicit multi-horizon target은 모두
H=25 collapse를 막지 못했다. 첫 raw-model positive는 transition
head 내부의 bounded delta parameterization이다. H=25 integrity는 bound `0.5`에서
`0.611900`, `0.25`에서 `0.652870`, `0.1`에서 `0.793182`, `0.05`에서
`0.870264`까지 오른다. selected `K`는 `4.354167`로 유지되고 correction, rollback,
rejection, dense fallback은 사용하지 않는다. 이는 P2를 순수 failure boundary에서 부분적
raw-stability 결과로 바꾼다. Simulator-resynchronized metric은 under-update 우려도 줄인다.
Bound `0.05`는 H=25 trajectory MSE `0.707117`, branch accuracy `0.729167`을 보이며,
finite projection의 trajectory MSE `1.695024`, branch accuracy `0.250000`보다 좋다.
남은 gap은 target-object trajectory MSE `361.358309`가 여전히 크다는 점이다. 후속으로
실험한 learned adaptive bounds, 수동 position/velocity split bounds, target-object delta
loss는 이 병목을 줄이지 못했다. target-object MSE는 `361-363` 근처에 남거나 branch
accuracy가 악화됐다. Full recurrent unrolled loss는 초기 probe에서 non-finite gradient를
만들었고, 안정화한 truncated H=2/4 unroll은 같은 lr/clip bounded-only baseline과 거의
같았다. 새 branch-weighted target-local transition head는 첫 positive follow-up이다. 선호하는
5-seed matched comparison에서 H=25 branch accuracy는 `0.612500`에서 `0.675000`으로,
trajectory MSE는 `1.063341`에서 `1.061493`으로, target-object MSE는 `544.710934`에서
`543.834596`으로 개선된다. 이는 transition-head 방향을 지지하지만 개선폭은 작고 state-integrity는
`0.868758`에서 `0.850574`로 낮아진다. 따라서 high-fidelity dynamics와 validity-preserving
target dynamics는 아직 해결되지 않았다. Channel-masked constrained target-head follow-up도
standalone 해결책은 아니다. H=25 branch accuracy는 `0.650000`, target-object MSE는
`544.588713`, integrity는 `0.846955`에 그친다. 따라서 다음 P2 목표는 단순 residual channel
masking이 아니라 state-validity-aware transition learning이다.

P1 candidate generation 증거는 단독 해결책으로는 명시적으로 negative result다.
Joint candidate-generator probe는 learned generated candidate가 oracle headroom을 만들 수
있음을 보인다. `K=16`에서 learned-generator oracle closure는 `0.361251`까지 도달하지만,
실제 deployed set evaluator closure는 `0.042951`에 그친다. 따라서 WPU v2는 candidate
generation만으로 working-set selection을 해결했다고 주장할 수 없다. 다음 단계는 candidate
generation, retrieval, propagation verification, calibrated no-harm training을 함께 묶는
joint objective다. 자세한 결과는
`docs/experiments/wpu_v2_joint_candidate_generator_results.ko.md`를 본다.

P1 verification-feature 증거도 단독 해결책으로는 negative result다. Verified
candidate-controller probe는 label-free sparse/local-dense propagation signature를
candidate descriptor에 추가했지만 best closure `0.024989`, safe-best closure
`0.023029`, train-selected closure `0.024989`에 그친다. 따라서 verification은 post-hoc
selector feature로 붙이는 것이 아니라 retrieval 및 propagation dynamics와 함께 학습해야
한다. 자세한 결과는
`docs/experiments/wpu_v2_verified_candidate_controller_results.ko.md`를 본다.

P1 shallow propagation-adapter 증거도 negative result다. Sparse/local-dense verification
feature에서 학습한 candidate-aware branch-logit adapter는 best/safe closure `0.092185`,
train-selected closure `0.069911`에 그친다. 따라서 작은 output adapter도 missing fix가
아니며, 남은 목표는 retrieval, candidate generation, propagation dynamics, propagation
verification, calibrated no-harm rejection의 더 깊은 공동학습이다. 자세한 결과는
`docs/experiments/wpu_v2_joint_propagation_adapter_results.ko.md`를 본다.

P1 joint utility-verifier 증거도 negative result다. Candidate object-set tensor,
sparse/local-dense verification signature, uncertainty, no-harm safety를 함께 쓰는
verifier는 best/safe closure `0.097845`, train-selected closure `0.077781`에
그친다. 따라서 fixed-propagator utility/safety head도 missing fix가 아니며, 남은
목표는 candidate generation, retrieval, propagation dynamics, verification,
calibrated no-harm rejection을 end-to-end로 결합하는 것이다. 자세한 결과는
`docs/experiments/wpu_v2_joint_utility_verifier_results.ko.md`를 본다.

P3 large-N simulator evidence는 강화됐지만 여전히 제한적이다. Medium-training N_bg=256
run은 total `N=261`에서 baseline-complete다. Best WPU accuracy는 `0.466667`, best
baseline accuracy는 `0.450000`이고, best WPU는 해당 best-accuracy baseline보다
`60.629526x` 빠르다. 이는 조건부 large-N state-native regime을 지지하지만, margin이
작고 task가 여전히 단일 cup family이므로 broad simulator superiority를 증명하지 않는다.
Higher-budget 5-seed N_bg=512 baseline-complete run은 total `N=517`까지 도달했다.
Best WPU accuracy는 `0.433333`, best baseline accuracy는 `0.425000`이고, best WPU는
해당 best-accuracy baseline보다 `57.595711x` 빠르다. 더 큰 training/evaluation
budget에서도 WPU edge는 유지되지만 margin은 줄어들기 때문에, 이는 broad simulator
superiority가 아니라 conditional large-N evidence로 인용해야 한다.

P3/P4 large-state mechanism diversity는 더 선명한 claim-boundary result가 됐다.
원본 total `N=517`의 N_bg=512 mechanism-diversity screens는 negative였다.
Nominal-train screen에서 WPU win/tie/loss는 `2/1/4`, 평균 margin은 `-0.047619`이고,
multi-mechanism-train screen에서는 `2/0/5`, 평균 margin `-0.095238`이었다. 이 audit은
state-input contract 결함을 드러냈다. PyBullet event의 `catch_action`과 objectified
state의 physical scalar가 존재했지만 tensorization에서 빠져 있었다. `catch_action`,
`edge_distance`, `hand_distance`, `fall_risk`, `angular_speed`를 보존한 후 nominal-train
screen은 `4/0/3`, 평균 margin `+0.002976`으로 회복됐다. 그러나 multi-mechanism
screen은 `2/2/3`, 평균 margin `-0.032738`로 여전히 mixed/negative다. 이 결과는
WPU가 full-state tensorization을 피할 수 있다는 systems claim을 약화하지는 않지만,
large N, 작은 K, 또는 mechanism diversity만으로 accuracy 우위가 보장된다는 주장은
약화한다. Mechanism-aware propagation과 adaptation이 필요하다.

P2 correction-trigger frontier는 C7의 한계를 더 명확히 한다. 테스트한 trigger policy 중
integrity >= `0.8`과 correction rate <= `0.25`를 동시에 만족한 경우는 `0`개이며,
최고 low-correction trigger는 `selective_corrected_entropy035`로 integrity `0.653668`에
그친다. 따라서 현재 증거는 low-frequency stable sparse dynamics가 아니라 bounded
memory-safety layer다.

P2 learned correction-trigger 증거도 hard seed split에서 negative result다.
`docs/experiments/pybullet_learned_correction_trigger_results.ko.md`의 audit은 integrity
>= `0.8`과 correction rate <= `0.25`를 동시에 만족한 summary policy가 `0`개임을
보인다. 최고 learned trigger integrity는 `0.958931`이지만 correction rate가
`0.791667`이고, correction rate <= `0.25` 조건의 최고 integrity는 `0.523279`에
그친다. 따라서 단순 learned trigger가 missing fix라는 가설은 약해졌고, 다음 단계는
state-validity 및 correction objective를 포함한 stable transition training이다.

P2 stable-transition loss 증거는 partial positive이지만 해결책은 아니다.
`docs/experiments/pybullet_stable_transition_sweep_results.ko.md`의 sweep에서
`delta_norm_strong`은 raw finite-clamped integrity를 `0.633398`,
selective-correction low-disruption score를 `0.809071`까지 올리고 correction rate를
`0.598333`까지 낮춘다. 하지만 integrity >= `0.8` 및 correction rate <= `0.25`를
동시에 만족하는 row는 `0`개다. 따라서 P2에는 one-step loss weight tuning이 아니라
multi-step 또는 simulator-resynchronized transition training이 필요하다.

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

# WPU 주장 관리표

이 문서는 WPU 논문 수준의 주장을 현재 증거, 한계, 반증 조건과 연결한다. 목적은
과장 주장을 막고, 심사 대응 시 어떤 주장이 어떤 실험으로 뒷받침되는지 명확히
보여주는 것이다.
외부 공표 준비도와 미해결 gap은 `docs/publication_readiness.ko.md`를 함께 본다.
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
| C6 | large `N`은 causal working set `K`가 작고 tensorization 전에 식별될 때만 WPU에 유리하다. | 조건부 systems hypothesis로 초기 증거가 있음. | `docs/experiments/cws_balanced_branch_8m_gpu_event_conditioned_final_results.md`, `docs/experiments/wpu_v2_pre_tensor_indexed_n_sweep_results.md`, `docs/versions/wpu_v1_closure.md`. | `N`이 큰 것만으로는 충분하지 않다. retrieval이 `O(N)`에 가깝거나 causal state를 놓치면 실패한다. |
| C7 | explicit state는 propagation 이전의 working-set control을 노출한다. | 현재 가장 강한 v2 mechanism claim으로 지지되지만 dashboard 기준으로는 아직 fail gap이 남아 있다. | `docs/experiments/wpu_v2_retriever_regret_distillation_results.md`, `docs/experiments/wpu_v2_invariant_set_scorer_results.md`, `docs/experiments/wpu_v2_candidate_oracle_gap_decomposition.ko.md`, `docs/experiments/wpu_v2_candidate_noharm_gate_results.ko.md`, `docs/experiments/wpu_v2_candidate_regret_gate_results.ko.md`, `docs/experiments/wpu_v2_candidate_regret_gate_penalty_results.ko.md`, `docs/experiments/wpu_v2_candidate_regret_gate_perturbed_results.ko.md`, `docs/experiments/wpu_v2_candidate_regret_crossfit_results.ko.md`, `docs/experiments/wpu_v2_candidate_safety_gate_results.ko.md`, `docs/experiments/wpu_v2_candidate_invariant_gate_results.ko.md`, `docs/experiments/wpu_v2_candidate_joint_gate_results.ko.md`, `docs/experiments/wpu_v2_candidate_joint_gate_regression_heavy_k16_results.ko.md`, `docs/experiments/wpu_v2_end_to_end_candidate_selector_results.ko.md`, `docs/experiments/wpu_v2_candidate_safety_frontier_results.ko.md`, `docs/experiments/wpu_v2_joint_candidate_generator_results.ko.md`, `docs/experiments/wpu_v2_verified_candidate_controller_results.ko.md`, `docs/experiments/wpu_v2_joint_propagation_adapter_results.ko.md`, `docs/experiments/pybullet_branch_prior_shift_results.ko.md`, `docs/experiments/pybullet_mechanism_prior_adaptation_results.ko.md`, `docs/experiments/pybullet_prior_strength_sweep_results.ko.md`, `docs/experiments/pybullet_selected_prior_adaptation_results.ko.md`, `docs/experiments/pybullet_fewshot_mechanism_adaptation_results.ko.md`, `docs/experiments/pybullet_mechanism_adaptive_policy_results.ko.md`, `docs/experiments/pybullet_shift_detector_policy_results.ko.md`, `docs/experiments/pybullet_shift_generalization_n512_results.ko.md`, `docs/experiments/pybullet_uncertainty_gated_recompute_results.ko.md`, `docs/experiments/pybullet_learned_uncertainty_gate_results.ko.md`, `docs/experiments/pybullet_mechanism_selective_calibration_gate_results.ko.md`, `docs/experiments/pybullet_calibration_cost_frontier_results.ko.md`, `docs/experiments/pybullet_correction_trigger_frontier_results.ko.md`, `docs/experiments/wpu_v2_priority_dashboard.ko.md`. | candidate oracle은 여전히 더 강하다. Direct candidate-regret deployment는 test sweep에서 `0.329950`, train-selected deployment에서 `0.328025`까지 도달했지만 목표 `0.5`에는 못 미치고 harmful accept가 safety limit 근처에 남아 있다. Safety-frontier audit은 tradeoff를 직접 보인다. Harmful limit `0.25`에서는 direct best closure가 약 `0.327-0.330`이지만, harmful limit `0.10`에서는 direct closure `0.081898`, perturbed closure `0.154320`으로 낮아진다. Cross-fit ensemble regret gate는 train-selected overfit이 missing fix인지 검사했지만 negative result다. 최고 closure는 `0.287268`, safe best는 `0.279738`, cross-fit selected closure는 `0.270989`로 direct gate보다 낮다. Harmful-accept/ranking penalty 학습은 harmful accept를 낮추지만 closure를 `0.081253`까지 떨어뜨린다. Feature perturbation은 safe test-sweep closure를 `0.329756`까지 올리지만 train-selected deployment는 `0.312586`으로 낮아진다. 별도 safety/utility head도 negative result다. Best closure는 `0.147450`, safe best는 `0.090719`, train-selected closure는 `0.144863`에 그친다. Descriptor standardization과 group-DRO no-harm training도 단독 해결책으로는 negative result다. Best closure와 safe best는 모두 `0.110889`, train-selected closure는 `0.093863`이다. Joint object-set candidate gate도 negative result다. Best/safe closure는 `0.101454`, train-selected closure는 `0.072167`, regression-heavy K=16 ablation은 `0.034751`에 그친다. Fixed-candidate/fixed-propagator downstream-loss selector도 negative result다. Best closure는 `0.106927`, harmful accept <= `0.25`를 만족하는 deployment는 없고, train-selected closure는 `0.096833`에 그친다. Learned candidate generation은 oracle headroom(`K=16`에서 `0.361251`)을 만들지만 deployed evaluator closure는 `0.042951`에 그친다. Label-free sparse/local-dense verification signature는 best closure `0.024989`에 그치고, shallow joint propagation adapter도 best/safe closure `0.092185`, train-selected closure `0.069911`에 그친다. 이 negative probe들은 generator-only, verification-feature-only, shallow output-adapter fix를 배제한다. Branch-prior audit은 P4/P5의 failure boundary를 추가한다. `catch_heavy` majority prior accuracy는 `0.753968`이고 best WPU는 `0.408730`에 그친다. Mechanism-prior adaptation은 shifted WPU win-rate를 `0.666667`로 올리지만 shifted mean WPU ECE를 `0.024819` 악화시킨다. Prior-strength sweep은 accuracy-best 비영점 strength(`0.75`)도 `strength=0` 대비 ECE를 악화시킴을 확인한다. Calibration-selected prior strength는 shifted mean WPU ECE를 `-0.046204`, Brier를 `-0.105470` 개선하지만 shifted WPU-vs-baseline win-rate는 `0.333333`에 머문다. Few-shot mechanism adaptation은 shifted WPU-vs-baseline win-rate `1.000000`, mean margin change `0.050264`까지 도달하지만 mechanism-specific calibration sample을 사용하므로 zero-shot generalization은 아니다. Mechanism-aware adaptive policy는 adapted regime을 더 개선한다. Shifted WPU win-rate는 `1.000000`, mean accuracy change는 `0.198412`, mean margin change는 `0.058201`, mean ECE change는 `-0.099347`, mean Brier change는 `-0.155443`이다. Calibration-statistic shift detector는 base ECE와 majority-prior gap으로 같은 safe policy를 복원하고 nominal false adaptation `0`을 달성한다. 이는 mechanism-name oracle 의존을 줄인 detect-and-adapt evidence이지 zero-shot evidence는 아니다. N_bg=512 mechanism-diversity screens는 large-state failure boundary를 추가한다. Nominal-train WPU win/tie/loss는 `2/1/4`, mean margin은 `-0.047619`이고, multi-mechanism-train WPU win/tie/loss는 `2/0/5`, mean margin은 `-0.095238`이다. 이는 작은 identifiable K와 large-N sparse execution만으로 mechanism-law learning이 해결되지 않음을 보인다. Uncertainty-gated local-dense recompute는 aggregate accuracy를 `0.071428`, ECE를 `-0.016396` 개선하지만 dense recompute rate가 `0.985450`이고, low-cost gate는 ECE를 `0.005395` 악화시킨다. Learned sparse-output benefit gate는 recompute rate `0.205027`에서 source low-cost accuracy를 `0.052910` 개선하지만 ECE를 `0.010769` 악화시킨다. Calibration-cost frontier audit은 mechanism-selective routing을 포함하면 `cost_proxy <= 0.25`에서 non-reference calibration-safe policy `1`개가 있음을 보인다. Accuracy delta는 `0.029100`, ECE delta는 `-0.001652`, Brier delta는 `-0.030758`, cost는 `0.247355`다. Correction-trigger frontier audit은 integrity >= `0.8`과 correction rate <= `0.25`를 동시에 만족한 trigger policy가 `0`개임을 보인다. Cross-seed robust scoring, calibration-safe 저비용 uncertainty gating, long-horizon model-delta stability도 남아 있다. |
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

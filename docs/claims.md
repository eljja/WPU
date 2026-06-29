# WPU Claim Ledger

This ledger maps each paper-level WPU claim to the evidence that currently
supports it, the boundary of that evidence, and the result that would weaken or
falsify the claim. It is the repository's guardrail against overclaiming.
For external-release readiness and unresolved gaps, see
`docs/publication_readiness.md`.
For the compact repository-level thesis and novelty boundary, see
`docs/research_thesis.md`.
For the process-unit release audit, see `docs/process_unit_release_audit.md`.
For the current v2 priority pass/partial/fail dashboard, see
`docs/experiments/wpu_v2_priority_dashboard.md`.
For the formal objectification definition, see `docs/objectification.md`.

## Claim Status

| ID | Claim | Current status | Primary evidence | Boundary |
|---|---|---|---|---|
| C1 | Token, state, and objectified state are different operational primitives. | Supported as a framing claim. | `docs/objectification.md`, `docs/arxiv/state_is_all_you_need_en.tex`, `docs/arxiv/state_is_all_you_need_ko.md`, `wpu/core/state.py`, `wpu/memory/state_store.py`. | This is not a claim that tokens cannot encode state; it is a claim that persistent identity, role/affordance state, relation traversal, delta patching, and branch overlays are native operations only after objectification. |
| C2 | Explicit objectified world-state processing is implementable in a trainable neural model. | Supported for the synthetic object-physics prototype. | `docs/objectification.md`, `wpu/core/objectification.py`, `wpu/models/world_state_processor.py`, `wpu/models/causal_working_set_processor.py`, `wpu/data/object_physics.py`, `demos/robot_cup_demo.py`, passing test suite. | Does not prove general physical understanding, end-to-end object construction, or perception-to-state construction. |
| C3 | Sparse/hybrid/dense routing is a measurable execution regime, not just a diagram. | Supported for v1 routing instrumentation. | `wpu/engines/scheduler.py`, `docs/experiments/b_sweep_v1_results.md`, `docs/experiments/n_sweep_v1_results.md`, `docs/experiments/baseline_and_regime_results.md`. | Fixed `rho` thresholds are engineering defaults, not an optimal scheduler. |
| C4 | v1 WPU shows a real accuracy-runtime tension. | Supported. | `docs/experiments/robust_v1_results.md`, `docs/experiments/n_sweep_v1_results.md`, `docs/versions/wpu_v1_closure.md`. | WPU accuracy advantage ends before runtime advantage in v1; this is a failure boundary, not a win. |
| C5 | WPU-hybrid is robust to irrelevant relation noise in the synthetic task. | Supported for the tested stress regime. | `docs/experiments/controlled_stress_v1_results.md`. | Does not imply superiority under all state-delta or affected-background regimes. |
| C6 | Large `N` only helps WPU when causal working set `K` is small and identifiable before tensorization. | Supported as a conditional systems hypothesis with early evidence. | `docs/experiments/cws_balanced_branch_8m_gpu_event_conditioned_final_results.md`, `docs/experiments/wpu_v2_pre_tensor_indexed_n_sweep_results.md`, `docs/experiments/large_n_target_frontier_pooling_probe_results.md`, `docs/experiments/world_copy_index_probe_results.md`, `docs/experiments/world_copy_causal_index_stress_results.md`, `docs/experiments/world_copy_escalation_correction_probe_results.md`, `docs/experiments/world_copy_learned_correction_probe_results.md`, `docs/experiments/world_copy_baseline_comparison_probe_results.md`, `docs/experiments/world_copy_streaming_region_guard_probe_results.md`, `docs/experiments/world_copy_dual_index_escalation_probe_results.md`, `docs/experiments/pybullet_shift_generalization_n1024_mechanism_relation_results.md`, `docs/experiments/pybullet_shift_generalization_n2048_mechanism_relation_results.md`, `docs/experiments/pybullet_shift_generalization_n4096_mechanism_relation_results.md`, `docs/versions/wpu_v1_closure.md`. | Larger `N` alone is not sufficient; if retrieval scans `O(N)` or misses causal state, the claim fails. The updated v3 baseline-comparison and streaming screens are positive only under bounded reliable local regions. The dual-index escalation probe narrows the condition further: if missing causal objects remain in a bounded adjacent correction pool, WPU can recover sublinearly; if they are absent from all local indexes and observation pools, dense/external observation still wins raw accuracy. |
| C7 | Explicit state exposes pre-propagation working-set control. | Supported as the strongest current v2 mechanism claim, with a conditional P1-positive sub-regime and remaining larger-K gaps. | `docs/experiments/wpu_v2_retriever_regret_distillation_results.md`, `docs/experiments/wpu_v2_invariant_set_scorer_results.md`, `docs/experiments/wpu_v2_candidate_oracle_gap_decomposition.md`, `docs/experiments/wpu_v2_candidate_noharm_gate_results.md`, `docs/experiments/wpu_v2_candidate_regret_gate_results.md`, `docs/experiments/wpu_v2_candidate_regret_gate_penalty_results.md`, `docs/experiments/wpu_v2_candidate_regret_gate_perturbed_results.md`, `docs/experiments/wpu_v2_candidate_regret_crossfit_results.md`, `docs/experiments/wpu_v2_candidate_safety_gate_results.md`, `docs/experiments/wpu_v2_candidate_invariant_gate_results.md`, `docs/experiments/wpu_v2_candidate_joint_gate_results.md`, `docs/experiments/wpu_v2_candidate_joint_gate_regression_heavy_k16_results.md`, `docs/experiments/wpu_v2_end_to_end_candidate_selector_results.md`, `docs/experiments/wpu_v2_candidate_safety_frontier_results.md`, `docs/experiments/wpu_v2_joint_candidate_generator_results.md`, `docs/experiments/wpu_v2_verified_candidate_controller_results.md`, `docs/experiments/wpu_v2_joint_propagation_adapter_results.md`, `docs/experiments/wpu_v2_joint_selector_propagator_results.md`, `docs/experiments/wpu_v2_joint_selector_propagator_geometry_results.md`, `docs/experiments/wpu_v2_joint_selector_propagator_budget8_results.md`, `docs/experiments/wpu_v2_joint_selector_propagator_relation_results.md`, `docs/experiments/wpu_v2_joint_selector_propagator_pairwise_noharm_w03_results.md`, `docs/experiments/wpu_v2_joint_selector_propagator_structured_candidates_no_margin_results.md`, `docs/experiments/wpu_v2_joint_selector_propagator_score_regression_results.md`, `docs/experiments/pybullet_branch_prior_shift_results.md`, `docs/experiments/pybullet_mechanism_prior_adaptation_results.md`, `docs/experiments/pybullet_prior_strength_sweep_results.md`, `docs/experiments/pybullet_selected_prior_adaptation_results.md`, `docs/experiments/pybullet_fewshot_mechanism_adaptation_results.md`, `docs/experiments/pybullet_mechanism_adaptive_policy_results.md`, `docs/experiments/pybullet_shift_detector_policy_results.md`, `docs/experiments/pybullet_shift_generalization_n512_results.md`, `docs/experiments/pybullet_uncertainty_gated_recompute_results.md`, `docs/experiments/pybullet_learned_uncertainty_gate_results.md`, `docs/experiments/pybullet_mechanism_selective_calibration_gate_results.md`, `docs/experiments/pybullet_calibration_cost_frontier_results.md`, `docs/experiments/pybullet_correction_trigger_frontier_results.md`, `docs/experiments/wpu_v2_priority_dashboard.md`. | Earlier direct candidate-regret deployment reached only `0.328025` train-selected closure, and many post-hoc/object-set/generator/verification/adapter probes were negative. The minimal joint selector-propagator objective is the first P1-positive result: at `N=2048`, 5 held-out seeds, and `K=8`, safe closure is `0.877854` with harmful accept `0.075555`. This does not solve P1 generally. At `K=16`, unconstrained closure is only `0.171659` with harmful accept `0.444444`, and train-selected closure is `-0.026759`; at `K=32`, train-selected closure is `0.059919`. Larger-K follow-ups show that geometry/force descriptor concatenation and budget=8 are not enough, while relation-conditioned propagation improves `K=32` closure to `0.266805` but leaves harmful accept at `0.333333`. Pairwise no-harm score margins reduce harmful accept, with the best trade-off tested at weight `0.3`: `K=16` closure `0.175125` with harmful accept `0.195555`, and `K=32` closure `0.200230` with harmful accept `0.053333`. Deterministic structured candidates without no-harm margin improve K=16 confidence-selected closure to `0.241624` with harmful accept `0.200000`, but K=32 remains weak at `0.124512`; adding no-harm margin suppresses this headroom. Score-to-loss regression is also negative: best closure is `0.186333` at K=16 with harmful accept `0.435556`, and conservative best is only `0.095543` at K=32. Thus the claim is conditional: explicit state exposes a useful controllable working-set surface, but deployable oracle-gap closure still depends on smaller or easier larger-K regimes and needs learned safe candidate generation plus propagation-aware verification for broader K=16/32. Mechanism-shift, calibration, correction-trigger, and long-horizon model-delta stability also remain unsolved. |
| C8 | WPU is not yet a hardware/chiplet/IP result. | Explicitly not supported. | `docs/Review/review_response_and_differentiation.md`, `docs/paper/state_is_all_you_need.md`, `docs/experiments/pybullet_system_profile_results.md`, `docs/experiments/pybullet_matched_speedup_audit_results.md`, `docs/experiments/pybullet_matched_speedup_tolerance_results.md`, `docs/experiments/pybullet_pareto_frontier_results.md`, `docs/experiments/pybullet_system_energy_proxy_results.md`, `docs/experiments/pybullet_system_claim_boundary_results.md`, `README.md`. | CUDA random-forward profiling supports a large latency-reduction opportunity, and the screening-only energy proxy suggests where a real power study should be run. A corrected matched-or-better audit is positive at `N=133` against the best-accuracy non-WPU baseline, and the small Pareto audit puts WPU on the accuracy-latency frontier at `N=133`. The systems claim-boundary audit separates `4` supported proxy axes, `2` partial trained axes, and `1` unmeasured real-power/sparse-kernel axis; it records branch-overlay memory proxy reduction `0.874128` but weak CUDA peak-memory proxy reduction `0.304080`. WPU is still slower/dominated at `N=5`, and this is not hardware evidence. Hardware claims still require real sparse kernels, memory traffic, branch-overlay allocation, power/energy, and trained matched-or-better speedup evidence. |
| C9 | WPU propagation over objectified relations is a simplified local-causality prior, not full physics. | Supported as a bounded analogy. | `docs/objectification.md`, `docs/arxiv/state_is_all_you_need_en.tex`, `docs/arxiv/state_is_all_you_need_ko.md`, `docs/paper/state_is_all_you_need.md`, `docs/experiments/object_history_hidden_mechanism_probe_results.md`, `docs/experiments/object_relation_law_probe_results.md`, `docs/experiments/object_relation_law_ood_probe_results.md`, `docs/experiments/object_relation_law_revision_probe_results.md`. | The hidden-mechanism, local-law, OOD, and revision probes are synthetic. They show that history-derived relation variables and a simple inverse-distance law can transfer beyond nominal type names under a generated mechanism, that OOD stress can separate relation failure from law mis-specification, and that small calibration can revise some generated law shifts. Real physics competence or unknown-theory discovery requires simulator/robotics benchmarks, long-horizon stability, and evidence that learned relations generalize beyond the training generator. |
| C10 | Near-term WPU value is more plausible as software runtime/middleware than silicon. | Plausible direction, not experimentally proven. | `docs/reproducibility.md`, `docs/arxiv/README.md`, current PyTorch package under `wpu/`. | Requires digital-twin, simulation backend, game/server, or robotics middleware benchmarks. |
| C11 | Objectification quality is measurable and locally repairable as a contract before propagation. | Supported as an implementation claim. | `wpu/core/objectification.py`, `tests/test_objectification.py`, `tests/test_script_entrypoints.py`, `docs/experiments/objectification_relation_repair_probe_results.md`, `docs/experiments/pybullet_objectification_quality_results.md`, `docs/experiments/pybullet_objectification_loss_coupling_results.md`, `README.md`, `docs/objectification.md`. | Relation repair and `LocalLawHypothesis` produce conservative hypotheses and revision reports, not ground-truth physics. The latest probes show that learned repair can transfer across aliased type names, improve a toy downstream diagnostic, and report law-revision gaps. The PyBullet loss-coupling audit links selected-K/frontier degradation to MSE increase, but branch-accuracy movement remains small; this is not evidence that perception-to-object construction or unknown-theory discovery is solved. |

Latest C7 large-state correction: the original N_bg=512 mechanism-diversity
screens remain valid failure-boundary evidence, but they should no longer be
read alone. The current interpretation is that preserving action-conditioned
event state and physical object-state scalars recovers the nominal-train
large-N shift screen to `4/0/3` with mean margin `+0.002976`, while
multi-mechanism training remains mixed/negative at `2/2/3` with mean margin
`-0.032738`. This narrows the claim: WPU's large-state advantage requires
small identifiable `K`, faithful object/action state tensorization, and
mechanism-aware propagation or adaptation.

Latest C6 sparse-feasibility boundary: the N_bg=4096 mechanism-relation run
extends the large-state stress to total `N=4101`, but only as a WPU sparse
feasibility result. It supports the systems claim that indexed state can avoid
full recompute when the selected working set stays small. It does not establish
accuracy superiority because matched dense/token/graph baselines were not
completed under the same protocol.

Latest C6/C7 large-N readout update:
`docs/experiments/large_n_target_frontier_pooling_probe_results.md` isolates one
mechanism behind the v1 `N>=204` branch collapse: global mean branch readout
dilutes event-causal state with non-causal objects. With identical 3-seed,
80-step training at total `N=404`, original `wpu-sparse` reaches mean branch
accuracy `0.359375`, while `wpu-sparse-frontier` reaches `0.781250` and
slightly exceeds `serialized-token` at `0.778646` using work proxy `3` versus
`166464`. This supports event target/frontier readout as a state-native fix for
non-causal background scaling. It does not solve cases where causal `K` grows,
relations are wrong, or mechanism variables are missing.

Latest C6 v3 world-copy index update:
`docs/experiments/world_copy_index_probe_results.md` adds a hierarchical
world-copy substrate and `WorldCausalIndex`. In the controlled probe, selected
`K` remains `4` while total `N` grows from `104` to `10004`; no non-causal
background objects are selected, and the affected fraction at `N=10004` is
`0.00039984`. This supports pre-tensor causal-slice retrieval at large `N`.
It does not establish trained accuracy, noisy objectification robustness, or
real-world physical understanding.

Latest C6 v3 noisy-index update:
`docs/experiments/world_copy_causal_index_stress_results.md` adds the first P1
stress benchmark for noisy causal retrieval. It sweeps `N=128..8192`,
`K_ref=4/8/16`, missing relation rates `0/0.25/0.5`, and false-positive
relation rates `0/0.1/0.25`, relation confidence thresholds `0/0.3`, and true
relation confidence `0.95/0.2`. In this controlled region-scoped setup, recall
stays `1.000000` and the `N=8192` touch ratio stays below `0.004385`. Without
the confidence gate, false-positive relations reduce mean precision to
`0.800000`; with `min_relation_confidence=0.3`, mean precision returns to
`1.000000` while recall stays `1.000000`. When true causal relation confidence
is lowered to `0.2`, region scope recovers recall and precision, but mean
escalation reaches `0.981481`. This strengthens the index scalability claim and
moves the next problem to local dense/hybrid correction after uncertainty
escalation.

Latest C6 v3 escalation-correction update:
`docs/experiments/world_copy_escalation_correction_probe_results.md` tests that
next boundary at substrate level. With true relation confidence `0.2`,
`sparse_confident_relations` reaches mean recall `0.145833`, precision
`1.000000`, and F1 `0.246623`; `hybrid_escalation_region` reaches mean
recall/precision/F1 `1.000000` while keeping max selected `K=16` and max touch
ratio `0.24489796`. This supports local correction candidates after escalation.
It does not prove learned transition accuracy or token/graph superiority.

Latest C6/P2 v3 learned-correction update:
`docs/experiments/world_copy_learned_correction_probe_results.md` tests whether
the recovered local correction candidates can improve a learned delta update. A
small relation/state-conditioned MLP is trained over local candidates. With true
relation confidence `0.2`, `sparse_confident_relations` leaves mean delta MSE
`0.275312`; `hybrid_escalation_region` reduces mean delta MSE to `0.006365`
while keeping max selected `K=16` and max touch ratio `0.24489796`. This is a
controlled substrate positive for learned local correction, not a token/graph
baseline victory or long-horizon world-copy result.

Latest C6/P2 v3 baseline-comparison update:
`docs/experiments/world_copy_baseline_comparison_probe_results.md` now includes
a bounded local-region guard. The guard keeps max selected `K=16`, mean work
proxy `9.333333`, and mean bytes proxy `336.000000`, while reaching raw delta
MSE `0.002646`. This is lower than dense graph `0.003810` and serialized token
`0.003223` in this controlled same-task screen. The negative result is also
important: `wpu-hybrid-context` has MSE `0.020904`, so shallow context
concatenation does not solve the gap. The positive claim is therefore specific:
bounded reliable local regions can close missing-relation gaps without
full-state recompute.

Latest C6/P3/P4 v3 streaming update:
`docs/experiments/world_copy_streaming_region_guard_probe_results.md` extends
the region-guard result to H=25 controlled streams with object churn and region
migration. `wpu-region-guard` keeps max selected `K=8`, trajectory MSE
`0.000000`, integrity `1.000000`, correction cost `0.000000`, work proxy
`8.000000`, and bytes proxy `288.000000`. Dense state copy matches integrity
but uses full-state work/bytes that grow with `N`; relation-frontier-only WPU
requires correction rate around `0.416667`. This is controlled oracle-law
evidence, not real simulator or learned-transition evidence.

Latest C7 P1 verification-context update:
`docs/experiments/wpu_v2_joint_selector_propagator_verification_context_results.md`
adds label-free propagation signatures to the joint selector-propagator
selector. It is a useful but insufficient larger-K improvement. At `K=16`,
unconstrained closure rises to `0.409420` but harmful accept remains
`0.342222`; confidence-selected closure is `0.269216` with harmful accept
`0.115555`. At `K=32`, confidence-selected closure is `0.153386`. This supports
propagation-aware verification as a direction, but it does not yet solve safe
candidate generation.

Latest C7 P1 verification-head update:
`docs/experiments/wpu_v2_joint_selector_propagator_verification_head_results.md`
adds an explicit harmful-candidate head on top of the label-free verification
signatures. It is negative as a standalone fix. At `K=16`, unconstrained closure
falls to `0.345395` with harmful accept `0.391111`; confidence-selected closure
falls to `0.193197` with harmful accept `0.102222`. At `K=32`,
confidence-selected closure is only `0.060597`. This narrows the next P1 target
away from stronger rejection heads and toward learned safe candidate generation
with propagation-aware supervision.

Latest C7 P1 learned-safe-candidate update:
`docs/experiments/wpu_v2_joint_selector_propagator_learned_safe_candidates_results.md`
adds train-fold object-level generators that imitate interaction, proximity,
density, and axis teachers. It is not the missing larger-K fix. At `K=16`,
confidence-selected closure is `0.246071` with harmful accept `0.222222`; at
`K=32`, confidence-selected closure is `0.143398`. This is weaker than the
verification-context result, so the next candidate-generation target must learn
from propagation loss and no-harm transfer directly rather than only imitating
hand-built teachers.

Latest route-state contract correction: `docs/experiments/wpu_v2_route_physics_contract_smoke_results.md`
records a smaller implementation fix. The adaptive route-regret context now
receives pair geometry, target physical scalars, selected-set physical scalars,
`force`, and `catch_action`; previously it compressed this context to pair
distances, target xy, and event norm. This is not a solved P1 result, but it
prevents follow-up route experiments from silently discarding mechanism-defining
state variables.
The full 5-seed staged-regret rerun in
`docs/experiments/wpu_v2_regret_router_variant_results.md` confirms the boundary:
expanded physical/action context is neutral by simple concatenation
(`physics_hidden` routed loss `0.962987` versus internal `0.962894`), while
`state_only` remains worse (`0.982804`). Structured verification or joint route
training is still required.
`docs/experiments/pybullet_route_regret_training_smoke_results.md` then wires the
same explicit route-regret training into the PyBullet mechanism-shift path. It is
infrastructure evidence only: the tiny smoke run shows that route metrics are
emitted and that a configurable threshold is necessary to avoid all-dense
routing collapse. The follow-up selected-threshold smoke avoids trivial
all-dense/all-sparse endpoints, but does not improve shifted accuracy in the
tiny run.
`docs/experiments/pybullet_shift_generalization_n512_route_regret_selected_results.md`
extends this to a 3-seed N_bg=512 mechanism screen. The result is mixed/negative:
selected route-regret WPU uses low dense compute (`0.071429`) and helps on
`no_catch`, but best-WPU versus best-baseline win/tie/loss is only `2/1/4` and
graph-transformer retains higher macro accuracy.
`docs/experiments/pybullet_shift_generalization_n512_route_regret_adapted_screen_results.md`
adds matched mechanism-prior adaptation on four shifted mechanisms. It is also
negative for the route-regret WPU: win/tie/loss is `0/0/4` versus best baseline.
This narrows the next step from prior-bias adaptation to mechanism-conditioned
propagation dynamics.
`docs/experiments/pybullet_shift_generalization_n512_mechanism_conditioned_screen_results.md`
is the first positive follow-up in that direction. It conditions sparse
propagation directly on explicit physics/action context and disables dense
fallback. On the 3-seed N_bg=512 screen, mechanism-conditioned WPU reaches macro
accuracy `0.541667` versus `0.500000` for the best non-WPU baseline, with dense
compute `0.000000` and win/tie/loss `1/2/1`. This is a promising screen, not a
solved claim: `edge_shift` remains negative, and the result must be expanded to
larger seed/mechanism sweeps before being used as strong evidence.
`docs/experiments/pybullet_shift_generalization_n512_mechanism_adapter_multitrain_results.md`
then clarifies the boundary. The nominal-only 5-seed/7-mechanism expansion is
negative (`0.433333` macro accuracy versus `0.476190` best baseline), and an
object-wise adapter is also negative under nominal-only training. However, when
the adapter is trained on primitive mechanisms, it reaches 5-seed N_bg=512 macro
accuracy `0.497143` versus `0.472857` for the best baseline, with dense compute
`0.000000` and win/tie/loss `3/1/3`. This supports a narrower claim: WPU needs
object-wise mechanism-conditioned propagation trained on primitive mechanism
variation; large-N sparse state alone and nominal-only zero-shot extrapolation
are insufficient.
`docs/experiments/pybullet_shift_generalization_n512_mechanism_factorized_shuffled_results.md`
then corrects the multi-mechanism training protocol. The previous
`ConcatDataset` training loader was not shuffled, so small step budgets could be
mechanism-order sensitive. After adding seed-fixed training shuffle, a
factorized sparse mechanism adapter is negative at 5 seeds: macro accuracy is
`0.497143` versus graph-transformer `0.548571`, dense compute remains
`0.000000`, and win/tie/loss is `2/1/4`. This downgrades the previous
multi-mechanism positive to a screen and strengthens the failure boundary:
robust local-law composition, especially edge-conditioned composition, still
requires explicit composition supervision.
`docs/experiments/pybullet_shift_generalization_n512_target_local_loss_results.md`
then audits the most direct form of that supervision: target-local delta MSE on
the event target object. This exposes a real large-N loss-alignment problem, but
it is also a negative standalone fix. At target-local weight `1.0`, WPU target
MSE improves relative to lower weights, but macro branch accuracy drops to
`0.418571` versus `0.494286` for graph-transformer in the matched run. Lower
weights `0.25` and `0.5` also do not recover the branch-composition gap. The
updated boundary is therefore sharper: the next WPU improvement must change the
transition dynamics, for example branch-conditioned or mechanism-specific local
propagation, not merely reweight the delta loss.
`docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_results.md`
implements that architectural change. The new
`wpu-cws-indexed-mechanism-branch` model preserves indexed sparse execution and
zero dense fallback, but adds a mechanism-conditioned branch transition head. On
the 5-seed N_bg=512 shuffled multi-mechanism screen, it reaches macro branch
accuracy `0.568571` versus `0.548571` for graph-transformer, ECE `0.247101`
versus `0.254194`, dense compute `0.000000`, and win/tie/loss `4/0/3`. This is
the first positive follow-up after the corrected factorized and target-local
negative diagnostics. It is still not broad superiority: `catch_heavy`,
`edge_shift`, and `edge_high_force` remain below the best dense baseline.
`docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_stress_results.md`
then stress-tests that positive screen. After adding explicit
`--train-samples-per-mechanism` control, the larger train/eval pilot is negative
for WPU accuracy: h32 WPU reaches `0.534524` versus `0.598810` for the best
baseline, and a fair h64 capacity check gives WPU `0.603571` versus
serialized-token `0.622619`. Dense compute remains `0.000000`, so the efficiency
claim survives, but the accuracy claim must be downgraded to a short-budget
screen until transition-head expressivity improves.
`docs/experiments/pybullet_shift_generalization_n512_branch_expert_results.md`
tests the first expressivity fix: branch-specific output experts. It is a
negative standalone result. The expert model reaches macro accuracy `0.505952`
under the h32 stress protocol, below the prior mechanism-branch head
(`0.534524`) and graph-transformer (`0.598810`). It improves some edge/catch
composition cases but loses general mechanism accuracy. The next architecture
step should therefore move below branch logits into relation-type-conditioned
sparse propagation messages.
`docs/experiments/pybullet_shift_generalization_n512_mechanism_relation_results.md`
implements that propagation-level change. The new
`wpu-cws-indexed-mechanism-relation` route scatters learned messages across
selected working-set relations using source/target hidden states, relation
features, and route physics features. Under the h32 trainpool40/steps16/eval40
stress protocol, the 5-seed expansion reaches macro accuracy `0.639286` versus
`0.597143` for graph-transformer, with dense compute `0.000000` and
win/tie/loss `5/0/2` against the best baseline. The 3-seed h64 fair-capacity
check reaches `0.678571` versus `0.622619` for serialized-token. A larger
5-seed N=1029 distractor screen is also positive: WPU reaches `0.639286` versus
`0.577143` for graph-transformer, with dense compute `0.000000` and
win/tie/loss `6/0/1`. The N=2053 3-seed distractor screen further strengthens
the scaling trend: WPU reaches `0.644048` versus `0.516667` for
graph-transformer, with dense compute `0.000000` and win/tie/loss `7/0/0`.
This is the
strongest current WPU v2 evidence, but it is still PyBullet synthetic,
single-step, and non-causal-distractor-bounded; calibration-aware, rollout, and
harder causal large-N expansion remain required.
The relation-conditioned closed-loop rollout diagnostic sharpens that boundary.
In the latest N=517 audit, raw relation WPU has H=25 integrity `0.250368`,
trajectory MSE `6.975125`, and branch accuracy `0.208333` despite selected
`K = 4.354167`; finite projection lifts H=25 integrity to `0.876760` but keeps
trajectory MSE high (`1.695024`) and branch accuracy low (`0.250000`). This is a
safety-guard result, not learned long-horizon dynamics. Earlier scalar
learned-stability ablations were negative: delta-norm regularization, fixed
temporal scaling, short-stride simulator targets, and explicit multi-horizon
targets did not prevent H=25 collapse. The first raw-model positive result is bounded delta parameterization
inside the transition head. H=25 integrity rises to `0.593354` at bound `0.5`,
`0.652870` at `0.25`, `0.793182` at `0.1`, and `0.870264` at `0.05`, while
selected `K` stays `4.354167` and no correction, rollback, rejection, or dense
fallback is used. This changes P2 from a pure failure boundary into a partial
raw-stability result. Simulator-resynchronized metrics now reduce the
under-update concern: bound `0.05` reaches H=25 trajectory MSE `0.707117` and
branch accuracy `0.729167`, versus finite projection trajectory MSE `1.695024`
and branch accuracy `0.250000`. The remaining gap is high target-object
trajectory MSE (`361.358309`). Follow-up attempts with learned adaptive bounds,
manual position/velocity split bounds, and target-object delta loss do not
reduce that bottleneck: they either leave target-object MSE near `361-363` or
reduce branch accuracy. Full recurrent unrolled loss was non-finite in the
initial probe, and stabilized truncated H=2/4 unroll is neutral relative to the
same-lr bounded-only baseline. A branch-weighted target-local transition head is
the first positive follow-up. In the preferred 5-seed matched comparison at
H=25 it improves branch accuracy from `0.612500` to `0.675000`, trajectory MSE
from `1.063341` to `1.061493`, and target-object MSE from `544.710934` to
`543.834596`. This supports the transition-head direction, but the improvement
is small and state-integrity falls from `0.868758` to `0.850574`, so
high-fidelity dynamics and validity-preserving target dynamics are not solved.
A channel-masked constrained target-head follow-up is also negative as a
standalone fix: H=25 branch accuracy is `0.650000`, target-object MSE is
`544.588713`, and integrity is `0.846955`. The next P2 target is therefore
state-validity-aware transition learning, not only residual channel masking.

P1 candidate-generation evidence is now explicitly negative as a standalone fix.
The joint candidate-generator probe shows that learned generated candidates can
create oracle headroom, with learned-generator oracle closure reaching
`0.361251` at `K=16`, but the deployed set evaluator reaches only `0.042951`.
This means WPU v2 cannot claim solved working-set selection from candidate
generation alone; it needs joint candidate generation, retrieval, propagation
verification, and calibrated no-harm training. See
`docs/experiments/wpu_v2_joint_candidate_generator_results.md`.

P1 verification-feature evidence is also negative as a standalone fix. The
verified candidate-controller probe appends label-free sparse/local-dense
propagation signatures to candidate descriptors, but reaches only `0.024989`
best closure, `0.023029` safe-best closure, and `0.024989` train-selected
closure. This means verification must be trained jointly with retrieval and
propagation dynamics rather than appended as a post-hoc selector feature. See
`docs/experiments/wpu_v2_verified_candidate_controller_results.md`.

P1 shallow propagation-adapter evidence is also negative. A candidate-aware
branch-logit adapter trained from sparse/local-dense verification features
reaches `0.092185` best/safe closure and `0.069911` train-selected closure.
This rules out a small output adapter as the missing fix; the remaining target
is deeper joint training of retrieval, candidate generation, propagation
dynamics, propagation verification, and calibrated no-harm rejection. See
`docs/experiments/wpu_v2_joint_propagation_adapter_results.md`.

P1 joint utility-verifier evidence is also negative. A verifier that combines
candidate object-set tensors, sparse/local-dense verification signatures,
uncertainty, and no-harm safety reaches only `0.097845` best/safe closure and
`0.077781` train-selected closure. This rules out fixed-propagator utility and
safety heads as the missing fix; the remaining target is end-to-end coupling of
candidate generation, retrieval, propagation dynamics, verification, and
calibrated no-harm rejection. See
`docs/experiments/wpu_v2_joint_utility_verifier_results.md`.

P3 large-N simulator evidence is stronger but still bounded. The medium-training
N_bg=256 run is baseline-complete at total `N=261`: best WPU accuracy is
`0.466667`, best baseline accuracy is `0.450000`, and best WPU is `60.629526x`
faster than that best-accuracy baseline. This supports a conditional large-N
state-native regime, but the margin is small and the task is still one cup
family, so it does not establish broad simulator superiority. A higher-budget
5-seed N_bg=512 baseline-complete run now reaches total `N=517`: best WPU
accuracy is `0.433333`, best baseline accuracy is `0.425000`, and best WPU is
`57.595711x` faster than that best-accuracy baseline. The WPU edge persists
under a larger training/evaluation budget, but the margin shrinks, so this
should still be cited as conditional large-N evidence rather than broad
simulator superiority.

P3/P4 large-state mechanism diversity is now a sharper claim-boundary result.
The original N_bg=512 mechanism-diversity screens at total `N=517` were negative:
nominal-train WPU win/tie/loss was `2/1/4` with mean margin `-0.047619`, and
multi-mechanism-train was `2/0/5` with mean margin `-0.095238`. That audit
exposed a state-input contract bug: `catch_action` and physical object-state
scalars were present in the objectified PyBullet state but omitted during
tensorization. After preserving `catch_action`, `edge_distance`, `hand_distance`,
`fall_risk`, and `angular_speed`, the nominal-train screen recovers to `4/0/3`
with mean margin `+0.002976`. The multi-mechanism screen remains mixed/negative
at `2/2/3` with mean margin `-0.032738`. This does not weaken the systems claim
that WPU can avoid full-state tensorization; it weakens any accuracy claim that
large N, small K, or mechanism diversity alone is sufficient. Mechanism-aware
propagation and adaptation remain required.

P2 learned correction-trigger evidence is also negative on the hard seed split.
The audit in `docs/experiments/pybullet_learned_correction_trigger_results.md`
finds `0` summary policies meeting integrity >= `0.8` and correction rate <=
`0.25`. The best learned trigger reaches integrity `0.958931`, but only at
correction rate `0.791667`; the best policy under correction rate <= `0.25`
reaches integrity `0.523279`. This rules out a simple learned trigger as the
missing fix and points to stable transition training with state-validity and
correction objectives.

P2 stable-transition loss evidence is a partial positive, not a solution. The
sweep in `docs/experiments/pybullet_stable_transition_sweep_results.md` shows
that `delta_norm_strong` raises raw finite-clamped integrity to `0.633398`,
raises selective-correction low-disruption score to `0.809071`, and lowers
correction rate to `0.598333`. However, it still finds `0` rows meeting
integrity >= `0.8` and correction rate <= `0.25`, so P2 needs multi-step or
simulator-resynchronized transition training rather than more one-step loss
weight tuning.

## Falsification Tests

- If serialized-token or graph baselines match WPU at equal compute across
  controlled identity, locality, and branching benchmarks, WPU is not a
  necessary primitive for those regimes.
- If pre-tensor retrieval cost grows close to `O(N)` in realistic state stores,
  the large-`N` systems claim collapses.
- If risk-adjusted mechanism selection loses its held-out-seed gain under
  larger seed/model sweeps, the current v2 working-set-control claim must be
  downgraded to a diagnostic result.
- If sample-level no-harm gates continue to show negative closure under
  held-out seeds, candidate selection must be supervised by calibrated
  candidate-regret and uncertainty targets rather than deployment thresholds
  alone.
- If direct candidate-regret gates improve average loss but keep high harmful
  accept rates, P1 remains unsolved until the selector can reject harmful
  candidates across K and held-out seeds.
- If cross-fit or ensemble candidate-regret gates reduce in-sample optimism but
  lower closure, the P1 bottleneck is not only deployment-threshold selection;
  the candidate score itself must become more invariant and jointly trained
  with retrieval and propagation.
- If sparse routing harms accuracy before producing meaningful latency or
  memory savings, propagation is not the right central operation for that
  workload.
- If geometry-derived relation repair adds spurious edges that reduce
  downstream prediction or working-set precision, repair must be gated or
  replaced by learned candidate scoring.
- If learned relation scoring only works for nominal type labels and fails when
  role/affordance variables are preserved under aliasing, objectification has
  not escaped brittle type classification.
- If delta overlays accumulate unrecoverable state corruption in long-horizon
  rollout, persistent state becomes a liability without verification and
  rollback mechanisms.
- If finite-safe or selective correction must still trigger on most sparse
  updates, the memory layer can protect applied state but the raw transition
  model is still not stable. The correction-trigger frontier currently confirms
  this boundary: `0` tested trigger policies meet integrity >= `0.8` with
  correction rate <= `0.25`. Stronger claims require learned low-frequency
  correction, not only shrinking the corrected object set or eliminating
  rollback.

## Submission Posture

The defensible paper posture is:

```text
WPU is a state-native execution model whose advantages are regime-specific:
large explicit state, small identifiable causal working set, local relation
propagation, branchable uncertainty, and state reuse across events.
```

The paper should not claim:

- universal superiority over token, graph, or latent world models;
- real-world physical understanding;
- end-to-end perception-to-state construction;
- hardware-level speed or energy advantage;
- a closed candidate-oracle gap.

Region locality alone is insufficient. The selective-region stress probe bounds
K at 16 through N=8192 under heavy contamination and reduces state corruption,
but it does not remove causal-recall loss under dual index omission.

# WPU Claim Ledger

This ledger maps each paper-level WPU claim to the evidence that currently
supports it, the boundary of that evidence, and the result that would weaken or
falsify the claim. It is the repository's guardrail against overclaiming.
For external-release readiness and unresolved gaps, see
`docs/publication_readiness.md`.
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
| C6 | Large `N` only helps WPU when causal working set `K` is small and identifiable before tensorization. | Supported as a conditional systems hypothesis with early evidence. | `docs/experiments/cws_balanced_branch_8m_gpu_event_conditioned_final_results.md`, `docs/experiments/wpu_v2_pre_tensor_indexed_n_sweep_results.md`, `docs/versions/wpu_v1_closure.md`. | Larger `N` alone is not sufficient; if retrieval scans `O(N)` or misses causal state, the claim fails. |
| C7 | Explicit state exposes pre-propagation working-set control. | Supported as the strongest current v2 mechanism claim, but the dashboard still records failing gaps. | `docs/experiments/wpu_v2_retriever_regret_distillation_results.md`, `docs/experiments/wpu_v2_invariant_set_scorer_results.md`, `docs/experiments/wpu_v2_candidate_oracle_gap_decomposition.md`, `docs/experiments/wpu_v2_candidate_noharm_gate_results.md`, `docs/experiments/wpu_v2_candidate_regret_gate_results.md`, `docs/experiments/wpu_v2_candidate_regret_gate_penalty_results.md`, `docs/experiments/wpu_v2_candidate_regret_gate_perturbed_results.md`, `docs/experiments/wpu_v2_candidate_regret_crossfit_results.md`, `docs/experiments/wpu_v2_candidate_safety_gate_results.md`, `docs/experiments/wpu_v2_candidate_invariant_gate_results.md`, `docs/experiments/wpu_v2_candidate_joint_gate_results.md`, `docs/experiments/wpu_v2_candidate_joint_gate_regression_heavy_k16_results.md`, `docs/experiments/wpu_v2_end_to_end_candidate_selector_results.md`, `docs/experiments/wpu_v2_candidate_safety_frontier_results.md`, `docs/experiments/wpu_v2_joint_candidate_generator_results.md`, `docs/experiments/wpu_v2_verified_candidate_controller_results.md`, `docs/experiments/wpu_v2_joint_propagation_adapter_results.md`, `docs/experiments/pybullet_branch_prior_shift_results.md`, `docs/experiments/pybullet_mechanism_prior_adaptation_results.md`, `docs/experiments/pybullet_prior_strength_sweep_results.md`, `docs/experiments/pybullet_selected_prior_adaptation_results.md`, `docs/experiments/pybullet_fewshot_mechanism_adaptation_results.md`, `docs/experiments/pybullet_mechanism_adaptive_policy_results.md`, `docs/experiments/pybullet_shift_detector_policy_results.md`, `docs/experiments/pybullet_shift_generalization_n512_results.md`, `docs/experiments/pybullet_uncertainty_gated_recompute_results.md`, `docs/experiments/pybullet_learned_uncertainty_gate_results.md`, `docs/experiments/pybullet_mechanism_selective_calibration_gate_results.md`, `docs/experiments/pybullet_calibration_cost_frontier_results.md`, `docs/experiments/pybullet_correction_trigger_frontier_results.md`, `docs/experiments/wpu_v2_priority_dashboard.md`. | Candidate oracle remains better; direct candidate-regret deployment reaches `0.329950` in the test sweep and `0.328025` under train-selected deployment, but this remains below the `0.5` target and harmful accepts remain near the safety limit. The safety-frontier audit shows the tradeoff directly: at harmful limit `0.25`, best direct closure is about `0.327-0.330`, while at harmful limit `0.10`, closure drops to `0.081898` for direct and `0.154320` for perturbed. Cross-fit ensemble regret gating tests whether train-selected overfit is the missing fix; it is negative, with best closure `0.287268`, safe best `0.279738`, and cross-fit selected closure `0.270989`. Harmful-accept/ranking penalty training lowers harmful accepts but collapses closure to `0.081253`; feature perturbation raises safe test-sweep closure to `0.329756` but lowers train-selected deployment to `0.312586`. A separate safety/utility head is also negative: best closure is `0.147450`, safe best is `0.090719`, and train-selected closure is `0.144863`. Descriptor standardization plus group-DRO no-harm training is negative as a standalone fix: best closure and safe best are both `0.110889`, and train-selected closure is `0.093863`. Joint object-set candidate gating is also negative: best/safe closure is `0.101454`, train-selected closure is `0.072167`, and a regression-heavy K=16 ablation reaches only `0.034751`. Fixed-candidate/fixed-propagator downstream-loss selector training is also negative: best closure is `0.106927`, no deployment satisfies harmful accept <= `0.25`, and train-selected closure is `0.096833`. Learned candidate generation creates oracle headroom (`0.361251` at `K=16`) but deployed evaluator closure is only `0.042951`; label-free sparse/local-dense verification signatures reach only `0.024989` best closure; and the shallow joint propagation adapter reaches only `0.092185` best/safe closure and `0.069911` train-selected closure. These negative probes rule out generator-only, verification-feature-only, and shallow output-adapter fixes. The branch-prior audit adds a P4/P5 failure boundary: `catch_heavy` majority prior accuracy is `0.753968`, while best WPU reaches `0.408730`. Mechanism-prior adaptation raises shifted WPU win-rate to `0.666667`, but worsens shifted mean WPU ECE by `0.024819`. Prior-strength sweep confirms that the accuracy-best nonzero strength (`0.75`) still increases ECE relative to `strength=0`. Calibration-selected prior strength improves shifted mean WPU ECE by `-0.046204` and Brier by `-0.105470`, but leaves shifted WPU-vs-baseline win-rate at `0.333333`. Few-shot mechanism adaptation reaches shifted WPU-vs-baseline win-rate `1.000000` with mean margin change `0.050264`, but it uses mechanism-specific calibration samples and is not zero-shot generalization. A mechanism-aware adaptive policy improves the adapted regime further: shifted WPU win-rate is `1.000000`, mean accuracy change is `0.198412`, mean margin change is `0.058201`, mean ECE change is `-0.099347`, and mean Brier change is `-0.155443`; a calibration-statistic shift detector recovers the same safe policy from base ECE and majority-prior gap with nominal false adaptation `0`. This is stronger detect-and-adapt evidence with less mechanism-name oracle dependence, not zero-shot evidence. The N_bg=512 mechanism-diversity screens add a large-state failure boundary: nominal-train WPU win/tie/loss is `2/1/4` with mean margin `-0.047619`, and multi-mechanism-train WPU win/tie/loss is `2/0/5` with mean margin `-0.095238`. This shows that small identifiable K and large-N sparse execution do not by themselves solve mechanism-law learning. Uncertainty-gated local-dense recompute improves aggregate accuracy by `0.071428` and ECE by `-0.016396`, but only at dense recompute rate `0.985450`; the low-cost gate worsens ECE by `0.005395`. Learned sparse-output benefit gating improves source low-cost accuracy by `0.052910` at recompute rate `0.205027`, but worsens ECE by `0.010769`. Calibration-cost frontier auditing now finds `1` mechanism-selective non-reference calibration-safe policy under `cost_proxy <= 0.25`: accuracy delta `0.029100`, ECE delta `-0.001652`, Brier delta `-0.030758`, and cost `0.247355`. Correction-trigger frontier auditing finds `0` trigger policies meeting integrity >= `0.8` and correction rate <= `0.25`; the best low-correction trigger is `selective_corrected_entropy035` with integrity `0.653668`. Cross-seed robust scoring, calibration-safe low-cost uncertainty gating, and long-horizon model-delta stability are also not solved. |
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

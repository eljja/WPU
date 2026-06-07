# WPU Experiment Reports

This directory contains the experiment evidence behind the current WPU research
paper and the v1-to-v2 evidence trail. The paper claim is intentionally
regime-specific: WPU is not shown to be universally superior to token or graph
baselines.

For new experiments, objectified state is now a measured input contract. When a
result depends on explicit objects, sparse retrieval, or a perception/state
adapter, log `ObjectificationReport` fields alongside accuracy, loss, latency,
work proxy, calibration, and route statistics. This separates propagation
failure from objectification failure.

When relation repair is used, log repaired edge count, repaired relation types,
repair precision/recall when ground truth is available, and downstream loss
with and without repair. Repaired edges are hypotheses for frontier recovery,
not ground-truth physical relations.

## V2 Priority Dashboard

Use `wpu_v2_priority_dashboard.md` as the conservative status summary for the
current v2 program. It recomputes priorities 1-7 from committed CSV evidence and
marks unresolved items explicitly. The current dashboard is intentionally not a
success-only table: candidate-oracle gap closure and long-horizon state
integrity remain failing criteria, while simulator grounding, shift
generalization, calibration, systems profiling, and objectification quality are
partial.

Reproduce it with:

```bash
python scripts/audit_v2_priority_dashboard.py
```

## Current Evidence Hierarchy

Use these reports for paper-level claims:

- `wpu_v2_priority_dashboard.md` and `wpu_v2_priority_dashboard.ko.md`:
  conservative cross-experiment dashboard for v2 priorities 1-7. It is the
  fastest way to see which claims are ready, partial, or still failing.
- `objectification_relation_repair_probe_results.md`: objectification repair
  probe showing that geometry-derived relation hypotheses can recover a missing
  sparse frontier, while type-aware objectification is needed to avoid
  distractor-induced spurious edges. The shifted probe separates nominal type
  labels from role-bearing object state: a learned relation scorer transfers
  across aliased type names when role/affordance variables are preserved, but it
  fails when both type and role information are removed. The same probe now
  includes a toy downstream branch diagnostic: role-aware learned repair improves
  aliased-type branch accuracy from `0.343750` to `0.671875` and lowers loss
  from `1.319667` to `0.885275`, while ungated dense-distractor repair worsens
  loss despite frontier recall `1.000000`.
- `object_history_hidden_mechanism_probe_results.md`: toy hidden-mechanism probe
  showing that relation candidates learned from object histories can transfer
  beyond nominal type names. A history scorer trained on `contact_transfer` and
  `support_transfer` reaches five-seed mean relation precision/recall `0.987500`
  on held-out `hidden_field`, improving downstream accuracy from `0.494531` to
  `0.992188`.
- `object_relation_law_probe_results.md`: toy local-law probe showing the next
  step after relation discovery. A history-derived relation selector plus an
  interpretable inverse-distance law transfers to renamed held-out
  `hidden_inverse`, reaching five-seed mean relation precision/recall `0.988281`
  and delta MSE `0.000828` versus `0.445909` for no relation or type prior. This
  is still synthetic and does not establish real physical-law discovery.
- `object_relation_law_ood_probe_results.md`: OOD stress version of the local-law
  probe. It shows that objectified relations remain useful under distance, gain,
  and denominator shifts, but also exposes the right failure boundaries:
  far-distance relation recall drops to `0.658594`, and gain/law-form shifts
  leave residual MSE even with oracle relations.
- `object_relation_law_revision_probe_results.md`: revision follow-up showing
  that small calibration sets can reduce OOD law residuals over objectified
  relations. Gain calibration cuts `hidden_inverse_gain_shift` MSE from
  `0.115978` to `0.000342`; form revision cuts `hidden_power_shift` MSE from
  `0.054596` to `0.008887`, while the oracle relation result `0.000232` exposes
  the remaining relation-selection gap.
- `pybullet_cup_benchmark_results.md`: first simulator-grounded benchmark.
  PyBullet rigid-body rollouts are converted into explicit `WorldState`
  objects. The initial two-seed result supports a limited systems claim:
  pre-tensor indexed WPU keeps `K ~= 4-5` as background state grows, while the
  full-state graph transformer slows sharply. It does not yet support accuracy
  dominance over token or graph baselines.
- `pybullet_objectification_stress_results.md`: first perception-like
  robustness test on simulator-derived state. Clean-trained models are evaluated
  on corrupted objectified state. The result exposes a missing metric: current
  objectification score detects confidence degradation but not missing expected
  causal frontier edges or semantic identity swaps; the benchmark therefore logs
  pre-projection `frontier_causal_recall_mean`.
- `pybullet_objectification_quality_results.md`: objectification-quality
  decomposition benchmark on PyBullet state. It compares corrupted state against
  clean simulator-derived state and records identity recall, semantic identity
  consistency, relation precision/recall, event-frontier recall, selected `K`,
  and `ObjectificationReport` fields. The public report now includes frontier
  completeness and semantic consistency, while the benchmark shows why the
  components still matter: relation-drop drives event-frontier recall to
  `0.585417`, and position noise drops semantic consistency to `0.675541`.
- `pybullet_matched_baseline_benchmark_results.md`: parameter-matched PyBullet
  pilot using `--target-params`. At an approximate 50k-parameter budget, WPU
  sparse preserves branch accuracy from background N=0 to N=128 while full-state
  graph/token baselines drop in this short run. The latency result remains
  mixed because the serialized-token baseline is still fastest at this scale.
- `pybullet_cup_benchmark_7seed_results.md`: 7-seed extension of the PyBullet
  cup benchmark at `N=5` and `N=133`. Sparse WPU preserves accuracy as
  background objects grow and is best at `N=133`, while serialized-token remains
  the fastest model.
- `pybullet_cup_benchmark_7seed_results.ko.md`: Korean companion for the
  7-seed PyBullet cup benchmark extension.
- `pybullet_closed_loop_rollout_results.md`: first PyBullet-derived closed-loop
  `WorldState` rollout diagnostic. Repeated delta application exposes a
  long-horizon WPU sparse failure: raw delta explosion and high constraint
  violations at horizon 25. Delta clipping reduces violations but does not fix
  the underlying raw prediction instability.
- `pybullet_state_integrity_audit_results.md`: derived audit over raw, clipped,
  guarded, regularized, unsafe-delta-rejected, consistency-regularized, and
  state-validity-regularized PyBullet closed-loop rollouts, now including
  rollback-only and corrected-rollback memory safety variants.
  It turns state integrity into a first-class score combining constraint
  validity, bounded applied-delta drift, branch stability, and rejection rate.
  Raw WPU sparse drops to integrity `0.084722` at horizon 25. Guarded state-store
  projection raises sparse WPU applied-state integrity to `0.958508`, but raw
  delta instability remains. A target-relative delta-norm regularized raw
  rollout only raises sparse H=25 integrity to `0.087153`; unsafe-delta
  rejection raises sparse integrity to `0.530270` only by rejecting `0.640000`
  of updates; naive rollout-consistency reaches only `0.084549`; state-validity
  and strong state-validity regularization both remain at `0.084722` for sparse
  H=25. Rollback-only raises sparse applied-state integrity to `0.988647` with
  rollback rate `0.812500`; corrected rollback lowers rollback rate to
  `0.564167` but lowers integrity to `0.900288`.
- `pybullet_local_law_revision_results.md`: first PyBullet-derived local-law
  revision probe. Simple candidate laws over objectified simulator state reduce
  cup-delta MSE under shifted `high_force` and `edge_shift` mechanisms, but
  nominal and `catch_heavy` cases expose overfitting and candidate-selection
  gaps. This supports revisable bounded local-law hypotheses, not unknown
  physical-law discovery.
- `pybullet_system_profile_results.md`: PyBullet-derived systems profile that
  separates full-state tensorization, pre-tensor indexed WPU tensorization,
  branch-overlay memory proxies, a random CPU forward-latency proxy, and a
  random CUDA forward/peak-memory proxy. It
  shows tensor-byte reduction rising to `0.997454` at `N≈2052.6` while selected
  `K≈4.6`, sparse-forward latency reduction reaching `0.996975`, and CUDA
  sparse-forward latency reduction reaching `0.996216`. CUDA peak-memory
  reduction is much weaker at `0.304080`, and branch-overlay memory proxy
  reduction reaches `0.874128` at `B=8`. This is systems evidence, not energy
  or matched-accuracy proof.
- `pybullet_matched_speedup_audit_results.md`: matched-accuracy speedup audit
  connecting the parameter-matched PyBullet benchmark to the CUDA systems
  profile. It applies a matched-or-better criterion: `N=5` is
  accuracy-matched but WPU is slower than token, while `N=133` is positive
  against the best-accuracy non-WPU baseline because WPU is more accurate than
  graph-transformer and `19.184067x` faster. This is not Pareto dominance over
  every baseline.
- `pybullet_matched_speedup_audit_results.ko.md`: Korean companion for the
  matched-accuracy speedup audit.
- `pybullet_matched_speedup_tolerance_results.md`: tolerance sweep for the
  matched-or-better speedup audit, used to check whether the `N=133` conclusion
  depends on a single tolerance threshold.
- `pybullet_matched_speedup_tolerance_results.ko.md`: Korean companion for the
  matched-speedup tolerance sweep.
- `pybullet_system_energy_proxy_results.md`: screening-only system energy proxy
  derived from tensorization and CUDA forward profiles. It is useful for
  choosing future power/sparse-kernel measurement regimes, but it is not real
  energy evidence.
- `pybullet_system_energy_proxy_results.ko.md`: Korean companion for the
  screening-only energy proxy.
- `pybullet_shift_generalization_results.md`: PyBullet mechanism-family shift
  benchmark. Models train on nominal dynamics and evaluate on `high_force`,
  `edge_shift`, and `catch_heavy`, with ECE/Brier/NLL as first-class calibration
  outputs. In the 7-seed rerun it shows a WPU-positive `catch_heavy` regime and
  WPU-negative `edge_shift`/`high_force` regimes where serialized-token remains
  stronger.
- `pybullet_shift_generalization_mixture_calibrated_results.md`: 3-seed
  calibrated mixture-training probe. It shows that mixture training helps WPU
  on `edge_shift` but not on `catch_heavy`, and that post-hoc temperature
  calibration can worsen aggregate WPU-vs-baseline ECE ratio to `1.133834`.
- `pybullet_shift_generalization_mixture_calibrated_results.ko.md`: Korean
  companion for the calibrated mixture-training probe.
- `pybullet_shift_leave_family_out_results.md`: 3-seed leave-family-out shift
  probe. WPU wins three of four held-out families but still loses the
  `catch_heavy` branch-prior shift.
- `pybullet_shift_leave_family_out_results.ko.md`: Korean companion for the
  leave-family-out shift probe.
- `pybullet_shift_composition_stress_results.md`: 3-seed composition-shift
  stress probe over `no_catch`, `edge_high_force`, and `edge_catch_heavy`.
  WPU is accuracy-positive in this run, but `no_catch` exposes worse
  calibration.
- `pybullet_shift_composition_stress_results.ko.md`: Korean companion for the
  composition-shift stress probe.
- `pybullet_shift_composition_stress_bias_calibrated_results.md`: repeats the
  composition-shift stress probe with temperature+bias calibration. It improves
  `no_catch` ECE but does not solve calibration across all composition
  mechanisms.
- `pybullet_shift_composition_stress_bias_calibrated_results.ko.md`: Korean
  companion for the bias-calibrated composition-stress probe.
- `pybullet_shift_calibration_comparison_results.md`: direct comparison between
  temperature-only and temperature+bias composition-stress calibration. Mean
  ECE ratio improves, but only 1/3 mechanisms improve, so calibration remains
  mechanism-aware.
- `pybullet_shift_calibration_comparison_results.ko.md`: Korean companion for
  the calibration comparison.
- `wpu_v2_candidate_safety_frontier_results.md`: candidate-regret safety
  frontier showing that P1 is not solved by threshold search: stricter harmful
  accept limits sharply reduce gap closure.
- `wpu_v2_candidate_safety_frontier_results.ko.md`: Korean companion for the
  candidate safety frontier.
- `wpu_v2_regret_router_variant_results.md`: compares internal, physics-hidden,
  and state-only regret routers; rejects scalar state-only routing for the
  current v2 model.
- `wpu_v2_structured_verifier_probe_results.md`: evaluates state/diagnostic
  verifier gates and K-expansion upper bounds after staged regret routing.
- `wpu_v2_staged_verifier_hybrid_results.md`: stricter deployed verifier test
  with validation-selected rules and held-out test evaluation.
- `wpu_v2_staged_k_expansion_hybrid_results.md`: deployed sparse/dense
  K-expansion experiment after verifier-triggered routing.
- `wpu_v2_proximity_expansion_results.md`: follow-up showing that
  state-native retrieval ordering matters; proximity improves the initial
  under-complete working set in one regime but does not make K expansion
  generally beneficial.
- `wpu_v2_interaction_retrieval_results.md`: interaction-density retrieval
  follow-up showing that state-native retrieval should rank local causal
  structure, not only relation order or target proximity.
- `wpu_v2_selection_composition_results.md`: mechanism audit for indexed,
  proximity, and interaction retrieval composition.
- `wpu_v2_learned_retriever_probe_results.md`: tests whether the
  interaction-density retriever can be reproduced by a small state-native MLP
  under held-out seeds.
- `wpu_v2_learned_retriever_integrated_results.md`: inserts the learned
  retriever into the staged WPU propagation/regret/expansion pipeline and
  compares downstream loss.
- `wpu_v2_learned_retriever_cross_k_results.md`: tests whether learned
  retrieval generalizes across K regimes and identifies fanout context as a
  required state-native input.
- `wpu_v2_global_retriever_integrated_results.md`: evaluates one mixed-K
  learned retriever reused across K=8,16,32 inside the downstream staged WPU
  pipeline.
- `wpu_v2_retriever_regret_oracle_results.md`: measures the downstream
  retriever oracle gap and shows that teacher imitation is not the final
  retrieval objective.
- `wpu_v2_retriever_reranker_results.md`: trains a deployed state-native
  reranker over explicit retrieval candidates and identifies the current
  credit-assignment bottleneck.
- `wpu_v2_retriever_set_reranker_results.md`: upgrades reranking from
  aggregate selected-set summaries to explicit object-set encoding and improves
  loss across K=8,16,32.
- `wpu_v2_retriever_generated_candidates_results.md`: adds state-native
  generated working-set candidates and shows larger deployed improvements over
  fixed selector candidates.
- `wpu_v2_generated_candidate_sweep_results.md`: sweeps generated candidate
  count and shows that oracle quality improves with more candidates while
  deployed reranking peaks around 2-4 candidates.
- `wpu_v2_retriever_regret_distillation_results.md`: trains a state-native
  object retriever from downstream-regret oracle candidate sets instead of
  interaction-teacher labels.
- `wpu_v2_cross_seed_regret_distillation_results.md`: tests whether
  regret-distilled retrieval transfers across held-out seeds and diagnoses the
  large-K obstacle under-selection failure mode.
- `wpu_v2_cross_seed_composition_regret_results.md`: adds a state-conditioned
  working-set composition prior to cross-seed regret retrieval and fixes the
  K=32 obstacle under-selection failure.
- `wpu_v2_composition_variant_selector_results.md`: audits whether
  composition-regret policy variants can be selected without using the target
  seed's test performance.
- `wpu_v2_cross_seed_set_evaluator_results.md`: tests a set-level evaluator
  over base, generated, and composition-aware candidate sets; candidate oracle
  improves but the learned evaluator overfits cross-seed.
- `wpu_v2_candidate_oracle_gap_analysis.md`: decomposes the set-evaluator
  result into candidate-pool oracle gain and deployed selection failure,
  narrowing the v2 bottleneck to transfer-stable candidate scoring.
- `wpu_v2_conservative_set_evaluator_results.md`: tests train-loss and
  per-seed no-harm margin gates for the set evaluator; both fail to protect
  K=8/16 under held-out seeds.
- `wpu_v2_invariant_set_scorer_results.md`: replaces the opaque set evaluator
  with role/geometry/family descriptors and adds train-selected mechanism
  routing; descriptor-only scoring helps K=8/16, while risk-adjusted mechanism
  routing improves K=8/16/32.
- `wpu_v2_candidate_oracle_gap_v2_results.md`: latest candidate-oracle gap
  audit over the invariant-scorer experiment. Risk-adjusted mechanism routing
  recovers only part of the available oracle gain: gap closure is `0.195451`
  at `K=8`, `0.244220` at `K=16`, and `0.042131` at `K=32`. This makes the
  remaining selector gap a required metric, not a prose caveat.
- `wpu_v2_candidate_oracle_gap_decomposition.md` and
  `wpu_v2_candidate_oracle_gap_decomposition.ko.md`: priority-1 decomposition
  by feature variant, `K`, and deployed policy family. It shows that the gap is
  not caused by omitting one aggregate policy: the best non-oracle closure
  remains `0.244220`, and K=32 is selection-signal weak.
- `wpu_v2_candidate_noharm_gate_results.md` and
  `wpu_v2_candidate_noharm_gate_results.ko.md`: sample-level no-harm/margin
  gate audit over the conservative set evaluator. It rejects threshold-only
  gating as the missing P1 fix: best closure is only `0.082804`, and K=8/16
  show negative closure under held-out seeds.
- `wpu_v2_candidate_regret_gate_results.md` and
  `wpu_v2_candidate_regret_gate_results.ko.md`: direct candidate-regret gate
  probe. It predicts `candidate_loss - learned_loss` and deploys a candidate
  only when predicted regret is favorable. The current deployment reaches
  `0.329950` in the test sweep and `0.328025` under train-selected deployment,
  but remains below the `0.5` threshold.
- `wpu_v2_candidate_regret_gate_penalty_results.md` and
  `wpu_v2_candidate_regret_gate_penalty_results.ko.md`: harmful-accept/ranking
  penalty variant. It lowers train-selected harmful accept to `0.088889` but
  collapses train-selected closure to `0.081253`, showing that safety penalties
  alone do not solve P1.
- `wpu_v2_candidate_regret_gate_perturbed_results.md` and
  `wpu_v2_candidate_regret_gate_perturbed_results.ko.md`: feature-perturbed
  candidate-regret training. It slightly improves safe test-sweep closure to
  `0.329756` but lowers train-selected deployment to `0.312586`.
- `wpu_v2_pairwise_reranker_results.md`: tests pairwise ranking loss for the
  larger generated-candidate pool and rejects it as a standalone fix.
- `wpu_v2_cross_seed_reranker_results.md`: applies a stricter
  leave-one-seed-out protocol and shows that current reranker gains are not yet
  robust cross-seed.
- `wpu_v2_cross_seed_normalized_reranker_results.md`: tests per-example
  normalized candidate losses for cross-seed transfer and finds partial but
  insufficient improvement.
- `wpu_v2_cross_seed_context_ablation_results.md`: ablates candidate identity
  and generated/base type context, showing that identity overfit is not the
  sole cross-seed failure mode.
- `wpu_v2_cross_seed_diagnostic_reranker_results.md`: adds candidate-level
  branch entropy, max-probability, and logit-margin diagnostics, then compares
  context ablations under leave-one-seed-out transfer.
- `wpu_v2_diagnostic_variant_selector_results.md`: analyzes whether context
  variants can be selected using only train-seed evidence rather than held-out
  test performance.
- `wpu_v2_diagnostic_safety_gate_probe_results.md`: shows that diagnostic
  safety gates contain oracle signal but do not yet transfer as deployed
  thresholds.
- `wpu_v2_staged_regret_hybrid_5seed_results.md`: five-seed audit of staged
  internal regret routing with bounded selective dense compute.
- `causal_working_set_v2_plan.md`: next large-`N` comparison plan that separates
  total world size `N` from event-conditioned causal working set size `K`.
- `causal_working_set_8m_gpu_protocol.md`: CUDA protocol for the 8M-class
  parameter-matched WPU/token/graph comparison.
- `causal_working_set_v2_results.md`: generated summary for the current CWS
  sweep, when available.
- `robust_v1_results.md`: 5-seed baseline comparison, confidence intervals,
  route sweep, CPU latency, and the central v1 accuracy-runtime tension.
- `n_sweep_v1_results.md`: dense object-count sweep over
  `N=4,8,12,16,24,36,52,68,84,108,132,164,204,260` and estimated crossovers.
- `b_sweep_v1_results.md`: branch-pressure sweep showing that fixed `rho`
  thresholds are not a final scheduler.
- `step_sweep_v1_results.md`: training-step sweep showing that single-step
  comparisons can be misleading.
- `controlled_stress_v1_results.md`: relation-noise and affected-background
  stress tests.

Historical or preliminary reports:

- `v1_object_physics_results.md`: first small validation of
  `WorldStateProcessor`.
- `baseline_and_regime_results.md`: initial one-seed baseline and route sweep;
  superseded for claims by the robust reports.
- `wpu_v2_geometry_hybrid_pilot_results.md`: two-seed CPU pipeline check for
  geometry-only local interaction features versus dense interaction hybrid;
  useful as a compute-realism diagnostic, not paper evidence.
- `pybullet_cup_benchmark_results.ko.md`: Korean companion for the first
  simulator-grounded PyBullet cup benchmark.
- `pybullet_objectification_stress_results.ko.md`: Korean companion for the
  PyBullet objectification corruption stress benchmark.
- `pybullet_objectification_quality_results.ko.md`: Korean companion for the
  PyBullet objectification-quality decomposition benchmark.
- `pybullet_matched_baseline_benchmark_results.ko.md`: Korean companion for the
  parameter-matched PyBullet benchmark.
- `pybullet_closed_loop_rollout_results.ko.md`: Korean companion for the
  PyBullet closed-loop rollout diagnostic.
- `pybullet_state_integrity_audit_results.ko.md`: Korean companion for the
  PyBullet state-integrity audit.
- `pybullet_local_law_revision_results.ko.md`: Korean companion for the
  PyBullet local-law revision probe.
- `pybullet_system_profile_results.ko.md`: Korean companion for the PyBullet
  systems profile.
- `pybullet_shift_generalization_results.ko.md`: Korean companion for the
  PyBullet shift-generalization and calibration benchmark.
- `wpu_v2_experiment_plan.md`: running v2 experiment plan and decision log;
  useful for provenance, not a result claim by itself.
- `wpu_v2_candidate_oracle_gap_v2_results.ko.md`: Korean companion for the
  latest candidate-oracle gap audit.
- `wpu_v2_candidate_oracle_gap_decomposition.ko.md`: Korean companion for the
  priority-1 gap decomposition.
- `wpu_v2_adaptive_hybrid_pilot_results.md`: early adaptive hybrid pilot;
  superseded by staged regret and verifier experiments.
- `wpu_v2_clipped_diagnostic_probe_results.md`: clipped residual diagnostic
  probe; useful as a negative routing-calibration result.
- `wpu_v2_compute_aware_pairwise_pilot_results.md`: compute-aware pairwise
  pilot; superseded by later selective and regret-router audits.
- `wpu_v2_distilled_selective_pilot_results.md`: early selective-router
  distillation pilot; superseded by five-seed selective validation.
- `wpu_v2_distractor_sweep_pilot_results.md`: distractor sweep pilot for
  stress-testing retrieval sensitivity.
- `wpu_v2_interaction_hybrid_pilot_results.md`: early interaction-hybrid pilot;
  superseded by interaction retrieval and staged hybrid reports.
- `wpu_v2_k_sweep_pilot_results.md`: early K sweep used to locate useful
  working-set budgets.
- `wpu_v2_learned_hybrid_pilot_results.md`: learned hybrid pilot before the
  staged regret route was introduced.
- `wpu_v2_learned_selective_pilot_results.md`: first learned selective-router
  pilot.
- `wpu_v2_learned_selective_t015_pilot_results.md`: threshold-0.15 learned
  selective-router pilot.
- `wpu_v2_pairwise_interaction_pilot_results.md`: pairwise interaction pilot
  before the later pairwise reranker analysis.
- `wpu_v2_pre_tensor_indexed_n_sweep_results.md`: pre-tensor indexed
  large-`N` sweep; useful for systems framing, superseded by later CWS
  protocols for current claims.
- `wpu_v2_selective_5seed_validation_results.md`: five-seed selective-router
  validation preceding the stricter staged-regret analysis.
- `wpu_v2_selective_compute_pairwise_comparison_results.md`: selective
  compute-aware pairwise comparison pilot.
- `wpu_v2_selective_threshold_comparison_results.md`: selective threshold
  comparison pilot.
- `wpu_v2_selector_gap_results.md`: selector-gap diagnostic before generated
  candidate and invariant-scorer follow-ups.
- `wpu_v2_staged_regret_context_probe_results.md`: staged-regret context probe
  showing that simple diagnostic concatenation is not sufficient.
- `wpu_v2_staged_regret_hybrid_pilot_results.md`: first staged-regret hybrid
  pilot; superseded by the five-seed staged-regret report.
- `wpu_v2_staged_regret_margin_policy_results.md`: K-conditioned margin policy
  audit; useful as a negative deployment-calibration result.
- `wpu_v2_staged_regret_margin_sweep_results.md`: margin sweep supporting the
  later conservative staged-regret policy analysis.
- `cws_balanced_branch_8m_gpu_event_conditioned_dense_n_sweep_results.md`:
  dense N-sweep from the 8M-class event-conditioned CWS protocol.
- `cws_balanced_branch_8m_gpu_event_conditioned_final_results.md`: final
  promoted 8M-class event-conditioned CWS result.

## Main Takeaways

- WPU-family accuracy is competitive or best in small-to-medium local synthetic
  regimes, but fails at large `N`.
- WPU-family accuracy advantage disappears around `N≈120` in the dense N sweep.
- Routed WPU runtime becomes favorable around `N≈124` versus serialized-token
  and around `N≈178` versus dense-graph.
- WPU-hybrid is robust to irrelevant relation noise.
- Affected-background stress does not support broad WPU superiority; token
  baselines remain strong.
- V2 K expansion is useful only when the initial working set is under-complete
  and the expansion operator adds causally relevant objects. Always expanding,
  especially with dense recompute over a larger subgraph, is worse.
- Pre-tensor state retrieval is the key systems distinction: selecting the
  event-local working set before tensorization makes WPU latency weakly
  dependent on total `N` in the synthetic large-state CWS sweeps.
- Retrieval quality matters as much as propagation capacity. Proximity and
  interaction-density retrieval show that state frontier ordering is a
  scientific variable, not an implementation detail.
- Learned retrieval is viable but not solved. Teacher-distilled learned
  retrieval can reproduce hand-built interaction structure, mixed-K retrieval
  needs explicit fanout/context features, and global mixed-K retrieval can be
  reused across `K=8,16,32` with limited loss.
- Downstream-regret supervision is stronger than teacher imitation in same-seed
  validation-to-test evaluation: regret-distilled retrieval improves loss at
  `K=8,16,32` and wins 14 of 15 seed/K conditions against learned interaction.
- Cross-seed transfer is the hard problem. Regret distillation, composition
  constraints, generated candidates, rerankers, diagnostic gates, and normalized
  losses each expose useful signal but do not close the candidate-oracle gap.
- The current strongest cross-seed working-set result is risk-adjusted
  mechanism selection over explicit role/geometry/family descriptors. At
  `N=2048`, it improves held-out mean loss over static learned selection at
  `K=8,16,32`, but the candidate oracle remains substantially better.
- The latest candidate-oracle gap audit makes the remaining headroom explicit:
  risk-adjusted mechanism routing closes only `0.244220` of the available
  oracle gain at best, so candidate scoring remains the priority-1 bottleneck.
- The candidate no-harm gate audit narrows that bottleneck further: margin-only
  sample-level gates are not sufficient, because their confidence is not
  reliably aligned with held-out downstream regret.
- Direct candidate-regret supervision improves P1 to `0.328025` under
  train-selected deployment, with a test-sweep best of `0.329950`. It is still
  a fail: no-harm rejection is weak and the candidate oracle remains
  substantially stronger.
- The first PyBullet benchmark shows that the WPU state pipeline is not limited
  to hand-written synthetic labels: simulator state can be objectified and fed
  through the same WPU API. Current evidence is systems-level only; accuracy
  remains comparable rather than dominant.
- The PyBullet objectification-quality benchmark makes the object contract
  sharper: the public report now includes frontier completeness and semantic
  consistency, and the benchmark shows that relation-drop can drive
  event-frontier recall to `0.585417` while position noise reduces semantic
  consistency to `0.675541`.
- The PyBullet systems profile is the clearest current cost-separation result:
  when PyBullet background state grows to `N≈2052.6`, indexed WPU still
  tensorizes only `K≈4.6`, reducing tensor bytes by `0.997454`; the random CPU
  sparse-forward proxy reaches `0.996975` reduction. This supports the
  state-indexing premise but remains a proxy, not a hardware-power or
  Pareto-dominance result.
- The PyBullet state-integrity audit turns closed-loop rollout stability into
  a tracked metric. It confirms that guarded state-store projection can protect
  applied state, but it remains a safety layer, not a solution to raw WPU sparse
  delta instability. The regularized and consistency-regularized raw rollouts
  confirm that simple delta penalties are insufficient, and unsafe-delta
  rejection must be reported with rejection rate because it can protect state by
  declining unsafe updates.
- The PyBullet shift benchmark adds the first mechanism-family generalization
  and calibration table. It is mixed: WPU local-dense leads on `catch_heavy`,
  but `serialized-token` is stronger on `edge_shift` and `high_force`.
  Temperature+bias calibration reduces the worst `no_catch` ECE failure but
  does not improve all composition mechanisms.
- The defensible v2 claim is therefore architectural: explicit state exposes
  working-set generation, candidate description, mechanism routing, and
  risk-aware deployment as trainable pre-propagation control surfaces. It does
  not yet prove broad accuracy dominance over token or graph baselines.

The v2 target is to move the accuracy crossover beyond the runtime crossover
while preserving sparse routed work.

For publication readiness, unresolved gaps, and the evidence required for
stronger claims, see `docs/publication_readiness.md`.

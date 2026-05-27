# WPU Experiment Reports

This directory contains the experiment evidence behind the current WPU v1 paper.
The paper claim is intentionally regime-specific: WPU is not shown to be
universally superior to token or graph baselines.

## Current Evidence Hierarchy

Use these reports for paper-level claims:

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

## Main Takeaways

- WPU-family accuracy is competitive or best in small-to-medium local synthetic
  regimes, but fails at large `N`.
- WPU-family accuracy advantage disappears around `N≈120` in the dense N sweep.
- Routed WPU runtime becomes favorable around `N≈124` versus serialized-token
  and around `N≈178` versus dense-graph.
- WPU-hybrid is robust to irrelevant relation noise.
- Affected-background stress does not support broad WPU superiority; token
  baselines remain strong.
- V2 K expansion is only useful when the retriever misses causal state and the
  expansion operator adds the right objects. Proximity-ranked retrieval shows
  that object ordering inside the state frontier matters as much as K size.
- Interaction-density retrieval is the current best v2 retrieval result: it
  improves the best deployed loss in the pairwise CWS task while staying
  state-native and pre-tensor.
- A small learned state retriever can reproduce the interaction teacher's
  composition with high held-out overlap, so the hand-built rule is not the
  final mechanism.
- Integrated learned retrieval preserves most of the interaction retriever's
  downstream advantage and is now the preferred direction over fixed
  hand-written retrieval rules.
- Cross-K retrieval needs explicit state-index context. Mixed-K training with
  fanout features generalizes across K=8,16,32; K=16-only training fails at
  K=32.
- A global mixed-K learned retriever preserves most downstream WPU performance
  without training separate retrievers per K condition.
- The retriever-regret oracle probe shows the next bottleneck: interaction
  teacher imitation is rarely the downstream oracle, so v2 must train retrieval
  against task loss or regret rather than teacher overlap alone.
- A first deployed reranker recovers part of the oracle gap at K=8, but larger
  K remains unstable. The next mechanism should encode candidate object sets
  directly rather than using only aggregate selected-set features.
- Object-set reranking is the first learned retrieval policy that improves loss
  across all tested K values. The gain is still small at larger K, but it
  supports the direction of learned state working-set selection.
- Generated local state candidates further improve the object-set reranker and
  expand the oracle. This is the current strongest evidence for v2: explicit
  state enables learned working-set generation and scoring before propagation.
- Candidate-count sweep shows that generation and scoring are now separate
  bottlenecks: more candidates improve the oracle, but deployed reranking needs
  better capacity/calibration once the candidate pool grows.
- Pairwise ranking loss helps K=8 but hurts K=16/32 by over-selecting generated
  candidates. The next scoring work should focus on calibration/cross-seed
  generalization rather than objective swaps alone.
- Cross-seed reranker transfer is weak. The current reranker captures
  seed/model-specific validation behavior, so v2 needs invariant calibration or
  co-training before making robust deployment claims.
- Normalizing candidate losses improves cross-seed transfer at K=8/16, but does
  not close the same-seed gap. Loss scale is only part of the generalization
  problem; model-invariant scoring remains unsolved.
- Removing candidate identity does not reliably improve cross-seed transfer.
  The next missing signal is model-state diagnostics for each candidate, not a
  simple removal of selector context.

The v2 target is to move the accuracy crossover beyond the runtime crossover
while preserving sparse routed work.

# WPU Claim Ledger

This ledger maps each paper-level WPU claim to the evidence that currently
supports it, the boundary of that evidence, and the result that would weaken or
falsify the claim. It is the repository's guardrail against overclaiming.
For external-release readiness and unresolved gaps, see
`docs/publication_readiness.md`.
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
| C7 | Explicit state exposes pre-propagation working-set control. | Supported as the strongest current v2 mechanism claim, but the dashboard still records failing gaps. | `docs/experiments/wpu_v2_retriever_regret_distillation_results.md`, `docs/experiments/wpu_v2_invariant_set_scorer_results.md`, `docs/experiments/wpu_v2_candidate_oracle_gap_decomposition.md`, `docs/experiments/wpu_v2_candidate_noharm_gate_results.md`, `docs/experiments/wpu_v2_candidate_regret_gate_results.md`, `docs/experiments/wpu_v2_candidate_regret_gate_penalty_results.md`, `docs/experiments/wpu_v2_candidate_regret_gate_perturbed_results.md`, `docs/experiments/wpu_v2_candidate_regret_crossfit_results.md`, `docs/experiments/wpu_v2_candidate_safety_gate_results.md`, `docs/experiments/wpu_v2_candidate_invariant_gate_results.md`, `docs/experiments/wpu_v2_candidate_safety_frontier_results.md`, `docs/experiments/pybullet_branch_prior_shift_results.md`, `docs/experiments/pybullet_mechanism_prior_adaptation_results.md`, `docs/experiments/pybullet_prior_strength_sweep_results.md`, `docs/experiments/pybullet_selected_prior_adaptation_results.md`, `docs/experiments/pybullet_fewshot_mechanism_adaptation_results.md`, `docs/experiments/pybullet_mechanism_adaptive_policy_results.md`, `docs/experiments/pybullet_uncertainty_gated_recompute_results.md`, `docs/experiments/pybullet_learned_uncertainty_gate_results.md`, `docs/experiments/wpu_v2_priority_dashboard.md`. | Candidate oracle remains better; direct candidate-regret deployment reaches `0.329950` in the test sweep and `0.328025` under train-selected deployment, but this remains below the `0.5` target and harmful accepts remain near the safety limit. The safety-frontier audit shows the tradeoff directly: at harmful limit `0.25`, best direct closure is about `0.327-0.330`, while at harmful limit `0.10`, closure drops to `0.081898` for direct and `0.154320` for perturbed. Cross-fit ensemble regret gating tests whether train-selected overfit is the missing fix; it is negative, with best closure `0.287268`, safe best `0.279738`, and cross-fit selected closure `0.270989`. Harmful-accept/ranking penalty training lowers harmful accepts but collapses closure to `0.081253`; feature perturbation raises safe test-sweep closure to `0.329756` but lowers train-selected deployment to `0.312586`. A separate safety/utility head is also negative: best closure is `0.147450`, safe best is `0.090719`, and train-selected closure is `0.144863`. Descriptor standardization plus group-DRO no-harm training is negative as a standalone fix: best closure and safe best are both `0.110889`, and train-selected closure is `0.093863`. The branch-prior audit adds a P4/P5 failure boundary: `catch_heavy` majority prior accuracy is `0.753968`, while best WPU reaches `0.408730`. Mechanism-prior adaptation raises shifted WPU win-rate to `0.666667`, but worsens shifted mean WPU ECE by `0.024819`. Prior-strength sweep confirms that the accuracy-best nonzero strength (`0.75`) still increases ECE relative to `strength=0`. Calibration-selected prior strength improves shifted mean WPU ECE by `-0.046204` and Brier by `-0.105470`, but leaves shifted WPU-vs-baseline win-rate at `0.333333`. Few-shot mechanism adaptation reaches shifted WPU-vs-baseline win-rate `1.000000` with mean margin change `0.050264`, but it uses mechanism-specific calibration samples and is not zero-shot generalization. A mechanism-aware adaptive policy improves the adapted regime further: shifted WPU win-rate is `1.000000`, mean accuracy change is `0.198412`, mean margin change is `0.058201`, mean ECE change is `-0.099347`, and mean Brier change is `-0.155443`; this is detect-and-adapt evidence, not zero-shot evidence. Uncertainty-gated local-dense recompute improves aggregate accuracy by `0.071428` and ECE by `-0.016396`, but only at dense recompute rate `0.985450`; the low-cost gate worsens ECE by `0.005395`. Learned sparse-output benefit gating improves source low-cost accuracy by `0.052910` at recompute rate `0.205027`, but worsens ECE by `0.010769`. Cross-seed robust scoring, calibration-safe low-cost uncertainty gating, and long-horizon model-delta stability are also not solved. |
| C8 | WPU is not yet a hardware/chiplet/IP result. | Explicitly not supported. | `docs/Review/review_response_and_differentiation.md`, `docs/paper/state_is_all_you_need.md`, `docs/experiments/pybullet_system_profile_results.md`, `docs/experiments/pybullet_matched_speedup_audit_results.md`, `docs/experiments/pybullet_matched_speedup_tolerance_results.md`, `docs/experiments/pybullet_pareto_frontier_results.md`, `docs/experiments/pybullet_system_energy_proxy_results.md`, `README.md`. | CUDA random-forward profiling supports a large latency-reduction opportunity, and the screening-only energy proxy suggests where a real power study should be run. A corrected matched-or-better audit is positive at `N=133` against the best-accuracy non-WPU baseline, and the small Pareto audit puts WPU on the accuracy-latency frontier at `N=133`. WPU is still slower/dominated at `N=5`, and this is not hardware evidence. Hardware claims still require real sparse kernels, memory traffic, branch-overlay allocation, power/energy, and trained matched-or-better speedup evidence. |
| C9 | WPU propagation over objectified relations is a simplified local-causality prior, not full physics. | Supported as a bounded analogy. | `docs/objectification.md`, `docs/arxiv/state_is_all_you_need_en.tex`, `docs/arxiv/state_is_all_you_need_ko.md`, `docs/paper/state_is_all_you_need.md`, `docs/experiments/object_history_hidden_mechanism_probe_results.md`, `docs/experiments/object_relation_law_probe_results.md`, `docs/experiments/object_relation_law_ood_probe_results.md`, `docs/experiments/object_relation_law_revision_probe_results.md`. | The hidden-mechanism, local-law, OOD, and revision probes are synthetic. They show that history-derived relation variables and a simple inverse-distance law can transfer beyond nominal type names under a generated mechanism, that OOD stress can separate relation failure from law mis-specification, and that small calibration can revise some generated law shifts. Real physics competence or unknown-theory discovery requires simulator/robotics benchmarks, long-horizon stability, and evidence that learned relations generalize beyond the training generator. |
| C10 | Near-term WPU value is more plausible as software runtime/middleware than silicon. | Plausible direction, not experimentally proven. | `docs/reproducibility.md`, `docs/arxiv/README.md`, current PyTorch package under `wpu/`. | Requires digital-twin, simulation backend, game/server, or robotics middleware benchmarks. |
| C11 | Objectification quality is measurable and locally repairable as a contract before propagation. | Supported as an implementation claim. | `wpu/core/objectification.py`, `tests/test_objectification.py`, `tests/test_script_entrypoints.py`, `docs/experiments/objectification_relation_repair_probe_results.md`, `README.md`, `docs/objectification.md`. | Relation repair and `LocalLawHypothesis` produce conservative hypotheses and revision reports, not ground-truth physics. The latest probes show that learned repair can transfer across aliased type names, improve a toy downstream diagnostic, and report law-revision gaps; this is not evidence that perception-to-object construction or unknown-theory discovery is solved. |

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
  model is still not stable; stronger claims require reducing correction
  trigger frequency, not only shrinking the corrected object set or eliminating
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

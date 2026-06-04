# WPU Claim Ledger

This ledger maps each paper-level WPU claim to the evidence that currently
supports it, the boundary of that evidence, and the result that would weaken or
falsify the claim. It is the repository's guardrail against overclaiming.
For external-release readiness and unresolved gaps, see
`docs/publication_readiness.md`.
For the formal objectification definition, see `docs/objectification.md`.

## Claim Status

| ID | Claim | Current status | Primary evidence | Boundary |
|---|---|---|---|---|
| C1 | Token, state, and objectified state are different operational primitives. | Supported as a framing claim. | `docs/objectification.md`, `docs/arxiv/state_is_all_you_need_en.tex`, `docs/arxiv/state_is_all_you_need_ko.md`, `wpu/core/state.py`, `wpu/memory/state_store.py`. | This is not a claim that tokens cannot encode state; it is a claim that persistent identity, relation traversal, delta patching, and branch overlays are native operations only after objectification. |
| C2 | Explicit objectified world-state processing is implementable in a trainable neural model. | Supported for the synthetic object-physics prototype. | `docs/objectification.md`, `wpu/core/objectification.py`, `wpu/models/world_state_processor.py`, `wpu/models/causal_working_set_processor.py`, `wpu/data/object_physics.py`, `demos/robot_cup_demo.py`, passing test suite. | Does not prove general physical understanding, end-to-end object construction, or perception-to-state construction. |
| C3 | Sparse/hybrid/dense routing is a measurable execution regime, not just a diagram. | Supported for v1 routing instrumentation. | `wpu/engines/scheduler.py`, `docs/experiments/b_sweep_v1_results.md`, `docs/experiments/n_sweep_v1_results.md`, `docs/experiments/baseline_and_regime_results.md`. | Fixed `rho` thresholds are engineering defaults, not an optimal scheduler. |
| C4 | v1 WPU shows a real accuracy-runtime tension. | Supported. | `docs/experiments/robust_v1_results.md`, `docs/experiments/n_sweep_v1_results.md`, `docs/versions/wpu_v1_closure.md`. | WPU accuracy advantage ends before runtime advantage in v1; this is a failure boundary, not a win. |
| C5 | WPU-hybrid is robust to irrelevant relation noise in the synthetic task. | Supported for the tested stress regime. | `docs/experiments/controlled_stress_v1_results.md`. | Does not imply superiority under all state-delta or affected-background regimes. |
| C6 | Large `N` only helps WPU when causal working set `K` is small and identifiable before tensorization. | Supported as a conditional systems hypothesis with early evidence. | `docs/experiments/cws_balanced_branch_8m_gpu_event_conditioned_final_results.md`, `docs/experiments/wpu_v2_pre_tensor_indexed_n_sweep_results.md`, `docs/versions/wpu_v1_closure.md`. | Larger `N` alone is not sufficient; if retrieval scans `O(N)` or misses causal state, the claim fails. |
| C7 | Explicit state exposes pre-propagation working-set control. | Supported as the strongest current v2 mechanism claim. | `docs/experiments/wpu_v2_retriever_regret_distillation_results.md`, `docs/experiments/wpu_v2_invariant_set_scorer_results.md`. | Candidate oracle remains better; cross-seed robust scoring is not solved. |
| C8 | WPU is not yet a hardware/chiplet/IP result. | Explicitly not supported. | `docs/Review/review_response_and_differentiation.md`, `docs/paper/state_is_all_you_need.md`, `README.md`. | Hardware claims require real sparse-kernel, memory-traffic, branch-overlay, and matched-accuracy speedup evidence. |
| C9 | WPU propagation over objectified relations is a simplified local-causality prior, not full physics. | Supported as a bounded analogy. | `docs/objectification.md`, `docs/arxiv/state_is_all_you_need_en.tex`, `docs/arxiv/state_is_all_you_need_ko.md`, `docs/paper/state_is_all_you_need.md`. | Real physics competence or unknown-theory discovery requires simulator/robotics benchmarks, long-horizon stability, and evidence that learned relations generalize beyond the training generator. |
| C10 | Near-term WPU value is more plausible as software runtime/middleware than silicon. | Plausible direction, not experimentally proven. | `docs/reproducibility.md`, `docs/arxiv/README.md`, current PyTorch package under `wpu/`. | Requires digital-twin, simulation backend, game/server, or robotics middleware benchmarks. |
| C11 | Objectification quality is measurable as a contract before propagation. | Supported as an implementation claim. | `wpu/core/objectification.py`, `tests/test_objectification.py`, `README.md`, `docs/objectification.md`. | This is not evidence that perception-to-object construction is solved; it only evaluates supplied objectified state. |

## Falsification Tests

- If serialized-token or graph baselines match WPU at equal compute across
  controlled identity, locality, and branching benchmarks, WPU is not a
  necessary primitive for those regimes.
- If pre-tensor retrieval cost grows close to `O(N)` in realistic state stores,
  the large-`N` systems claim collapses.
- If risk-adjusted mechanism selection loses its held-out-seed gain under
  larger seed/model sweeps, the current v2 working-set-control claim must be
  downgraded to a diagnostic result.
- If sparse routing harms accuracy before producing meaningful latency or
  memory savings, propagation is not the right central operation for that
  workload.
- If delta overlays accumulate unrecoverable state corruption in long-horizon
  rollout, persistent state becomes a liability without verification and
  rollback mechanisms.

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

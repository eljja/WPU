# WPU Claim Ledger

This ledger maps each paper-level WPU claim to the evidence that currently
supports it, the boundary of that evidence, and the result that would weaken or
falsify the claim. It is the repository's guardrail against overclaiming.

## Claim Status

| ID | Claim | Current status | Primary evidence | Boundary |
|---|---|---|---|---|
| C1 | Token and state are different operational primitives. | Supported as a framing claim. | `docs/arxiv/state_is_all_you_need_en.tex`, token/state schematic, state model APIs. | This is not a claim that tokens cannot encode state; it is a claim about native update operations. |
| C2 | Explicit world-state processing is implementable in a trainable neural model. | Supported for the synthetic object-physics prototype. | `WorldStateProcessor`, `CausalWorkingSetProcessor`, passing test suite, robot-cup validation. | Does not prove general physical understanding or perception-to-state construction. |
| C3 | Sparse/hybrid/dense routing is a measurable execution regime, not just a diagram. | Supported for v1 routing instrumentation. | Route sweep, dense `N` sweep, `rho` thresholds, `selected_paths`. | Fixed `rho` thresholds are engineering defaults, not an optimal scheduler. |
| C4 | v1 WPU shows a real accuracy-runtime tension. | Supported. | Robust 5-seed suite and dense `N` sweep. | WPU accuracy advantage ends before runtime advantage in v1; this is a failure boundary, not a win. |
| C5 | WPU-hybrid is robust to irrelevant relation noise in the synthetic task. | Supported for the tested stress regime. | `docs/experiments/controlled_stress_v1_results.md`. | Does not imply superiority under all state-delta or affected-background regimes. |
| C6 | Large `N` only helps WPU when causal working set `K` is small and identifiable before tensorization. | Supported as a conditional systems hypothesis with early evidence. | CWS GPU/CPU reports, pre-tensor indexed sweep, v1 closure. | Larger `N` alone is not sufficient; if retrieval scans `O(N)` or misses causal state, the claim fails. |
| C7 | Explicit state exposes pre-propagation working-set control. | Supported as the strongest current v2 mechanism claim. | Regret-distilled retrieval; invariant descriptors; risk-adjusted mechanism selection at `N=2048`, `K=8,16,32`. | Candidate oracle remains better; cross-seed robust scoring is not solved. |
| C8 | WPU is not yet a hardware/chiplet/IP result. | Explicitly not supported. | Review response, arXiv discussion, README application boundary. | Hardware claims require real sparse-kernel, memory-traffic, branch-overlay, and matched-accuracy speedup evidence. |
| C9 | WPU propagation is a simplified local-causality prior, not full physics. | Supported as a bounded analogy. | ArXiv propagation section and limitations. | Real physics competence requires simulator/robotics benchmarks and long-horizon stability. |
| C10 | Near-term WPU value is more plausible as software runtime/middleware than silicon. | Plausible direction, not experimentally proven. | Application boundary docs and current PyTorch prototype. | Requires digital-twin, simulation backend, game/server, or robotics middleware benchmarks. |

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

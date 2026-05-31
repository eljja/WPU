# WPU Review Response and Differentiation

This note maps the two external reviews to the current manuscript state after the
May 11, 2026 revision.

## Review Coverage Matrix

| Review concern | Current status | Manuscript/code response |
|---|---|---|
| Experiments too small, single seed, no confidence intervals | Substantially addressed | Added 5-seed robust baseline suite, dense N sweep, B sweep, step sweep, relation-noise stress, affected-count stress, CPU latency reporting. |
| Serialized-token baseline beats WPU in some regimes | Addressed by narrowing claim | Manuscript now explicitly rejects universal WPU dominance and reports the N≈120 accuracy crossover. |
| Need stronger graph/token baselines | Partially addressed | Added dense graph, graph-transformer, serialized-token baselines. Still missing full DreamerV3/IRIS/GNS matched training. |
| Sparse/dense route points too coarse | Addressed | Route and dense N sweep now use N=4,8,12,16,24,36,52,68,84,108,132,164,204,260. |
| Existing work overlap with GNNs/message passing | Addressed in framing | Added Related Work section. Manuscript now states WPU novelty is not a new message-passing equation but state-memory execution abstraction. |
| Missing object discovery/state construction | Not solved, explicitly acknowledged | Added limitation that v1 assumes object/relation state; roadmap now includes Slot Attention/IODINE-style front-end. |
| Fixed rho thresholds arbitrary | Addressed empirically, not solved | B sweep shows hard threshold weakness; text states thresholds are engineering defaults and learned routing is required. |
| Hardware/commercial claim unclear | Addressed in framing | Text now says WPU is a workload/execution abstraction and PyTorch reference model, not a completed chip design or proven chiplet/IP product. Near-term applications are framed as software runtime/middleware targets. |
| Complexity analysis omits graph construction and sparse hardware overhead | Partially addressed | Added caveat paragraph listing graph construction, dynamic edge repair, irregular memory access, frontier queue, scatter/gather, kernel overhead. |
| Physical analogy overclaimed | Addressed by narrowing | Text frames propagation as a simplified local-causality prior, not full physics. |
| State corruption/security missing | Not implemented, explicitly acknowledged | Added limitation and roadmap item for checkpoint, rollback, uncertainty gates, and consistency checks. |
| Need real/simulator benchmarks | Not solved | Roadmap now explicitly lists learned-physics, MuJoCo, Isaac Gym, robotics manipulation, and matched baselines. |
| Accuracy-runtime crossover does not overlap | Not solved, made central | v2 target is now stated as creating a non-empty regime where routed WPU is faster at matched or acceptable accuracy. New v2 work adds pre-tensor retrieval and risk-adjusted working-set mechanism selection, but does not yet close the accuracy-runtime gap. |
| References too sparse | Addressed | Bibliography expanded from 4 to 22 references, including object-centric learning, GNN simulators, world models, sparse world models, graph world models, and accelerator context. |
| Manuscript reads like a roadmap rather than a paper | Addressed | Main paper now keeps only central claim, architecture, evidence, and discussion. Roadmap-style material moved to `docs/arxiv/README.md`; dense sweep tables and secondary figures moved to supplementary materials. |

## Added Reference Families

| Family | Added references | Why included |
|---|---|---|
| Object discovery | IODINE; Slot Attention | Addresses review criticism that WPU assumes state is already given. |
| Learned physics and GNNs | Interaction Networks; Relational Inductive Biases; GNS; GCN | Places propagation relative to graph/message-passing predecessors. |
| Set/graph attention | Set Transformer | Places dense fallback and graph baselines relative to attention over sets. |
| Latent world models | World Models; PlaNet; Dreamer; DreamerV3; IRIS | Positions WPU against established model-based RL/world-model baselines. |
| Object-centric world models | STICA; ObjectZero; SPARTAN; Graph World Models survey | Covers the closest recent work on object-centric, graph, sparse, and causal world models. |
| Accelerator context | RPU; Modern Neuromorphic AI | Clarifies WPU as workload/execution abstraction relative to token/reasoning/sparse processing hardware ideas. |

## Differentiation From Closest Work

| Prior line | What it primarily optimizes | WPU difference |
|---|---|---|
| Slot Attention / IODINE | Discover object slots from perceptual input | WPU assumes or consumes discovered state, then focuses on persistent update, propagation, branching, and routing. |
| Interaction Networks / GNS | Accurate learned physical simulation via graph message passing | WPU treats message passing as one execution path inside a broader state-memory system with sparse/hybrid/dense routing and branch overlays. |
| Set Transformer / Graph Transformer | Attention over sets or graph-structured inputs | WPU uses attention as dense fallback/baseline; its defining operation is event-conditioned state propagation and delta merge. |
| PlaNet / Dreamer / DreamerV3 | Learn latent dynamics for planning and policy optimization | WPU is not primarily an RL agent; it defines a structured state update substrate and execution regime for world maintenance. |
| IRIS / Transformer world models | Autoregressive token/world-model prediction | WPU argues that token serialization should not be the primary operational primitive for persistent identity and local delta updates. |
| SPARTAN | Sparse attention over object-factored tokens for world modeling | WPU stores state and deltas explicitly and routes execution by affected-state fraction; sparsity is a state-update policy, not only an attention regularizer. |
| STICA / ObjectZero | Object-centric world models for RL and planning | WPU's novelty is the processor abstraction: base state plus delta branches, route scheduler, and accuracy-compute-memory regime. |
| Graph World Models taxonomy | Survey and formalize graph world-model paradigms | WPU can be viewed as a systems/execution instantiation focused on state memory, routing, and branching. |
| RPU / token accelerators | Memory-bandwidth optimized token/reasoning inference | WPU targets graph-local state update, neighbor fetch, delta logs, branch overlays, and sparse-dense route switching. |

## Remaining Gaps That Should Not Be Claimed Solved

- No end-to-end perception-to-state system is implemented.
- No DreamerV3, IRIS, GNS, SPARTAN, STICA, or ObjectZero matched benchmark has been run.
- No real robot, MuJoCo, Isaac Gym, Atari, or autonomous-driving dataset result is available.
- No final learned, hardware-aware, end-to-end scheduler is implemented yet.
  Several learned, selective, staged-regret, and risk-adjusted routing probes
  exist, but they remain experimental components rather than a deployment-ready
  scheduler.
- No real GPU sparse-kernel or hardware simulation evidence exists.
- No state-integrity, rollback, or corruption-recovery mechanism is implemented.
- No software middleware benchmark for digital-twin, game/server synchronization,
  or robotics world-model maintenance exists yet.
- The current v1 accuracy-runtime crossovers do not yet overlap: WPU-family
  accuracy advantage ends around `N≈120`, while routed runtime advantage starts
  around `N≈124` versus serialized-token and `N≈178` versus dense-graph.

## v2 Adjustment Plan

The five concrete adjustments are:

- Reframe the central claim as a local-changing-world execution model, not a
  general replacement for token, graph, or latent world models.
- Define experiments around regime surfaces over `rho`, `N`, `B`, noise, and
  affected-region size instead of isolated wins.
- Prioritize risk-adjusted mechanism selection, learned routing, large-`N`
  sparse stability, regional dense correction, long-horizon branch consistency,
  and state-integrity mechanisms.
- Treat perception as an object-state adapter layer that can use slot/object
  discovery, supervised detectors, simulators, or logs.
- Keep commercialization claims at the software runtime/middleware level until
  hardware-aware profiling and matched-accuracy speedups exist.

## Revised Claim

The defensible claim is:

```text
WPU is a state-native execution abstraction for workloads dominated by
persistent identity, local causal updates, uncertainty, and branching.
The current evidence supports the route/regime hypothesis, noisy local-update
robustness, and state-native working-set control through risk-adjusted
mechanism selection, but it does not prove universal model-quality superiority
over token, graph, or latent world-model baselines.
```

# WPU Process-Unit Release Audit

This audit checks whether the current repository matches the original WPU plan,
where it diverges, and what must not be claimed in an external release. It is a
publication guardrail, not a marketing page.
For the compact thesis and novelty boundary, see `docs/research_thesis.md`.

## Release Classification

| Area | Current level | Public wording |
|---|---|---|
| Core idea | coherent research hypothesis | "state-native execution for objectified world state" |
| Software prototype | implemented | "PyTorch reference implementation and experimental runtime" |
| Scientific evidence | regime-limited | "promising when large `N` contains a small identifiable causal working set `K`" |
| Hardware/process-unit evidence | early systems proxy | "process-unit direction and workload abstraction, not a completed chip/IP claim" |
| Real-world autonomy | not established | "requires object-state adapters, broader simulators, and real deployment tests" |

The repository is suitable for a constrained public research preview and arXiv
preprint. It is not yet evidence that WPU is a finished processor, a universal
Transformer replacement, or a generally superior accelerator.
Its current originality is the process-unit abstraction: a proposed execution
interface for objectified state, causal working sets, delta overlays, and
branch-aware propagation. That abstraction can motivate future hardware, but
the present artifact is a PyTorch reference model and evidence program.

## Original Plan Conformance

| Planned item | Current status | Audit judgment |
|---|---|---|
| Python/PyTorch research package | Implemented with package, scripts, tests, demos, docs. | Conforms. |
| Explicit world-state types | Implemented under `wpu/core/` and used by store, batch, rollout, and demos. | Conforms. |
| StateGraphBatch and model API | Implemented for train/eval/demo paths. | Conforms for prototype scale. |
| Sparse, hybrid, dense routing | Implemented and tested through scheduler, model variants, and CWS experiments. | Conforms, but learned routing remains incomplete. |
| Branch rollout as base state plus deltas | Implemented at software abstraction level. | Conforms, but long-horizon raw delta stability is not solved. |
| Object physics first domain | Implemented with synthetic and PyBullet cup-family benchmarks. | Conforms, but domain breadth is narrow. |
| Paper and Korean/English documents | Implemented with claim ledger, readiness register, reproducibility guide, and arXiv draft. | Conforms, with constrained claims. |
| Demonstrable superiority over token/graph baselines | Not part of a defensible universal claim. | Must stay regime-specific. |
| Hardware WPU evidence | Not implemented as silicon, sparse kernel, or power study. | Diverges from any hardware interpretation of "process unit". |
| Perception-to-object-state adapter | Not solved. | Out of current evidence scope. |

## Problems Found and Corrections

| Problem | Risk | Correction applied |
|---|---|---|
| "End-to-end downstream-loss selector" can sound like full joint WPU training. | Readers may think fixed candidate generation and fixed propagation were jointly optimized. | Result and dashboard wording now state that this is fixed-candidate/fixed-propagator selector training, not full joint retriever-propagator training. |
| "Process unit" can be read as current hardware. | Unsupported hardware or energy claims would overstate the evidence. | This audit and the readiness register classify current evidence as software runtime plus systems proxy only. |
| Large-`N` language can drift into "WPU wins when `N` is large." | False: large `N` helps only when `K` is small, identifiable, and retrieved before tensorization. | Existing claim ledger and dashboard keep the conditional statement; this audit repeats it as the release rule. |
| P1 negative results can be hidden by adding more gates. | The research direction could become threshold tuning rather than solving the bottleneck. | Next action is narrowed to joint candidate generation, retrieval, propagation, and low-harm calibration objectives. |
| P2 high integrity can hide heavy correction. | Memory-layer correction may look like stable learned dynamics. | Readiness text keeps raw-delta instability and high correction-trigger frequency visible. |

## Current Scientific Claim

The strongest defensible claim is:

```text
WPU exposes an execution regime that token-serialized or dense graph processors
do not expose natively: when a large explicit world state contains a small,
identifiable causal working set, computation can be scheduled around object
identity, relations, deltas, uncertainty, and branches before full tensorization.
```

This is not a representation-universality claim. A token model can encode state
in principle. The WPU claim is operational: object identity, relation traversal,
delta patching, branch overlays, and causal working-set selection become native
execution operations only after objectification.

## Current Blockers

1. Candidate selection still leaves most oracle gain unused. The best deployed
   P1 closure remains below the target. Fixed-candidate downstream-loss selector
   training is negative, learned candidate generation creates headroom that the
   deployed evaluator still cannot use reliably, and label-free sparse/dense
   verification signatures are negative as post-hoc selector features.
2. Long-horizon raw sparse dynamics remain unstable without memory-layer
   correction, rollback, or fallback.
3. Calibration-safe low-cost adaptation exists only in a narrow
   mechanism-selective regime.
4. Baseline-complete simulator evidence is still mostly cup-family PyBullet.
5. Systems results are latency/memory proxies, not real sparse-kernel,
   allocator, wall-power, or silicon evidence.
6. Objectification quality is measured, but perception-to-object construction
   is not solved.

## External Release Rule

External documents may say:

- WPU is a state-native software research prototype and process-unit direction.
- WPU is promising in large-state/small-causal-working-set regimes.
- WPU currently provides stronger evidence for execution structure than for
  broad model-quality dominance.
- WPU's long-term goal is to learn relation/propagation structures that may
  approximate physical regularities, including regularities not yet explicitly
  known.

External documents must not say:

- WPU is a completed hardware accelerator, chiplet, IP block, or power-saving
  processor.
- WPU generally beats token, graph, GPU, TPU, NPU, or LPU systems.
- WPU has solved perception-to-state construction, real physical understanding,
  or unknown-law discovery.
- Large `N` alone makes WPU superior.

## Next Completion Bar

The next stronger process-unit claim requires all of the following:

- A non-empty matched-accuracy regime where WPU is faster than token/graph
  baselines using trained models, not only random-forward proxies.
- Candidate generation, retrieval, and propagation trained together with
  held-out seed selection and no-harm calibration.
- Long-horizon state integrity with low correction frequency, not only
  correction-heavy projection.
- Baseline-complete large-`N` simulator tasks beyond the current cup family.
- Real systems measurements: sparse kernels, allocator traffic, peak memory,
  and power/energy telemetry.

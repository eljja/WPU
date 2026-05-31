# State Is All You Need

This Markdown note is the compact research brief for the WPU paper. The
submission-oriented source is `docs/arxiv/state_is_all_you_need_en.tex`; the
Korean companion is `docs/arxiv/state_is_all_you_need_ko.md`.

## Central Claim

World processing should not be framed only as token continuation. A world model
must often maintain persistent entities, typed relations, local causal changes,
uncertainty, and multiple possible futures. A token sequence can describe these
objects, but serialization does not make object identity, relation traversal,
delta update, or branch overlay first-class operations.

The WPU claim is therefore operational rather than representational:

```text
Token = ordered evidence for append / attend
State = persistent substrate for patch / propagate / branch
```

The current paper does not claim universal superiority over Transformers,
Graph Transformers, GPUs, TPUs, NPUs, or LPUs. It proposes a falsifiable regime
hypothesis: state-native propagation should matter when persistent identity,
local causal change, uncertainty, and branchable futures dominate the workload.
The experimental goal is to map that regime over `rho`, `N`, branch pressure,
noise, and affected-region size, not to claim that WPU always beats token or
graph models.

## Relation To Existing Work

The novelty is not a new message-passing equation. WPU overlaps directly with
object-centric learning, graph neural simulators, Set/Graph Transformers, latent
world models, and sparse world-model systems. The difference is the execution
abstraction:

```text
persistent state memory
event frontier
sparse / hybrid / dense routing
delta overlay
branch sharing
accuracy-compute-memory regime surface
```

Slot Attention or IODINE can be used as perception-to-state front ends. Graph
Network-based Simulators can be used as propagation cores. Set/Graph
Transformers can be used as dense fallback or baselines. WPU asks what execution
substrate is appropriate once the system must repeatedly update an explicit
world state rather than only process a fresh sequence.

In this framing, perception-to-state is an adapter problem rather than a solved
part of v1. The core WPU should accept object-state graphs from supervised
detectors, slot/object discovery, simulators, or logs, then expose how update
work, prediction risk, and branch memory scale.

## State Primitive

The state is represented as:

```text
S_t = {O_t, R_t, T_t, P_t}
```

Where:

- `O_t`: persistent objects with typed attributes.
- `R_t`: typed relations between objects.
- `T_t`: temporal memory and event/delta history.
- `P_t`: uncertainty, confidence, and branch probabilities.

Events patch state rather than append to a sequence:

```text
S_{t+1} = S_t + Delta S_t
Branch = BaseState + DeltaState
```

This gives the model explicit operations for identity-preserving updates,
local graph traversal, copy-on-write futures, and partial state materialization.

## WPU Architecture

The v1 reference implementation is `WorldStateProcessor`, a PyTorch
`nn.Module`. It accepts a `StateGraphBatch` with object features, relation edge
indices, relation features, event/action features, masks, target indices, time
features, and scheduler metrics.

It returns a `StatePrediction` with object deltas, relation logits, uncertainty
updates, branch logits/probabilities, and selected execution paths.

The execution paths are:

- Sparse path: event-conditioned relation message passing from a frontier.
- Hybrid path: sparse propagation plus regional dense correction.
- Dense path: global object-set attention for full consistency recomputation.

The hard v1 scheduler uses:

```text
rho = (DeltaN * fanout^depth * branches) / N

rho < 0.05  -> sparse
rho < 0.25  -> hybrid
otherwise   -> dense
```

These thresholds are engineering defaults, not final constants. The B sweep
shows that fixed thresholds are not sufficient; learned accuracy-latency-aware
routing is required.

The v2 scheduler should optimize prediction risk and update cost jointly. It
should be allowed to choose regional dense correction even when `rho` looks
small, or sparse propagation even when raw object count is large, if uncertainty,
fanout, relation quality, or branch divergence make that route preferable.

## Propagation Rather Than Attention

Attention asks which token or state element should be read. Propagation asks
what consequence a state delta causes through typed relations.

```text
Delta o_i^{l+1}
  = f(o_i, e_t, sum_j g(o_i, r_ij, o_j, Delta o_j^l))
```

Propagation can use attention-like neural blocks, but its intended semantics are
different. It is a simplified local-causality prior: physical and world-state
changes often begin at a small set of entities and spread through contact,
support, proximity, constraints, or causal relations. Dense fallback is the
global correction path when locality is no longer a good approximation.

## Validation Task

The first task is intentionally small: a synthetic robot-cup object physics
scene with cup, table, robot hand, table edge, and configurable background
objects. The event is a hand touch with force. The targets are next-state object
deltas and one of three branch labels: `stable`, `falls`, or `caught`.

This is not a benchmark of general physical reasoning. It is a unit test for the
WPU hypothesis: can explicit state graphs, event-conditioned propagation,
sparse/dense routing, and branch probabilities be trained and inspected in one
model?

Primary validation after 200 steps on CPU:

| Model | Next-state MSE | Branch NLL | Branch accuracy |
|---|---:|---:|---:|
| Untrained WSP | 0.8111 | 1.2070 | 0.1289 |
| Majority baseline | n/a | n/a | 0.6680 |
| Trained WSP | 0.0005 | 0.8074 | 0.7188 |

## Stronger Experiments

The reviewer-driven experiment package adds five seeds, stronger baselines,
denser sweeps, and runtime profiling. The strongest current conclusion is mixed
and therefore more defensible.

Robust comparison over 5 seeds, 150 steps, 256 evaluation samples:

| N | Best WPU | Accuracy | Best non-WPU | Accuracy | Interpretation |
|---:|---|---:|---|---:|---|
| 4 | WPU-hybrid | 0.7242 +/- 0.0260 | Dense graph | 0.6398 +/- 0.1257 | WPU wins; routed scheduler can fail by choosing dense. |
| 24 | WPU-hybrid | 0.7320 +/- 0.0280 | Graph Transformer | 0.6609 +/- 0.0680 | WPU wins in a medium local regime. |
| 84 | WPU-hybrid | 0.7508 +/- 0.0244 | Graph Transformer | 0.6953 +/- 0.0388 | WPU remains best in this synthetic regime. |
| 204 | WPU-routed/sparse | 0.4516 +/- 0.1957 | Graph Transformer | 0.7172 +/- 0.0615 | WPU fails; graph/token baselines dominate accuracy. |

Dense N sweep:

- Measured `N`: `4, 8, 12, 16, 24, 36, 52, 68, 84, 108, 132, 164, 204, 260`.
- Route crossover: dense to hybrid at measured `N=16`.
- Route crossover: hybrid to sparse at measured `N=68`.
- Accuracy crossover: WPU-family advantage disappears around `N≈120`.
- Runtime crossover: routed WPU becomes faster than serialized-token around
  `N≈124`.
- Runtime crossover: routed WPU becomes faster than dense-graph around `N≈178`.

The central v1 gap is therefore precise:

```text
WPU efficiency advantage appears at large N.
WPU accuracy advantage currently appears at medium N.
The next model must push the accuracy crossover beyond the runtime crossover.
```

Controlled stress tests:

- Relation noise: WPU-hybrid accuracy drop is `0.0250` from 0 to 128 irrelevant
  edges, while Graph Transformer drops `0.3438`.
- Affected-background deltas: serialized-token has the best affected-background
  MSE at the largest affected count; WPU is not broadly superior.

## V2 Working-Set Control Evidence

The main v2 lesson is that working-set control should be trained and selected
against downstream regret, not against a hand-written teacher. Earlier learned
retrievers imitated the interaction-density selector. That kept the mechanism
state-native, but it optimized teacher overlap rather than branch prediction
loss.

The first improvement was regret-distilled retrieval. On a validation split,
the system evaluates base candidates and generated candidates, then treats the
candidate set with the lowest downstream branch cross-entropy as a pseudo-label
object set. A small state-native object scorer is trained to recover that set
from object/event features.

Mean over five seeds at `N=2048`:

| K | Static learned interaction loss | Regret-distilled loss | Accuracy before | Accuracy after |
|---:|---:|---:|---:|---:|
| 8 | 0.988432 | 0.977017 | 0.506667 | 0.542222 |
| 16 | 0.966183 | 0.955077 | 0.504444 | 0.513333 |
| 32 | 1.004095 | 0.999112 | 0.475556 | 0.513333 |

The regret-distilled retriever wins 14 of 15 seed/K conditions against the
learned interaction retriever. The newer cross-seed result is stricter. It adds
candidate-level role/geometry/family descriptors and selects among static,
composition-aware, and invariant-scoring mechanisms using risk-adjusted
train-seed evidence.

Mean over five held-out seeds at `N=2048`:

| K | Static learned loss | Risk-adjusted mechanism loss | Accuracy before | Accuracy after |
|---:|---:|---:|---:|---:|
| 8 | 0.988432 | 0.982002 | 0.506667 | 0.522222 |
| 16 | 0.966183 | 0.951243 | 0.504444 | 0.517778 |
| 32 | 1.004095 | 1.002597 | 0.475556 | 0.522222 |

This is currently the strongest v2 working-set-control result. It supports a
sharper WPU claim: explicit state is useful not only because it enables sparse
propagation, but because it exposes object working-set selection and mechanism
routing as trainable pre-propagation control problems.

Negative results are equally important. Opaque set evaluators, score-margin
confidence gates, and strict no-harm seed-stable gates do not solve cross-seed
selection. The unresolved v2 problem is therefore not simply candidate
generation; it is invariant candidate descriptors, risk-aware mechanism
selection, and joint retriever-propagator training.

## Current Evidence Boundary

Supported:

- Explicit state-first modeling is implementable and trainable.
- The hard scheduler produces the expected sparse/hybrid/dense route crossover.
- WPU-family models are competitive or best in small-to-medium local synthetic
  regimes.
- WPU-hybrid is robust under irrelevant relation noise.
- Routed sparse execution can reduce CPU latency at large `N`.
- Regret-distilled state retrieval improves downstream loss over
  interaction-teacher retrieval in same-seed validation-to-test experiments.
- Risk-adjusted state-native mechanism selection improves held-out-seed loss
  over static learned selection at `K=8,16,32` in the current `N=2048` CWS
  setting.

Not supported:

- Universal WPU superiority over token or graph baselines.
- General physical understanding.
- End-to-end perception-to-state construction.
- Hardware-level advantage over GPU/NPU/TPU/LPU.
- Fixed `rho` thresholds as a final routing policy.
- Closed oracle gap for cross-seed candidate scoring; current risk-adjusted
  mechanism selection is positive but still far from the candidate oracle.

## Application Boundary

The commercial implication should be stated conservatively. WPU is not yet a
chiplet, robot OS core, or proven edge-AI processor. The nearer-term target is a
software runtime or middleware layer for systems where state is large, identity
persists, and each event changes a local subset: digital twin simulation,
simulator backends, game/server state synchronization, and robotics world-model
maintenance.

Hardware remains a possible future target, but only after the software version
measures the real costs that the current work proxies away: frontier queues,
relation fetch, scatter/gather, sparse kernel overhead, delta logs, branch
overlay memory, and state recovery.

## Research Direction

The correct high-impact framing is not a universal dominance claim. It is a
testable claim about when a different computational primitive is appropriate.

The next decisive step is experimental, not rhetorical:

- learned routing instead of fixed thresholds;
- regret-aware retrieval trained from downstream branch loss;
- invariant candidate descriptors and risk-adjusted mechanism selection across
  seeds and model instances;
- joint retriever-propagator training;
- stronger sparse propagation capacity at large `N`;
- regional dense correction for cases where pure sparse propagation loses
  global consistency;
- long-horizon rollout, branch consistency, and branch calibration;
- matched Dreamer/GNS/object-centric baselines;
- simulator-backed object dynamics and object-state adapters for perception;
- explicit state integrity: checkpoint, rollback, and consistency checks;
- hardware-aware profiling of frontier queues, relation fetch, scatter/gather,
  delta logs, and branch overlays.

Nature/Science-level direction requires showing a new computational principle
that works in a clearly specified regime and fails outside it in predictable
ways.

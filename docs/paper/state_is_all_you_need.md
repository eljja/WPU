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

WPU calls the required conversion **objectification**: a world observation or
simulator/database state is converted into persistent, addressable objects with
typed attributes, role/affordance state, relations, uncertainty, admissible
deltas, and branch-local overlays. The formal definition is maintained in
`docs/objectification.md`.
The implementation now treats this as a measurable contract through
`evaluate_objectification`: invalid identities, broken relation endpoints,
invalid deltas, and poor causal locality are reported before propagation.
When identity exists but local relation extraction misses edges, the prototype
can also apply conservative geometry-derived relation repair. Repaired edges are
logged hypotheses, not ground-truth physics. The repair probe now separates
nominal type labels from relation-bearing object state. Ungated geometry repair
recovers frontier recall but attaches distractors, and the precision collapse
worsens under denser distractors. Hand-written type gating preserves precision
only while type names match the gate. A small learned scorer transfers across
aliased type names when role/affordance variables are preserved, but fails when
both type and role information are removed. This makes the objectification
boundary falsifiable: WPU does not need names alone; it needs persistent state
variables that can support relation hypotheses. A toy downstream diagnostic
closes the first repair-to-prediction loop: role-aware learned repair improves
aliased-type branch accuracy from `0.343750` to `0.671875` and lowers loss from
`1.319667` to `0.885275`, while ungated dense-distractor repair restores frontier
recall but worsens loss.

A second toy probe tests whether relation candidates can be learned from object
histories and transferred to a held-out mechanism family. A history scorer
trained on `contact_transfer` and `support_transfer` reaches five-seed mean
relation precision/recall `0.987500` on held-out `hidden_field`, improving
downstream accuracy from `0.494531` to `0.992188`. This is still synthetic
evidence; it is not a claim of physical-law discovery.

A third toy probe tests relation-to-law transfer. A history-derived relation
selector plus an interpretable inverse-distance local law transfers to renamed
held-out `hidden_inverse`, reaching five-seed mean relation precision/recall
`0.988281` and delta MSE `0.000828` versus `0.445909` for no relation or type
prior. This is a controlled synthetic bridge from objectification to approximate
local theory, not evidence of unknown-law discovery.

An OOD version maps the failure boundary: the relation/law stack remains useful
under distance, gain, and denominator shifts, but far-distance relation recall
drops to `0.658594`, and shifted gain or law form leaves residual MSE even with
oracle relations. Thus the intended WPU loop is not "fit a law once"; it is
objectify, propose a local rule, stress it, and revise the rule when OOD error
separates relation failure from law mis-specification.

A revision probe shows this loop can be made operational in the synthetic
setting. With 64 calibration samples, gain calibration reduces
`hidden_inverse_gain_shift` MSE from `0.115978` to `0.000342`; form revision
reduces `hidden_power_shift` MSE from `0.054596` to `0.008887`. Oracle-relation
form revision reaches `0.000232`, identifying the remaining gap as relation
selection and calibration noise rather than only law-form capacity.
The implementation exposes this loop as reporting primitives:
`LocalLawHypothesis` stores an interpretable rule candidate, and
`LawRevisionReport` records base/revised error, revision decision, and optional
relation-selection versus law-residual attribution.

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

Slot Attention or IODINE can be used as perception-to-object-state front ends.
Graph Network-based Simulators can be used as propagation cores. Set/Graph
Transformers can be used as dense fallback or baselines. WPU asks what execution
substrate is appropriate once the system must repeatedly update objectified
world state rather than only process a fresh sequence.

In this framing, perception-to-state is an adapter problem rather than a solved
part of v1. The core WPU should accept object-state graphs from supervised
detectors, slot/object discovery, simulators, or logs, then expose how update
work, prediction risk, objectification quality, relation-repair precision, and
branch memory scale.

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
| 204 | WPU-sparse/routed | 0.4516 +/- 0.1957 | Graph Transformer | 0.7172 +/- 0.0615 | WPU fails; token/graph baselines dominate accuracy. |

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
| 8 | 0.988727 | 0.977017 | 0.508889 | 0.542222 |
| 16 | 0.966098 | 0.955077 | 0.504444 | 0.513333 |
| 32 | 1.004100 | 0.999112 | 0.480000 | 0.513333 |

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
  Cross-fit ensemble regret gating also fails to improve the gap, and
  descriptor-standardized group-DRO/no-harm gating is weaker than direct
  candidate-regret gating. The next issue is transfer-stable candidate scoring
  learned with retrieval and propagation rather than another detached deployment
  threshold.
- Robust branch-prior adaptation under mechanism shift. The PyBullet
  branch-prior audit shows that `catch_heavy` is prior-dominated: majority
  accuracy is `0.753968`, while best WPU reaches `0.408730`. This makes
  mechanism-aware branch priors and uncertainty-gated recompute required v2
  components, not optional calibration polish. A small 7-seed mechanism-prior
  adaptation probe raises shifted WPU win-rate to `0.666667`, but worsens mean
  shifted WPU ECE by `0.024819`. A prior-strength sweep finds an accuracy-best
  nonzero strength (`0.75`) but no nonzero strength that preserves or improves
  win-rate relative to `strength=0` without increasing ECE, so
  calibration-safe adaptation remains open. A calibration-selected prior
  follow-up improves shifted mean WPU ECE by `-0.046204` and Brier by
  `-0.105470`, but it leaves shifted WPU-vs-baseline win-rate at `0.333333`.
  This separates branch-probability calibration from robust mechanism
  generalization. A few-shot mechanism adaptation follow-up reaches shifted
  WPU-vs-baseline win-rate `1.000000` and mean margin change `0.050264`, but it
  uses mechanism-specific calibration samples. It is evidence for an adapted
  regime, not for zero-shot physical generalization. A mechanism-aware adaptive
  policy that chooses selected-prior adaptation for high prior-shift cases and
  few-shot parameter adaptation otherwise improves the adapted regime further:
  shifted win-rate remains `1.000000`, mean accuracy changes by `0.198412`,
  margin by `0.058201`, ECE by `-0.099347`, and Brier by `-0.155443`. This is
  detect-and-adapt evidence, not zero-shot evidence. A separate 7-seed
  composition-shift stress probe is a stronger zero-shot positive sub-regime:
  WPU local-dense wins all three compound mechanisms with mean accuracy delta
  `0.071428`, but mean ECE ratio is `1.014879`, so accuracy and calibration
  remain separate claims.
- Baseline-complete large-`N` simulator superiority. The PyBullet coverage
  audit now includes a low-training matched screen at N_bg=256, total `N=261`,
  where WPU, graph, and token baselines all complete, but the setting is too
  small to support strong accuracy superiority. The WPU-only N_bg=512, total
  `N=517` extension remains systems feasibility evidence because the
  graph-transformer baseline did not complete under the attempted protocol.

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
- explicit state integrity: checkpoint, rollback, finite-safe correction, and
  consistency checks. The current selective-corrected rollout reaches sparse
  H=25 integrity `0.958735` with zero rollback/escalation and corrected-object
  fraction `0.027461`, but correction trigger frequency remains `0.784166`, so
  raw sparse dynamics are not solved;
- hardware-aware profiling of frontier queues, relation fetch, scatter/gather,
  delta logs, and branch overlays.

Nature/Science-level direction requires showing a new computational principle
that works in a clearly specified regime and fails outside it in predictable
ways.

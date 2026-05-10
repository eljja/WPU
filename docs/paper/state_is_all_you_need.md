# State Is All You Need

## Abstract

Modern sequence models process the world through dense token streams. This paper
explores a different primitive: persistent world state. We introduce the
World-State Processing Unit (WPU), a state-centric neural architecture that
represents a scene as objects, relations, time, uncertainty, events, and future
branches. The first reference implementation, `WorldStateProcessor`, updates an
explicit state graph through sparse event-conditioned propagation, switches to
hybrid or dense recomputation when the affected world fraction grows, and
predicts multiple future branches as deltas over a base state. On a synthetic
robot-cup object physics task, the v1 model reduces next-state prediction MSE
from `0.8111` to `0.0005` after 200 training steps and improves branch
classification from `0.1289` untrained accuracy to `0.7188`, exceeding the
`0.6680` majority baseline. These results are not evidence of broad capability;
they establish a minimal reproducible substrate for state-first neural modeling.

## 1. Motivation

The dominant abstraction in current foundation models is the token sequence.
This abstraction is powerful, but it is not the natural substrate for many
embodied and simulation-heavy tasks. Physical scenes contain persistent
entities, typed relations, heterogeneous attributes, temporal memory, uncertain
observations, and multiple plausible futures. A cup near the edge of a table is
not merely a span of text; it is an object with position, support relations,
confidence, possible motion, and causal neighbors.

The WPU hypothesis is that some classes of intelligence should be modeled as
state transition over a structured world memory:

```text
WorldState != TokenSequence
WorldState = ObjectGraph + TemporalMemory + BeliefDistribution
```

The goal is not to replace Transformers in text modeling. The goal is to define
a reference architecture for learning over explicit world states, analogous in
spirit to how early Transformer work made attention a first-class computation.

## 2. State Input

The model input is:

```text
S = {O, R, T, P}
```

Where:

- `O` is a set of objects with typed, heterogeneous attributes.
- `R` is a set of typed relations between objects.
- `T` is temporal state, including current time and delta history.
- `P` is uncertainty, represented initially as confidence and branch
  probabilities.

The reference implementation batches state graphs as:

```text
StateGraphBatch = {
  object_features,
  relation_indices,
  relation_features,
  event_features,
  object_mask,
  relation_mask,
  target_indices,
  time_features,
  scheduler_metrics
}
```

This projection is a practical PyTorch interface. It does not reduce the theory
to token modeling: object identity, relation structure, event target, masks, and
state deltas remain explicit.

## 3. Model

`WorldStateProcessor` is a PyTorch `nn.Module` with three execution paths.

The sparse path performs event-conditioned message passing from the affected
frontier. Relation encodings are used as edge-conditioned messages, and an event
encoding biases which objects receive local updates.

The dense path applies global attention over the object set. In v1 this is a
small multi-head attention block over object embeddings. Its role is not to be a
Transformer over text, but to provide a learned global consistency operator when
local propagation becomes inefficient.

The hybrid path mixes sparse and dense representations around the affected
region. It is the transition regime between local graph propagation and full
state recomputation.

The model returns:

```text
StatePrediction = {
  object_delta,
  relation_logits,
  uncertainty,
  branch_logits,
  branch_probabilities,
  selected_paths
}
```

Branches are represented as:

```text
Branch = BaseState + DeltaState
```

This avoids copying the entire world state for every future hypothesis.

## 4. Sparse-Dense Routing

The scheduler selects the execution path using the crossover metric:

```text
rho = (delta_n * fanout ** depth * branches) / total_n
```

The v1 policy is:

```text
rho < 0.05       -> sparse
rho < 0.25       -> hybrid
otherwise        -> dense
```

This is a hard routing policy. Later versions should learn routing or calibrate
thresholds from runtime cost and prediction quality. The important point is that
state update cost is measured against the affected fraction of the world, not
against sequence length alone.

## 5. Initial Task

The first benchmark is a synthetic robot-cup object physics task.

Scene:

- `cup_001`
- `table_001`
- `hand_001`
- `edge_001`
- configurable background context objects

Relations:

- cup `on_top_of` table
- hand `near` cup
- cup `near` table edge

Event:

```text
hand_touched_cup(target=cup_001, force=f)
```

Targets:

- next object feature delta
- branch label: `stable`, `falls`, or `caught`

The task is intentionally small. Its purpose is to validate the architecture
surface: explicit state input, event-conditioned sparse update, branch
prediction, and route switching.

## 6. Results

Environment:

- Python 3.11.5
- PyTorch 2.11.0
- CPU execution
- synthetic object physics dataset
- 256 evaluation samples, seed `101`
- training: 200 steps, batch size 16, seed `13`

Main evaluation:

| Model | Next-State MSE | Branch NLL | Branch Accuracy |
|---|---:|---:|---:|
| Untrained WSP | 0.8111 | 1.2070 | 0.1289 |
| Majority baseline | n/a | n/a | 0.6680 |
| Trained WSP | 0.0005 | 0.8074 | 0.7188 |

Label distribution:

| Label | Count |
|---|---:|
| stable | 171 |
| falls | 33 |
| caught | 52 |

Routing sweep:

| Background Objects | Sparse Ratio | Hybrid Ratio | Dense Ratio |
|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0000 | 1.0000 |
| 20 | 0.0000 | 1.0000 | 0.0000 |
| 80 | 1.0000 | 0.0000 | 0.0000 |

The routing result captures the WPU intuition: the same local event becomes
dense when the entire world is small, hybrid at intermediate world size, and
sparse when the event affects only a small fraction of a larger state memory.

Baseline and ablation branch accuracy after 100 training steps:

| Model | N=4 | N=24 | N=84 |
|---|---:|---:|---:|
| WPU routed | 0.6719 | 0.7969 | 0.6719 |
| WPU sparse | 0.7812 | 0.8047 | 0.6719 |
| WPU hybrid | 0.7812 | 0.7969 | 0.7969 |
| WPU dense | 0.6719 | 0.6719 | 0.6719 |
| Dense graph | 0.4219 | 0.5000 | 0.5938 |
| Serialized token | 0.5625 | 0.6641 | 0.7891 |

## 7. Interpretation

The next-state MSE result shows that the model can learn the synthetic local
delta rule quickly. This is expected and should not be overstated.

The branch result is more relevant. The trained model exceeds the majority
baseline despite class imbalance, which means the event force, object position,
and hand-cup geometry are being used rather than simply predicting `stable`.
The margin is modest, so the next benchmark needs stronger branch calibration,
balanced labels, and held-out scenario families.

The route sweep is the clearest architectural signal. It demonstrates that WPU
routing depends on the fraction of affected world state. This differentiates the
prototype from ordinary dense sequence processing: the same event does not imply
the same compute pattern under different world memory sizes.

Propagation should be interpreted as a simplified local-physics prior. It is not
an exact physical law, but it captures a common operational structure: physical
changes often begin at a small set of entities and spread through contact,
support, proximity, or causal relations before any global correction is needed.
In that sense, WPU propagation is closer to a useful low-order physical
approximation than to an arbitrary message-passing trick.

The baseline results constrain the claim. WPU does not dominate every condition:
the serialized-token baseline is strongest at `N=84` in the 100-step suite. The
current evidence therefore supports the route-regime hypothesis more strongly
than it supports universal quality dominance. The next paper-quality benchmark
must compare matched models over accuracy, compute, and memory.

## 8. Current Limitations

This is a v1 research scaffold, not a finished model family.

- The dataset is synthetic and rule-generated.
- The scheduler is hand-coded, not learned.
- Dense fallback is a small attention block, not a hardware-optimized global
  recompute kernel.
- Uncertainty is limited to confidence heads and branch probabilities.
- The physical prior has not yet been validated on real robotics data, video
  perception, or long-horizon physical simulation.
- The current branch task has class imbalance and only three branch labels.

## 9. Next Experiments

The next paper-quality experiments should add:

- A balanced synthetic benchmark with controllable graph size, fanout, and
  causal depth.
- Ablations: sparse-only, dense-only, hybrid-only, and routed WPU.
- A learned routing head compared against the hard `rho` scheduler.
- Calibration metrics for branch probabilities.
- Long-horizon rollout error over multiple events.
- A real or simulator-backed object dynamics dataset.
- Runtime and memory profiling as `N`, `fanout`, `depth`, and `branches` vary.

## 10. Claim Boundary

The current claim should be narrow:

> Explicit world-state processing can be implemented as a learnable architecture
> with sparse event propagation, dense fallback, and delta-based branching, and
> the resulting reference implementation can be trained and evaluated
> reproducibly on a small object-physics task.

The current implementation should not claim general intelligence, broad physical
reasoning, or superiority over Transformers. It establishes the substrate needed
to test those claims later.

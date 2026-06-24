# WPU World-Copy Model

This document defines the long-term WPU target: a persistent executable copy of
the world that is updated through object state and causal propagation rather
than through repeated token serialization.

## Definition

A WPU world copy is a persistent, queryable, updateable state substrate:

```text
WorldCopy(t) =
  Objects(t)
  + Relations(t)
  + Regions(t)
  + EventLog(<=t)
  + Uncertainty(t)
  + DeltaLog(<=t)
  + Branches(t -> t+k)
```

It is not a perfect duplicate of physical reality. It is an executable state
model that maintains the identities, attributes, relations, uncertainty, and
possible futures needed to predict, update, and correct a world over time.

The word "copy" has an operational meaning:

- The same object can be referred to across events and time.
- Object attributes can be patched without rewriting the entire world.
- Relations determine which objects can propagate change.
- Events create local deltas and can open alternative branches.
- New observations can correct the stored state and record why correction was
  needed.
- Uncertainty controls whether the system uses sparse propagation, local dense
  recompute, global recompute, or external observation.

## Object-Oriented Processing Unit

WPU is object-oriented in the computational sense, not in the narrow programming
language sense. A WPU object is an execution unit with:

- persistent identity;
- mutable typed state;
- relation-bearing variables such as geometry, role, affordance, confidence,
  history, and physical/action context;
- admissible local deltas;
- typed relation channels for propagation;
- branch-local overlays;
- uncertainty and validity checks.

In a token processor, an object is usually represented indirectly as evidence in
a sequence. In WPU, an object is addressable state. A local event behaves more
like a method call on a set of related objects:

```text
event(object_i, action) -> delta(object_i)
delta(object_i) --relation_j--> delta(object_k)
BaseState + DeltaState -> updated WorldCopy
```

The intended performance benefit is not that state is always more accurate than
tokens. The benefit appears when the world is large, but each event only needs a
small causal working set. The processor should then update:

```text
retrieve(K) + propagate(K) + patch(K)
```

instead of repeatedly processing:

```text
serialize(N) + attend/recompute(N)
```

## Non-Negotiable Goal Alignment

The ultimate WPU goal is not to build a better graph benchmark model. It is to
build a processing model that can maintain an executable world copy. The
following requirements are non-negotiable for that goal:

- The persistent unit of computation is objectified state, not token order.
- Propagation must occur through typed relations and causal mechanisms, not
  through unrelated global averaging.
- Large-`N` claims require evidence that non-causal background state stays out
  of the event-local working set.
- Accuracy improvements must not be purchased by silently returning to full
  token serialization as the primary processing path.
- Sparse execution is allowed only when causal retrieval is reliable; otherwise
  the system must escalate and report why.
- World-copy quality must be evaluated over time, because a state model that
  wins one-step accuracy but drifts over long horizons is not a world copy.

This keeps WPU distinct from an LPU-style token processor. An LPU may emulate
state by processing long serialized contexts, but WPU should make identity,
relation traversal, local mutation, branch overlays, and correction native
operations. The comparison target is therefore not only token/sec, but
event-local state update quality per unit of latency, memory traffic, and
correction cost.

## Natural Propagation

Natural propagation means that state changes should flow through object
relations that approximate the structure of the world. It is a learned local
causality prior, not a claim that the current implementation has discovered full
physics.

For physical scenes, examples include:

- support: a table constrains a cup;
- contact: a hand can apply force to a cup;
- proximity: an edge changes fall risk only when near enough;
- containment: an object inside a container follows container motion;
- occlusion: a sensor-visible state can become uncertain;
- constraint: impossible positions or velocities should be corrected.

For non-physical state worlds, analogous propagation can represent dependency,
ownership, connectivity, permission, health, supply, or causal influence.

The v3 goal is to learn relation-conditioned local mechanisms:

```text
message = f_relation(source_state, target_state, relation_state, event_state)
delta_target = g(target_state, aggregate(messages), uncertainty)
```

The learned mechanism should be auditable: if prediction fails, the system
should expose whether the failure came from objectification, causal retrieval,
propagation dynamics, uncertainty calibration, or state correction.

## Required Runtime Components

The world-copy runtime must include the following components.

| Component | Role |
|---|---|
| State store | Persistent object, relation, region, uncertainty, and delta memory. |
| Causal index | Retrieves event-local working set `K` before tensor projection. |
| Propagation core | Applies learned relation-conditioned updates over `K`. |
| Branch manager | Maintains alternative futures as `BaseState + DeltaState`. |
| Correction loop | Reconciles predictions with new observations. |
| Integrity monitor | Detects invalid state, drift, stale objects, and uncertainty growth. |
| Dense fallback | Recomputes local or global state when sparse evidence is unsafe. |

## Benchmark Definition

A valid world-copy benchmark must provide:

- total world size `N`;
- event-local causal working set reference `K_ref` when available;
- object identities across time;
- object attributes and uncertainty;
- typed relations and relation noise settings;
- event/action stream;
- ground-truth future observations or simulator state;
- branch labels or distributional future outcomes when branching is present.

The benchmark should vary:

- non-causal background size;
- causal working set size;
- relation missing rate;
- relation false-positive rate;
- object creation/deletion;
- region migration;
- occlusion or confidence degradation;
- mechanism shift;
- horizon length.

## Success Metrics

WPU v3 should be judged with state-native metrics, not only token-like accuracy:

- causal slice recall and precision;
- selected `K` and affected fraction `K/N`;
- event latency;
- state updates/sec;
- bytes moved per update;
- next-state error;
- branch accuracy, NLL, Brier, and ECE;
- identity continuity;
- relation consistency;
- state-integrity score;
- correction cost;
- long-horizon trajectory error;
- dense fallback rate.

## Success Criteria

A v3 result should be considered strong only if it satisfies all of these
conditions in at least one large-`N` benchmark:

- `N` grows by at least an order of magnitude while selected `K` remains bounded
  or grows sublinearly.
- Causal slice recall remains high enough that sparse propagation does not drop
  necessary state.
- WPU matches or exceeds the best token/graph baseline on state or branch
  accuracy under a matched budget.
- WPU uses materially lower event latency, memory traffic, or dense compute.
- Long-horizon state integrity remains stable without constant global recompute.
- Failure cases are traced to objectification, retrieval, propagation,
  uncertainty, or correction rather than hidden in aggregate scores.

## Failure Criteria

The WPU v3 claim should be weakened if:

- selected `K` grows linearly with `N`;
- causal retrieval must scan the full world for every event;
- missing or noisy relations destroy causal slice recall;
- sparse propagation is fast but less accurate than token baselines at the same
  budget;
- long-horizon deltas drift without frequent dense recompute;
- object identity is unstable under streaming updates;
- correction cost approaches full-state recompute cost.

## Current Status

The current repository has the first v3 world-copy substrate pieces:

- `HierarchicalWorldState` for region/object membership.
- `WorldCausalIndex` for multi-signal causal retrieval.
- `world_copy_index_probe` showing selected `K = 4` as total `N` grows from
  `104` to `10004` in a controlled non-causal-background setting.
- `world_copy_causal_index_stress` showing controlled noisy-index behavior
  through `N=8192` with missing and false-positive relations.
- `world_copy_escalation_correction_probe` showing that local region correction
  candidates can recover causal update-set recall after low-confidence
  relation escalation.
- `world_copy_learned_correction_probe` showing that those bounded local
  correction candidates can reduce learned delta MSE in a controlled synthetic
  local-law setting.
- `world_copy_baseline_comparison_probe` providing the first same-task
  WPU/token/graph/dense comparison screen. The updated `wpu-region-guard` path
  keeps bounded selected `K` and much lower work/bytes proxy while beating the
  controlled dense/token baselines on raw delta MSE. The result depends on
  bounded, reliable local regions.
- `world_copy_streaming_region_guard_probe` extending the region-guard path to
  H=25 controlled streams with object churn and region migration. It preserves
  state integrity and zero correction cost in the bounded-region setup while
  avoiding full-state work/bytes.

This is necessary but not sufficient. It proves that the repository can express
large-world causal indexing and a first learned local correction diagnostic. It
does not prove full trained world modeling, real physical understanding,
perception-to-state construction, raw token/graph accuracy superiority, or
real-simulator long-horizon world-copy stability.

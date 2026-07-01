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

### Selective-region failure boundary

At N=8192 with 128 contaminants, an ordinary region guard selects about
134--136 objects and reaches MSE 0.361--0.477. A typed, confidence-ranked
`2*K_ref` guard keeps K at 16 and lowers MSE to 0.034--0.115. Region membership
is therefore a noisy index, not causal truth; objects absent from both region
and relation evidence remain unrecoverable by local retrieval.

### Dual-index omission correction boundary

The `world_copy_dual_index_escalation_probe` tests that unrecoverable boundary
directly. When causal objects are omitted from both the active region and
relation frontier, `wpu-selective-region-guard` loses recall. A bounded
`wpu-escalating-neighbor-guard` can recover much of that loss if the missing
objects remain in an adjacent observation/correction pool. At `N=8192`,
`dual_omission=0.75`, and `escape_rate=0.0`, the escalating guard improves
trajectory MSE from `0.416213` to `0.084905` while selected `K` rises only from
16 to 24. With `escape_rate=0.25`, it still improves MSE from `0.377478` to
`0.163802`, but dense state copy remains the raw-accuracy upper bound. This
fixes the next v3 boundary: bounded correction can recover omitted state only
when objectification exposes a nearby correction candidate; fully unobserved
causal objects require external observation, broader escalation, or dense
recompute.

### Uncertainty-triggered observation boundary

The `world_copy_uncertainty_observation_policy_probe` tests the next step:
causal objects that are absent even from the adjacent correction pool. The WPU
policy does not serialize the whole world. Instead, when local support evidence
is insufficient, it requests a bounded external observation probe and patches
only the returned objects into the causal slice. At `N=8192`, `escape_rate=0.75`,
and observation budget `8`, this improves trajectory MSE from the neighbor-only
range around `0.323295` to `0.098747`, while selected `K` remains `32` rather
than growing with `N`. At `escape_rate=0.50`, budget `8` improves MSE from
`0.255797` to `0.083280`. Dense state copy remains exact in this controlled
setup, so the result does not prove raw-accuracy superiority. It shows the
state-native correction pattern WPU needs for real world copies: uncertainty
must be able to buy bounded observation when local indexes are insufficient.

### Adaptive observation-budget boundary

The `world_copy_adaptive_observation_budget_probe` moves observation from a
fixed human-chosen budget to a bounded WPU decision. The policy combines local
support deficit with a cheap anomaly signal, so it does not spend observation
when missing support can be explained by harmless index noise. At `N=8192` and
`escape_rate=0.0`, adaptive observation spends budget `0.0`, keeps selected
`K=24`, and avoids the fixed-budget objective penalty (`0.082287` versus
`0.206153`). At `escape_rate=0.75`, it spends mean budget `4.3125` instead of
`8`, keeps selected `K=32`, and reaches MSE `0.079298` versus fixed-budget
`0.079620`, lowering the cost-aware objective from `0.199620` to `0.143985`.
This is not yet a learned policy, but it fixes the next runtime contract:
observation is a state-native correction decision with a bounded budget, not a
constant full-world fallback.

### Learned observation-policy boundary

The `world_copy_learned_observation_policy_probe` replaces the hand-specified
budget rule with a small learned classifier over WPU uncertainty summaries. In
the clean setting it can approximate the hand adaptive policy: at `N=8192` and
`escape_rate=0.75`, learned observation spends mean budget `4.625` versus hand
adaptive `4.71875`, keeps selected `K` bounded, and reaches objective
`0.163047` versus `0.163946`. At `escape_rate=0.50`, it reaches objective
`0.132664` versus hand adaptive `0.133466`. The failure boundary is anomaly
shift. Under `noisy_anomaly`, the learned policy over-spends budget and has
worse objective than the hand rule; under `weak_anomaly`, both learned and hand
adaptive policies under-observe and miss causal objects. The next correction
problem is therefore calibration under shifted observation signals, not simply
learning the clean rule.

### Calibrated learned observation-policy boundary

The `world_copy_calibrated_observation_policy_probe` adds a small calibration
set and neighbor-support credit before the learned budget decision. This attacks
the two shift failures directly: false high anomaly scores under
`noisy_anomaly`, and suppressed causal anomaly scores under `weak_anomaly`. At
`N=8192`, `escape_rate=0.75`, `noisy_anomaly` improves from learned objective
`0.258949` to calibrated objective `0.203604`, with mean budget reduced from
`6.0` to `4.625` and recall increased from `0.789062` to `0.871094`. Under
`weak_anomaly`, the same setting improves from `0.327296` to `0.184247`, with
recall rising from `0.414062` to `0.863281`. This is still not perfect: dense
state copy remains exact, and low-escape calibration can slightly perturb cases
where neighbor correction was already sufficient. The next failure is robust
calibration without relying on labeled shift calibration sets.

### Online observation-calibration boundary

The `world_copy_online_calibration_policy_probe` removes the labeled
calibration-set assumption and updates anomaly calibration from bounded
observation hit/miss feedback. The current version evaluates all modes on
paired event streams, adds a conservative stability gate, and adds
`wpu-verified-online-observation`: a bounded correction-policy verifier that
requests at most two extra observations only when the estimated marginal
correction value exceeds observation cost. This is closer to the intended
world-copy correction loop: WPU observes a small candidate set, measures whether
the observation repaired missing causal state, and adjusts future observation
sensitivity without serializing the full world.

At `N=8192`, `escape_rate=0.75`, `noisy_anomaly` improves from learned
objective `0.266230` to verified online `0.193618`, with recall rising from
`0.800781` to `0.957031`. Under `weak_anomaly`, verified online improves
learned objective `0.334783` to `0.202765`, improving on unverified online
`0.211687` and approaching labeled calibration `0.196455`; recall rises from
`0.390625` to `0.822266`. In the clean paired stream, verified online improves
learned objective `0.166575` to `0.159478`, moving toward hand adaptive
`0.154890`. The mean verifier top-up remains bounded and value-gated:
`0.171875` in clean, `0.0` in noisy anomaly, and `1.09375` in weak anomaly at
this setting.
Dense state copy remains exact but touches all `8192` state units.

The remaining boundary is now sharper, but the first safe base-budget result is
positive. Naive value trimming was negative because it removed tail observations
before measuring whether they repaired hidden causal state. The sequential
online verifier instead observes ranked candidates one at a time, stops only
after hit/miss evidence suggests that remaining marginal value is low, and falls
back to the full proposed budget when calibration is unstable. At the same
`N=8192`, `escape_rate=0.75` noisy setting, it reduces mean base observation
budget from `6.796875` to `6.140625`, keeps recall slightly higher
(`0.957031` to `0.960938`), and improves objective from online/verified
`0.193618` to `0.181400`, nearly matching the small labeled-calibration
objective `0.180837` without using a labeled calibration set.

The first composition policy is now implemented as
`wpu-composed-online-observation`: when online calibration indicates
under-observation (`offset > 0.03`), it selects the verified top-up path;
otherwise it selects sequential base-budget stopping. This selector preserves
the noisy sequential result (`0.181400`) and the weak verified result
(`0.202765`) while keeping `K` bounded near `32`. It is not a universal best
policy: in clean streams it is neutral (`0.166575`) and worse than verified
top-up (`0.159478`), while labeled calibration still wins under weak anomaly
(`0.196455`). The next failure is therefore not path composition itself, but
learning a no-harm composition gate that can also recover clean misses without
reintroducing noisy over-observation.

The learned no-harm composition gate is the next positive step. It is trained
from paired sequential-versus-verified outcomes using WPU-native feedback
features: calibration offset/scale, streak state, proposed budget, sequential
trim, observed hit precision, support deficit, and background anomaly pressure.
A conservative clean-recovery prior opens a small top-up only when the stream
has no sequential trim, high observed precision, remaining support deficit, and
low high-anomaly background fraction. At `N=8192`, `escape_rate=0.75`,
learned-composed recovers clean misses to the verified level (`0.159211` versus
sequential `0.166043`), preserves noisy no-harm at `0.181089` instead of
regressing to verified `0.193234`, and improves weak anomaly to `0.194131`,
better than verified `0.200840`, hand-composed `0.207904`, and sequential
`0.215912`. The remaining gap is no longer clean recovery, but replacing the
hand-coded clean-recovery prior with a learned safety-calibrated gate and
closing the weak gap to labeled calibration (`0.191310`) and hand adaptive
clean performance (`0.154867`).

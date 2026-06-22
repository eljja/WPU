# WPU Version 3 Plan

WPU v3 is the transition from a sparse object-physics prototype to a
world-copy processor. This document fixes the v3 definition, benchmark plan,
success criteria, and roadmap before further implementation.

## V3 Thesis

WPU v3 should test the following claim:

> A world-processing system should maintain a persistent object-state copy of
> the world, retrieve event-local causal working sets before tensorization,
> propagate state through learned relation-conditioned mechanisms, and correct
> the copy over time. In large-`N`, local-event regimes, this can produce a
> better accuracy/latency/memory/consistency trade-off than repeatedly
> serializing the world into token streams or densely recomputing the full graph.

This is still a conditional claim. v3 must not claim that WPU always beats LPU,
GPU, NPU, TPU, graph transformers, or token models.

## What Changes From V2

| Area | v2 | v3 target |
|---|---|---|
| State | Object graph for synthetic/PyBullet tasks | Hierarchical world copy with regions, recency, stale state, and streaming deltas |
| Retrieval | Event/frontier and CWS probes | Causal index with recall/precision under noisy, missing, and moving state |
| Propagation | Relation-conditioned sparse messages | Learned local mechanisms with validity, uncertainty, and correction feedback |
| Horizon | Mostly one-step plus selected rollout diagnostics | H=25/H=100 world-copy stability and correction cost |
| Baselines | Token/graph accuracy and latency screens | Matched token/graph comparisons on state-update/sec, memory traffic, identity continuity, and rollout stability |
| Claim | WPU helps in small identifiable `K` regimes | WPU is an object-oriented state-processing runtime for executable world copies |

## Fixed Definitions

The canonical definition of world copy is in `docs/world_copy_model.md`.

For v3, the core terms are:

- `N`: total maintained world state size.
- `K`: selected event-local causal working set.
- `K_ref`: reference causal working set when simulator or oracle labels exist.
- `causal slice`: object/relation subset selected before tensor projection.
- `object-oriented processing`: direct processing over persistent identity,
  mutable state, typed relations, local deltas, and branch overlays.
- `natural propagation`: learned local-causality updates over relation channels.
- `correction cost`: how much external observation, dense recompute, or state
  rewrite is needed to keep the copy valid.

## Architecture Target

```text
Observation / simulator / state source
        |
        v
Objectification adapter
        |
        v
Hierarchical WorldState
        |
        v
WorldCausalIndex(event) -> causal slice K
        |
        v
Relation-conditioned propagation core
        |
        v
DeltaState + Branch overlays
        |
        v
Integrity / uncertainty / correction loop
        |
        v
Updated WorldCopy(t+1)
```

The implementation should preserve one rule: sparse execution is only allowed
when the causal slice is reliable enough. Otherwise WPU must escalate to hybrid
or dense recompute and record why.

## Benchmark Program

### P1. Causal Index Stress

Goal: prove that v3 can retrieve `K` without full-state recompute as `N` grows.

Required sweeps:

- `N`: 128, 256, 512, 1024, 2048, 4096, 8192, 16384.
- `K_ref`: 4, 8, 16, 32.
- relation missing rate: 0, 0.1, 0.25, 0.5.
- relation false-positive rate: 0, 0.1, 0.25.
- region movement: none, local, cross-region.
- object churn: create/delete 0%, 1%, 5%.

Metrics:

- causal slice recall/precision;
- selected `K`;
- affected fraction;
- retrieval latency;
- bytes touched;
- false non-causal selection rate.

Success:

- recall remains high under moderate noise;
- selected `K` remains sublinear in `N`;
- retrieval cost is lower than serialized full-state scan.

### P2. Learned Mechanism Propagation

Goal: replace hand-shaped sparse updates with learned relation-conditioned
local mechanisms.

Required models:

- sparse WPU with relation-conditioned propagation;
- sparse WPU with mechanism-specific local heads;
- hybrid WPU with uncertainty-triggered local dense recompute;
- serialized-token baseline;
- graph-transformer baseline;
- dense graph baseline.

Metrics:

- next-state MSE/classification accuracy;
- branch accuracy/NLL/ECE;
- propagation no-harm rate;
- accuracy per byte touched;
- accuracy per millisecond.

Success:

- WPU equals or beats the best token/graph baseline in at least one large-`N`
  local-event regime;
- WPU retains materially lower event latency or memory traffic;
- failure cases are attributable to retrieval or propagation rather than hidden
  aggregate collapse.

### P3. Streaming World Store

Goal: maintain a world copy across a stream, not a single static graph.

Required events:

- object attribute update;
- object creation/deletion;
- identity merge/split;
- region migration;
- relation creation/deletion;
- confidence decay;
- stale-state eviction.

Metrics:

- identity continuity;
- stale-object rate;
- correction frequency;
- delta-log growth;
- branch-overlay memory;
- state-integrity score.

Success:

- state remains valid over long streams without full rewrite;
- correction cost stays below full-state recompute cost;
- identity and relation consistency remain measurable and auditable.

### P4. Long-Horizon World-Copy Rollout

Goal: verify that WPU state does not only predict one step, but remains useful
over time.

Required horizons:

- H=10, H=25, H=50, H=100.

Modes:

- no correction;
- uncertainty-triggered correction;
- scheduled observation correction;
- hybrid/dense escalation.

Metrics:

- trajectory MSE;
- target-object MSE;
- branch accuracy over horizon;
- state-integrity score;
- correction cost;
- rollback/escalation rate.

Success:

- WPU improves over current bounded-delta rollout;
- long-horizon integrity remains stable without constant global recompute;
- target-object error is reduced, not only aggregate background error.

### P5. Objectification Adapter Baseline

Goal: stop assuming perfect objectified state.

Initial acceptable sources:

- simulator-provided object labels;
- supervised segmentation/tracking;
- slot/object discovery as an optional harder path.

Metrics:

- identity recall;
- relation precision/recall;
- causal frontier recall;
- objectification score;
- downstream WPU loss under adapter errors.

Success:

- WPU can separate objectification failure from propagation failure;
- state repair and uncertainty escalation improve downstream behavior without
  hiding missing causal objects.

### P6. Token/LPU-Oriented Comparison

Goal: compare WPU to token processing on the right axes.

Required metrics:

- event latency;
- state updates/sec;
- memory bytes/update;
- accuracy at matched parameter budget;
- accuracy at matched latency budget;
- long-horizon consistency;
- identity persistence;
- branch rollout/sec.

Success:

- WPU demonstrates a regime where it is not only lower-work but also equal or
  better in task quality under matched budget;
- token baselines are allowed to win outside that regime.

## Success Gate For Calling It V3

The repository should not call the next milestone "WPU v3 complete" until all
of the following are true:

- The world-copy formalism is implemented in public APIs.
- The causal index benchmark reports recall/precision and latency through at
  least `N=8192`.
- At least one learned propagation benchmark is baseline-complete.
- At least one streaming benchmark includes object churn and region migration.
- At least one H=25 or longer rollout improves over the current v2 stability
  baseline.
- Documentation reports negative results and failure modes alongside wins.

## Immediate Implementation Order

1. Add benchmark scaffolding for streaming world-copy events.
2. Extend `WorldCausalIndex` to report candidate relation paths and retrieval
   cost.
3. Add noisy-relation causal-index stress script.
4. Add relation-conditioned mechanism module that consumes causal slices.
5. Add long-horizon world-copy rollout runner with correction accounting.
6. Add token/graph matched baselines for the same world-copy stream.
7. Update paper/claim docs only after the benchmark evidence exists.

## Claim Boundary

The publishable v3 claim should be:

> WPU v3 implements an object-oriented state-processing runtime for executable
> world copies. It shows that large persistent worlds can be updated through
> event-local causal slices when objectification and causal retrieval are
> reliable, and it measures the regimes where this beats token or dense graph
> processing on latency, memory traffic, and state consistency.

The non-publishable claim is:

> WPU has solved real-world physical understanding or universally replaces
> token/LPU-style processing.


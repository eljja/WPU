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

## Alignment Audit

The fixed v3 plan is aligned with the ultimate WPU goal only if each work cycle
preserves these constraints:

- The primary representation remains persistent objectified state.
- Token baselines are used for comparison, not as the hidden implementation
  path for WPU.
- Every large-`N` improvement reports whether selected `K` stays bounded,
  sublinear, or grows with `N`.
- Every accuracy claim is paired with latency, memory traffic, dense fallback,
  or correction-cost evidence.
- Every world-copy claim includes horizon or streaming evidence, not only
  one-step prediction.
- Every failure is assigned to objectification, retrieval, propagation,
  uncertainty, correction, or systems overhead.

If a proposed improvement violates these constraints, it should be treated as a
baseline or ablation, not as WPU v3 progress.

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

## Current V3 Evidence Boundary

The v3 substrate now includes controlled evidence for causal indexing,
noisy-index stress, escalation-region correction candidates, and a first
learned local-delta correction probe. The latest learned-correction probe is a
P2 substrate positive: after low-confidence relation escalation, bounded local
region candidates reduce synthetic delta MSE from `0.275312` to `0.006365` at
max selected `K=16`.

The updated same-task baseline-comparison screen is conditionally positive.
`wpu-region-guard` keeps max selected `K=16`, mean work proxy `9.333333`, and
mean bytes proxy `336.000000`, while reaching raw delta MSE `0.002646` versus
dense graph `0.003810` and serialized token `0.003223`. The negative ablation
is also important: `wpu-hybrid-context` remains weak at MSE `0.020904`, so the
useful mechanism is bounded local-region guarding, not generic context
concatenation.

The first streaming region-guard screen extends this into a controlled H=25
world-copy setting with object churn and region migration. `wpu-region-guard`
keeps max selected `K=8`, trajectory MSE `0.000000`, integrity `1.000000`, and
correction cost `0.000000`, while dense state copy matches integrity only with
full-state work/bytes that scale with `N`. This is still oracle-law evidence;
real learned dynamics and mis-objectified regions remain unsolved.

The dual-index omission escalation probe attacks the next mis-objectification
boundary. When causal objects are absent from both the active region and
relation frontier, the bounded selective guard loses recall. If those objects
remain in a bounded adjacent observation pool, `wpu-escalating-neighbor-guard`
recovers much of the loss while keeping selected `K` bounded. At `N=8192`,
`dual_omission=0.75`, and `escape_rate=0.0`, MSE improves from `0.416213` to
`0.084905` with selected `K=24`; at `escape_rate=0.25`, MSE improves from
`0.377478` to `0.163802`. Dense state copy still wins raw accuracy. The
scientific boundary is therefore sharper: WPU can stay sublinear only if
uncertainty escalation has access to bounded correction candidates; if causal
objects are absent from all local indexes and observation pools, sparse WPU must
ask for external observation or escalate beyond the local budget.

The uncertainty-observation policy probe implements that request path in a
bounded form. When local support evidence is insufficient, WPU asks for a small
external observation budget and patches only the observed objects into the
causal slice. At `N=8192`, `escape_rate=0.75`, and observation budget `8`,
`wpu-uncertainty-observation` improves MSE from the neighbor-only range around
`0.323295` to `0.098747` while keeping selected `K=32`. At `escape_rate=0.50`,
the same budget improves MSE from `0.255797` to `0.083280`. Dense state copy
still wins exact raw accuracy, so this is not a universal superiority result.
It is a v3 correction-loop result: WPU can remain sublinear after local-index
failure if uncertainty buys bounded observation rather than full-state
serialization.

The adaptive observation-budget probe then removes the fixed observation budget
assumption. It uses local support deficit plus a cheap anomaly signal to decide
whether and how much bounded observation to request. At `N=8192`,
`escape_rate=0.0`, it spends no observation budget and avoids the fixed-budget
objective penalty (`0.082287` versus `0.206153`). At `escape_rate=0.75`, it
spends mean budget `4.3125` rather than `8`, keeps selected `K=32`, and reaches
MSE `0.079298` versus fixed-budget `0.079620`, improving the cost-aware
objective from `0.199620` to `0.143985`. This is still hand-specified
uncertainty logic, not a learned policy, but it makes observation budget a
native WPU correction-loop variable.

The learned observation-policy probe replaces that hand rule with a small
budget classifier over WPU uncertainty summaries. In the clean `N=8192`
setting, learned policy matches the hand adaptive rule closely: at
`escape_rate=0.75`, objective is `0.163047` versus hand adaptive `0.163946`;
at `escape_rate=0.50`, objective is `0.132664` versus `0.133466`. The negative
boundary is shift calibration. Under noisy anomaly signals, the learned policy
over-spends budget; under weak anomaly signals, both hand and learned policies
under-observe. The next unresolved criterion is therefore calibrated learned
uncertainty under shifted observation signals.

The calibrated observation-policy probe reduces that shift failure with a small
calibration set and neighbor-support credit. At `N=8192`,
`escape_rate=0.75`, `noisy_anomaly` improves from learned objective `0.258949`
to calibrated objective `0.203604`; `weak_anomaly` improves from `0.327296` to
`0.184247`. Selected `K` remains bounded by the local correction/observation
budget. The remaining boundary is important: this uses labeled calibration
samples and does not yet solve unlabeled or online calibration.

The online observation-calibration probe attacks that boundary by using
post-observation hit/miss feedback instead of a labeled shift calibration set.
The updated version evaluates modes on paired event streams, adds conservative
stability gating, and adds a value-gated bounded correction-policy verifier. At
`N=8192`, `escape_rate=0.75`, verified online improves `noisy_anomaly`
objective from learned `0.266740` to `0.193756`, and `weak_anomaly` from
`0.333978` to `0.201456`, while keeping selected work near `32` rather than
dense `8192`. On the paired clean stream, verified online improves learned
objective `0.165791` to `0.159096`, approaching hand adaptive `0.154890`. The
verifier spends mean top-up budget `0.171875` in clean, `0.0` in noisy anomaly,
and `1.09375` in weak anomaly; the corresponding estimated top-up values are
`0.006895`, `0.0`, and `0.043876`. This is positive WPU-native correction
evidence, but not a closed solution: noisy shift still trails labeled
calibration because the base online budget is too large. The next criterion is
base-budget value calibration, not simply top-up gating.

This does not complete P2. The missing next step is a baseline-complete learned
propagation benchmark that compares WPU against token/graph/dense baselines on
state accuracy, latency, memory traffic, and long-horizon integrity under the
same world-copy stream.

## Iteration Protocol

Each repeated improvement cycle should follow the same loop:

1. Select the highest-priority failing criterion from this document.
2. Inspect the current code, tests, reports, and claim docs before editing.
3. Implement the smallest WPU-native change that attacks the failure.
4. Add or update a benchmark that can falsify the change.
5. Run the benchmark and record both positive and negative results.
6. Compare against token/graph baselines when the claim involves superiority.
7. Update claim boundaries before strengthening any public claim.
8. Run tests, commit, and push only the relevant files.

Per-cycle deliverables:

- code or benchmark change;
- reproducible command;
- CSV or markdown report;
- English and Korean documentation update when claims change;
- explicit statement of what improved, what failed, and what remains next.

The next cycle should start from the remaining failure with the highest impact
on the world-copy goal, not from the easiest figure to improve.

## Reusable Continuation Prompt

Use this prompt to continue WPU v3 work in a new session:

```text
Continue WPU v3 toward the ultimate goal: an object-oriented world-state
processing unit that maintains an executable world copy through persistent
objectified state, causal indexing, learned relation-conditioned propagation,
delta/branch overlays, uncertainty, and correction.

Do not turn WPU back into a token processor. Token/graph/LPU-style models are
baselines only. WPU progress must preserve object identity, relation traversal,
event-local causal working sets, state patching, and long-horizon world-copy
integrity as native operations.

First inspect docs/world_copy_model.md and docs/versions/wpu_v3_plan.md, then
inspect the current code, tests, and latest experiment reports. Pick the
highest-impact unresolved v3 criterion, implement the smallest WPU-native
improvement, add a falsifiable benchmark or test, run it, record positive and
negative results, update English/Korean docs if claims change, run the relevant
tests, and commit/push the work.

Always report:
- what WPU failure was targeted;
- what changed in code or benchmark;
- whether selected K remains bounded or sublinear as N grows;
- whether accuracy, latency, memory traffic, correction cost, or state integrity
  improved;
- whether token/graph baselines still win in any regime;
- what remains the next highest-priority failure.
```

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

The selective-region stress result adds a concrete boundary: ranking and
bounding contaminated region candidates prevents K growth, but dense state copy
still wins raw accuracy and dual omission from membership and relation evidence
remains unresolved.

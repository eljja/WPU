# WPU Version 2 Architecture

WPU v2 is not a return to token processing. It is a stronger state-native model
that treats world processing as indexed causal retrieval, local propagation,
delta construction, and branch-aware rollout.

## V2 Goal

WPU v2 should expand the regime where WPU is better than token and dense graph
baselines. The target regime is:

```text
large total state N
small or moderately bounded causal working set K
identifiable affected state
local relation-mediated propagation
branching future uncertainty
```

The v2 success criterion is not only higher accuracy. It is higher accuracy per
unit of routed work, lower latency growth with N, and better long-horizon state
stability.

## Architecture Overview

```text
WorldState
  -> Causal Index
  -> Event-Conditioned Retriever
  -> Adaptive Working Set K
  -> Local Propagation Core
  -> Delta/Branch Engine
  -> Consistency and Uncertainty Manager
  -> Updated BaseState + DeltaState overlays
```

## 1. State Store

The state store is identity-addressed memory, not a serialized sequence.

It contains:

- Object records: id, type, attributes, confidence, last update time.
- Relation records: source, target, relation type, strength, confidence.
- Temporal records: event log, recent changes, object update age.
- Uncertainty records: object confidence, relation confidence, branch entropy.
- Delta records: branch-local patches against a shared base state.

The v2 implementation should expose this memory through retrieval APIs rather
than forcing every model call to scan every object.

## 2. Causal Index

The causal index is the main missing system component in v1. It should support
retrieval by:

- Direct event target id.
- Relation adjacency and relation type.
- Spatial bucket or proximity.
- Object type compatibility with event type.
- Recent change set.
- High-uncertainty hot set.
- Branch-local delta overlays.

The target complexity is:

```text
selector cost ~= O(log N + K + local fanout)
```

not:

```text
selector cost ~= O(N)
```

This is the core difference between WPU as an architecture and a graph neural
network that simply masks objects after scanning them all.

## 3. Event-Conditioned Causal Retriever

The retriever chooses the initial affected set. V1 shows that oracle CWS can
outperform baselines but learned CWS remains unstable. Therefore v2 should treat
retrieval as a first-class learned-and-indexed module.

Inputs:

- Event type and event vector.
- Target object id.
- Target object type.
- Relation neighborhood.
- Spatial neighborhood.
- Uncertainty and recency features.

Outputs:

- Ranked object ids.
- Ranked relation ids.
- Initial K.
- Retrieval confidence.
- Expected propagation depth and fanout.

Training signals:

- Binary causal membership loss.
- Top-k recall.
- Branch accuracy downstream loss.
- Delta reconstruction downstream loss.
- Penalty for unnecessary K growth.

## 4. Adaptive K Scheduler

V1 uses mostly fixed working-set size. V2 should make K adaptive:

```text
K = f(event type, uncertainty, fanout, branch entropy, relation density)
```

The scheduler should start from a small K and expand only if needed.

Expansion triggers:

- Low selector confidence.
- High branch entropy.
- Constraint violation after local update.
- Missing support relation.
- Collision/contact fanout.
- Uncertainty growth above threshold.

The scheduler should produce a budgeted decision:

```text
SPARSE(K small)
LOCAL_DENSE(K moderate)
EXPAND(K grows by frontier)
FALLBACK(global or large subgraph only when unavoidable)
```

## 5. Local Propagation Core

The propagation core should not be a global attention block. It should operate
on the selected causal subgraph.

Recommended structure:

1. Relation-typed message passing.
2. Event injection at every propagation layer.
3. Local dense block inside the selected K-subgraph.
4. Constraint heads for support, contact, containment, attachment, and motion.
5. Delta heads that predict object, relation, and uncertainty updates.

Relation types should have different update priors:

- `near`: weak candidate propagation.
- `touching`: strong force/contact propagation.
- `on/supports`: stability and gravity-relevant propagation.
- `holding/attached`: constrained motion propagation.
- `inside/contains`: containment and collision propagation.

## 6. Delta/Branch Engine

V2 should treat branch prediction as future delta generation, not as a detached
classification head.

A branch is:

```text
Branch = BaseState + DeltaState trajectory + probability + uncertainty
```

The branch engine should predict:

- Candidate delta states.
- Branch probabilities.
- Branch-specific uncertainty.
- Pruning and merge decisions.
- Probability trajectory over horizon.

This is where WPU should differentiate strongly from token models. The model
does not reserialize a whole world for every possible future; it overlays
branch-specific deltas on the same base state.

## 7. Consistency and Uncertainty Manager

The manager decides when the local update is sufficient.

It checks:

- Object constraints.
- Relation constraints.
- Physical plausibility proxies.
- Branch entropy.
- Selector confidence.
- Delta magnitude.
- Accumulated rollout drift.

If consistency fails, it requests:

- K expansion.
- Relation frontier expansion.
- Local dense recompute.
- Branch split.
- Uncertainty increase.

This mechanism is how v2 should widen WPU's useful regime without falling back
to token-like global processing.

## V2 Public Interfaces

Planned APIs:

```python
class CausalIndex:
    def query(event, state, budget) -> CausalWorkingSet: ...

class AdaptiveScheduler:
    def choose(event, state, retrieval, metrics) -> RouteDecision: ...

class WorldStateProcessorV2(nn.Module):
    def forward(state, event, horizon=1, branch_budget=3, work_budget=None): ...

def closed_loop_rollout(model, state, actions, horizon, branch_budget): ...
```

The model should report:

- Selected K.
- Causal recall estimate.
- Work estimate.
- Route path.
- Branch entropy.
- Consistency failures.
- Expansion count.

## V2 Success Criteria

V2 should be considered meaningfully better than v1 if it shows:

- Smaller learned-vs-oracle selector gap.
- Stable accuracy across dense N-sweeps.
- Lower latency growth with N than token and dense graph baselines.
- Better robustness under adversarial distractors.
- Better long-horizon branch/delta stability.
- Evidence that indexed retrieval reduces selector cost from O(N)-like behavior.


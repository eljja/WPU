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

Implementation status: v2 now includes a pre-tensor indexed path that projects
`WorldState` to the event-local subgraph before `StateGraphBatch`
construction. This is the required direction for WPU; full-state tensorization
followed by masking is only a research fallback.

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

Current prototype:

- `wpu.core.causal_index.CausalIndex`
- `collate_indexed_working_set_samples`
- `--pre-tensor-indexed` in `scripts/causal_working_set_experiment.py`

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

Current implementation status:

- Implemented: hard sparse/local-dense routing in
  `wpu-cws-indexed-adaptive-hybrid`.
- Implemented: differentiable sparse/local-dense gating in
  `wpu-cws-indexed-learned-hybrid`.
- Implemented: learned selective routing prototype in
  `wpu-cws-indexed-learned-selective-hybrid`, with route compute and
  distillation losses.
- Implemented: interaction-aware sparse/local-dense routing in
  `wpu-cws-indexed-interaction-hybrid`.
- Implemented: selective interaction-aware execution in
  `wpu-cws-indexed-selective-interaction-hybrid`, which executes local dense
  only for samples above an interaction threshold.
- Route signal: selected K pressure and selector confidence.
- Interaction route signal: pairwise spatial density inside the indexed
  working set.
- Reported metrics: sparse ratio, local-dense ratio, selector confidence, and
  selected execution path.
- Compute-aware metric: `dense_compute_ratio` reports whether the local dense
  block was actually executed. This is distinct from `local_dense_ratio`, which
  reports representation mixing.
- Not yet complete: learned K expansion, violation-triggered frontier growth,
  compute-regularized gating, and calibrated route probabilities.

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

Current experimental stressors:

- `interaction_mode=standard`: mostly sparse-solvable object physics.
- `interaction_mode=pairwise`: branch labels depend on pairwise obstacle
  spacing inside the causal working set, creating a direct test for when local
  dense recompute helps.

Current lesson:

State-local interaction structure improves accuracy in the pairwise stress
test, but dense recompute is only a valid WPU advantage if it is selectively
executed. A low dense mixing ratio is not enough. The first selective
interaction prototype now executes dense only for high-interaction samples,
preserving most of the interaction-hybrid accuracy in a two-seed pilot while
reducing `dense_compute_ratio`. A threshold sweep shows a real
accuracy-compute frontier, with threshold 0.15 as the current fixed operating
point. A five-seed validation supports the Pareto claim: selective execution
keeps accuracy within the full interaction confidence intervals while reducing
actual dense execution, especially at K=8 and K=16. The remaining architecture
need is learned or calibrated routing rather than a fixed threshold. The first
learned-selective prototype is a negative result: downstream loss, route cost,
and simple distillation are not enough to reproduce the fixed interaction
frontier. Counterfactual sparse-vs-dense diagnostics show why: dense fixes some
sparse mistakes but also breaks some sparse-correct cases. Future routing needs
stronger supervision from branch correctness, uncertainty, and
constraint-failure signals. A seed-heldout route-label probe confirms that
interaction density alone is not enough to identify dense-needed samples; the
rank metric is below chance, so it is actively misleading as a dense-needed
proxy. The first scalar sparse-diagnostic probe also fails to generalize, and
train-calibrated thresholds do not transfer reliably. The next router needs
calibrated model-internal route representations and counterfactual dense-needed
supervision, not only hand-built scalar diagnostics.

The shared-model counterfactual probe further tightens this requirement. When
sparse and local-dense paths share parameters and are both trained, dense
recompute is still not a default improvement. It can fix sparse failures, but
it can also break sparse-correct predictions. The scheduler should therefore
estimate route regret:

```text
route_regret = dense_loss - sparse_loss + compute_cost
```

Dense fallback should be executed only when expected regret is negative, not
merely when local interaction density is high.

The first shared-route sample probes support this direction. Binary
dense-needed labels remain hard to identify, but the loss-based
dense-beneficial signal is more predictable from state features. V2 routing
should therefore move from binary dense-needed classification toward
cost-sensitive regret estimation while preserving hard sparse/dense execution
at inference time.

The continuous regret probe gives the first positive learned-routing signal:
state features can reduce expected loss relative to always-sparse under a
moderate dense compute cost. This suggests the architectural target should be a
jointly trained route-regret head. Post-hoc scalar sparse diagnostics are not
enough; they over-route to dense and increase loss in the current probe.

The first internal regret-head attempt shows that this target is non-trivial.
A naive jointly trained head collapses to always-sparse in the current pilot.
The architecture should therefore treat regret routing as a staged optimization
problem rather than a single auxiliary loss attached to an unstable propagation
core.

The staged route-regret pilot supports this design change. Training sparse and
local-dense propagation first, then freezing the propagation core and fitting a
calibrated route-regret head, reduces held-out prediction loss relative to
always-sparse while executing dense recompute on only a minority of samples.
This does not yet prove accuracy superiority, but it makes the scheduler
operational: dense fallback is now a cost-sensitive decision over state
representations, not a fixed heuristic or an unconditional dense path.

The five-seed audit narrows the requirement further. The route-regret head has
a repeatable positive ranking signal, but validation threshold calibration is
not uniformly stable. The scheduler should therefore evolve from a fixed scalar
threshold to a cost-conditioned utility decision with a sparse-favoring margin:

```text
execute_dense if E[dense_loss - sparse_loss + compute_cost] < -margin
```

Near-tie samples should remain sparse because dense recompute consumes work and
can overwrite a correct sparse prediction.

## 6. Delta/Branch Engine

V2 treats branch prediction as future delta generation, not as a detached
classification head. The current CWS implementation now conditions branch
logits on the predicted selected-object delta summary, so branch supervision
trains the delta path instead of only a pooled classifier.

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

Current implementation status:

- Implemented: one-step delta-conditioned branch scoring in
  `CausalWorkingSetProcessor`.
- Not yet complete: branch-specific multi-step delta trajectories, branch
  merge/prune operators, and calibration-aware branch uncertainty.

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

Current implementation status:

- Implemented: confidence/K-triggered sparse-to-local-dense fallback.
- Not yet complete: closed-loop constraint violation feedback into K expansion,
  branch split, or uncertainty growth.

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

Implemented model names:

- `wpu-cws-indexed`
- `wpu-cws-indexed-sparse`
- `wpu-cws-indexed-local-dense`
- `wpu-cws-indexed-adaptive-hybrid`
- `wpu-cws-indexed-learned-hybrid`
- `wpu-cws-indexed-learned-selective-hybrid`
- `wpu-cws-indexed-interaction-hybrid`
- `wpu-cws-indexed-selective-interaction-hybrid`
- `wpu-cws-indexed-geometry-hybrid`
- `wpu-cws-indexed-regret-hybrid`

## V2 Success Criteria

V2 should be considered meaningfully better than v1 if it shows:

- Smaller learned-vs-oracle selector gap.
- Stable accuracy across dense N-sweeps.
- Lower latency growth with N than token and dense graph baselines.
- Better robustness under adversarial distractors.
- Better long-horizon branch/delta stability.
- Evidence that indexed retrieval reduces selector cost from O(N)-like behavior.
- Evidence that adaptive dense fallback reduces actual dense execution cost,
  not only dense-output mixing.
- Evidence that the route-regret oracle gap shrinks under staged calibration,
  not only that dense fallback sometimes helps.
- Evidence that the scheduler is stable under fixed thresholds, held-out seeds,
  or explicit cost-conditioned margins.

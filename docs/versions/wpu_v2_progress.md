# WPU V2 Progress

This document records what has been completed after the v1 closure and what
the current v2 evidence implies. Already completed dense N-sweep work is not
repeated here.

## Implemented V2 Components

### Causal Index

Added `wpu.core.causal_index.CausalIndex`, a first state-indexed retrieval
primitive. It retrieves the event target and bounded relation frontier from
state graph structure rather than relying only on learned global scoring.

Current limitation: the PyTorch model still encodes the full object tensor
before applying indexed selection unless the experiment uses
`--pre-tensor-indexed`. The v2 code now contains both paths: full-tensor
selection for controlled ablations and pre-tensor indexing for the actual WPU
systems claim.

### Indexed WPU Models

Added model names:

- `wpu-cws-indexed`
- `wpu-cws-indexed-sparse`
- `wpu-cws-indexed-local-dense`
- `wpu-cws-indexed-adaptive-hybrid`
- `wpu-cws-indexed-learned-hybrid`
- `wpu-cws-indexed-interaction-hybrid`
- `wpu-cws-indexed-geometry-hybrid`

These support priority 5 and 6:

- indexed retrieval
- local sparse propagation
- local dense propagation within the selected state subset
- hard adaptive routing between sparse and local-dense propagation
- learned differentiable routing between sparse and local-dense propagation
- interaction-aware routing from state-local pairwise geometry
- geometry-only local interaction features without executing the local dense
  transformer block

### Closed-Loop Rollout

Added `scripts/cws_closed_loop_rollout.py`.

Unlike the previous diagnostic long-horizon evaluator, this script applies
predicted `DeltaState` overlays back to `WorldState` at each step. This makes
branch entropy, branch switching, changed-object count, delta norm, and
constraint violations visible over time.

### Delta-Conditioned Branch Head

Updated `CausalWorkingSetProcessor` so branch logits are conditioned on the
candidate object delta predicted for the selected working set. Branch loss now
backpropagates into the delta head, making the implementation closer to the v2
claim:

```text
Branch = BaseState + DeltaState trajectory + probability + uncertainty
```

This replaces the earlier detached pooled-state classifier for the CWS model.
It is still a compact one-step branch scorer, not a full multi-step branch
generator, but the model now couples branch probabilities to predicted state
patches.

### Adaptive Sparse/Local-Dense Route

Added `wpu-cws-indexed-adaptive-hybrid`.

This model starts from the indexed working set and selects a sparse or
local-dense propagation path per sample. The first policy is deliberately hard
and inspectable:

```text
use local dense if selected K is large or selector confidence is low
otherwise use sparse propagation
```

The model reports `sparse_ratio`, `local_dense_ratio`, and
`mean_selector_confidence` through `WorkingSetStats`, and returns `SPARSE` or
`HYBRID` in `StatePrediction.selected_paths`.

Added `wpu-cws-indexed-learned-hybrid`.

This model keeps the same pre-tensor indexed working set, but replaces the
hand threshold with a differentiable gate:

```text
state-local sparse representation
state-local dense representation
learned gate -> convex mixture -> delta and branch heads
```

The gate is trained through the same delta and branch losses. It does not
serialize the world or use global token attention; it learns whether local
dense recompute is useful inside the already selected causal state.

Added `wpu-cws-indexed-interaction-hybrid`.

This model computes a state-local interaction density from pairwise distances
inside the selected working set and uses that signal to choose how much
local-dense recompute to mix into the sparse representation. It is not a token
fallback and does not scan global state; the route decision is made from the
already indexed causal state.

Added `wpu-cws-indexed-geometry-hybrid`.

This model tests the stricter compute claim. It uses pairwise state geometry as
an explicit local feature, but it does not execute the local dense transformer.
It separates two quantities that were previously conflated:

```text
local_dense_ratio = how much dense-style representation is mixed into output
dense_compute_ratio = whether the dense transformer block was actually run
```

## Completed V2 Priority Experiments

### Compute-Aware Metric Fix

Output:

- `docs/experiments/wpu_v2_compute_aware_pairwise_pilot.csv`
- `docs/experiments/wpu_v2_compute_aware_pairwise_pilot_results.md`

Finding:

The previous `local_dense_ratio` metric was not enough. For
`wpu-cws-indexed-interaction-hybrid`, it measured the amount of dense result
mixed into the output, but the dense transformer block was still executed for
every sample. Therefore low `local_dense_ratio` is not evidence of low dense
compute cost.

The code now reports `dense_compute_ratio` separately. This is the metric that
must be used for compute-cost claims.

### Priority 1: Selector Gap

Output:

- `docs/experiments/wpu_v2_selector_gap.csv`
- `docs/experiments/wpu_v2_selector_gap_results.md`

Finding:

Frontier and oracle are effectively identical on the current CWS task because
the synthetic causal core is directly relation-connected to the event target.
Learned selection is often faster but less stable. This means the current
dataset is too easy for relation-indexed retrieval and should be made harder
with ambiguous distractors, hidden support relations, and multi-hop effects.

### Priority 2: K Sweep Pilot

Output:

- `docs/experiments/wpu_v2_k_sweep_pilot.csv`
- `docs/experiments/wpu_v2_k_sweep_pilot_results.md`

Setup:

- N = 4096
- K = 4, 8, 16, 32, 64
- Seeds = 11, 13
- Steps = 100
- Working-set cap = 64

Finding:

One-step accuracy is noisy. The most important result is not a clean monotonic
accuracy curve but that K growth increases latency and exposes stability
differences between local sparse and local dense variants. The K sweep must be
rerun with five seeds and closed-loop consistency metrics before it becomes
paper evidence.

### Priority 3: Distractor Sweep Pilot

Output:

- `docs/experiments/wpu_v2_distractor_sweep_pilot.csv`
- `docs/experiments/wpu_v2_distractor_sweep_pilot_results.md`

Setup:

- N = 4096
- causal K = 8
- adversarial distractors = 0, 32, 128, 256
- Seeds = 11, 13
- Steps = 100

Finding:

Latency increases sharply with distractor count because the current prototype
still tensorizes and scores large portions of the state. This supports the
need for a pre-tensor CausalIndex. Learned selection sometimes benefits from
distractors, which suggests the current synthetic labels are not yet hard
enough to isolate true false-positive retrieval failures.

### Priority 4: Closed-Loop Rollout Pilot

Output:

- `docs/experiments/wpu_v2_closed_loop_pilot.csv`

Finding:

`wpu-cws-indexed-sparse` can look strong in one-step accuracy but becomes
unstable in closed-loop rollout, showing large delta norms and constraint
violations. This is a key v2 lesson: WPU evaluation must include rollout
consistency, not only next-step branch accuracy.

### Priority 5: Indexed Selector

Implemented and now supported by a pre-tensor indexed N-sweep.

Output:

- `docs/experiments/wpu_v2_pre_tensor_indexed_n_sweep.csv`
- `docs/experiments/wpu_v2_pre_tensor_indexed_n_sweep_results.md`

The v2 code now supports:

```text
WorldState -> CausalIndex query -> selected objects/relations -> tensorization
```

instead of only:

```text
WorldState -> full tensorization -> index/mask selected objects
```

Result:

| N | pre-tensor indexed accuracy | pre-tensor indexed ms/sample | token ms/sample | graph ms/sample |
| --- | --- | --- | --- | --- |
| 64 | 0.6775 | 1.077 | 0.187 | 2.563 |
| 128 | 0.6775 | 1.112 | 0.344 | 3.745 |
| 256 | 0.6775 | 1.300 | 0.432 | 3.550 |
| 512 | 0.6775 | 1.210 | 0.727 | 3.831 |
| 1024 | 0.6775 | 1.497 | 1.412 | 3.660 |
| 2048 | 0.6775 | 1.553 | 3.063 | 6.386 |
| 4096 | 0.6775 | 1.513 | 7.728 | 9.958 |
| 8192 | 0.6775 | 1.840 | 24.803 | 27.096 |

This is the strongest v2-positive signal so far. It directly supports the
architectural claim that WPU should index into world state before neural
tensorization. Once the model receives only the selected causal subgraph,
latency grows weakly with total N while token and dense graph baselines grow
rapidly.

Remaining limitation:

The current pre-tensor indexed path is still a synthetic relation-frontier
index. It must be extended with spatial buckets, uncertainty hot sets, and
harder distractor cases before being treated as a final result.

### Priority 6: Local Dense Hybrid

Implemented as:

- `wpu-cws-indexed-sparse`
- `wpu-cws-indexed-local-dense`
- `wpu-cws-indexed-adaptive-hybrid`

Current finding:

Local dense propagation is not automatically better. Sparse local propagation
can produce higher one-step accuracy in pilot runs, but closed-loop stability
can be worse. The correct v2 direction is adaptive:

```text
start sparse
check uncertainty/constraints
use local dense only when needed
expand K only when local dense is insufficient
```

Additional implementation update:

`CausalWorkingSetProcessor` now scores branches from selected working-set
deltas rather than only from a pooled embedding. The added regression test
verifies that branch loss trains the delta head.

The adaptive variant now performs hard sparse/local-dense routing per sample.
This is not yet a learned scheduler, but it makes the v2 consistency/fallback
claim executable and measurable.

Adaptive pilot output:

- `docs/experiments/wpu_v2_adaptive_hybrid_pilot.csv`
- `docs/experiments/wpu_v2_adaptive_hybrid_pilot_results.md`
- `docs/experiments/wpu_v2_learned_hybrid_pilot.csv`
- `docs/experiments/wpu_v2_learned_hybrid_pilot_results.md`

Setup:

- N = 4096
- K = 4, 8, 16, 32, 64
- Seeds = 11, 13
- Hidden dim = 256
- Pre-tensor indexed input enabled

Finding:

The hard adaptive route is measurable but does not yet dominate sparse or
local-dense variants. It can reduce local-dense usage at intermediate K, but
the current confidence/K rule is not calibrated enough to reliably choose the
best path. This is a useful v2 result: adaptive fallback should be learned or
calibrated from rollout/constraint metrics, not hand-tuned from K alone.

The learned hybrid improves the pilot accuracy at larger K while routing almost
entirely to the sparse representation:

| K | learned-hybrid accuracy | learned local-dense ratio |
| --- | --- | --- |
| 4 | 0.725 | 0.013 |
| 8 | 0.750 | 0.015 |
| 16 | 0.771 | 0.028 |
| 32 | 0.833 | 0.002 |
| 64 | 0.833 | 0.007 |

Interpretation:

The current synthetic CWS task does not require frequent local dense recompute.
The useful v2 behavior is therefore not "always use dense locally"; it is
"make dense recompute available, but learn to suppress it when sparse causal
propagation is sufficient." This strengthens the WPU claim because the model
can preserve state-local compute instead of paying dense cost by default.

### Priority 6b: Pairwise Local-Interaction Stress

Added `interaction_mode=pairwise` to the CWS dataset and experiment runner.

This mode keeps the WPU premise intact: the model still receives explicit
objects and relations, and the indexed working set is still selected before
tensorization. The difference is that the branch label depends on pairwise
spacing among causal obstacles, making the task less reducible to independent
object updates.

Output:

- `docs/experiments/wpu_v2_pairwise_interaction_pilot.csv`
- `docs/experiments/wpu_v2_pairwise_interaction_pilot_results.md`
- `docs/experiments/wpu_v2_interaction_hybrid_pilot.csv`
- `docs/experiments/wpu_v2_interaction_hybrid_pilot_results.md`

Setup:

- N = 2048
- K = 8, 16, 32
- Seeds = 11, 13
- Hidden dim = 128
- Interaction mode = pairwise
- Pre-tensor indexed input enabled

Finding:

Local-dense propagation improves over sparse at K=8 and K=16, but sparse wins
again at K=32 in this short pilot:

| K | sparse accuracy | local-dense accuracy | learned-hybrid accuracy | interaction-hybrid accuracy | interaction local-dense mix |
| --- | --- | --- | --- | --- | --- |
| 8 | 0.450 | 0.489 | 0.483 | 0.550 | 0.148 |
| 16 | 0.506 | 0.544 | 0.522 | 0.578 | 0.164 |
| 32 | 0.550 | 0.517 | 0.478 | 0.711 | 0.176 |

Interpretation:

This is the first evidence in the repo that dense local recompute can help in
an explicitly interaction-heavy state task. However, after adding
`dense_compute_ratio`, the interpretation is stricter: the interaction-aware
route improves accuracy, but the current implementation still executes the
dense block for every sample. It is therefore an accuracy result, not yet a
compute-efficiency result.

### Priority 6c: Compute-Aware Pairwise Hybrid Check

Output:

- `docs/experiments/wpu_v2_compute_aware_pairwise_pilot.csv`
- `docs/experiments/wpu_v2_compute_aware_pairwise_pilot_results.md`

Setup:

- N = 2048
- K = 8, 16, 32
- Seeds = 11, 13
- Hidden dim = 128
- Interaction mode = pairwise
- Pre-tensor indexed input enabled

Result:

| K | geometry-hybrid accuracy | geometry dense compute | interaction-hybrid accuracy | interaction dense compute |
| --- | --- | --- | --- | --- |
| 8 | 0.467 | 0.000 | 0.561 | 1.000 |
| 16 | 0.522 | 0.000 | 0.594 | 1.000 |
| 32 | 0.483 | 0.000 | 0.722 | 1.000 |

Interpretation:

The compute-realistic geometry-hybrid proves that pairwise state geometry can
be injected without dense execution, but it does not recover the accuracy of
the dense interaction-hybrid. Conversely, the interaction-hybrid is the best
accuracy result in this stress test, but it is not yet sparse in actual compute.

This narrows the next v2 objective:

```text
learn or implement selective dense execution, not merely selective dense mixing
```

The practical route is to execute local dense only for samples whose
interaction-density, branch entropy, or constraint-violation signals exceed a
threshold, while using geometry-enhanced sparse propagation for the rest.

## Updated V2 Direction

The seven architecture directions remain valid, but their priorities are now
clearer:

1. State Store: keep BaseState + DeltaState as the core memory abstraction.
2. Causal Index: move retrieval before tensorization; this is the biggest v2
   systems milestone.
3. Event-Conditioned Retriever: make learned retrieval compete with indexed and
   oracle retrieval under distractors.
4. Adaptive K Scheduler: expose K growth as a controlled decision, not a fixed
   hyperparameter; hard, learned, and interaction-aware local-route variants
   now exist.
5. Local Propagation Core: support both sparse and local dense updates.
6. Delta/Branch Engine: implemented the first delta-conditioned branch scorer;
   the next step is full branch-specific delta trajectories.
7. Consistency/Uncertainty Manager: confidence/K fallback and learned local
   route are implemented; closed-loop violation-triggered expansion remains
   open.

## What V2 Should Claim Now

WPU v2 is now concrete enough to claim a direction, not a final result:

> WPU v2 turns the v1 regime observation into an architecture: state-indexed
> causal retrieval, adaptive local propagation, and delta-based closed-loop
> rollout. Early pilots show that one-step accuracy is insufficient; rollout
> consistency and pre-tensor indexed retrieval are necessary to widen the WPU
> advantage region. The pre-tensor indexed N-sweep is the first evidence that
> WPU latency can be made weakly dependent on total world size N when K is
> retrieved before tensorization. The adaptive-hybrid pilot shows that fallback
> decisions must be trained or calibrated; hard K/confidence rules are useful
> instrumentation but not yet a final scheduler. The learned-hybrid pilot
> suggests the correct default may be sparse-first with learned suppression of
> unnecessary dense recompute, not dense recompute as a universal local upgrade.
> The pairwise-interaction pilot gives a more realistic stress case where
> local-dense recompute can help, but only in specific K regimes. The
> interaction-aware route is the strongest v2 scheduler result so far for
> accuracy, but compute-aware measurement shows that it still executes dense
> recompute in the current implementation. The next claim must be earned by
> selective dense execution or relation-typed sparse propagation that preserves
> the accuracy gain without paying dense cost for every sample.

## Next Required Work

Before claiming v2 as a strong experimental result:

- Rerun K sweep with five seeds and baselines.
- Rerun distractor sweep with harder false-positive distractors and five seeds.
- Rerun pairwise local-interaction stress with five seeds and stronger
  baselines.
- Extend the interaction-aware route with actual selective dense execution,
  compute regularization, and violation-triggered K expansion.
- Extend delta-conditioned branch scoring into branch-specific delta
  trajectories and calibration losses.
- Evaluate closed-loop rollout with trained checkpoints, not only random or
  newly initialized models.

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
before applying indexed selection. This is acceptable for v2-prototype
semantics, but not yet a true sublinear implementation. The next step is to
move indexing before tensorization.

### Indexed WPU Models

Added model names:

- `wpu-cws-indexed`
- `wpu-cws-indexed-sparse`
- `wpu-cws-indexed-local-dense`

These support priority 5 and 6:

- indexed retrieval
- local sparse propagation
- local dense propagation within the selected state subset

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

## Completed V2 Priority Experiments

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

Implemented as `wpu-cws-indexed-local-dense`.

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

## Updated V2 Direction

The seven architecture directions remain valid, but their priorities are now
clearer:

1. State Store: keep BaseState + DeltaState as the core memory abstraction.
2. Causal Index: move retrieval before tensorization; this is the biggest v2
   systems milestone.
3. Event-Conditioned Retriever: make learned retrieval compete with indexed and
   oracle retrieval under distractors.
4. Adaptive K Scheduler: expose K growth as a controlled decision, not a fixed
   hyperparameter.
5. Local Propagation Core: support both sparse and local dense updates.
6. Delta/Branch Engine: implemented the first delta-conditioned branch scorer;
   the next step is full branch-specific delta trajectories.
7. Consistency/Uncertainty Manager: make closed-loop violations trigger K
   expansion or local dense fallback.

## What V2 Should Claim Now

WPU v2 is now concrete enough to claim a direction, not a final result:

> WPU v2 turns the v1 regime observation into an architecture: state-indexed
> causal retrieval, adaptive local propagation, and delta-based closed-loop
> rollout. Early pilots show that one-step accuracy is insufficient; rollout
> consistency and pre-tensor indexed retrieval are necessary to widen the WPU
> advantage region. The pre-tensor indexed N-sweep is the first evidence that
> WPU latency can be made weakly dependent on total world size N when K is
> retrieved before tensorization.

## Next Required Work

Before claiming v2 as a strong experimental result:

- Rerun K sweep with five seeds and baselines.
- Rerun distractor sweep with harder false-positive distractors and five seeds.
- Add adaptive sparse/local-dense fallback.
- Extend delta-conditioned branch scoring into branch-specific delta
  trajectories and calibration losses.
- Evaluate closed-loop rollout with trained checkpoints, not only random or
  newly initialized models.

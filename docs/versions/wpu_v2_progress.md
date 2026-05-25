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
- `wpu-cws-indexed-learned-selective-hybrid`
- `wpu-cws-indexed-interaction-hybrid`
- `wpu-cws-indexed-selective-interaction-hybrid`
- `wpu-cws-indexed-geometry-hybrid`

These support priority 5 and 6:

- indexed retrieval
- local sparse propagation
- local dense propagation within the selected state subset
- hard adaptive routing between sparse and local-dense propagation
- learned differentiable routing between sparse and local-dense propagation
- learned selective routing with a differentiable route cost and optional
  distillation to the analytic interaction route
- interaction-aware routing from state-local pairwise geometry
- selective interaction-aware execution that runs local dense only for samples
  whose state-local interaction score exceeds a threshold
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

Added `wpu-cws-indexed-selective-interaction-hybrid`.

This model keeps the interaction-aware soft mixing score, but only executes the
local dense transformer for samples whose interaction score exceeds a threshold.
This is the first v2 implementation that directly targets the compute-aware
claim:

```text
preserve interaction accuracy while reducing actual dense execution
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

### Priority 6d: Selective Dense Execution

Output:

- `docs/experiments/wpu_v2_selective_interaction_pairwise_pilot.csv`
- `docs/experiments/wpu_v2_selective_compute_pairwise_comparison.csv`
- `docs/experiments/wpu_v2_selective_compute_pairwise_comparison_results.md`

Setup:

- N = 2048
- K = 8, 16, 32
- Seeds = 11, 13
- Hidden dim = 128
- Interaction mode = pairwise
- Pre-tensor indexed input enabled

Result:

| K | interaction accuracy | interaction dense compute | selective accuracy | selective dense compute | selective ms/sample |
| --- | --- | --- | --- | --- | --- |
| 8 | 0.561 | 1.000 | 0.556 | 0.367 | 1.786 |
| 16 | 0.594 | 1.000 | 0.611 | 0.572 | 3.098 |
| 32 | 0.722 | 1.000 | 0.711 | 0.817 | 5.083 |

Interpretation:

This is the first positive compute-aware hybrid result. The selective model
preserves nearly all interaction-hybrid accuracy at K=8 and K=32, slightly
improves K=16 in this two-seed pilot, and lowers actual dense execution from
1.0 to 0.37-0.82 depending on K. At K=32, latency also drops from 7.74 to 5.08
ms/sample.

The result is not final proof. Dense execution still rises with K, the run uses
only two seeds, and the threshold is fixed rather than learned. But the
direction is now technically correct: WPU can distinguish dense-output mixing
from dense execution and can make dense fallback conditional on state-local
interaction structure.

### Priority 6e: Selective Threshold Sweep

Output:

- `docs/experiments/wpu_v2_selective_threshold_sweep.csv`
- `docs/experiments/wpu_v2_selective_threshold_summary.csv`
- `docs/experiments/wpu_v2_selective_threshold_comparison.csv`
- `docs/experiments/wpu_v2_selective_threshold_comparison_results.md`
- `docs/figures/wpu_v2_selective_threshold_pareto.png`

Setup:

- N = 2048
- K = 8, 16, 32
- Thresholds = 0.05, 0.10, 0.15, 0.20, 0.30
- Seeds = 11, 13
- Hidden dim = 128
- Interaction mode = pairwise
- Pre-tensor indexed input enabled

Result:

| K | threshold | accuracy | dense compute | dense mix |
| --- | --- | --- | --- | --- |
| 8 | 0.05 | 0.561 | 1.000 | 0.148 |
| 8 | 0.10 | 0.556 | 0.756 | 0.127 |
| 8 | 0.15 | 0.556 | 0.367 | 0.079 |
| 8 | 0.20 | 0.494 | 0.167 | 0.044 |
| 8 | 0.30 | 0.467 | 0.039 | 0.014 |
| 16 | 0.05 | 0.594 | 1.000 | 0.164 |
| 16 | 0.10 | 0.617 | 0.989 | 0.163 |
| 16 | 0.15 | 0.611 | 0.572 | 0.108 |
| 16 | 0.20 | 0.567 | 0.167 | 0.038 |
| 16 | 0.30 | 0.517 | 0.006 | 0.002 |
| 32 | 0.05 | 0.722 | 1.000 | 0.176 |
| 32 | 0.10 | 0.722 | 1.000 | 0.176 |
| 32 | 0.15 | 0.711 | 0.822 | 0.151 |
| 32 | 0.20 | 0.578 | 0.206 | 0.044 |
| 32 | 0.30 | 0.472 | 0.000 | 0.000 |

Interpretation:

The threshold sweep exposes a real accuracy-compute frontier rather than a
single cherry-picked point. The current best practical threshold is 0.15: it
keeps accuracy close to full interaction routing while materially reducing
actual dense execution at K=8 and K=16, and still reduces dense execution at
K=32 with only a small accuracy loss.

This result also shows where the current approach is weak. At K=32, aggressive
thresholds collapse accuracy, so large causal working sets require either more
dense execution or a better sparse interaction operator. Latency should be
treated cautiously in this pilot because boolean-indexed dense execution and
small batch profiling add noise; the primary compute metric is
`dense_compute_ratio`.

### Priority 6f: Five-Seed Selective Validation

Output:

- `docs/experiments/wpu_v2_selective_5seed_validation.csv`
- `docs/experiments/wpu_v2_selective_5seed_validation_results.md`
- `docs/experiments/wpu_v2_selective_5seed_summary.csv`
- `docs/figures/wpu_v2_selective_5seed_validation.png`

Setup:

- N = 2048
- K = 8, 16, 32
- Seeds = 11, 13, 17, 19, 23
- Selective threshold = 0.15
- Hidden dim = 128
- Interaction mode = pairwise
- Pre-tensor indexed input enabled

Result:

| K | full interaction accuracy | full interaction CI95 | selective accuracy | selective CI95 | selective dense compute |
| --- | --- | --- | --- | --- | --- |
| 8 | 0.580 | 0.029 | 0.558 | 0.031 | 0.402 |
| 16 | 0.589 | 0.056 | 0.604 | 0.015 | 0.576 |
| 32 | 0.691 | 0.034 | 0.678 | 0.036 | 0.833 |

Interpretation:

The five-seed validation supports the more cautious v2 claim. Selective WPU
does not universally beat full interaction routing, but it preserves comparable
accuracy within confidence intervals while reducing actual dense execution. The
effect is strongest at K=8 and K=16. At K=32, dense execution remains high and
accuracy is slightly lower, so larger working sets still need either better
sparse interaction propagation or a learned router.

This result is academically useful because it turns the WPU claim into a
measurable Pareto statement:

```text
WPU is not simply "more accurate"; it can trade a small amount of interaction
accuracy for a measurable reduction in actual dense execution inside an
explicit state-processing regime.
```

### Priority 6g: Learned Selective Router Attempt

Output:

- `docs/experiments/wpu_v2_learned_selective_pilot.csv`
- `docs/experiments/wpu_v2_learned_selective_t015_pilot.csv`
- `docs/experiments/wpu_v2_distilled_selective_pilot.csv`
- `docs/experiments/wpu_v2_learned_selective_router_pilots.csv`
- `docs/experiments/wpu_v2_learned_selective_router_pilots_summary.csv`

Implemented:

- `wpu-cws-indexed-learned-selective-hybrid`
- `route_compute_loss()`
- `route_distillation_loss()`
- `--route-compute-loss-weight`
- `--route-distill-loss-weight`

Result:

| router variant | K | accuracy | dense compute |
| --- | --- | --- | --- |
| learned selective, t=0.50, compute loss 0.01 | 8 | 0.433 | 0.000 |
| learned selective, t=0.50, compute loss 0.01 | 16 | 0.456 | 0.000 |
| learned selective, t=0.50, compute loss 0.01 | 32 | 0.417 | 0.000 |
| learned selective, t=0.15, compute loss 0.01 | 8 | 0.433 | 0.000 |
| learned selective, t=0.15, compute loss 0.01 | 16 | 0.450 | 0.133 |
| learned selective, t=0.15, compute loss 0.01 | 32 | 0.417 | 0.000 |
| distilled selective, t=0.15, distill loss 1.0 | 8 | 0.428 | 0.000 |
| distilled selective, t=0.15, distill loss 1.0 | 16 | 0.467 | 0.161 |
| distilled selective, t=0.15, distill loss 1.0 | 32 | 0.406 | 0.061 |

Interpretation:

This is an important negative result. A naive learned router does not recover
the fixed interaction threshold frontier. Even with interaction density as an
input and distillation to the analytic interaction teacher, the learned route
collapses toward too little dense execution and loses accuracy.

The next router should therefore not be presented as solved. The required
research problem is calibrated routing:

```text
learn when dense execution is worth its cost, with explicit supervision for
branch correctness, uncertainty, and constraint failure, not only MSE
distillation to a soft interaction score.
```

### Priority 6h: Counterfactual Dense-Needed Labels

Output:

- `scripts/counterfactual_route_labels.py`
- `docs/experiments/wpu_v2_counterfactual_route_labels.csv`
- `docs/experiments/wpu_v2_counterfactual_route_labels_summary.csv`

Setup:

- N = 2048
- K = 8, 16, 32
- Seeds = 11, 13
- Hidden dim = 128
- Interaction mode = pairwise
- Pre-tensor indexed input enabled
- Compare separately trained sparse and local-dense WPU on the same held-out
  samples.

Result:

| K | sparse acc | dense acc | dense-needed rate | dense fix rate | dense break rate | dense lower-loss rate |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 0.456 | 0.467 | 0.272 | 0.083 | 0.072 | 0.483 |
| 16 | 0.539 | 0.511 | 0.311 | 0.094 | 0.122 | 0.650 |
| 32 | 0.456 | 0.383 | 0.200 | 0.083 | 0.156 | 0.439 |

Interpretation:

This explains why the naive learned router failed. Dense is not uniformly
better. It fixes some sparse mistakes, but it also breaks some sparse-correct
samples, especially at K=16 and K=32. A router trained only from branch loss,
compute penalty, or analytic interaction density does not know which individual
samples are dense-beneficial.

The next router should be trained from counterfactual labels:

```text
dense_needed = dense fixes sparse branch error
            or dense materially reduces branch/constraint loss
```

This makes the WPU claim more precise. Selective dense execution is valuable
only if route supervision identifies the samples where dense recompute improves
state prediction enough to justify its cost.

### Priority 6i: Dense-Needed Route Label Probe

Output:

- `docs/experiments/wpu_v2_counterfactual_route_examples.csv`
- `docs/experiments/wpu_v2_counterfactual_route_labels_with_examples.csv`
- `docs/experiments/wpu_v2_route_label_probe.csv`
- `docs/experiments/wpu_v2_route_label_probe_summary.csv`
- `scripts/route_label_probe.py`
- `scripts/summarize_route_label_probe.py`

Question:

```text
Are dense-needed labels identifiable from state-only and sparse-diagnostic features?
```

Setup:

- Use the counterfactual route examples from N=2048, K=8/16/32, seeds 11 and
  13.
- Train small MLP probes on one seed and evaluate on the held-out seed.
- Compare state-only features against sparse-output diagnostics: branch
  entropy, branch margin, top-branch confidence, object-delta norm, and mean
  uncertainty.
- Compare against raw interaction-density threshold heuristics.
- Report threshold metrics, train-calibrated threshold transfer, ROC-AUC,
  average precision, Brier score, and expected calibration error.

Result:

| probe | threshold | dense label rate | predicted dense rate | balanced accuracy | F1 | ROC-AUC | AP | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| interaction density | 0.15 | 0.261 | 0.839 | 0.488 | 0.390 | 0.182 | 0.460 | 0.102 |
| interaction density, train-calibrated | calibrated | 0.261 | 0.637 | 0.451 | 0.267 | 0.182 | 0.460 | 0.102 |
| MLP state probe | 0.20 | 0.261 | 0.561 | 0.561 | 0.414 | 0.565 | 0.303 | 0.255 |
| MLP state probe, train-calibrated | calibrated | 0.261 | 0.400 | 0.530 | 0.350 | 0.565 | 0.303 | 0.255 |
| MLP sparse diagnostics | 0.50 | 0.261 | 0.050 | 0.481 | 0.036 | 0.482 | 0.255 | 0.284 |
| MLP sparse diagnostics, train-calibrated | calibrated | 0.261 | 0.050 | 0.481 | 0.036 | 0.482 | 0.255 | 0.284 |

Interpretation:

Interaction density alone does not identify dense-needed samples. Its ROC-AUC
is below chance, meaning dense-needed is not simply "more local interaction."
The state-only MLP improves balanced accuracy above chance, but only weakly,
and its ROC-AUC remains modest. Train-calibrated threshold selection does not
transfer well to the held-out seed, so the failure is not just a fixed-threshold
choice.

Adding scalar sparse diagnostics does not solve the problem. The sparse
diagnostic probe stays near chance AUC and collapses toward almost never
selecting dense execution. This is likely a calibration and model-instance
shift problem: entropy, margin, delta norm, and uncertainty are not yet stable
route signals across separately trained sparse models.

This strengthens the negative learned-router result. The route decision is not
captured by simple state heuristics or by post-hoc scalar sparse diagnostics.
A useful router likely needs calibrated model-internal route representations,
counterfactual dense-needed supervision, branch-correctness costs,
constraint-violation signals, and hard selective-execution evaluation.

### Priority 6j: Shared-Model Sparse/Dense Counterfactual

Output:

- `scripts/shared_route_counterfactual.py`
- `scripts/summarize_shared_route_counterfactual.py`
- `docs/experiments/wpu_v2_shared_route_counterfactual.csv`
- `docs/experiments/wpu_v2_shared_route_counterfactual_summary.csv`

Question:

```text
Is the dense-needed signal still present when sparse and dense paths share the
same model parameters?
```

Setup:

- N = 2048
- K = 8, 16, 32
- Seeds = 11, 13
- Hidden dim = 128
- Steps = 40
- Samples = 90 per seed
- Interaction mode = pairwise
- Pre-tensor indexed working set
- Train one `wpu-cws-indexed-local-dense` model with both forced sparse and
  forced local-dense losses, then evaluate both paths on the same samples.
- `force_route="sparse"` now skips the local dense encoder instead of computing
  it and discarding the result, so forced-route compute accounting matches the
  executed path.

Result:

| K | sparse accuracy | dense accuracy | dense-needed rate | dense fix rate | dense break rate | branch disagreement |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 0.489 | 0.467 | 0.167 | 0.050 | 0.072 | 0.172 |
| 16 | 0.556 | 0.433 | 0.233 | 0.083 | 0.206 | 0.367 |
| 32 | 0.494 | 0.422 | 0.200 | 0.106 | 0.178 | 0.400 |

Interpretation:

This resolves an important ambiguity in the previous counterfactual labels.
The earlier sparse-vs-dense comparison used separately trained models, so route
labels could be contaminated by model-instance differences. The shared-model
probe still finds dense-beneficial samples, but it also shows that dense breaks
sparse-correct samples more often than it fixes sparse mistakes in this short
pilot.

The v2 conclusion should therefore be stricter:

```text
local dense recompute is an available correction operator, not a default
improvement operator
```

A useful WPU router must predict both sides of the decision: when dense can fix
a sparse failure and when dense is likely to damage an already-correct sparse
state update. This points toward regret-style route supervision:

```text
route_target = dense_loss - sparse_loss
execute dense only when expected regret is negative enough to justify cost
```

## Updated V2 Direction

The seven architecture directions remain valid, but their priorities are now
clearer:

1. State Store: keep BaseState + DeltaState as the core memory abstraction.
2. Causal Index: move retrieval before tensorization; this is the biggest v2
   systems milestone.
3. Event-Conditioned Retriever: make learned retrieval compete with indexed and
   oracle retrieval under distractors.
4. Adaptive K Scheduler: expose K growth as a controlled decision, not a fixed
   hyperparameter; hard, learned, interaction-aware, and forced counterfactual
   route variants now exist.
5. Local Propagation Core: support both sparse and local dense updates.
6. Delta/Branch Engine: implemented the first delta-conditioned branch scorer;
   the next step is full branch-specific delta trajectories.
7. Consistency/Uncertainty Manager: confidence/K fallback and learned local
   route are implemented; regret-style route supervision and closed-loop
   violation-triggered expansion remain open.

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
  compute regularization, and violation-triggered K expansion. The first
  selective dense execution prototype now exists, but it needs threshold
  sweeps, five-seed validation, and learned/calibrated routing.
- Extend delta-conditioned branch scoring into branch-specific delta
  trajectories and calibration losses.
- Evaluate closed-loop rollout with trained checkpoints, not only random or
  newly initialized models.

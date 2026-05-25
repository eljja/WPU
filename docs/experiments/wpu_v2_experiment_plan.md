# WPU V2 Experiment Plan

This plan converts the WPU v2 architecture into falsifiable experiments. The
goal is to expand the proven WPU regime without returning to token processing.

## Common Setup

Unless otherwise stated:

- Hidden dimension: 512
- Layers: 2
- Heads: 8
- Working-set cap: 16 unless swept
- Seeds: 11, 13, 17, 19, 23
- Labels: balanced
- Device: CUDA
- Primary metric: branch accuracy
- Secondary metrics: latency, peak CUDA memory, delta MSE, causal recall,
  selected K, calibration/entropy where available
- Compute-routing metrics: `local_dense_ratio` for representation mixing and
  `dense_compute_ratio` for actual dense block execution

## Priority 1: Oracle vs Learned Selector Gap

Question:

```text
How much of WPU's error is caused by selection rather than propagation?
```

Run:

- `wpu-cws-oracle`
- `wpu-cws-learned`
- `wpu-cws-frontier`

Sweep:

- `N = 64, 128, 256, 512, 1024, 2048, 4096, 8192`
- `K = 8`

Evidence needed:

- Accuracy gap between oracle and learned.
- Causal recall of learned selector.
- Latency difference between oracle, frontier, and learned.

Interpretation:

- If oracle wins and learned fails, selector is bottleneck.
- If both fail, propagation core is bottleneck.
- If learned catches oracle, WPU v2 can move to harder distractors and long
  horizon tests.

## Priority 2: K Sweep

Question:

```text
When does WPU stop being better as the causal working set grows?
```

Run:

- `wpu-cws-oracle`
- `wpu-cws-learned`
- `serialized-token`
- `graph-transformer`

Sweep:

- Fixed `N = 4096`
- `K = 4, 8, 16, 32, 64`

Evidence needed:

- Accuracy versus K.
- Latency versus K.
- The K/N boundary where WPU stops being favorable.

Expected result:

WPU should be strongest when `K/N` is small. If K grows toward dense state,
token/dense graph baselines should become more competitive.

## Priority 3: Adversarial Distractor Sweep

Question:

```text
Can WPU still find the causal working set when many similar fake objects exist?
```

Run:

- `wpu-cws-oracle`
- `wpu-cws-learned`
- `serialized-token`
- `graph-transformer`

Sweep:

- Fixed `N = 4096`
- Fixed causal K = 8
- Fake causal-looking distractors: `0, 8, 16, 32, 64, 128, 256`

Evidence needed:

- Learned selector causal recall.
- Branch accuracy degradation.
- False positive selected-K composition.
- Latency under distractor pressure.

Interpretation:

- If oracle remains strong but learned falls, improve retrieval.
- If both fall, propagation is using insufficient state or labels are
  underspecified.

## Priority 4: Closed-Loop Long-Horizon Rollout

Question:

```text
Does WPU preserve state consistency better than token/dense baselines over time?
```

Current status:

`scripts/cws_closed_loop_rollout.py` now applies predicted `DeltaState`
overlays back onto branch-local states. The earlier
`scripts/cws_long_horizon_eval.py` remains a diagnostic one-step stability
evaluator.

Run:

- Horizon: 10, 25, 50, 100
- N: 1024, 4096, 8192
- Models: WPU oracle, WPU learned, token, graph

Evidence needed:

- Branch flip rate.
- Delta drift.
- Constraint violation count.
- Branch probability entropy over time.
- Runtime per step.

Remaining v2 requirement:

Run closed-loop evaluation on trained checkpoints and include constraint
violation metrics in the main v2 evidence table.

## Priority 5: Indexed Selector

Question:

```text
Can WPU retrieval become sublinear in N?
```

Current limitation:

The learned selector still computes object features for all objects in the
batch. That is useful for a neural prototype, but it weakens the hardware and
architecture claim.

V2 implementation:

- Build `CausalIndex`.
- Query target id directly.
- Expand through relation adjacency.
- Add spatial bucket retrieval.
- Add uncertainty hot-set retrieval.
- Return candidate ids before neural scoring.

Experiment:

- Compare `learned-scan` versus `indexed-retrieval`.
- Sweep `N = 1024, 2048, 4096, 8192, 16384`.
- Keep `K = 8`.

Evidence needed:

- Selector latency versus N.
- End-to-end latency versus N.
- Causal recall.
- Branch accuracy.

Success criterion:

Selector time should grow much more slowly than N, ideally close to local
fanout plus selected K.

Current v2 status:

The first pre-tensor indexed path is implemented. It projects `WorldState` to
the event-local indexed subgraph before tensorization. Initial N-sweep evidence
is stored in:

- `docs/experiments/wpu_v2_pre_tensor_indexed_n_sweep.csv`
- `docs/experiments/wpu_v2_pre_tensor_indexed_n_sweep_results.md`

The next indexed-selector experiment should make the index harder by adding
spatial distractors and hidden multi-hop support relations.

## Priority 6: Local Dense Hybrid

Question:

```text
Can local dense reasoning improve accuracy without returning to global token attention?
```

V2 implementation:

- Retrieve K using causal index or learned selector.
- Run relation-typed propagation.
- Run a local dense block only over selected K.
- Predict deltas.
- Score branches from candidate deltas rather than only pooled embeddings.

Experiment:

- `wpu-cws-indexed-sparse`
- `wpu-cws-indexed-local-dense`
- `wpu-cws-indexed-adaptive-hybrid`
- `wpu-cws-indexed-learned-hybrid`
- `wpu-cws-indexed-learned-selective-hybrid`
- `wpu-cws-indexed-interaction-hybrid`
- `wpu-cws-indexed-selective-interaction-hybrid`
- token and graph baselines

Sweep:

- N: 1024, 2048, 4096, 8192
- K: 8, 16, 32
- Distractors: 0, 32, 128

Evidence needed:

- Accuracy improvement over sparse WPU.
- Latency relative to token/graph.
- Expansion/fallback frequency.
- Actual dense execution frequency through `dense_compute_ratio`, not only
  local dense mixing.

Success criterion:

Local dense WPU should close the accuracy gap without matching the latency
growth of global token or dense graph baselines.

Current v2 status:

`CausalWorkingSetProcessor` now includes a delta-conditioned branch head. The
local dense/sparse variants and a first hard adaptive switch are implemented.
The adaptive switch routes to local dense when selected K is large or selector
confidence is low; the next experiment should measure whether this expands the
favorable WPU regime without reintroducing global attention cost.

Pilot evidence:

- `docs/experiments/wpu_v2_adaptive_hybrid_pilot.csv`
- `docs/experiments/wpu_v2_adaptive_hybrid_pilot_results.md`
- `docs/experiments/wpu_v2_learned_hybrid_pilot.csv`
- `docs/experiments/wpu_v2_learned_hybrid_pilot_results.md`

The hard route is not yet reliably better than always-sparse or
always-local-dense. The next version should train or calibrate the route from
closed-loop consistency failures, branch entropy, and delta magnitude rather
than treating K/confidence thresholds as final.

The first learned route improves accuracy in this pilot but suppresses dense
usage almost completely. That is not a failure by itself: it indicates the
current task is mostly sparse-solvable and that WPU should learn when dense
local recompute is worth its cost. The next experiment should add cases where
local dense consistency is genuinely required.

Pairwise local-interaction stress:

- `docs/experiments/wpu_v2_pairwise_interaction_pilot.csv`
- `docs/experiments/wpu_v2_pairwise_interaction_pilot_results.md`
- `docs/experiments/wpu_v2_interaction_hybrid_pilot.csv`
- `docs/experiments/wpu_v2_interaction_hybrid_pilot_results.md`

This stress mode makes branch labels depend on pairwise obstacle spacing inside
the causal working set. The short pilot shows local-dense gains at K=8 and
K=16, but not K=32. This is the right kind of evidence: it identifies a
conditional dense-recompute regime instead of claiming dense local propagation
is universally better.

The interaction-aware hybrid is the current best scheduler variant on this
stress test. It routes from state-local pairwise geometry, reaches the best
accuracy at K=8, K=16, and K=32, and uses only about 15-18% local-dense mixing.
This gives a more realistic v2 hypothesis: dense fallback should be triggered
by measured interaction structure inside the causal working set, not by total
world size N or fixed K thresholds alone.

Compute-aware correction:

The 15-18% number is a mixing ratio, not a compute ratio. The dense transformer
is still executed for every sample in `wpu-cws-indexed-interaction-hybrid`.
The new required metric is `dense_compute_ratio`.

Follow-up implementation target:

- `wpu-cws-indexed-geometry-hybrid` proves that state-local pairwise geometry
  can be used without dense execution, but current accuracy is lower.
- The next hybrid must conditionally execute the dense block only for selected
  samples or replace dense recompute with relation-typed pairwise propagation.
- Success requires preserving most of the interaction-hybrid accuracy while
  driving `dense_compute_ratio` materially below 1.0.

Current selective-execution pilot:

- `docs/experiments/wpu_v2_selective_interaction_pairwise_pilot.csv`
- `docs/experiments/wpu_v2_selective_compute_pairwise_comparison_results.md`

The first selective interaction model preserves almost all of the
interaction-hybrid accuracy while reducing actual dense execution in the
pairwise stress pilot:

| K | interaction acc | interaction dense compute | selective acc | selective dense compute |
| --- | --- | --- | --- | --- |
| 8 | 0.561 | 1.000 | 0.556 | 0.367 |
| 16 | 0.594 | 1.000 | 0.611 | 0.572 |
| 32 | 0.722 | 1.000 | 0.711 | 0.817 |

This should now be promoted to a five-seed threshold sweep. The key falsifiable
question is whether there is a stable Pareto frontier where selective WPU keeps
accuracy close to interaction-hybrid while reducing dense execution and latency.

Threshold sweep pilot:

- `docs/experiments/wpu_v2_selective_threshold_sweep.csv`
- `docs/experiments/wpu_v2_selective_threshold_summary.csv`
- `docs/figures/wpu_v2_selective_threshold_pareto.png`

The two-seed threshold sweep confirms that the selective model has an
accuracy-compute frontier. Threshold 0.15 is the current best fixed operating
point:

| K | t=0.15 accuracy | t=0.15 dense compute | full interaction accuracy |
| --- | --- | --- | --- |
| 8 | 0.556 | 0.367 | 0.561 |
| 16 | 0.611 | 0.572 | 0.594 |
| 32 | 0.711 | 0.822 | 0.722 |

The next experiment should not just rerun this threshold. It should test
whether a learned or calibrated router can choose per-sample dense execution
better than a fixed threshold, especially for K=32 where aggressive sparsity
causes a sharp accuracy drop.

Five-seed validation:

- `docs/experiments/wpu_v2_selective_5seed_validation.csv`
- `docs/experiments/wpu_v2_selective_5seed_validation_results.md`
- `docs/experiments/wpu_v2_selective_5seed_summary.csv`
- `docs/figures/wpu_v2_selective_5seed_validation.png`

The fixed threshold remains meaningful over five seeds, but the conclusion is
Pareto-style rather than dominance:

| K | full interaction acc | selective acc | selective dense compute |
| --- | --- | --- | --- |
| 8 | 0.580 +/- 0.029 | 0.558 +/- 0.031 | 0.402 |
| 16 | 0.589 +/- 0.056 | 0.604 +/- 0.015 | 0.576 |
| 32 | 0.691 +/- 0.034 | 0.678 +/- 0.036 | 0.833 |

The next router experiment should optimize this frontier directly: minimize
dense execution subject to an accuracy or calibration constraint, instead of
claiming universal accuracy improvement.

Learned-router negative result:

- `docs/experiments/wpu_v2_learned_selective_router_pilots_summary.csv`

The first learned-selective router did not reproduce the fixed threshold
frontier. It collapsed toward too little dense execution and lost branch
accuracy, even with a simple distillation loss against the analytic interaction
teacher. The next router experiment should add explicit route supervision from
counterfactual dense-vs-sparse correctness:

```text
run sparse and dense local paths during training
label dense-needed when dense fixes a sparse branch error or reduces
constraint/uncertainty loss
train the route as a cost-sensitive classifier
evaluate with hard selective execution
```

Counterfactual label pilot:

- `docs/experiments/wpu_v2_counterfactual_route_labels_summary.csv`

The first counterfactual diagnostic shows that dense is not uniformly better:

| K | dense fix rate | dense break rate | dense-needed rate |
| --- | --- | --- | --- |
| 8 | 0.083 | 0.072 | 0.272 |
| 16 | 0.094 | 0.122 | 0.311 |
| 32 | 0.083 | 0.156 | 0.200 |

This should become the next supervised-router dataset. The route label should
not be "interaction density is high"; it should be "dense improves this
sample's outcome enough to justify cost."

Route-label identifiability probe:

- `docs/experiments/wpu_v2_counterfactual_route_examples.csv`
- `docs/experiments/wpu_v2_route_label_probe_summary.csv`

Dense-needed labels are only weakly predictable from the current state-only
features. Interaction density alone is near chance balanced accuracy. A small
MLP improves to roughly 0.58 balanced accuracy, but this is not enough for a
reliable router.

Sparse-output diagnostics have now been added to the probe: branch entropy,
branch margin, top-branch confidence, object-delta norm, and mean uncertainty.
They do not improve the route decision under seed-heldout evaluation. The probe
collapses to almost never selecting dense execution, which suggests that
post-hoc scalar diagnostics are poorly calibrated across independently trained
sparse models.

The calibrated-threshold and rank-metric update makes the diagnosis stricter.
Interaction density has below-chance ROC-AUC, so dense-needed is not equivalent
to high local interaction density. The state-only probe has only weak rank
signal, and train-selected thresholds do not transfer cleanly to the held-out
seed. Therefore the immediate bottleneck is route-label identifiability under
model/seed shift, not only threshold tuning.

The next supervised router should therefore not only append more scalar
features. It should train a route head on model-internal state representations
with counterfactual dense-needed labels, calibration losses, seed/domain
augmentation, and hard selective-execution evaluation. Useful route evidence
should include:

- Sparse branch entropy and margin.
- Local constraint violation scores.
- Delta magnitude and uncertainty growth.
- Training-time sparse/dense disagreement labels.
- Cost-sensitive route objective evaluated by hard selective execution.

Shared-model counterfactual check:

- `docs/experiments/wpu_v2_shared_route_counterfactual.csv`
- `docs/experiments/wpu_v2_shared_route_counterfactual_summary.csv`

The first shared-model sparse/dense counterfactual removes a major confound in
the previous labels. Sparse and dense paths are trained inside one
`wpu-cws-indexed-local-dense` model and then forced at evaluation time on the
same samples. The result is stricter than the separate-model diagnostic:
dense-beneficial samples exist, but dense also breaks sparse-correct samples
more often than it fixes sparse failures in this pilot.

The sample-level shared-route probes add an important distinction. The strict
`dense_needed` label remains weakly identifiable, while the loss-based
`dense_beneficial` label is more learnable from state features. This suggests
that router supervision should use dense-vs-sparse regret as a continuous or
cost-sensitive target rather than only a binary "dense fixed sparse" label.

The next router should therefore predict dense regret, not only dense need:

```text
regret = dense_loss - sparse_loss
execute dense when expected regret + compute_cost < 0
```

This converts routing into a cost-sensitive correction problem. It also gives a
clear failure mode: if expected regret cannot be predicted under seed/domain
shift, WPU should keep sparse propagation and expand the causal working set or
invoke a different relation-typed operator rather than blindly using dense
recompute.

Continuous regret probe:

- `docs/experiments/wpu_v2_shared_route_regret_probe.csv`
- `docs/experiments/wpu_v2_shared_route_regret_probe_summary.csv`

The first continuous regret probe is more encouraging than binary dense-needed
classification. A small state-feature regressor predicts enough
`dense_loss - sparse_loss` structure to reduce average loss versus always
sparse under moderate compute costs, while using dense on only part of the
batch. This does not yet solve routing: R2 is low, oracle excess remains, and
post-hoc sparse diagnostic features over-route to dense. But it gives a more
academically defensible next target:

```text
train a model-internal regret head and optimize routed expected loss, not only
binary dense-needed accuracy
```

Internal regret-head attempt:

- `docs/experiments/wpu_v2_internal_regret_hybrid_pilot/k-sweep.csv`

The first internal regret-head implementation is a negative result. The
post-hoc state-feature regressor can reduce loss, but the jointly trained
`wpu-cws-indexed-regret-hybrid` collapses to always-sparse in a short pilot:
predicted regret remains positive and the negative-regret route ratio is zero.
This means the next experiment should not simply increase model size. It should
separate optimization phases:

```text
train sparse/dense propagation -> train regret head on counterfactual losses
-> fine-tune with routed expected loss and compute cost
```

This is the more falsifiable route to an operational WPU scheduler.

Staged internal regret-head pilot:

- `scripts/staged_regret_hybrid.py`
- `docs/experiments/wpu_v2_staged_regret_hybrid_pilot.csv`
- `docs/experiments/wpu_v2_staged_regret_hybrid_pilot_results.md`

The staged version trains sparse and local-dense propagation first, then freezes
the propagation core and trains only the route-regret head on counterfactual
losses. A held-out validation split calibrates the route threshold before test
evaluation. This removes the previous always-sparse collapse and produces a
bounded selective-dense policy:

| K | sparse acc | routed acc | sparse loss | routed loss | dense compute | routed loss delta |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 0.490 | 0.490 | 0.977 | 0.953 | 0.133 | -0.024 |
| 16 | 0.500 | 0.500 | 0.958 | 0.912 | 0.272 | -0.047 |
| 32 | 0.483 | 0.500 | 1.019 | 0.989 | 0.272 | -0.030 |

This should be treated as partial support, not a final win. The correct v2
claim is that internal cost-sensitive routing can reduce expected loss with
less than full dense recompute. The remaining open problem is the oracle gap:
the routed loss is still about 0.05 above the per-sample best sparse/dense
choice. The next scheduler experiment should train calibration and compute cost
end-to-end, then repeat across five seeds and a denser N/K/D grid.

Five-seed regret-routing audit:

- `docs/experiments/wpu_v2_staged_regret_hybrid_5seed.csv`
- `docs/experiments/wpu_v2_staged_regret_hybrid_5seed_results.md`

The five-seed audit confirms the narrower result and rejects the stronger one.
Selective regret routing repeatedly reduces expected loss versus always-sparse,
but it does not yet produce broad accuracy gains and it does not close the
oracle selective-routing gap.

| K | sparse loss | routed loss | zero-threshold loss | dense compute | regret corr | oracle excess |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 0.991 | 0.960 | 0.972 | 0.204 | 0.377 | 0.042 |
| 16 | 0.967 | 0.949 | 0.949 | 0.204 | 0.289 | 0.049 |
| 32 | 1.008 | 0.979 | 1.001 | 0.302 | 0.388 | 0.043 |

The experiment plan should therefore prioritize threshold-free or
threshold-stable scheduling before larger model scaling. Otherwise a larger WPU
may only amplify an unstable routing boundary.

Fixed-margin regret scheduler sweep:

- `scripts/staged_regret_margin_sweep.py`
- `docs/experiments/wpu_v2_staged_regret_margin_sweep.csv`
- `docs/experiments/wpu_v2_staged_regret_margin_sweep_results.md`

The fixed-margin sweep gives a more deployable scheduler direction. A fixed
sparse-favoring margin does not match validation-calibrated routing, but it
does preserve part of the loss gain without seed-specific threshold selection.

| policy | routed loss | loss delta | dense compute | oracle excess |
| --- | --- | --- | --- | --- |
| calibrated | 0.960 | -0.030 | 0.240 | 0.040 |
| margin 0.02 | 0.970 | -0.020 | 0.209 | 0.054 |
| margin 0.05 | 0.972 | -0.018 | 0.127 | 0.054 |
| margin 0.10 | 0.975 | -0.013 | 0.063 | 0.057 |

This suggests the scheduler should expose margin as a compute-quality tradeoff
rather than hide it as an opaque validation threshold. The next experiment
should learn or analytically set:

```text
margin = f(K, confidence, interaction_density, rollout_drift, compute_budget)
```

Leave-one-seed-out margin policy selection:

- `scripts/analyze_regret_margin_sweep.py`
- `docs/experiments/wpu_v2_staged_regret_margin_policy_summary.csv`
- `docs/experiments/wpu_v2_staged_regret_margin_policy_results.md`

The first stricter policy-selection test rejects a K-only scheduler. Choosing a
margin per K on four seeds and evaluating on the held-out seed does not improve
over a fixed global margin:

| policy | routed loss | loss delta | dense compute | oracle excess |
| --- | --- | --- | --- | --- |
| fixed global margin | 0.972 | -0.016 | 0.127 | 0.054 |
| LOSO K-conditioned margin | 0.973 | -0.015 | 0.147 | 0.055 |

Therefore the next scheduler should not be `margin = f(K)` alone. It should use
state-derived evidence:

```text
margin = f(K, selector_confidence, interaction_density, regret_uncertainty,
           sparse_entropy, rollout_drift, compute_budget)
```

## Combined V2 Regime Diagram

The final v2 paper figure should be a regime diagram over:

```text
N = total world state size
K = causal working set size
D = distractor ambiguity
H = rollout horizon
```

WPU should be shown as favorable only where evidence supports it:

```text
N large
K small or moderate
D manageable by retrieval
H benefits from delta overlays
```

This framing is stronger than universal superiority because it is falsifiable
and explains where WPU should lose.

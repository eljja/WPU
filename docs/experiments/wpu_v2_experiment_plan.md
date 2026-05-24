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
- token and graph baselines

Sweep:

- N: 1024, 2048, 4096, 8192
- K: 8, 16, 32
- Distractors: 0, 32, 128

Evidence needed:

- Accuracy improvement over sparse WPU.
- Latency relative to token/graph.
- Expansion/fallback frequency.

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

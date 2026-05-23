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

`scripts/cws_long_horizon_eval.py` is a diagnostic evaluator, not a full
closed-loop state overlay rollout. V2 needs a true closed-loop version.

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

V2 implementation requirement:

The evaluator must apply predicted `DeltaState` back onto branch-local
overlays, not sample independent one-step datasets.

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

## Priority 6: Local Dense Hybrid

Question:

```text
Can local dense reasoning improve accuracy without returning to global token attention?
```

V2 implementation:

- Retrieve K using causal index or learned selector.
- Run relation-typed propagation.
- Run a local dense block only over selected K.
- Predict deltas and branch probabilities.

Experiment:

- `wpu-cws-sparse`
- `wpu-cws-local-dense`
- `wpu-cws-adaptive-hybrid`
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


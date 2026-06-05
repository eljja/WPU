# PyBullet Matched-Baseline Benchmark

This experiment reruns the simulator-grounded PyBullet cup benchmark with a
parameter budget. The benchmark script now supports `--target-params`, which
selects a per-model hidden dimension whose trainable parameter count is closest
to the requested budget.

Source CSV:

- `docs/experiments/pybullet_matched_baseline_benchmark.csv`

## Protocol

- Simulator: PyBullet `DIRECT` rigid-body rollout.
- Base task: balanced cup impulse branch prediction.
- Target parameter budget: `50,000`.
- Models: `wpu-cws-indexed-sparse`, `wpu-cws-indexed-local-dense`,
  `graph-transformer`, `serialized-token`.
- Seeds: `11, 13`.
- Background objects: `0, 128`.
- Training: 20 steps, batch 8.
- Evaluation: 36 samples per condition.
- WPU input: pre-tensor indexed event-local subgraph.
- Baseline input: full simulator-derived state graph/token sequence.

## Summary

| background objects | model | params | hidden dim | accuracy | latency ms/sample | pre-tensor indexed |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | graph-transformer | 47,622 | 40 | 0.528 | 2.199 | false |
| 0 | serialized-token | 58,530 | 48 | 0.569 | 0.145 | false |
| 0 | wpu-cws-indexed-local-dense | 48,300 | 40 | 0.500 | 1.430 | true |
| 0 | wpu-cws-indexed-sparse | 52,924 | 112 | 0.569 | 1.299 | true |
| 128 | graph-transformer | 47,622 | 40 | 0.472 | 41.766 | false |
| 128 | serialized-token | 58,530 | 48 | 0.472 | 0.294 | false |
| 128 | wpu-cws-indexed-local-dense | 48,300 | 40 | 0.500 | 1.438 | true |
| 128 | wpu-cws-indexed-sparse | 52,924 | 112 | 0.569 | 2.177 | true |

## Interpretation

Parameter matching makes the simulator benchmark more favorable to the WPU
regime claim, but still not a universal win. At the same approximate parameter
budget, `wpu-cws-indexed-sparse` keeps accuracy stable as irrelevant background
state grows from 0 to 128 objects. The full-state graph and serialized-token
baselines both drop in this small run.

The systems result is mixed. WPU is much faster than the full-state graph
transformer at `N=128`, because the WPU path tensorizes only the event-local
subgraph. However, the current serialized-token implementation remains faster
than WPU at this scale. Therefore the defensible claim is not "WPU is faster
than every token baseline." The defensible claim is narrower:

```text
With objectified state and identifiable local K, pre-tensor WPU retrieval can
preserve accuracy while avoiding full-state graph processing cost.
```

## Issues Found

- Parameter count alone does not equal compute fairness. The serialized-token
  model has efficient small-sequence PyTorch execution and remains very fast.
- WPU sparse received a larger hidden dimension than WPU local-dense under the
  same parameter budget because local-dense includes a Transformer encoder.
  This is fair by parameter count but not identical by operator type.
- The run uses two seeds and short training. It should be treated as a
  pilot-level matched-baseline correction, not final paper evidence.

## Next Steps

- Repeat with five seeds, larger training budgets, and `N=0,32,128,512`.
- Add compute-normalized metrics, not only parameter matching.
- Add memory traffic and tensorized-object count as first-class metrics.
- Add long-horizon rollout to test whether stable one-step WPU accuracy holds
  under repeated delta application.

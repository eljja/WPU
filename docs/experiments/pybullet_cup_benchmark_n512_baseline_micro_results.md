# PyBullet Cup N512 Baseline-Complete Micro-Screen

This run extends the simulator-backed large-state comparison to `N_bg=512`,
total `N=517`, with WPU, graph-transformer, and serialized-token baselines in
the same protocol. It is deliberately small (`3` seeds, `2` training steps,
`8` samples per seed), so it should be treated as baseline-complete coverage and
systems feasibility evidence, not as strong simulator-superiority evidence.

Source CSV:

- `docs/experiments/pybullet_cup_benchmark_n512_baseline_micro.csv`

## Aggregate Results

| Model | Seeds | Mean branch accuracy | Mean forward latency ms/sample | Mean params | Mean MSE | Mean selected K |
|---|---:|---:|---:|---:|---:|---:|
| `wpu-cws-indexed-sparse` | 3 | 0.375000 | 2.006700 | 2,236 | 0.435292 | 4.292 |
| `graph-transformer` | 3 | 0.333333 | 151.449383 | 5,030 | 0.339243 | 4.292 |
| `serialized-token` | 3 | 0.333333 | 0.663283 | 3,954 | 0.451093 | 4.292 |
| `wpu-cws-indexed-local-dense` | 3 | 0.333333 | 2.481933 | 5,516 | 0.681147 | 4.292 |

## Interpretation

- The best WPU model is `wpu-cws-indexed-sparse`, with branch accuracy
  `0.375000`.
- The best-accuracy non-WPU baseline is `graph-transformer`, with branch
  accuracy `0.333333`.
- Against that best-accuracy baseline, the best WPU is `0.041667` more accurate
  and `75.471861x` faster in forward latency.
- `serialized-token` is the fastest model, but it is tied with the graph
  baseline at `0.333333` branch accuracy in this tiny run.
- This result closes a coverage gap at total `N=517`, but it does not close P3:
  the training budget is too small, the domain is still a single cup family, and
  the evaluation is one-step rather than long-horizon.

## Reproduction

```bash
python scripts/pybullet_cup_benchmark.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --background-objects 512 --seeds 11 13 17 --steps 2 --sim-steps 120 --samples 8 --batch-size 2 --hidden-dim 16 --layers 1 --num-heads 2 --working-set-size 12 --runtime-repeats 1 --balanced-labels --out docs/experiments/pybullet_cup_benchmark_n512_baseline_micro.csv
```

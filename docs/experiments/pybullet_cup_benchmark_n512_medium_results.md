# PyBullet Cup N512 Baseline-Complete Medium Benchmark

This run upgrades the previous `N_bg=512`, total `N=517` micro-screen to a
5-seed baseline-complete benchmark with `6` training steps and `16` evaluation
samples per seed. It keeps WPU, graph-transformer, and serialized-token
baselines in the same protocol. The result is stronger than the micro-screen,
but it is still a single cup-family, one-step benchmark with a small accuracy
margin.

Source CSV:

- `docs/experiments/pybullet_cup_benchmark_n512_medium.csv`

## Aggregate Results

| Model | Seeds | Mean branch accuracy | Majority baseline | Mean forward latency ms/sample | Mean params | Mean MSE | Mean selected K | CUDA peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `wpu-cws-indexed-sparse` | 5 | 0.387500 | 0.375000 | 2.167110 | 3,996 | 0.314859 | 4.350 | 17.369 |
| `graph-transformer` | 5 | 0.362500 | 0.375000 | 146.064080 | 10,606 | 0.256670 | 4.350 | 35.335 |
| `serialized-token` | 5 | 0.337500 | 0.375000 | 8.614540 | 8,226 | 0.241068 | 4.350 | 154.959 |
| `wpu-cws-indexed-local-dense` | 5 | 0.325000 | 0.375000 | 2.488510 | 11,220 | 0.497430 | 4.350 | 17.495 |

## Interpretation

- The best WPU model is `wpu-cws-indexed-sparse`, with branch accuracy
  `0.387500`.
- The best-accuracy non-WPU baseline is `graph-transformer`, with branch
  accuracy `0.362500`.
- Against that best-accuracy baseline, the best WPU is `0.025000` more accurate
  and `67.400400x` faster in forward latency.
- This improves the P3 evidence over the micro-screen, because all baselines
  complete at `N=517` under a larger seed and training budget.
- This still does not prove broad simulator superiority: the margin is small,
  the task is one cup family, and the evaluation is one-step rather than
  long-horizon.

## Reproduction

```bash
python scripts/pybullet_cup_benchmark.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --background-objects 512 --seeds 11 13 17 19 23 --steps 6 --sim-steps 120 --samples 16 --batch-size 2 --hidden-dim 24 --layers 1 --num-heads 4 --working-set-size 12 --runtime-repeats 1 --balanced-labels --out docs/experiments/pybullet_cup_benchmark_n512_medium.csv
```

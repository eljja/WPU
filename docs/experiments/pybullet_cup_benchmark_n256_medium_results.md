# PyBullet Cup N256 Medium-Training Benchmark

This run strengthens the previous `N_bg=256`, total `N=261` screen by increasing
the training budget from `steps=2`, `samples=12` to `steps=8`, `samples=24`,
while keeping WPU, graph-transformer, and serialized-token baselines in the same
5-seed protocol. It is stronger P3 evidence than the low-training screen, but it
is still a single cup-family benchmark and should not be promoted into broad
simulator superiority.

Source CSV:

- `docs/experiments/pybullet_cup_benchmark_n256_medium.csv`

## Aggregate Results

| Model | Seeds | Mean branch accuracy | Mean forward latency ms/sample | Mean params | Mean MSE |
|---|---:|---:|---:|---:|---:|
| `wpu-cws-indexed-local-dense` | 5 | 0.466667 | 1.784500 | 31,612 | 0.341193 |
| `graph-transformer` | 5 | 0.450000 | 108.193390 | 30,934 | 0.087779 |
| `wpu-cws-indexed-sparse` | 5 | 0.408333 | 1.595585 | 6,204 | 0.302353 |
| `serialized-token` | 5 | 0.266667 | 1.012280 | 26,738 | 0.079201 |

## Interpretation

- The best WPU model is `wpu-cws-indexed-local-dense`, with branch accuracy
  `0.466667`.
- The best-accuracy non-WPU baseline is `graph-transformer`, with branch
  accuracy `0.450000`.
- Against the best-accuracy baseline, the best WPU is `0.016667` more accurate
  and `60.629526x` faster in forward latency.
- `serialized-token` remains the fastest model, but its branch accuracy falls to
  `0.266667` in this run.
- This is positive large-N simulator evidence, not a completed P3 claim: the
  margin is small, the domain is still a single cup family, and the evaluation
  is one-step rather than long-horizon.

## Reproduction

```bash
python scripts/pybullet_cup_benchmark.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --background-objects 256 --seeds 11 13 17 19 23 --steps 8 --sim-steps 120 --samples 24 --batch-size 4 --hidden-dim 32 --num-heads 4 --working-set-size 12 --runtime-repeats 1 --balanced-labels --out docs/experiments/pybullet_cup_benchmark_n256_medium.csv
```

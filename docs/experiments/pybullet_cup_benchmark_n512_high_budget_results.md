# PyBullet Cup N512 Baseline-Complete Higher-Budget Benchmark

This run increases the `N_bg=512`, total `N=517` baseline-complete protocol from
the medium setting (`6` training steps, `16` evaluation samples per seed,
hidden dim `24`) to `10` training steps, `24` evaluation samples per seed, and
hidden dim `32`. It keeps WPU, graph-transformer, and serialized-token baselines
in the same 5-seed protocol.

Source CSV:

- `docs/experiments/pybullet_cup_benchmark_n512_high_budget.csv`

## Aggregate Results

| Model | Seeds | Mean branch accuracy | Majority baseline | Mean forward latency ms/sample | Mean params | Mean MSE | Mean selected K | CUDA peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `wpu-cws-indexed-local-dense` | 5 | 0.433333 | 0.333333 | 2.551960 | 18,908 | 0.456292 | 4.367 | 17.595 |
| `graph-transformer` | 5 | 0.425000 | 0.333333 | 146.981950 | 18,230 | 0.110532 | 4.367 | 19.530 |
| `wpu-cws-indexed-sparse` | 5 | 0.416666 | 0.333333 | 2.361880 | 6,204 | 0.341901 | 4.367 | 17.389 |
| `serialized-token` | 5 | 0.266666 | 0.333333 | 0.623770 | 14,034 | 0.103926 | 4.367 | 20.230 |

## Interpretation

- The best WPU model is `wpu-cws-indexed-local-dense`, with branch accuracy
  `0.433333`.
- The best-accuracy non-WPU baseline is `graph-transformer`, with branch
  accuracy `0.425000`.
- Against that best-accuracy baseline, the best WPU is `0.008333` more accurate
  and `57.595711x` faster in forward latency.
- The WPU edge persists under a higher budget, but the margin shrinks compared
  with the medium run. This is useful conditional evidence, not broad simulator
  superiority.
- The next P3 bottleneck is no longer another small cup-only budget increase;
  it is mechanism diversity, long-horizon simulator rollout, and
  perception-to-state objectification.

## Reproduction

```bash
python scripts/pybullet_cup_benchmark.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --background-objects 512 --seeds 11 13 17 19 23 --steps 10 --sim-steps 120 --samples 24 --batch-size 2 --hidden-dim 32 --layers 1 --num-heads 4 --working-set-size 12 --runtime-repeats 1 --balanced-labels --out docs/experiments/pybullet_cup_benchmark_n512_high_budget.csv
```

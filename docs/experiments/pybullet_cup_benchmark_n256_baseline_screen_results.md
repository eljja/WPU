# PyBullet Cup N256 Baseline Screen

This screen extends the PyBullet cup benchmark to `N_bg=256`, total `N=261`,
while keeping WPU, graph-transformer, and serialized-token baselines in the
same run. It is intentionally labeled as a screen because the training budget is
small (`steps=2`, `samples=12`, 5 seeds). The result is useful as matched
large-N feasibility evidence, not as a strong accuracy-superiority claim.

Source CSV:

- `docs/experiments/pybullet_cup_benchmark_n256_baseline_screen.csv`

## Aggregate Results

| Model | Seeds | Mean branch accuracy | Mean forward latency ms/sample | Mean params |
|---|---:|---:|---:|---:|
| `wpu-cws-indexed-sparse` | 5 | 0.350000 | 1.872940 | 6,204 |
| `wpu-cws-indexed-local-dense` | 5 | 0.316667 | 2.435880 | 31,612 |
| `graph-transformer` | 5 | 0.333333 | 114.401640 | 30,934 |
| `serialized-token` | 5 | 0.316667 | 1.484675 | 26,738 |

## Interpretation

- The matched baseline boundary is now exercised at total `N=261`, but only as
  a low-training screen.
- Sparse WPU is the most accurate model in this screen, but the margin is too
  small and the training budget too low to support a strong superiority claim.
- The graph-transformer baseline becomes much slower at this state size,
  consistent with the cost of dense graph processing.
- Serialized token processing remains competitive in latency in this small
  screened configuration, so WPU should not claim universal speed dominance.
- The strongest supported claim remains conditional: WPU is useful when the
  objectified state exposes a small identifiable causal working set before
  dense tensorization.

## Reproduction

```bash
python scripts/pybullet_cup_benchmark.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --background-objects 256 --seeds 11 13 17 19 23 --steps 2 --sim-steps 120 --samples 12 --batch-size 4 --hidden-dim 32 --num-heads 4 --working-set-size 12 --runtime-repeats 1 --balanced-labels --out docs/experiments/pybullet_cup_benchmark_n256_baseline_screen.csv
```

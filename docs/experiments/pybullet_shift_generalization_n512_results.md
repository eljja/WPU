# PyBullet N512 Mechanism-Diversity Screens

These screens test whether the large-N WPU advantage observed in the cup-only
benchmark survives mechanism variation at the same large explicit state size.
Both runs use `N_bg=512`, total `N=517`, 3 seeds, 6 training steps, 16 evaluation
samples per mechanism, hidden dim `24`, and WPU/graph/token baselines.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization_n512_screen.csv`
- `docs/experiments/pybullet_shift_generalization_n512_multimech.csv`

## Aggregate Result

| Protocol | Train mechanisms | Eval mechanisms | WPU win/tie/loss | Mean best-WPU minus best-baseline margin | Best macro WPU accuracy | Best macro baseline accuracy |
|---|---|---|---:|---:|---:|---:|
| Nominal-train screen | nominal | 7 mechanisms | 2/1/4 | -0.047619 | 0.380952 | 0.404762 |
| Multi-mechanism-train screen | 7 mechanisms | 7 mechanisms | 2/0/5 | -0.095238 | 0.369048 | 0.419643 |

## Per-Mechanism Best Comparison

### Nominal-Train Screen

| Eval mechanism | Best WPU | Best WPU accuracy | Best baseline | Best baseline accuracy | Margin |
|---|---|---:|---|---:|---:|
| catch_heavy | `wpu-cws-indexed-sparse` | 0.333333 | `serialized-token` | 0.375000 | -0.041667 |
| edge_catch_heavy | `wpu-cws-indexed-local-dense` | 0.562500 | `graph-transformer` | 0.520833 | +0.041667 |
| edge_high_force | `wpu-cws-indexed-sparse` | 0.291667 | `graph-transformer` | 0.458333 | -0.166666 |
| edge_shift | `wpu-cws-indexed-local-dense` | 0.479167 | `graph-transformer` | 0.541667 | -0.062500 |
| high_force | `wpu-cws-indexed-local-dense` | 0.312500 | `graph-transformer` | 0.312500 | +0.000000 |
| no_catch | `wpu-cws-indexed-local-dense` | 0.437500 | `graph-transformer` | 0.625000 | -0.187500 |
| nominal | `wpu-cws-indexed-local-dense` | 0.437500 | `graph-transformer` | 0.354167 | +0.083333 |

### Multi-Mechanism-Train Screen

| Eval mechanism | Best WPU | Best WPU accuracy | Best baseline | Best baseline accuracy | Margin |
|---|---|---:|---|---:|---:|
| catch_heavy | `wpu-cws-indexed-local-dense` | 0.020833 | `serialized-token` | 0.333333 | -0.312500 |
| edge_catch_heavy | `wpu-cws-indexed-sparse` | 0.479167 | `graph-transformer` | 0.583333 | -0.104166 |
| edge_high_force | `wpu-cws-indexed-local-dense` | 0.500000 | `graph-transformer` | 0.437500 | +0.062500 |
| edge_shift | `wpu-cws-indexed-sparse` | 0.520833 | `graph-transformer` | 0.604167 | -0.083334 |
| high_force | `wpu-cws-indexed-local-dense` | 0.312500 | `graph-transformer` | 0.291667 | +0.020833 |
| no_catch | `wpu-cws-indexed-sparse` | 0.500000 | `graph-transformer` | 0.625000 | -0.125000 |
| nominal | `wpu-cws-indexed-local-dense` | 0.250000 | `graph-transformer` | 0.375000 | -0.125000 |

## Interpretation

- This is a negative/claim-boundary result, not a WPU superiority result.
- The previous N=517 cup-only benchmarks show that sparse object-state execution
  can be much faster while maintaining a small accuracy edge in a narrow
  one-step cup setting.
- These mechanism screens show that the same large-N compute advantage does not
  automatically transfer to mechanism generalization.
- The important condition is therefore stricter: WPU is favored when the causal
  working set is small and identifiable, and when the local propagation law over
  that working set is learned or adapted well enough for the mechanism family.
- Multi-mechanism exposure alone did not solve the issue in this low-budget
  screen. The next model change should be mechanism-aware propagation, not merely
  adding more mechanism names to the training mixture.

## Reproduction

```bash
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --train-mechanisms nominal --eval-mechanisms nominal high_force edge_shift catch_heavy no_catch edge_high_force edge_catch_heavy --background-objects 512 --seeds 11 13 17 --steps 6 --sim-steps 120 --samples 16 --batch-size 2 --hidden-dim 24 --layers 1 --num-heads 4 --working-set-size 12 --balanced-labels --out docs/experiments/pybullet_shift_generalization_n512_screen.csv
```

```bash
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --train-mechanisms nominal high_force edge_shift catch_heavy no_catch edge_high_force edge_catch_heavy --eval-mechanisms nominal high_force edge_shift catch_heavy no_catch edge_high_force edge_catch_heavy --background-objects 512 --seeds 11 13 17 --steps 6 --sim-steps 120 --samples 16 --batch-size 2 --hidden-dim 24 --layers 1 --num-heads 4 --working-set-size 12 --balanced-labels --out docs/experiments/pybullet_shift_generalization_n512_multimech.csv
```

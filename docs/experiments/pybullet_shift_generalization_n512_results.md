# PyBullet N512 Mechanism-Diversity Screens

These screens test whether the large-N WPU advantage observed in the cup-only
benchmark survives mechanism variation at the same explicit state size. All
runs use `N_bg=512`, total `N=517`, 3 seeds, 6 training steps, 16 evaluation
samples per mechanism, hidden dim `24`, and WPU/graph/token baselines.

The latest rerun also fixes an input-contract bug: `catch_action` was present
in the PyBullet `Event` but was not encoded into `StateGraphBatch.event_features`.
The object encoder also now preserves physical state scalars used by the
objectified simulator state: `edge_distance`, `hand_distance`, `fall_risk`, and
`angular_speed`.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization_n512_screen.csv`
- `docs/experiments/pybullet_shift_generalization_n512_multimech.csv`
- `docs/experiments/pybullet_shift_generalization_n512_event_action_screen.csv`
- `docs/experiments/pybullet_shift_generalization_n512_event_action_multimech.csv`
- `docs/experiments/pybullet_shift_generalization_n512_event_physical_screen.csv`
- `docs/experiments/pybullet_shift_generalization_n512_event_physical_multimech.csv`

## Aggregate Result

| Protocol | Train mechanisms | Feature contract | WPU win/tie/loss | Mean best-WPU minus best-baseline margin | Best macro WPU accuracy | Best macro baseline accuracy |
|---|---|---|---:|---:|---:|---:|
| Original nominal-train screen | nominal | event action omitted, physical scalars omitted | 2/1/4 | -0.047619 | 0.407738 | 0.455357 |
| Action-event nominal-train screen | nominal | `catch_action` preserved | 1/1/5 | -0.083333 | 0.419643 | 0.502976 |
| Event+physical nominal-train screen | nominal | `catch_action` and physical scalars preserved | 4/0/3 | +0.002976 | 0.485119 | 0.482143 |
| Original multi-mechanism screen | 7 mechanisms | event action omitted, physical scalars omitted | 2/0/5 | -0.095238 | 0.369048 | 0.464286 |
| Action-event multi-mechanism screen | 7 mechanisms | `catch_action` preserved | 3/1/3 | -0.044643 | 0.377976 | 0.422619 |
| Event+physical multi-mechanism screen | 7 mechanisms | `catch_action` and physical scalars preserved | 2/2/3 | -0.032738 | 0.357143 | 0.389881 |

## Per-Mechanism Best Comparison After Event+Physical Encoding

### Nominal-Train Screen

| Eval mechanism | Best WPU | Best WPU accuracy | Best baseline | Best baseline accuracy | Margin |
|---|---|---:|---|---:|---:|
| catch_heavy | `wpu-cws-indexed-local-dense` | 0.333333 | `graph-transformer` | 0.625000 | -0.291667 |
| edge_catch_heavy | `wpu-cws-indexed-sparse` | 0.479167 | `graph-transformer` | 0.229167 | +0.250000 |
| edge_high_force | `wpu-cws-indexed-sparse` | 0.500000 | `graph-transformer` | 0.354167 | +0.145833 |
| edge_shift | `wpu-cws-indexed-sparse` | 0.541667 | `graph-transformer` | 0.375000 | +0.166667 |
| high_force | `wpu-cws-indexed-local-dense` | 0.416667 | `graph-transformer` | 0.541667 | -0.125000 |
| no_catch | `wpu-cws-indexed-sparse` | 0.666667 | `serialized-token` | 0.625000 | +0.041667 |
| nominal | `wpu-cws-indexed-local-dense` | 0.458333 | `graph-transformer` | 0.625000 | -0.166667 |

### Multi-Mechanism-Train Screen

| Eval mechanism | Best WPU | Best WPU accuracy | Best baseline | Best baseline accuracy | Margin |
|---|---|---:|---|---:|---:|
| catch_heavy | `wpu-cws-indexed-local-dense` | 0.041667 | `graph-transformer` | 0.062500 | -0.020833 |
| edge_catch_heavy | `wpu-cws-indexed-sparse` | 0.437500 | `graph-transformer` | 0.437500 | +0.000000 |
| edge_high_force | `wpu-cws-indexed-sparse` | 0.500000 | `graph-transformer` | 0.479167 | +0.020833 |
| edge_shift | `wpu-cws-indexed-sparse` | 0.458333 | `serialized-token` | 0.458333 | +0.000000 |
| high_force | `wpu-cws-indexed-local-dense` | 0.333333 | `graph-transformer` | 0.312500 | +0.020833 |
| no_catch | `wpu-cws-indexed-sparse` | 0.520833 | `serialized-token` | 0.625000 | -0.104167 |
| nominal | `wpu-cws-indexed-sparse` | 0.208333 | `serialized-token` | 0.354167 | -0.145834 |

## Interpretation

- This remains a claim-boundary result, not a broad WPU superiority result.
- The original negative result partly exposed an implementation flaw: the
  objectified state contained action and physical variables that the tensor
  encoder discarded.
- Preserving those variables materially improves the nominal-train large-N
  shift screen, moving it from negative margin to near tie/slight WPU edge.
- The multi-mechanism screen is still mixed/negative. Better state encoding
  reduces the gap but does not solve mechanism-family law learning.
- The defensible conclusion is stricter: WPU is favored when `N` is large, the
  causal working set `K` is small and identifiable, relevant action/physical
  state is preserved before tensorization, and the local propagation law is
  learned or adapted for the mechanism family.

## Reproduction

```bash
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --train-mechanisms nominal --eval-mechanisms nominal high_force edge_shift catch_heavy no_catch edge_high_force edge_catch_heavy --background-objects 512 --seeds 11 13 17 --steps 6 --sim-steps 120 --samples 16 --batch-size 2 --hidden-dim 24 --layers 1 --num-heads 4 --working-set-size 12 --balanced-labels --out docs/experiments/pybullet_shift_generalization_n512_event_physical_screen.csv
```

```bash
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --train-mechanisms nominal high_force edge_shift catch_heavy no_catch edge_high_force edge_catch_heavy --eval-mechanisms nominal high_force edge_shift catch_heavy no_catch edge_high_force edge_catch_heavy --background-objects 512 --seeds 11 13 17 --steps 6 --sim-steps 120 --samples 16 --batch-size 2 --hidden-dim 24 --layers 1 --num-heads 4 --working-set-size 12 --balanced-labels --out docs/experiments/pybullet_shift_generalization_n512_event_physical_multimech.csv
```

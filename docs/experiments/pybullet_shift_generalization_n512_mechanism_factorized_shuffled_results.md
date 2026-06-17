# N_bg=512 Shuffled Multi-Mechanism Factorized Adapter Results

This report corrects an important training-protocol issue in the earlier
multi-mechanism screens. The training dataset was built as a `ConcatDataset`,
but the DataLoader did not shuffle it. With small step budgets, training could
see mechanisms in dataset order rather than a balanced mixture. The script now
uses a seed-fixed shuffled DataLoader for training.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization_n512_mechanism_factorized_multitrain_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_factorized_shuffled_multitrain_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_factorized_shuffled_multitrain_5seed.csv`

## Model Change

The new `wpu-cws-indexed-mechanism-factorized` model keeps sparse execution but
replaces the object-wise residual adapter with a factorized mechanism adapter.
It encodes route physics context and uses it to produce per-object scale and
shift updates over selected object embeddings.

This is still a state-native propagation model: it does not serialize the world
as tokens and it does not use dense fallback.

## Corrected 5-Seed Result

| Model | Mean branch accuracy | Mean ECE | Mean Brier | Mean NLL | Mean dense compute |
|---|---:|---:|---:|---:|---:|
| `graph-transformer` | 0.548571 | 0.254194 | 0.581732 | 0.960382 | 1.000000 |
| `serialized-token` | 0.394286 | 0.256186 | 0.638318 | 1.050593 | 1.000000 |
| `wpu-cws-indexed-mechanism-factorized` | 0.497143 | 0.256011 | 0.639074 | 1.056679 | 0.000000 |

The 3-seed shuffled screen looked positive for the factorized adapter, but the
5-seed screen does not hold up. Graph-transformer has higher macro accuracy,
lower Brier, and lower NLL. WPU still uses zero dense compute, but this is a
compute property, not an accuracy win.

## Per-Mechanism Boundary

| Mechanism | Factorized WPU | Best baseline | Delta | Boundary |
|---|---:|---:|---:|---|
| `catch_heavy` | 0.720000 | 0.720000 | +0.000000 | Tie. |
| `edge_catch_heavy` | 0.270000 | 0.450000 | -0.180000 | Failure; edge+catch composition remains weak. |
| `edge_high_force` | 0.370000 | 0.570000 | -0.200000 | Failure; edge+force composition remains weak. |
| `edge_shift` | 0.420000 | 0.540000 | -0.120000 | Failure; edge law remains weak. |
| `high_force` | 0.440000 | 0.520000 | -0.080000 | Failure. |
| `no_catch` | 0.590000 | 0.470000 | +0.120000 | Positive. |
| `nominal` | 0.670000 | 0.570000 | +0.100000 | Positive. |

Win/tie/loss is `2/1/4`, with mean margin `-0.051429`.

## Interpretation

This is a useful negative result. It corrects the earlier optimistic
multi-mechanism result and shows that mechanism-aware sparse adapters are not
yet enough for robust composition generalization. The main failure is still the
edge-conditioned family, especially composed edge mechanisms.

The current WPU v2 direction remains valid, but the claim must be narrowed:

- WPU has a sparse compute advantage when `K` is small and indexed.
- WPU can inject mechanism state into local object propagation.
- WPU does not yet learn robust local-law composition across held-out edge
  mechanisms under this small training budget.

## Next Step

The next experiment should stop relying on branch-label supervision alone. It
should add explicit local-law or composition supervision, such as edge-distance
auxiliary targets, force/edge/catch factor losses, or simulator-derived
counterfactual pairs that isolate one mechanism factor at a time.

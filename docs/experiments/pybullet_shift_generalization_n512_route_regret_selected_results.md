# PyBullet N_bg=512 Selected Route-Regret Shift Screen

Source CSV:

- `docs/experiments/pybullet_shift_generalization_n512_route_regret_selected.csv`

This experiment reruns the large-state PyBullet mechanism-shift screen with a
validation-selected route-regret WPU. It tests whether explicit sparse/dense
counterfactual route supervision plus threshold selection improves the
large-`N`, small-`K` mechanism-shift regime.

## Protocol

- Total objects: `N=517` (`N_bg=512` plus cup, table, hand, edge, goal).
- Training mechanism: `nominal`.
- Evaluation mechanisms: `nominal`, `high_force`, `edge_shift`,
  `catch_heavy`, `no_catch`, `edge_high_force`, `edge_catch_heavy`.
- Seeds: `11`, `13`, `17`.
- Compared models: sparse WPU, local-dense WPU, selected route-regret WPU,
  graph transformer, serialized token.
- Route-regret threshold candidates: `-0.75`, `-0.5`, `-0.25`, `0.0`, `0.25`;
  selected on a held-out nominal validation split with compute-adjusted NLL.

## Aggregate Result

| model | macro accuracy | macro ECE | dense compute ratio |
|---|---:|---:|---:|
| graph-transformer | 0.508929 | 0.244585 | 1.000000 |
| serialized-token | 0.425595 | 0.268479 | 1.000000 |
| wpu-cws-indexed-local-dense | 0.377976 | 0.186405 | 1.000000 |
| wpu-cws-indexed-physics-regret-hybrid | 0.366071 | 0.243926 | 0.071429 |
| wpu-cws-indexed-sparse | 0.351190 | 0.209293 | 0.000000 |

Best WPU versus best non-WPU over seven mechanisms is `2 / 1 / 4`
win/tie/loss. The selected route-regret WPU uses little dense recompute
(`0.071429`) but does not beat the best baseline on macro accuracy.

## Per-Mechanism Boundary

| mechanism | best WPU | best baseline | selected route-regret WPU |
|---|---:|---:|---:|
| catch_heavy | 0.500000 | 0.645833 | 0.041667 |
| edge_catch_heavy | 0.375000 | 0.520833 | 0.187500 |
| edge_high_force | 0.458333 | 0.458333 | 0.458333 |
| edge_shift | 0.437500 | 0.416667 | 0.375000 |
| high_force | 0.354167 | 0.562500 | 0.354167 |
| no_catch | 0.770833 | 0.520833 | 0.770833 |
| nominal | 0.500000 | 0.604167 | 0.375000 |

## Interpretation

This is a useful negative/mixed result. Validation-selected route-regret routing
does not solve mechanism-shift generalization. It creates a low-compute route
that is strong on `no_catch` and ties the baseline on `edge_high_force`, but it
fails badly on `catch_heavy` and does not recover graph-transformer macro
accuracy.

The main implication is architectural: route-regret threshold selection is not
enough. The next P4/P5 step should train mechanism-aware propagation or
adaptation, not only a sparse/dense route selector. In particular, `catch_heavy`
remains a strong counterexample where the branch-prior/mechanism law changes
more than the local working-set route can handle.

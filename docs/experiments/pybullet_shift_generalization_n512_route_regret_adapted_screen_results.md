# PyBullet N_bg=512 Route-Regret With Mechanism-Prior Adaptation Screen

Source CSV:

- `docs/experiments/pybullet_shift_generalization_n512_route_regret_adapted_screen.csv`

This follow-up tests whether the selected route-regret WPU improves when it is
given the same mechanism-prior adaptation interface used by the baselines. It is
an adapted setting, not zero-shot mechanism generalization.

## Protocol

- Total objects: `N=517`.
- Training mechanism: `nominal`.
- Evaluation mechanisms: `high_force`, `edge_shift`, `catch_heavy`, `no_catch`.
- Seeds: `11`, `13`, `17`.
- Models: selected route-regret WPU, graph transformer, serialized token.
- Mechanism-prior strengths: `0.0`, `0.25`, `0.5`, `0.75`, `1.0`.
- Selection metric: `nll_ece` on mechanism-specific calibration samples.

## Aggregate Result

| model | macro accuracy | macro ECE | macro Brier | dense compute ratio |
|---|---:|---:|---:|---:|
| graph-transformer | 0.527778 | 0.337050 | 0.601170 | 1.000000 |
| serialized-token | 0.416667 | 0.275198 | 0.658986 | 1.000000 |
| wpu-cws-indexed-physics-regret-hybrid | 0.312500 | 0.274279 | 0.663490 | 0.097222 |

Selected route-regret WPU versus best non-WPU is `0 / 0 / 4`
win/tie/loss over the four shifted mechanisms. It preserves low dense compute
but does not preserve accuracy.

## Per-Mechanism Boundary

| mechanism | route-regret WPU | best baseline | delta |
|---|---:|---:|---:|
| catch_heavy | 0.333333 | 0.944444 | -0.611111 |
| edge_shift | 0.194445 | 0.222222 | -0.027777 |
| high_force | 0.333333 | 0.500000 | -0.166667 |
| no_catch | 0.388889 | 0.527778 | -0.138889 |

## Interpretation

This is a negative adapted-screen result. Mechanism-prior adaptation is useful
as an interface, but it does not rescue the selected route-regret WPU when the
same adaptation evidence is available to dense/token baselines. The sharp
`catch_heavy` gap shows that branch-prior correction can strongly help a dense
baseline while the sparse route-regret model still lacks mechanism-aware
propagation capacity.

The next P4/P5 step should therefore not be another prior-bias or threshold
selection pass. It should modify the WPU propagation model itself: mechanism
state must condition local transition dynamics, branch logits, and uncertainty,
not only post-hoc route or output biases.

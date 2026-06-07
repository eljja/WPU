# PyBullet Branch-Prior Shift Audit

This analysis separates relation/propagation failure from branch-label prior shift in the PyBullet mechanism-family benchmark. `majority_accuracy` is the non-learned baseline that always predicts the most frequent branch in the evaluation mechanism.

Source CSV:

- `docs/experiments/pybullet_shift_generalization.csv`

Derived CSV:

- `docs/experiments/pybullet_branch_prior_shift.csv`

| mechanism | best WPU | best baseline | WPU acc | baseline acc | majority acc | WPU-baseline | majority-WPU | prior dominated |
|---|---|---|---:|---:|---:|---:|---:|---|
| catch_heavy | `wpu-cws-indexed-local-dense` | `serialized-token` | 0.408730 | 0.349206 | 0.753968 | 0.059524 | 0.345238 | True |
| edge_shift | `wpu-cws-indexed-sparse` | `serialized-token` | 0.527778 | 0.571428 | 0.515873 | -0.043650 | -0.011905 | False |
| high_force | `wpu-cws-indexed-local-dense` | `serialized-token` | 0.432540 | 0.460318 | 0.424603 | -0.027778 | -0.007937 | False |
| nominal | `wpu-cws-indexed-sparse` | `serialized-token` | 0.444445 | 0.500000 | 0.468254 | -0.055555 | 0.023809 | False |

## Interpretation

- Mean WPU-baseline accuracy delta over shifted mechanisms is `-0.003968`.
- Mean majority-prior gap over the best WPU on shifted mechanisms is `0.108465`.
- `1/3` shifted mechanisms are prior-dominated, meaning the majority prior beats both the best WPU and the best non-WPU baseline.
- In prior-dominated mechanisms, mechanism-aware branch priors, branch-frequency shift detection, and uncertainty-gated recompute are higher-priority fixes than simply enlarging the propagation block.
- This narrows the WPU claim: objectification and sparse propagation are not enough when the branch prior itself shifts without explicit prior adaptation.

## Mechanism Consequence

The prior-dominated mechanisms are `catch_heavy`. These should remain explicit P4/P5 counterexamples in WPU v2.

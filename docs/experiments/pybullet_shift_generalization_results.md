# PyBullet Shift Generalization and Calibration

This benchmark trains on nominal PyBullet cup dynamics and evaluates on held-out
mechanism families. It addresses cross-generator-family generalization and adds
calibration metrics as first-class outputs.

Source CSV:

- `docs/experiments/pybullet_shift_generalization.csv`

## Protocol

- Train mechanism: `nominal`.
- Eval mechanisms: `nominal`, `high_force`, `edge_shift`, `catch_heavy`.
- Models: `wpu-cws-indexed-sparse`, `wpu-cws-indexed-local-dense`,
  `graph-transformer`, `serialized-token`.
- Seeds: `11, 13, 17, 19, 23, 29, 31`.
- Background objects: `32`.
- Training steps: `20`.
- Eval samples: `36` per seed/mechanism.
- Calibration metrics: ECE, Brier score, NLL.

## Summary

| eval mechanism | model | accuracy | ECE | Brier | NLL | selected K |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| nominal | wpu-cws-indexed-sparse | 0.444445 | 0.118385 | 0.628308 | 1.033729 | 4.350000 |
| nominal | wpu-cws-indexed-local-dense | 0.388889 | 0.196291 | 0.646500 | 1.035757 | 4.350000 |
| nominal | graph-transformer | 0.444444 | 0.187391 | 0.648265 | 1.036947 | 4.350000 |
| nominal | serialized-token | 0.500000 | 0.131319 | 0.609825 | 0.982766 | 4.350000 |
| high_force | wpu-cws-indexed-sparse | 0.428571 | 0.146666 | 0.624832 | 1.036275 | 4.350000 |
| high_force | wpu-cws-indexed-local-dense | 0.432540 | 0.313820 | 0.736105 | 1.222798 | 4.350000 |
| high_force | graph-transformer | 0.452381 | 0.313074 | 0.758171 | 1.291259 | 4.350000 |
| high_force | serialized-token | 0.460318 | 0.210711 | 0.663026 | 1.092347 | 4.350000 |
| edge_shift | wpu-cws-indexed-sparse | 0.527778 | 0.212969 | 0.607690 | 1.012082 | 4.350000 |
| edge_shift | wpu-cws-indexed-local-dense | 0.456349 | 0.195355 | 0.598732 | 0.992563 | 4.350000 |
| edge_shift | graph-transformer | 0.531746 | 0.210478 | 0.604425 | 1.022356 | 4.350000 |
| edge_shift | serialized-token | 0.571428 | 0.187070 | 0.592148 | 0.987664 | 4.350000 |
| catch_heavy | wpu-cws-indexed-sparse | 0.321429 | 0.241991 | 0.655905 | 1.067589 | 4.803571 |
| catch_heavy | wpu-cws-indexed-local-dense | 0.408730 | 0.264469 | 0.688479 | 1.076357 | 4.803571 |
| catch_heavy | graph-transformer | 0.341270 | 0.294828 | 0.718105 | 1.111870 | 4.803571 |
| catch_heavy | serialized-token | 0.349206 | 0.219188 | 0.650279 | 1.020129 | 4.803571 |

## Interpretation

This is a mixed but useful cross-generator result.

The 7-seed result keeps the same regime boundary. WPU wins on `catch_heavy`
through the local-dense path (`0.408730` versus the best non-WPU `0.349206`),
but loses on `edge_shift` and `high_force`. The older two-seed `edge_shift`
advantage was therefore not stable enough to support a broad shift claim.

Calibration remains slightly favorable in aggregate: mean WPU ECE is lower than
mean baseline ECE in the dashboard calculation, but the ratio weakens to
`0.963449` after the 7-seed rerun. This is useful but not a solved calibration
claim, because accuracy remains mixed and the test is still single-step rather
than multi-step rollout calibration.

## Consequence

Priority 4 and 5 are now instrumented but not solved:

- Cross-generator evaluation exists through mechanism-family shifts.
- ECE, Brier, and NLL are first-class outputs.
- WPU still needs mechanism-aware branch priors, uncertainty-gated fallback, and
  shift-aware calibration before claiming robust world-state generalization.

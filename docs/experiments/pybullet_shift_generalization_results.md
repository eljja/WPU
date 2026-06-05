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
- Seeds: `11, 13, 17, 19, 23`.
- Background objects: `32`.
- Training steps: `20`.
- Eval samples: `36` per seed/mechanism.
- Calibration metrics: ECE, Brier score, NLL.

## Summary

| eval mechanism | model | accuracy | ECE | Brier | NLL | selected K |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| nominal | wpu-cws-indexed-sparse | 0.427778 | 0.124159 | 0.636457 | 1.046428 | 4.3600 |
| nominal | wpu-cws-indexed-local-dense | 0.394444 | 0.200036 | 0.658903 | 1.054336 | 4.3600 |
| nominal | graph-transformer | 0.411111 | 0.221604 | 0.675927 | 1.077000 | 4.3600 |
| nominal | serialized-token | 0.455556 | 0.140774 | 0.628618 | 1.012293 | 4.3600 |
| high_force | wpu-cws-indexed-sparse | 0.416667 | 0.112684 | 0.638699 | 1.055790 | 4.3600 |
| high_force | wpu-cws-indexed-local-dense | 0.416667 | 0.342607 | 0.777555 | 1.295899 | 4.3600 |
| high_force | graph-transformer | 0.416667 | 0.382426 | 0.828363 | 1.420038 | 4.3600 |
| high_force | serialized-token | 0.433333 | 0.232135 | 0.703167 | 1.154525 | 4.3600 |
| edge_shift | wpu-cws-indexed-sparse | 0.522222 | 0.178233 | 0.622615 | 1.033803 | 4.3600 |
| edge_shift | wpu-cws-indexed-local-dense | 0.477778 | 0.185243 | 0.603099 | 0.999774 | 4.3600 |
| edge_shift | graph-transformer | 0.522222 | 0.224137 | 0.614147 | 1.033391 | 4.3600 |
| edge_shift | serialized-token | 0.555555 | 0.167665 | 0.603833 | 0.997661 | 4.3600 |
| catch_heavy | wpu-cws-indexed-sparse | 0.300000 | 0.252880 | 0.662538 | 1.078916 | 4.8050 |
| catch_heavy | wpu-cws-indexed-local-dense | 0.366667 | 0.313698 | 0.728429 | 1.135705 | 4.8050 |
| catch_heavy | graph-transformer | 0.322222 | 0.355048 | 0.763415 | 1.185473 | 4.8050 |
| catch_heavy | serialized-token | 0.327778 | 0.229288 | 0.670691 | 1.058393 | 4.8050 |

## Interpretation

This is a mixed but useful cross-generator result.

The 5-seed result changes the regime boundary. WPU now wins on `catch_heavy`
through the local-dense path (`0.366667` versus the best non-WPU `0.327778`),
but loses on `edge_shift` and `high_force`. The older two-seed `edge_shift`
advantage was therefore not stable enough to support a broad shift claim.

Calibration improves in aggregate: mean WPU ECE is lower than mean baseline ECE
in the dashboard calculation. This is useful but not a solved calibration
claim, because accuracy remains mixed and the test is still single-step rather
than multi-step rollout calibration.

## Consequence

Priority 4 and 5 are now instrumented but not solved:

- Cross-generator evaluation exists through mechanism-family shifts.
- ECE, Brier, and NLL are first-class outputs.
- WPU still needs mechanism-aware branch priors, uncertainty-gated fallback, and
  shift-aware calibration before claiming robust world-state generalization.

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
- Seeds: `11, 13`.
- Background objects: `32`.
- Training steps: `20`.
- Eval samples: `36` per seed/mechanism.
- Calibration metrics: ECE, Brier score, NLL.

## Summary

| eval mechanism | model | accuracy | ECE | Brier | NLL | selected K |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| nominal | wpu-cws-indexed-sparse | 0.486111 | 0.216639 | 0.638486 | 1.044608 | 4.3625 |
| nominal | wpu-cws-indexed-local-dense | 0.416666 | 0.221161 | 0.666338 | 1.065497 | 4.3625 |
| nominal | graph-transformer | 0.361111 | 0.217941 | 0.719271 | 1.137500 | 4.3625 |
| nominal | serialized-token | 0.402778 | 0.100940 | 0.630391 | 1.009570 | 4.3625 |
| high_force | wpu-cws-indexed-sparse | 0.444445 | 0.110487 | 0.643738 | 1.063413 | 4.3625 |
| high_force | wpu-cws-indexed-local-dense | 0.444445 | 0.369688 | 0.804497 | 1.376495 | 4.3625 |
| high_force | graph-transformer | 0.430555 | 0.393302 | 0.846081 | 1.453774 | 4.3625 |
| high_force | serialized-token | 0.458334 | 0.188049 | 0.686918 | 1.124732 | 4.3625 |
| edge_shift | wpu-cws-indexed-sparse | 0.597222 | 0.171235 | 0.624835 | 1.037711 | 4.3625 |
| edge_shift | wpu-cws-indexed-local-dense | 0.527778 | 0.168204 | 0.626533 | 1.051272 | 4.3625 |
| edge_shift | graph-transformer | 0.472222 | 0.195111 | 0.667463 | 1.130049 | 4.3625 |
| edge_shift | serialized-token | 0.472222 | 0.119432 | 0.644900 | 1.072424 | 4.3625 |
| catch_heavy | wpu-cws-indexed-sparse | 0.194445 | 0.248698 | 0.686505 | 1.106100 | 4.8125 |
| catch_heavy | wpu-cws-indexed-local-dense | 0.277778 | 0.383692 | 0.779896 | 1.207206 | 4.8125 |
| catch_heavy | graph-transformer | 0.361112 | 0.355464 | 0.789159 | 1.211036 | 4.8125 |
| catch_heavy | serialized-token | 0.402778 | 0.198037 | 0.637309 | 1.004722 | 4.8125 |

## Interpretation

This is a mixed but useful cross-generator result.

The positive regime is `edge_shift`: sparse WPU reaches `0.597222` accuracy,
above local-dense WPU, graph, and serialized-token baselines. This is consistent
with the WPU premise when the event-local object graph remains identifiable and
the mechanism shift is still local.

The negative regime is `catch_heavy`: WPU sparse falls to `0.194445`, while the
serialized-token baseline reaches `0.402778`. This indicates that WPU's current
state/retrieval/branch head underuses the changed catch-action prior. The result
should not be hidden; it is exactly the kind of regime boundary needed for a
credible WPU claim.

Calibration is not solved. ECE varies substantially by model and mechanism.
Sparse WPU has low ECE on `high_force` (`0.110487`) but remains poorly accurate
on `catch_heavy`. Serialized-token is often better calibrated in this short
pilot. Therefore future WPU claims should report accuracy and calibration
together.

## Consequence

Priority 4 and 5 are now instrumented but not solved:

- Cross-generator evaluation exists through mechanism-family shifts.
- ECE, Brier, and NLL are first-class outputs.
- WPU still needs mechanism-aware branch priors, uncertainty-gated fallback, and
  shift-aware calibration before claiming robust world-state generalization.

# PyBullet Mechanism-Selective Calibration Gate

This audit tests whether selecting different WPU recompute policies by mechanism can satisfy accuracy, calibration, and low-cost constraints when a single global threshold cannot.

This is not zero-shot routing. It assumes a mechanism or mechanism-level detector has already identified the shifted family. The purpose is to test whether P5 should move from global confidence gates toward mechanism-aware calibration routing.

Source CSV: `docs/experiments/pybullet_learned_uncertainty_gate.csv`

Derived CSV: `docs/experiments/pybullet_mechanism_selective_calibration_gate.csv`

## Interpretation

- There are `4` low-cost (`cost <= 0.25`), accuracy-safe, calibration-safe non-reference combinations.
- The best safe policy has accuracy delta `0.029100`, ECE delta `-0.001652`, Brier delta `-0.030758`, and cost `0.247355`.
- This suggests P5 is not impossible; the next path is mechanism-aware selective routing rather than another global confidence threshold.
- The caveat is decisive: mechanism identification and calibration samples remain required, so this is not zero-shot calibration-safe routing.

## Per-Mechanism Safe Candidates

| Mechanism | Selected policy | Cost | Accuracy delta | ECE delta | Brier delta |
|---|---|---:|---:|---:|---:|
| `edge_catch_heavy` | `wpu_sparse` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `edge_high_force` | `fewshot_learned_p0.01` | 0.742064 | 0.087301 | -0.004957 | -0.092275 |
| `no_catch` | `wpu_sparse` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Top Combinations

| Policy | Cost | Accuracy delta | ECE delta | Brier delta | Safe |
|---|---:|---:|---:|---:|---|
| `edge_catch_heavy=wpu_sparse; edge_high_force=fewshot_learned_p0.01; no_catch=wpu_sparse` | 0.247355 | 0.029100 | -0.001652 | -0.030758 | True |
| `edge_catch_heavy=fewshot_learned_p0.04; edge_high_force=source_learned_p0.12; no_catch=wpu_sparse` | 0.182540 | 0.027778 | -0.000240 | -0.023432 | True |
| `edge_catch_heavy=source_learned_p0.12; edge_high_force=source_learned_p0.12; no_catch=wpu_sparse` | 0.146826 | 0.025132 | -0.000068 | -0.021758 | True |
| `edge_catch_heavy=wpu_sparse; edge_high_force=source_learned_p0.12; no_catch=wpu_sparse` | 0.107143 | 0.017196 | -0.001504 | -0.020073 | True |
| `edge_catch_heavy=wpu_sparse; edge_high_force=wpu_sparse; no_catch=wpu_sparse` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | True |
| `edge_catch_heavy=fewshot_learned_p0.08; edge_high_force=source_learned_p0.12; no_catch=source_learned_p0.08` | 0.236773 | 0.072751 | 0.024656 | -0.040653 | False |
| `edge_catch_heavy=fewshot_learned_p0.12; edge_high_force=fewshot_learned_p0.12; no_catch=source_learned_p0.12` | 0.231481 | 0.072751 | 0.029206 | -0.041222 | False |
| `edge_catch_heavy=source_learned_p0.12; edge_high_force=fewshot_learned_p0.12; no_catch=source_learned_p0.12` | 0.246032 | 0.068783 | 0.022250 | -0.040211 | False |
| `edge_catch_heavy=fewshot_learned_p0.08; edge_high_force=source_learned_p0.08; no_catch=source_learned_p0.12` | 0.240741 | 0.068783 | 0.025923 | -0.041486 | False |
| `edge_catch_heavy=wpu_sparse; edge_high_force=fewshot_learned_p0.12; no_catch=source_learned_p0.08` | 0.231481 | 0.067460 | 0.023576 | -0.042539 | False |
| `edge_catch_heavy=fewshot_learned_p0.12; edge_high_force=source_learned_p0.12; no_catch=source_learned_p0.04` | 0.243387 | 0.067460 | 0.025643 | -0.040855 | False |
| `edge_catch_heavy=fewshot_learned_p0.08; edge_high_force=source_learned_p0.12; no_catch=source_learned_p0.12` | 0.211640 | 0.066138 | 0.021894 | -0.036640 | False |
| `edge_catch_heavy=fewshot_learned_p0.08; edge_high_force=wpu_sparse; no_catch=fewshot_learned_p0.12` | 0.165344 | 0.066138 | 0.035705 | -0.026279 | False |
| `edge_catch_heavy=fewshot_learned_p0.08; edge_high_force=wpu_sparse; no_catch=fewshot_learned_p0.08` | 0.206349 | 0.066138 | 0.036903 | -0.028049 | False |
| `edge_catch_heavy=fewshot_learned_p0.12; edge_high_force=source_learned_p0.08; no_catch=source_learned_p0.08` | 0.244709 | 0.066137 | 0.024516 | -0.043420 | False |
| `edge_catch_heavy=fewshot_learned_p0.12; edge_high_force=source_learned_p0.12; no_catch=source_learned_p0.08` | 0.215609 | 0.063492 | 0.020487 | -0.038574 | False |
| `edge_catch_heavy=wpu_sparse; edge_high_force=fewshot_learned_p0.08; no_catch=source_learned_p0.12` | 0.239418 | 0.062169 | 0.020705 | -0.040118 | False |
| `edge_catch_heavy=wpu_sparse; edge_high_force=source_learned_p0.12; no_catch=fewshot_learned_p0.12` | 0.226191 | 0.062169 | 0.021639 | -0.041576 | False |
| `edge_catch_heavy=fewshot_learned_p0.12; edge_high_force=source_learned_p0.04; no_catch=source_learned_p0.12` | 0.243386 | 0.062169 | 0.021918 | -0.042439 | False |
| `edge_catch_heavy=source_learned_p0.01; edge_high_force=wpu_sparse; no_catch=fewshot_learned_p0.12` | 0.239418 | 0.062169 | 0.033028 | -0.025379 | False |

## Best Safe Policy

`edge_catch_heavy=wpu_sparse; edge_high_force=fewshot_learned_p0.01; no_catch=wpu_sparse`

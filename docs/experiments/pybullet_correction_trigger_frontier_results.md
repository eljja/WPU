# PyBullet Correction-Trigger Frontier Results

This audit isolates the remaining P2 bottleneck. Selective correction reduces the corrected-object fraction, but correction trigger frequency remains high. The frontier tests whether lower trigger frequency preserves state integrity.

Source CSV: `docs/experiments/pybullet_state_integrity_audit.csv`

Derived CSV: `docs/experiments/pybullet_correction_trigger_frontier.csv`

The best integrity is `0.988647` (`rollback`) at correction rate `0.000000`. The best low-disruption score is `0.958508` (`guarded`). Among correction-trigger policies under correction rate <= `0.25`, the best integrity is `0.653668` (`selective_corrected_entropy035`). Rows meeting the joint target: `0`.

| run | family | integrity | low-disruption | violations/step | correction | corrected objects | rollback | escalation | joint target |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `finite_clamped` | `no_correction_baseline` | 0.527391 | 0.527391 | 0.784166 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `guarded` | `safety_baseline` | 0.958508 | 0.958508 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `rollback` | `safety_baseline` | 0.988647 | 0.744897 | 0.000000 | 0.000000 | 0.000000 | 0.812500 | 0.000000 | 0 |
| `selective_corrected_margin1` | `correction_trigger` | 0.527391 | 0.527391 | 0.784166 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `selective_corrected_rawdelta2m` | `correction_trigger` | 0.527391 | 0.527391 | 0.784166 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `selective_corrected_stride2` | `correction_trigger` | 0.535190 | 0.527543 | 0.770000 | 0.014166 | 0.027371 | 0.000000 | 0.000000 | 0 |
| `selective_corrected_entropy045` | `correction_trigger` | 0.642658 | 0.586039 | 0.574167 | 0.210000 | 0.027458 | 0.000000 | 0.000000 | 0 |
| `selective_corrected_entropy035` | `correction_trigger` | 0.653668 | 0.592049 | 0.554167 | 0.230000 | 0.027458 | 0.000000 | 0.000000 | 0 |
| `finite_corrected` | `correction_trigger` | 0.958735 | 0.645068 | 0.000000 | 0.784166 | 0.784166 | 0.000000 | 0.000000 | 0 |
| `selective_corrected` | `correction_trigger` | 0.958735 | 0.758574 | 0.000000 | 0.784166 | 0.027461 | 0.000000 | 0.000000 | 0 |

## Interpretation

- Entropy gating lowers correction rate but does not preserve the integrity target.
- Raw-delta thresholding is not a useful trigger in this setting; it removes correction and fails.
- The next P2 step is not a stricter threshold, but a more stable transition model or a learned trigger.

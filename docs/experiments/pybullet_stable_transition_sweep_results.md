# PyBullet Stable Transition Sweep Results

This experiment tests the next P2 bottleneck directly: whether transition training objectives can stabilize sparse WPU rollouts before correction. It trains WPU sparse with combinations of delta-norm, rollout-consistency, and state-validity losses, then evaluates both raw finite-clamped rollout and selective-correction rollout.

Source CSV: `docs/experiments/pybullet_stable_transition_sweep.csv`

Best raw finite-clamped integrity is `0.633398` (`delta_norm_strong`), with violations/step `0.598333`. Best selective-correction low-disruption score is `0.809071` (`delta_norm_strong`), with integrity `0.962773` and correction rate `0.598333`. Rows meeting the joint target (integrity >= 0.8 and correction_rate <= 0.25): `0`.

| config | eval mode | integrity | low-disruption | violations/step | correction | corrected objects | delta norm | joint target |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `delta_norm_strong` | `raw_finite_clamped` | 0.633398 | 0.633398 | 0.598333 | 0.000000 | 0.000000 | 0.670200 | 0 |
| `combined_strong` | `raw_finite_clamped` | 0.628103 | 0.628103 | 0.605833 | 0.000000 | 0.000000 | 0.663909 | 0 |
| `delta_norm_mid` | `raw_finite_clamped` | 0.564610 | 0.564610 | 0.720000 | 0.000000 | 0.000000 | 0.693878 | 0 |
| `combined_mid` | `raw_finite_clamped` | 0.560365 | 0.560365 | 0.725000 | 0.000000 | 0.000000 | 0.691949 | 0 |
| `baseline_finite` | `raw_finite_clamped` | 0.527391 | 0.527391 | 0.784166 | 0.000000 | 0.000000 | 0.709270 | 0 |
| `validity_mid` | `raw_finite_clamped` | 0.527391 | 0.527391 | 0.784166 | 0.000000 | 0.000000 | 0.709270 | 0 |
| `validity_strong` | `raw_finite_clamped` | 0.527391 | 0.527391 | 0.784166 | 0.000000 | 0.000000 | 0.709270 | 0 |
| `delta_norm_strong` | `selective_corrected` | 0.962773 | 0.809071 | 0.000000 | 0.598333 | 0.027466 | 0.661827 | 0 |
| `combined_strong` | `selective_corrected` | 0.961608 | 0.806030 | 0.000000 | 0.605833 | 0.027462 | 0.655450 | 0 |
| `delta_norm_mid` | `selective_corrected` | 0.960800 | 0.776680 | 0.000000 | 0.720000 | 0.027463 | 0.683509 | 0 |
| `combined_mid` | `selective_corrected` | 0.958957 | 0.773588 | 0.000000 | 0.725000 | 0.027462 | 0.681580 | 0 |
| `baseline_finite` | `selective_corrected` | 0.958735 | 0.758574 | 0.000000 | 0.784166 | 0.027461 | 0.697858 | 0 |
| `validity_mid` | `selective_corrected` | 0.958735 | 0.758574 | 0.000000 | 0.784166 | 0.027461 | 0.697858 | 0 |
| `validity_strong` | `selective_corrected` | 0.958735 | 0.758574 | 0.000000 | 0.784166 | 0.027461 | 0.697858 | 0 |

## Interpretation

- Higher raw finite-clamped integrity is evidence that the transition model itself is more stable.
- Lower selective-correction rate at preserved integrity would mean the P2 correction-frequency bottleneck is reduced.
- If both fail, P2 needs architecture or multi-step/simulator-resynchronized training rather than more loss-weight tuning.

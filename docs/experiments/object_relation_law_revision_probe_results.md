# Object-Relation Local Law Revision Probe

This probe tests the next step after OOD stress. If an objectified relation/law
stack fails under shifted dynamics, can a small calibration set revise the local
law without returning to token serialization?

The base model is unchanged from the local-law probe: relation candidates are
selected from object histories, and the initial law is trained on
`contact_inverse` and `support_inverse`. The revision step receives 64
calibration samples from the held-out mechanism and then evaluates on 256 test
samples per seed.

Revision policies:

- `base_history_law`: no revision; use the original inverse-distance law.
- `gain_calibrated_history_law`: fit one scalar multiplier on the base law.
- `form_revised_history_law`: choose and fit a small law form over selected
  objectified relations.
- `form_revised_oracle_law`: same revision with oracle relations, measuring the
  remaining gap caused by relation selection.

Source CSV: `docs/experiments/object_relation_law_revision_probe.csv`

Reproduce:

```bash
python scripts/object_relation_law_revision_probe.py --train-samples 768 --calibration-samples 64 --eval-samples 256 --seeds 31 37 41 43 47 --candidates 8 --history-steps 14 --train-steps 180 --threshold 0.5 --mechanisms hidden_inverse_gain_shift hidden_power_shift --out docs/experiments/object_relation_law_revision_probe.csv
```

## Result

The table reports five-seed means over 1,280 held-out test samples per mechanism.

| mechanism | policy | selected_form | relation_precision | relation_recall | delta_mse | calibration_mse |
|---|---|---|---:|---:|---:|---:|
| hidden_inverse_gain_shift | base_history_law | trained_base | 0.998438 | 0.998438 | 0.115978 | 0.000000 |
| hidden_inverse_gain_shift | gain_calibrated_history_law | gain_scaled_base | 0.998438 | 0.998438 | 0.000342 | 0.001854 |
| hidden_inverse_gain_shift | form_revised_history_law | mixed | 0.998438 | 0.998438 | 0.000323 | 0.001723 |
| hidden_inverse_gain_shift | form_revised_oracle_law | mixed | 1.000000 | 1.000000 | 0.000229 | 0.000218 |
| hidden_power_shift | base_history_law | trained_base | 0.956250 | 0.956250 | 0.054596 | 0.000000 |
| hidden_power_shift | gain_calibrated_history_law | gain_scaled_base | 0.956250 | 0.956250 | 0.022787 | 0.030868 |
| hidden_power_shift | form_revised_history_law | mixed | 0.956250 | 0.956250 | 0.008887 | 0.014951 |
| hidden_power_shift | form_revised_oracle_law | mixed | 1.000000 | 1.000000 | 0.000232 | 0.000218 |

## Interpretation

The result supports a narrow but important WPU loop:

```text
objectify -> propose local law -> stress -> observe residual -> revise law
```

For `hidden_inverse_gain_shift`, the residual is mostly a gain error. A scalar
calibration reduces MSE from `0.115978` to `0.000342`; form revision gives a
similar `0.000323`.

For `hidden_power_shift`, the residual is a law-form error. Gain calibration
helps but remains at `0.022787`; form revision improves to `0.008887`. The
oracle relation result `0.000232` shows that most remaining error is not the
candidate law family itself but relation-selection and noisy calibration.

This is still a generated synthetic benchmark. It does not show discovery of an
unknown physical theory. It does show how WPU-style objectification can make
theory revision operational: residuals are attached to persistent objects,
relations, geometry, and local deltas, so the system can test whether failure is
caused by relation retrieval or by the propagation law.

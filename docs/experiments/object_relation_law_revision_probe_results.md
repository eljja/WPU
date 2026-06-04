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

| mechanism | policy | selected_form | decision | delta_mse | relative_improvement | relation_selection_gap | law_residual_gap |
|---|---|---|---|---:|---:|---:|---:|
| hidden_inverse_gain_shift | base_history_law | trained_base | baseline | 0.115978 | 0.000000 | 0.115749 | 0.000229 |
| hidden_inverse_gain_shift | gain_calibrated_history_law | gain_scaled_base | accept_revision | 0.000342 | 0.997063 | 0.000112 | 0.000229 |
| hidden_inverse_gain_shift | form_revised_history_law | mixed | accept_revision | 0.000323 | 0.997164 | 0.000093 | 0.000229 |
| hidden_inverse_gain_shift | form_revised_oracle_law | mixed | accept_revision | 0.000229 | 0.998006 | 0.000000 | 0.000229 |
| hidden_power_shift | base_history_law | trained_base | baseline | 0.054596 | 0.000000 | 0.054364 | 0.000232 |
| hidden_power_shift | gain_calibrated_history_law | gain_scaled_base | accept_revision | 0.022787 | 0.580086 | 0.022555 | 0.000232 |
| hidden_power_shift | form_revised_history_law | mixed | accept_revision | 0.008887 | 0.840020 | 0.008655 | 0.000232 |
| hidden_power_shift | form_revised_oracle_law | mixed | accept_revision | 0.000232 | 0.995656 | 0.000000 | 0.000232 |

## Interpretation

The result supports a narrow but important WPU loop:

```text
objectify -> propose local law -> stress -> observe residual -> revise law
```

For `hidden_inverse_gain_shift`, the residual is mostly a gain error. A scalar
calibration reduces MSE from `0.115978` to `0.000342`; form revision gives a
similar `0.000323`. Both revisions are accepted by `evaluate_law_revision`, with
relative improvement above `0.997`.

For `hidden_power_shift`, the residual is a law-form error. Gain calibration
helps but remains at `0.022787`; form revision improves to `0.008887`. The
oracle relation result `0.000232` shows that most remaining error is not the
candidate law family itself but relation-selection and noisy calibration. The
reported gap makes this explicit: form revision leaves relation-selection gap
`0.008655` and law-residual gap `0.000232`.

This is still a generated synthetic benchmark. It does not show discovery of an
unknown physical theory. It does show how WPU-style objectification can make
theory revision operational: residuals are attached to persistent objects,
relations, geometry, and local deltas, so the system can test whether failure is
caused by relation retrieval or by the propagation law.

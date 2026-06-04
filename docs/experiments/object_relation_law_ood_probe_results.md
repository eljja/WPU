# Object-Relation Local Law OOD Probe

This probe stress-tests the local-law result beyond the nominal held-out
`hidden_inverse` mechanism. It keeps the objectification pipeline fixed:
history-derived relation selection plus an interpretable inverse-distance local
law fitted on `contact_inverse` and `support_inverse`.

The evaluation mechanisms are:

- `hidden_inverse`: renamed object types with the same generated law family.
- `hidden_inverse_far`: renamed types and distances outside the training range.
- `hidden_inverse_gain_shift`: renamed types with the same law form but larger
  response gain.
- `hidden_power_shift`: renamed types with a changed response denominator.

Source CSV: `docs/experiments/object_relation_law_ood_probe.csv`

Reproduce:

```bash
python scripts/object_relation_law_probe.py --train-samples 768 --eval-samples 256 --seeds 31 37 41 43 47 --candidates 8 --history-steps 14 --train-steps 180 --threshold 0.5 --geometry-threshold 0.85 --eval-mechanisms hidden_inverse hidden_inverse_far hidden_inverse_gain_shift hidden_power_shift --out docs/experiments/object_relation_law_ood_probe.csv
```

## Result

The table reports five-seed means over 1,280 samples per mechanism.

| mechanism | policy | relation_precision | relation_recall | delta_mse | sign_accuracy |
|---|---|---:|---:|---:|---:|
| hidden_inverse | no_relation | 0.000000 | 0.000000 | 0.445909 | 0.487500 |
| hidden_inverse | history_relation_law | 0.988281 | 0.988281 | 0.000828 | 1.000000 |
| hidden_inverse | oracle_relation_law | 1.000000 | 1.000000 | 0.000260 | 1.000000 |
| hidden_inverse_far | no_relation | 0.000000 | 0.000000 | 0.008515 | 0.487500 |
| hidden_inverse_far | history_relation_law | 0.658594 | 0.658594 | 0.000834 | 0.998438 |
| hidden_inverse_far | oracle_relation_law | 1.000000 | 1.000000 | 0.000355 | 0.997656 |
| hidden_inverse_gain_shift | no_relation | 0.000000 | 0.000000 | 1.003059 | 0.487500 |
| hidden_inverse_gain_shift | history_relation_law | 0.996094 | 0.996094 | 0.112252 | 1.000000 |
| hidden_inverse_gain_shift | oracle_relation_law | 1.000000 | 1.000000 | 0.112155 | 1.000000 |
| hidden_power_shift | no_relation | 0.000000 | 0.000000 | 0.739034 | 0.487500 |
| hidden_power_shift | history_relation_law | 0.953125 | 0.953125 | 0.052089 | 1.000000 |
| hidden_power_shift | oracle_relation_law | 1.000000 | 1.000000 | 0.047200 | 1.000000 |

## Interpretation

The OOD probe strengthens the claim only in a bounded way. Objectified histories
still identify useful causal relations under renamed objects, distance shift,
gain shift, and denominator shift. The local law also remains much better than
no relation in all tested mechanisms.

The same result exposes two failure boundaries:

- Under `hidden_inverse_far`, relation precision/recall falls to `0.658594`.
  The delta MSE remains low because far-distance effects are small, but the
  relation selector is no longer robust.
- Under `hidden_inverse_gain_shift` and `hidden_power_shift`, oracle and history
  policies have similar residual MSE. The remaining error is therefore mostly
  law mis-specification, not relation retrieval.

This is the desired falsifiable direction for WPU: objectification supplies
persistent identities, histories, geometry, and candidate relations; propagation
rules must then be tested for when they extrapolate and when they need revision.
It is not evidence that WPU has discovered an unknown physical theory.

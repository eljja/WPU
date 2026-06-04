# Object-History Hidden Mechanism Probe

This probe tests the next objectification claim after relation repair:
relations should not only be hand-specified or type-gated. A WPU-style state
system should be able to infer useful relation candidates from object histories
and transfer them to a held-out mechanism family.

The training mechanisms are `contact_transfer` and `support_transfer`. The
evaluation mechanism is `hidden_field`, which uses different nominal object
types. The shared signal is not the type name; it is a lagged influence pattern
in object histories. Policies are compared on relation precision/recall and on
a downstream binary branch prediction derived from the selected relation set.

Source CSV: `docs/experiments/object_history_hidden_mechanism_probe.csv`

Reproduce:

```bash
python scripts/object_history_hidden_mechanism_probe.py --train-samples 512 --eval-samples 256 --seed 23 --candidates 8 --history-steps 12 --train-steps 160 --threshold 0.5 --out docs/experiments/object_history_hidden_mechanism_probe.csv
```

## Result

| policy | relation_precision | relation_recall | mean_selected_k | downstream_accuracy | downstream_loss |
|---|---:|---:|---:|---:|---:|
| no_relation | 0.000000 | 0.000000 | 0.000000 | 0.496094 | 0.693147 |
| geometry_only | 0.137369 | 0.460938 | 3.355469 | 0.765625 | 0.565862 |
| type_prior | 0.000000 | 0.000000 | 0.000000 | 0.496094 | 0.693147 |
| history_scorer | 0.988281 | 0.988281 | 1.000000 | 0.996094 | 0.275466 |
| oracle_relation | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.270057 |

## Interpretation

This is a toy hidden-mechanism result, not evidence of real physical-law
discovery. It does, however, test a stricter condition than the earlier relation
repair probe. The learned history scorer is trained on two visible mechanism
families and evaluated on a renamed held-out mechanism. Type prior fails because
the evaluation types are unseen. Geometry-only retrieval partially helps but
selects too many distractors. The history scorer nearly matches the oracle by
using lagged object influence, improving downstream accuracy from `0.496094` to
`0.996094` and lowering loss from `0.693147` to `0.275466`.

The result supports a bounded next-step claim: objectification can expose
history-derived relation variables that are more transferable than nominal type
names. The next required evidence is a simulator-backed version where the
history signal is produced by real dynamics rather than by a synthetic generator.

# Object-Relation Local Law Probe

This probe tests the step after hidden-mechanism relation discovery. It asks
whether objectified relation histories can support a simple transferable local
law, rather than only selecting a useful edge.

The training mechanisms are `contact_inverse` and `support_inverse`. The held-out
evaluation mechanism is `hidden_inverse`, which renames the causal and distractor
object types. The shared structure is a local inverse-distance response:

```text
target_delta ~= gain * current_impulse / (distance^2 + c)
```

The learned law is intentionally small and interpretable: a ridge-fitted linear
model over inverse-distance impulse features. The relation selector is a linear
history scorer trained from lagged object histories. This is a synthetic
diagnostic for objectified relation-to-law transfer, not evidence of real
physical-law discovery.

Source CSV: `docs/experiments/object_relation_law_probe.csv`

Reproduce:

```bash
python scripts/object_relation_law_probe.py --train-samples 768 --eval-samples 256 --seeds 31 37 41 43 47 --candidates 8 --history-steps 14 --train-steps 180 --threshold 0.5 --geometry-threshold 0.85 --out docs/experiments/object_relation_law_probe.csv
```

## Result

The table reports five-seed means over 1,280 held-out `hidden_inverse` samples.

| policy | relation_precision | relation_recall | mean_selected_k | delta_mse | sign_accuracy |
|---|---:|---:|---:|---:|---:|
| no_relation | 0.000000 | 0.000000 | 0.000000 | 0.445909 | 0.487500 |
| geometry_law | 0.130097 | 0.342969 | 2.637500 | 6.049853 | 0.979688 |
| type_prior_law | 0.000000 | 0.000000 | 0.000000 | 0.445909 | 0.487500 |
| history_relation_law | 0.988281 | 0.988281 | 1.000000 | 0.000828 | 1.000000 |
| oracle_relation_law | 1.000000 | 1.000000 | 1.000000 | 0.000260 | 1.000000 |

## Interpretation

The result supports a bounded claim: after objectification, relation histories
can provide enough structure to learn a local propagation law and transfer it
across renamed object types in a controlled synthetic setting.

The negative controls matter. Type prior fails because the held-out mechanism
uses unseen names. Geometry-only retrieval gets the sign mostly right because
all candidates share the current impulse direction, but false selected relations
accumulate large magnitude error, producing much worse MSE. The oracle shows
that most remaining error for `history_relation_law` is relation-selection error
rather than the local law fit.

This is not a discovery of physics. It is a falsifiable intermediate benchmark:
objectification should expose persistent identities, histories, geometry, and
candidate relations that make local law fitting possible. The next required
evidence is a simulator-backed version where the law is not hand-generated and
where learned relations are tested under out-of-distribution geometry, mass,
contact, friction, or delayed-force regimes.

# Objectification Relation Repair Probe

This probe tests a narrow but important objectification failure mode: objects
are present, but local relation edges are missing, so sparse propagation cannot
reach the causal working set. It also tests the next failure mode:
geometry-only repair can attach many nearby distractors. Finally, it separates
type labels from role-bearing object state by evaluating shifted scenarios in
which object type names are aliased while objectification role variables are
either preserved or removed.

`repair_objectification_relations` is evaluated as ungated geometry repair,
hand-written type-gated repair, and a small learned relation-candidate scorer
trained on generated object-pair features. The learned scorer receives geometry,
relation type, nominal type counts, and objectification role-pair features such
as dynamic-manipulator, dynamic-support, dynamic-boundary, and context
participation.

Source CSV: `docs/experiments/objectification_relation_repair_probe.csv`

Reproduce:

```bash
python scripts/objectification_relation_repair_probe.py --samples 64 --seed 17 --near-distance 0.25 --contact-distance 0.08 --background-objects 32 --near-distractors 8 --train-samples 128 --learned-steps 80 --learned-threshold 0.5 --out docs/experiments/objectification_relation_repair_probe.csv
```

## Result

| scenario | repair_policy | samples | background_objects | near_distractors | mean_before_frontier_recall | mean_after_frontier_recall | mean_added_relations | repair_precision | repair_recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| in_distribution | ungated | 64 | 32 | 8 | 0.250000 | 1.000000 | 97.515625 | 0.078994 | 1.000000 |
| in_distribution | type_gated | 64 | 32 | 8 | 0.250000 | 1.000000 | 7.703125 | 1.000000 | 1.000000 |
| in_distribution | learned_scorer | 64 | 32 | 8 | 0.250000 | 1.000000 | 7.703125 | 1.000000 | 1.000000 |
| dense_distractors | ungated | 64 | 128 | 24 | 0.250000 | 1.000000 | 584.000000 | 0.013244 | 1.000000 |
| dense_distractors | type_gated | 64 | 128 | 24 | 0.250000 | 1.000000 | 7.734375 | 1.000000 | 1.000000 |
| dense_distractors | learned_scorer | 64 | 128 | 24 | 0.250000 | 1.000000 | 7.734375 | 1.000000 | 1.000000 |
| aliased_types_with_roles | ungated | 64 | 32 | 8 | 0.250000 | 1.000000 | 97.515625 | 0.078994 | 1.000000 |
| aliased_types_with_roles | type_gated | 64 | 32 | 8 | 0.250000 | 0.250000 | 0.000000 | 0.000000 | 0.000000 |
| aliased_types_with_roles | learned_scorer | 64 | 32 | 8 | 0.250000 | 1.000000 | 7.703125 | 1.000000 | 1.000000 |
| aliased_types_without_roles | ungated | 64 | 32 | 8 | 0.250000 | 1.000000 | 97.515625 | 0.078994 | 1.000000 |
| aliased_types_without_roles | type_gated | 64 | 32 | 8 | 0.250000 | 0.250000 | 0.000000 | 0.000000 | 0.000000 |
| aliased_types_without_roles | learned_scorer | 64 | 32 | 8 | 0.250000 | 0.250000 | 0.000000 | 0.000000 | 0.000000 |

## Interpretation

The result is not a physics-discovery claim. It shows that when object identity
is already correct and relation extraction misses local edges, explicit
relation repair can restore sparse frontier connectivity without returning to
token processing. It also shows why objectification cannot be reduced to
geometry alone: ungated repair recovers recall but creates many spurious
relations, and the problem becomes worse as dense distractors increase.

The stricter result is the type-alias split. Hand-written type gating succeeds
only while nominal type labels match the gate. It fails when the same physical
roles are renamed. The learned scorer transfers across those aliases when
role/affordance state variables remain available, preserving recall and
precision at 1.000000. When both type labels and role variables are removed, it
fails. This is the desired falsifiable boundary for WPU objectification:
performance depends on persistent identity plus relation-bearing state
variables, not on object names alone.

The next required experiment is downstream: measure prediction loss with and
without repaired edges, then test whether relation candidates learned from
object histories improve held-out physical regimes where the generator does not
explicitly supply the relation family.

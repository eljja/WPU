# Objectification Relation Repair Probe

This probe tests a narrow but important objectification failure mode: objects
are present, but local relation edges are missing, so sparse propagation cannot
reach the causal working set. It also tests the next failure mode:
geometry-only repair can attach many nearby distractors. Finally, it separates
type labels from role-bearing object state by evaluating shifted scenarios in
which object type names are aliased while objectification role variables are
either preserved or removed.

`repair_objectification_relations` is evaluated against a no-repair baseline,
ungated geometry repair, hand-written type-gated repair, and a small learned
relation-candidate scorer trained on generated object-pair features. The
learned scorer receives geometry, relation type, nominal type counts, and
objectification role-pair features such as dynamic-manipulator,
dynamic-support, dynamic-boundary, and context participation. The probe also
reports a simple downstream branch-prediction accuracy/loss computed from the
post-repair sparse frontier. This branch probe is not a learned physics model;
it is a diagnostic for whether repaired relations improve a local causal
prediction instead of merely increasing frontier recall.

Source CSV: `docs/experiments/objectification_relation_repair_probe.csv`

Reproduce:

```bash
python scripts/objectification_relation_repair_probe.py --samples 64 --seed 17 --near-distance 0.25 --contact-distance 0.08 --background-objects 32 --near-distractors 8 --train-samples 128 --learned-steps 80 --learned-threshold 0.5 --out docs/experiments/objectification_relation_repair_probe.csv
```

## Result

| scenario | repair_policy | frontier_recall | repair_precision | repair_recall | downstream_accuracy | downstream_loss |
|---|---|---:|---:|---:|---:|---:|
| in_distribution | no_repair | 0.250000 | 0.000000 | 0.000000 | 0.343750 | 1.319667 |
| in_distribution | ungated | 1.000000 | 0.078994 | 1.000000 | 0.343750 | 1.075615 |
| in_distribution | type_gated | 1.000000 | 1.000000 | 1.000000 | 0.671875 | 0.885275 |
| in_distribution | learned_scorer | 1.000000 | 1.000000 | 1.000000 | 0.671875 | 0.885275 |
| dense_distractors | no_repair | 0.250000 | 0.000000 | 0.000000 | 0.359375 | 1.297792 |
| dense_distractors | ungated | 1.000000 | 0.013244 | 1.000000 | 0.359375 | 1.712768 |
| dense_distractors | type_gated | 1.000000 | 1.000000 | 1.000000 | 0.656250 | 0.908302 |
| dense_distractors | learned_scorer | 1.000000 | 1.000000 | 1.000000 | 0.656250 | 0.908302 |
| aliased_types_with_roles | no_repair | 0.250000 | 0.000000 | 0.000000 | 0.343750 | 1.319667 |
| aliased_types_with_roles | ungated | 1.000000 | 0.078994 | 1.000000 | 0.343750 | 1.075615 |
| aliased_types_with_roles | type_gated | 0.250000 | 0.000000 | 0.000000 | 0.343750 | 1.319667 |
| aliased_types_with_roles | learned_scorer | 1.000000 | 1.000000 | 1.000000 | 0.671875 | 0.885275 |
| aliased_types_without_roles | no_repair | 0.250000 | 0.000000 | 0.000000 | 0.343750 | 1.319667 |
| aliased_types_without_roles | ungated | 1.000000 | 0.078994 | 1.000000 | 0.343750 | 1.075615 |
| aliased_types_without_roles | type_gated | 0.250000 | 0.000000 | 0.000000 | 0.343750 | 1.319667 |
| aliased_types_without_roles | learned_scorer | 0.250000 | 0.000000 | 0.000000 | 0.343750 | 1.319667 |

## Interpretation

The result is not a physics-discovery claim. It shows that when object identity
is already correct and relation extraction misses local edges, explicit
relation repair can restore sparse frontier connectivity without returning to
token processing. It also shows why objectification cannot be reduced to
geometry alone: ungated repair recovers recall but creates many spurious
relations, and the problem becomes worse as dense distractors increase. The
downstream diagnostic makes this concrete: in the dense-distractor scenario,
ungated repair reaches frontier recall 1.000000 but worsens branch loss to
1.712768, below the no-repair loss of 1.297792. Frontier recall without relation
precision is not enough.

The stricter result is the type-alias split. Hand-written type gating succeeds
only while nominal type labels match the gate. It fails when the same physical
roles are renamed. The learned scorer transfers across those aliases when
role/affordance state variables remain available, preserving recall and
precision at 1.000000. When both type labels and role variables are removed, it
fails. This is the desired falsifiable boundary for WPU objectification:
performance depends on persistent identity plus relation-bearing state
variables, not on object names alone.

The downstream result closes the first repair-to-prediction loop for this toy
probe: role-aware learned repair improves aliased-type branch accuracy from
0.343750 to 0.671875 and lowers loss from 1.319667 to 0.885275. The next
required experiment is stronger: learn relation candidates from object
histories, then test whether they improve held-out physical regimes where the
generator does not explicitly supply the relation family.

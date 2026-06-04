# Objectification Relation Repair Probe

This probe tests a narrow failure mode: objects are present, but local relation
edges are missing, so sparse propagation cannot reach the causal working set.
It also tests the next failure mode: geometry-only repair can attach many
nearby distractors. `repair_objectification_relations` is therefore evaluated
with and without a type-aware repair gate.

Source CSV: `docs/experiments/objectification_relation_repair_probe.csv`

Reproduce:

```bash
python scripts/objectification_relation_repair_probe.py --samples 64 --seed 17 --near-distance 0.25 --contact-distance 0.08 --background-objects 32 --near-distractors 8 --out docs/experiments/objectification_relation_repair_probe.csv
```

## Result

| repair_policy | samples | background_objects | near_distractors | mean_before_frontier_recall | mean_after_frontier_recall | mean_added_relations | repair_precision | repair_recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ungated | 64 | 32 | 8 | 0.250000 | 1.000000 | 97.515625 | 0.078994 | 1.000000 |
| type_gated | 64 | 32 | 8 | 0.250000 | 1.000000 | 7.703125 | 1.000000 | 1.000000 |

## Interpretation

The result is not a physics-discovery claim. It shows that when object identity
is already correct and relation extraction misses local edges, explicit
relation repair can restore sparse frontier connectivity without returning to
token processing. It also shows why object type is part of objectification:
geometry-only repair recovers recall but creates many spurious relations, while
type-gated repair preserves the same recall with much higher precision.

The next required experiment is stricter: introduce ambiguous distractors,
measure downstream prediction loss with and without repaired edges, and replace
the hand-written type gate with learned relation-candidate scoring.

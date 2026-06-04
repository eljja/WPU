# Objectification Relation Repair Probe

This probe tests a narrow failure mode: objects are present, but local relation
edges are missing, so sparse propagation cannot reach the causal working set.
`repair_objectification_relations` adds conservative geometry-derived `near`
edges before propagation.

Source CSV: `docs/experiments/objectification_relation_repair_probe.csv`

Reproduce:

```bash
python scripts/objectification_relation_repair_probe.py --samples 64 --seed 17 --near-distance 0.25 --contact-distance 0.08 --background-objects 32 --out docs/experiments/objectification_relation_repair_probe.csv
```

## Result

| samples | background_objects | mean_before_frontier_recall | mean_after_frontier_recall | mean_added_relations | repair_precision | repair_recall |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 32 | 0.250000 | 1.000000 | 7.640625 | 1.000000 | 1.000000 |

## Interpretation

The result is not a physics-discovery claim. It shows that when object identity
is already correct and relation extraction misses local edges, explicit
relation repair can restore sparse frontier connectivity without returning to
token processing.

The next required experiment is stricter: introduce ambiguous distractors,
measure repair precision/recall against simulator ground truth, and report
downstream prediction loss with and without repaired edges.

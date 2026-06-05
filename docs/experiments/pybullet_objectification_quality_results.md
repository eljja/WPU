# PyBullet Objectification Quality Benchmark

This benchmark separates objectification quality from downstream model
accuracy. It compares corrupted PyBullet `WorldState` inputs against the clean
simulator-derived state and records identity, semantic consistency, relation
precision/recall, event-frontier recall, selected `K`, and the existing
`ObjectificationReport`.

Source CSV:

- `docs/experiments/pybullet_objectification_quality.csv`

## Protocol

- Simulator: PyBullet `DIRECT` cup scene.
- Samples: `12` per seed/background setting.
- Seeds: `11, 13`.
- Background objects: `32, 128, 512`.
- Corruptions: `clean`, `drop_relations_heavy`, `drop_objects_light`,
  `position_noise`, `low_confidence`, `identity_swap`, `combined`.
- Indexed frontier: event target plus relation frontier, `max_nodes=12`,
  `max_depth=1`.

## Representative Summary at Background N=128

| corruption | contract score | identity recall | semantic consistency | relation recall | frontier recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 0.939715 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| combined | 0.837442 | 0.844246 | 0.814672 | 0.834180 | 0.716667 |
| drop_objects_light | 0.939794 | 0.796720 | 0.796720 | 0.795459 | 0.866667 |
| drop_relations_heavy | 0.938665 | 1.000000 | 1.000000 | 0.979556 | 0.585417 |
| identity_swap | 0.939715 | 1.000000 | 0.984895 | 1.000000 | 1.000000 |
| low_confidence | 0.786843 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| position_noise | 0.939715 | 1.000000 | 0.675541 | 1.000000 | 1.000000 |

## Interpretation

The result confirms that objectification must be evaluated as a multi-part
contract, not a single scalar.

`low_confidence` is correctly reflected in the existing contract score, which
drops from `0.939715` to `0.786843` at background `N=128`.

`drop_relations_heavy` exposes the opposite failure mode. The contract score is
almost unchanged (`0.938665`), and global relation recall is still high
(`0.979556`) because most background relations remain intact. But event-frontier
recall falls to `0.585417`, which is the metric that matters for sparse WPU
propagation.

`position_noise` also shows why semantic consistency should be separate from
syntactic identity validity. Object IDs and relations remain valid, so the
contract score stays high, but semantic consistency drops to `0.675541`.

`identity_swap` is milder in this PyBullet scene because only a small number of
role-bearing non-protected objects can be swapped. Even so, it is detected by
semantic consistency rather than by relation validity or identity coverage.

## WPU Implication

WPU does not merely need "objects." It needs objectified state with at least
four measurable qualities:

- stable identity coverage;
- semantic identity consistency over role, type, and geometry;
- relation precision/recall;
- event-frontier completeness for the causal working set.

The current `ObjectificationReport` is useful but incomplete. It should be
extended or paired with task-specific quality metrics before sparse WPU claims
are made under perception-like corruption.

## Next Steps

- Add frontier completeness and semantic identity consistency to the public
  objectification report.
- Evaluate downstream branch loss as a function of each objectification metric.
- Add simulator relation ground truth when available, not only clean-state
  relation comparison.
- Stress identity swaps in richer scenes with multiple role-bearing objects.

# WPU V2 Proximity Working-Set Retrieval Results

Source CSV: `docs/experiments/wpu_v2_staged_k_expansion_proximity_initial4_5seed.csv`

## Question

The previous deployed K-expansion experiment showed a narrow positive regime:
selective sparse expansion helped when the initial working set was
under-complete, but universal expansion and expanded dense recompute both hurt.

This left a more basic question:

```text
Is the bottleneck the expansion decision, or the ordering of objects admitted
into the working set?
```

The indexed retrieval path expands the event target's relation frontier in
insertion order. In the pairwise obstacle task, that means an initial budget of
four usually selects `cup`, `table`, `hand`, and `edge`, while the nearest
causal obstacles may be excluded. The new proximity path keeps the same
state-native sparse interface but ranks the relation frontier by state geometry
before tensorization.

## Implementation

Added:

- `collate_proximity_working_set_samples`
- `--selection-mode {indexed,proximity}` for staged regret and K-expansion
  experiments

The proximity selector:

1. Starts from the event target object.
2. Enumerates the bounded relation frontier.
3. Ranks frontier objects by distance to the event target plus small
   type-specific biases.
4. Projects only the selected objects and relations into `StateGraphBatch`.

This is not a token fallback. The model still operates on explicit object and
relation state, and retrieval still happens before tensorization.

## Setup

- N = 2048
- K = 8, 16, 32
- Seeds = 11, 13, 17, 19, 23
- Initial working set = 4
- Expanded working set = 32
- Interaction mode = pairwise
- Hidden dim = 128
- Propagation steps = 40
- Regret steps = 80
- Compute cost = 0.05
- Expansion cost = 0.02

Output:

- `docs/experiments/wpu_v2_staged_k_expansion_proximity_initial4_5seed.csv`

## Indexed vs Proximity Summary

| selection | policy | loss | loss delta | total compute | accuracy |
| --- | --- | --- | --- | --- | --- |
| indexed | initial calibrated regret | 0.968 | -0.032 | 0.215 | 0.493 |
| indexed | physical sparse expansion | 0.964 | -0.035 | 0.251 | 0.500 |
| proximity | initial calibrated regret | 0.965 | -0.025 | 0.256 | 0.498 |
| proximity | physical sparse expansion | 0.967 | -0.024 | 0.265 | 0.496 |
| proximity | structured sparse expansion | 0.966 | -0.025 | 0.278 | 0.497 |

## K-Specific Results

| selection | K | policy | loss | loss delta | total compute | accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| indexed | 8 | initial calibrated regret | 0.965 | -0.033 | 0.209 | 0.493 |
| indexed | 8 | physical sparse expansion | 0.966 | -0.033 | 0.231 | 0.493 |
| proximity | 8 | initial calibrated regret | 0.963 | -0.027 | 0.258 | 0.493 |
| proximity | 8 | physical sparse expansion | 0.965 | -0.025 | 0.264 | 0.496 |
| indexed | 16 | initial calibrated regret | 0.960 | -0.023 | 0.204 | 0.487 |
| indexed | 16 | physical sparse expansion | 0.947 | -0.036 | 0.269 | 0.504 |
| proximity | 16 | initial calibrated regret | 0.952 | -0.019 | 0.173 | 0.493 |
| proximity | 16 | physical sparse expansion | 0.952 | -0.019 | 0.187 | 0.493 |
| indexed | 32 | initial calibrated regret | 0.978 | -0.040 | 0.231 | 0.498 |
| indexed | 32 | physical sparse expansion | 0.980 | -0.037 | 0.253 | 0.502 |
| proximity | 32 | initial calibrated regret | 0.981 | -0.030 | 0.336 | 0.507 |
| proximity | 32 | physical sparse expansion | 0.983 | -0.028 | 0.344 | 0.500 |

## Interpretation

The result is mixed and useful.

Proximity retrieval improves the initial K=4 policy at K=16
(`0.960 -> 0.952` loss) because it admits nearby causal obstacles earlier than
the insertion-ordered relation frontier. This supports the WPU thesis that
state-native retrieval should use object identity, relation structure, and
geometry before neural tensorization.

However, proximity does not improve the deployed expansion gate. Once the
initial working set is already geometry-aware, the simple verifier-triggered
expansion rule has less remaining headroom. At K=32, proximity slightly worsens
loss while increasing compute, which means the current geometric heuristic is
not enough for dense local causal sets.

## Updated Claim

The correct claim is narrower than "increase K when uncertain":

```text
WPU benefits from large N when retrieval admits the right causal working set
early. Expansion helps only when the initial retrieved set is under-complete
and the expansion operator adds the missing causal objects without flooding the
propagation core.
```

This pushes v2 toward a stronger mechanism:

- relation-typed sparse propagation instead of relation-order BFS,
- proximity/spatial indexing as part of the causal index,
- learned state-native retrieval ranking,
- verifier-triggered expansion only after the initial retriever exposes
  uncertainty or missing-interaction evidence,
- sparse expansion by physically relevant frontier, not universal K growth.

## Negative Boundary

This experiment also rejects a tempting simplification:

```text
K expansion alone is not the solution.
```

If the selected objects are poorly ordered, expansion may help narrowly. If the
initial selection is better, expansion rules must become more selective. If K
is genuinely large and interaction-dense, sparse propagation itself needs a
better local operator; simply adding more objects or running expanded dense
attention is not enough.

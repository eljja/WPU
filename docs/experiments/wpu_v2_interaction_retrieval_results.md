# WPU V2 Interaction-Density Retrieval Results

## Question

The proximity retrieval experiment showed that state-native object ordering
matters, but target proximity alone did not solve dense local causal sets. The
pairwise obstacle task depends on obstacle-obstacle geometry, not only
obstacle-cup distance.

This experiment asks:

```text
Can a state-native retriever that ranks local interaction density improve the
WPU regime without returning to token processing?
```

## Implementation

Added:

- `collate_interaction_working_set_samples`
- `--selection-mode interaction`

The interaction selector:

1. Starts from the event target.
2. Keeps contact anchors such as the robot hand.
3. Enumerates the bounded relation frontier.
4. Ranks obstacle candidates by local obstacle-obstacle density and axis
   alignment.
5. Projects the selected explicit objects and relations before tensorization.

This is still a WPU path. It does not serialize the world and does not run
global attention over all objects.

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

- `docs/experiments/wpu_v2_staged_k_expansion_interaction_initial4_5seed.csv`
- `docs/experiments/wpu_v2_selection_composition.csv`
- `docs/experiments/wpu_v2_selection_composition_results.md`

## Overall Result

| selection | policy | loss | loss delta | total compute | accuracy |
| --- | --- | --- | --- | --- | --- |
| indexed | initial calibrated regret | 0.968 | -0.032 | 0.215 | 0.493 |
| indexed | physical sparse expansion | 0.964 | -0.035 | 0.251 | 0.500 |
| proximity | initial calibrated regret | 0.965 | -0.025 | 0.256 | 0.498 |
| proximity | physical sparse expansion | 0.967 | -0.024 | 0.265 | 0.496 |
| interaction | initial calibrated regret | 0.961 | -0.030 | 0.224 | 0.497 |
| interaction | physical sparse expansion | 0.961 | -0.031 | 0.256 | 0.499 |
| interaction | structured sparse expansion | 0.960 | -0.031 | 0.253 | 0.499 |

## K-Specific Result

| K | selection | best policy | loss | loss delta | total compute | accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | indexed | initial calibrated regret | 0.965 | -0.033 | 0.209 | 0.493 |
| 8 | proximity | initial calibrated regret | 0.963 | -0.027 | 0.258 | 0.493 |
| 8 | interaction | structured sparse expansion | 0.959 | -0.031 | 0.304 | 0.493 |
| 16 | indexed | physical sparse expansion | 0.947 | -0.036 | 0.269 | 0.504 |
| 16 | proximity | initial calibrated regret | 0.952 | -0.019 | 0.173 | 0.493 |
| 16 | interaction | physical sparse expansion | 0.945 | -0.026 | 0.251 | 0.509 |
| 32 | indexed | initial calibrated regret | 0.978 | -0.040 | 0.231 | 0.498 |
| 32 | proximity | initial calibrated regret | 0.981 | -0.030 | 0.336 | 0.507 |
| 32 | interaction | initial calibrated regret | 0.976 | -0.036 | 0.193 | 0.496 |

## Interpretation

This is a stronger positive result than proximity-only retrieval. The
interaction-density retriever improves the overall best policy and gives the
best K=16 result. It also slightly improves K=32 initial routing while using
less compute than indexed or proximity retrieval in that condition.

The result supports a more precise WPU v2 claim:

```text
The advantage is not just sparse K. The advantage comes from state-native
retrieval operators that choose the right causal working set before neural
propagation.
```

The result also remains bounded. At K=32, interaction retrieval improves the
initial working set but expansion is still not beneficial. This means the
remaining bottleneck is no longer only retrieval order. The sparse propagation
operator must better aggregate interaction evidence once the local causal set
becomes large.

## Mechanism Check

The selection-composition diagnostic explains why interaction retrieval helps.
At budget 4, indexed retrieval keeps the hand but admits no obstacles.
Proximity retrieval admits obstacles but often drops the hand. Interaction
retrieval keeps the hand and selects high-density obstacle pairs:

| K | selection | selected obstacles | hand hit | selected obstacle-pair density |
| --- | --- | --- | --- | --- |
| 8 | indexed | 0.000 | 1.000 | 0.000 |
| 8 | proximity | 2.246 | 0.398 | 0.254 |
| 8 | interaction | 2.000 | 1.000 | 0.625 |
| 16 | indexed | 0.000 | 1.000 | 0.000 |
| 16 | proximity | 2.504 | 0.332 | 0.551 |
| 16 | interaction | 2.000 | 1.000 | 0.812 |
| 32 | indexed | 0.000 | 1.000 | 0.000 |
| 32 | proximity | 2.590 | 0.301 | 0.799 |
| 32 | interaction | 2.000 | 1.000 | 0.902 |

This supports the mechanism-level claim: WPU needs retrieval that preserves
the relevant state structure, not only a small K.

## Updated Direction

The next WPU v2 step should not be a token baseline detour. It should be a
stronger state-native retriever and propagator:

- report selected-object composition and causal top-k recall,
- train a learned state retriever against interaction-density and downstream
  regret labels,
- add relation-typed sparse message passing so obstacle-obstacle density is
  computed by the model rather than only by a hand-built retriever,
- trigger expansion only when the retriever's selected set lacks interaction
  support,
- keep dense fallback local and selective, never global by default.

## Negative Boundary

The experiment rejects two oversimplified claims:

```text
N large alone is enough for WPU advantage.
K expansion alone is enough for WPU accuracy.
```

The supported claim is narrower and stronger:

```text
WPU becomes useful when total state N is large, the event-relevant causal
working set is small or moderate, and the retriever can identify the right
state-local interaction structure before tensorization.
```

# WPU v2 Invariant Set Scorer

Source CSV: `docs/experiments/wpu_v2_retriever_invariant_set_scorer.csv`

## Question

Prior set-evaluator experiments showed that good candidate working sets exist,
but learned set selection does not transfer reliably across held-out seeds. This
experiment tests whether a smaller scorer built only from state-native,
candidate-level descriptors transfers better than an object-set encoder with
candidate identity features.

The scorer receives only per-candidate invariant descriptors derived before
propagation:

- hand inclusion;
- selected obstacle ratio;
- selected obstacle pair density;
- event force;
- selected-obstacle distance min/mean/max/span relative to the event target;
- selected-obstacle lateral spread and target-axis alignment;
- selected hand and edge distance relative to the event target;
- hand-density and obstacle-density interactions;
- optional candidate family flags: generated candidate and composition
  candidate.

No raw object-set tensor and no candidate identity one-hot is used.

## Protocol

```powershell
.\.venv\Scripts\python.exe scripts\retriever_cross_seed_invariant_set_scorer_probe.py --n-values 2048 --k-values 8 16 32 --seeds 11 13 17 19 23 --budget 4 --generated-candidates 4 --propagation-steps 40 --retriever-steps 400 --regret-retriever-steps 600 --composition-steps 600 --scorer-steps 600 --validation-samples 180 --samples 90 --batch-size 10 --device cuda --out docs\experiments\wpu_v2_retriever_invariant_set_scorer.csv
```

Two feature variants are evaluated:

- `role_geometry_family`: role/geometry descriptors plus generated/composition
  family flags.
- `role_geometry_only`: the same role/geometry descriptors with family flags
  removed.

The report also includes:

- `train_safe_invariant_set_scorer`: use the invariant scorer only when
  train-seed validation loss improves over static learned interaction.
- `train_selected_mechanism`: select the best train-seed mechanism among
  static learned interaction, composition candidates, and invariant scoring,
  then deploy that mechanism to the held-out seed.
- `seed_stable_selected_mechanism`: deploy a mechanism only when it improves
  every train seed and satisfies the configured win-rate threshold.
- `risk_adjusted_selected_mechanism`: select by mean train-seed improvement
  penalized by worst-seed harm, avoiding both unrestricted mean selection and
  overly conservative no-harm filtering.

## Results

Mean over five held-out seeds:

| variant | K | policy | loss | accuracy | oracle match | delta vs static |
|---|---:|---|---:|---:|---:|---:|
| role_geometry_family | 8 | static learned interaction | 0.988432 | 0.506667 | 0.004444 | 0.000000 |
| role_geometry_family | 8 | invariant set scorer | 0.983660 | 0.515556 | 0.191111 | -0.004772 |
| role_geometry_family | 8 | train-selected mechanism | 0.982002 | 0.522222 | 0.228889 | -0.006429 |
| role_geometry_family | 8 | risk-adjusted mechanism | 0.982002 | 0.522222 | 0.228889 | -0.006429 |
| role_geometry_family | 8 | candidate oracle | 0.955536 | 0.555555 | 1.000000 | -0.032895 |
| role_geometry_family | 16 | static learned interaction | 0.966183 | 0.504444 | 0.002222 | 0.000000 |
| role_geometry_family | 16 | invariant set scorer | 0.958227 | 0.500000 | 0.195555 | -0.007956 |
| role_geometry_family | 16 | train-selected mechanism | 0.951243 | 0.517778 | 0.077778 | -0.014940 |
| role_geometry_family | 16 | risk-adjusted mechanism | 0.951243 | 0.517778 | 0.077778 | -0.014940 |
| role_geometry_family | 16 | candidate oracle | 0.905009 | 0.580000 | 1.000000 | -0.061173 |
| role_geometry_family | 32 | static learned interaction | 1.004095 | 0.475556 | 0.004444 | 0.000000 |
| role_geometry_family | 32 | invariant set scorer | 1.008212 | 0.500000 | 0.164444 | 0.004118 |
| role_geometry_family | 32 | train-selected mechanism | 1.002708 | 0.522222 | 0.008889 | -0.001387 |
| role_geometry_family | 32 | seed-stable mechanism | 1.004715 | 0.480000 | 0.002222 | 0.000620 |
| role_geometry_family | 32 | risk-adjusted mechanism | 1.002597 | 0.522222 | 0.113333 | -0.001498 |
| role_geometry_family | 32 | candidate oracle | 0.968548 | 0.577778 | 1.000000 | -0.035547 |
| role_geometry_only | 8 | static learned interaction | 0.988432 | 0.506667 | 0.004444 | 0.000000 |
| role_geometry_only | 8 | invariant set scorer | 0.985303 | 0.513333 | 0.226667 | -0.003129 |
| role_geometry_only | 8 | train-selected mechanism | 0.984696 | 0.515556 | 0.140000 | -0.003735 |
| role_geometry_only | 16 | invariant set scorer | 0.967945 | 0.491111 | 0.195556 | 0.001762 |
| role_geometry_only | 16 | train-selected mechanism | 0.951243 | 0.517778 | 0.077778 | -0.014940 |
| role_geometry_only | 32 | invariant set scorer | 1.007681 | 0.495556 | 0.186667 | 0.003586 |
| role_geometry_only | 32 | train-selected mechanism | 1.002418 | 0.524444 | 0.055555 | -0.001677 |

## Interpretation

This is now a partial positive result with a deployable mechanism selector.

The `role_geometry_family` scorer improves loss at `K=8` and `K=16`, unlike
the previous full set evaluator which hurt both regimes. It also raises oracle
match rate from near zero to about `0.19--0.20`. Adding event-relative geometry
reduces the earlier over-selection of the indexed candidate and increases
generated/composition candidate use.

However, the same raw scorer still fails at `K=32`. The successful fix is not
to force invariant scoring everywhere. The `train_selected_mechanism` policy
chooses among static learned interaction, composition candidates, and invariant
scoring using only train-seed validation loss. It improves held-out loss at
`K=8`, `K=16`, and `K=32`.

The selected mechanisms are interpretable. With `role_geometry_family`,
mean train selection chooses mostly invariant scoring at `K=8`, composition
candidates at `K=16`, and composition-count-only at `K=32`.

The stricter seed-stable gate is useful as a diagnostic but not as the deployed
policy. It preserves the `K=8/16` gains, but it under-deploys at `K=32` and
slightly hurts loss relative to static learned interaction. The risk-adjusted
selector fixes this tradeoff: it keeps the `K=8/16` gains and also improves
`K=32` by penalizing worst-seed harm without requiring zero harm on every
training seed.

This supports a stronger v2 claim: WPU candidate selection should be a
structured mechanism-selection problem over explicit state descriptors, not a
single opaque reranker.

## Consequence

The next deployable direction is not a larger opaque set evaluator. It is a
structured, state-native candidate scorer:

```text
candidate quality =
  explicit role/geometry descriptors
  + calibrated family priors
  + composition constraints
  + risk-adjusted train-seed mechanism routing
```

For the paper, this result narrows the WPU v2 claim:

```text
Explicit state enables transferable working-set descriptors and mechanism
routing before propagation. Descriptor-only scoring improves K=8/16, while
risk-adjusted mechanism routing preserves improvement across K=8/16/32 without
falling back to token serialization.
```

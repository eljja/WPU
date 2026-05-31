# WPU v2 Invariant Set Scorer

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

The report also includes a `train_safe_invariant_set_scorer` policy that uses
the invariant scorer only when train-seed validation loss improves over static
learned interaction.

## Results

Mean over five held-out seeds:

| variant | K | policy | loss | accuracy | oracle match | delta vs static |
|---|---:|---|---:|---:|---:|---:|
| role_geometry_family | 8 | static learned interaction | 0.988432 | 0.506667 | 0.004444 | 0.000000 |
| role_geometry_family | 8 | invariant set scorer | 0.983969 | 0.522222 | 0.213333 | -0.004462 |
| role_geometry_family | 8 | train-safe invariant scorer | 0.983969 | 0.522222 | 0.213333 | -0.004462 |
| role_geometry_family | 8 | candidate oracle | 0.955278 | 0.555555 | 1.000000 | -0.033154 |
| role_geometry_family | 16 | static learned interaction | 0.966183 | 0.504444 | 0.002222 | 0.000000 |
| role_geometry_family | 16 | invariant set scorer | 0.962028 | 0.506667 | 0.222222 | -0.004154 |
| role_geometry_family | 16 | train-safe invariant scorer | 0.966864 | 0.508889 | 0.060000 | 0.000681 |
| role_geometry_family | 16 | candidate oracle | 0.904985 | 0.580000 | 1.000000 | -0.061198 |
| role_geometry_family | 32 | static learned interaction | 1.004095 | 0.475556 | 0.004444 | 0.000000 |
| role_geometry_family | 32 | invariant set scorer | 1.010686 | 0.497778 | 0.180000 | 0.006592 |
| role_geometry_family | 32 | train-safe invariant scorer | 1.004095 | 0.475556 | 0.004444 | 0.000000 |
| role_geometry_family | 32 | candidate oracle | 0.968505 | 0.580000 | 1.000000 | -0.035589 |
| role_geometry_only | 8 | static learned interaction | 0.988432 | 0.506667 | 0.004444 | 0.000000 |
| role_geometry_only | 8 | invariant set scorer | 0.987314 | 0.508889 | 0.253333 | -0.001117 |
| role_geometry_only | 16 | invariant set scorer | 0.970004 | 0.491111 | 0.220000 | 0.003821 |
| role_geometry_only | 32 | invariant set scorer | 1.011841 | 0.497778 | 0.186667 | 0.007746 |

## Interpretation

This is a partial positive result and a useful constraint.

The `role_geometry_family` scorer improves loss at `K=8` and `K=16`, unlike
the previous full set evaluator which hurt both regimes. It also raises oracle
match rate from near zero to about `0.21--0.22`. This supports the hypothesis
that transferable candidate scoring should depend on explicit state descriptors
rather than raw candidate identity or score margin confidence.

However, the same scorer fails at `K=32`. The failure is not random: the
train-seed loss also worsens at `K=32`, so a train-safe gate can reject it and
avoid harm. The gate also rejects most `K=16` cases even though the raw scorer
improves held-out loss, showing that train validation evidence is conservative
but not perfectly aligned with held-out transfer.

Removing family flags makes the scorer weaker. This means candidate family
information is not merely harmful identity leakage; it carries useful structural
signal when combined with state descriptors.

## Consequence

The next deployable direction is not a larger opaque set evaluator. It is a
structured, state-native candidate scorer:

```text
candidate quality =
  explicit role/geometry descriptors
  + calibrated family priors
  + composition constraints
  + held-out-aware safety criteria
```

For the paper, this result narrows the WPU v2 claim:

```text
Explicit state enables transferable working-set descriptors, and those
descriptors improve small-to-medium causal working-set regimes. They do not yet
solve large-K candidate selection.
```


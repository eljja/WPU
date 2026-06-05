# PyBullet Local-Law Revision

This experiment moves the local-law revision idea from synthetic hidden
mechanisms to PyBullet-derived objectified state. It fits simple interpretable
candidate laws for cup `delta_x` from `WorldState` fields such as hand impulse,
edge distance, hand distance, catch action, and relation strengths.

Source CSV:

- `docs/experiments/pybullet_local_law_revision.csv`

## Protocol

- Simulator: PyBullet `DIRECT` rigid-body rollout.
- Target: cup position delta on the x axis.
- Training distribution: nominal PyBullet cup samples.
- Calibration: small mechanism-specific calibration set.
- Evaluation mechanisms: `nominal`, `high_force`, `edge_shift`, `catch_heavy`.
- Seeds: `11, 13`.
- Train/calibration/eval samples: `64/24/48`.
- Policies:
  - `base_law`: linear law fit on nominal samples.
  - `gain_calibrated_law`: scalar gain correction of the base law.
  - `form_revised_law`: candidate-form selection from the calibration set.
  - `oracle_form_law`: test-set upper bound for candidate-form quality, not deployable.

## Summary

| mechanism | policy | selected form | delta MSE | relative improvement | decision |
| --- | --- | --- | ---: | ---: | --- |
| nominal | base_law | base | 0.025154 | 0.000000 | baseline |
| nominal | gain_calibrated_law | gain_scaled_base | 0.026402 | 0.000000 | keep_base_or_collect_data |
| nominal | form_revised_law | edge_form | 0.035359 | 0.000000 | keep_base_or_collect_data |
| nominal | oracle_form_law | edge_form | 0.010712 | 0.574098 | accept_revision |
| high_force | base_law | base | 0.086944 | 0.000000 | baseline |
| high_force | gain_calibrated_law | gain_scaled_base | 0.077267 | 0.111118 | accept_revision |
| high_force | form_revised_law | catch_form | 0.037060 | 0.573785 | accept_revision |
| high_force | oracle_form_law | catch_form | 0.018673 | 0.785224 | accept_revision |
| edge_shift | base_law | base | 0.680045 | 0.000000 | baseline |
| edge_shift | gain_calibrated_law | gain_scaled_base | 0.641828 | 0.056191 | keep_base_or_collect_data |
| edge_shift | form_revised_law | catch_form | 0.563311 | 0.171645 | accept_revision |
| edge_shift | oracle_form_law | edge_form | 0.086721 | 0.872477 | accept_revision |
| catch_heavy | base_law | base | 0.012977 | 0.000000 | baseline |
| catch_heavy | gain_calibrated_law | gain_scaled_base | 0.012939 | 0.003018 | keep_base_or_collect_data |
| catch_heavy | form_revised_law | quadratic_form | 0.016676 | 0.000000 | keep_base_or_collect_data |
| catch_heavy | oracle_form_law | catch_form | 0.009083 | 0.300059 | accept_revision |

## Interpretation

The result supports a narrow but important claim: local-law revision over
objectified simulator state is operational, and OOD residuals can identify
conditions where a revised form improves prediction. `high_force` and
`edge_shift` are the positive regimes. In both, the deployable revised form
beats the nominal base law.

The result also exposes the next bottleneck. The oracle form is much better than
the deployed form under `edge_shift`, while `nominal` and `catch_heavy` show
that revision can overfit or fail to improve. Therefore the WPU claim should
not be "the model discovers physical law." The current defensible claim is:

```text
Objectified state makes local law hypotheses explicit, measurable, stressable,
and revisable under bounded candidate families.
```

## Issues Found

- Candidate-form selection from only 24 calibration samples is unstable.
- The best deployable form for `edge_shift` was not the same as the test-set
  oracle form, showing a mechanism-selection gap.
- This is still not unknown-law discovery. Candidate forms are hand-provided.
- The target is one scalar cup delta, not full rigid-body state evolution.

## Next Steps

- Add validation-split or risk-adjusted law-form selection instead of choosing
  the lowest calibration MSE directly.
- Add multi-output laws for position, velocity, and branch probability.
- Couple law residuals to the closed-loop verifier so unsafe deltas trigger
  law revision or local dense fallback.
- Replace hand-provided forms with generated candidate descriptors or symbolic
  regression over objectified variables.

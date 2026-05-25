# WPU V2 Structured Verifier Probe

This experiment tests the next scheduler direction after the failure of scalar
state-only regret routing.

## Question

Can WPU routing improve if physical state and sparse diagnostics are used as a
structured verifier rather than concatenated into a small MLP?

The probe separates three cases:

- `calibrated_regret_route`: route dense by the existing staged regret score.
- `structured_verifier_gate`: route dense only when the staged regret score
  asks for dense and a learned threshold rule verifies the condition.
- `structured_expansion_upper_bound`: if a learned verifier trigger fires,
  assume a future K-expansion/consistency step can choose the better local
  result, then charge an expansion cost.

The expansion policy is an upper bound, not a deployed model. It exists to test
whether verifier triggers identify samples where an actual K-expansion mechanism
would be valuable.

## Protocol

- Input: `docs/experiments/wpu_v2_staged_regret_context_samples.csv`.
- Evaluation: leave-one-seed-out over seeds `11, 13, 17, 19, 23`.
- World size: `N = 2048`.
- Causal working sets: `K in {8, 16, 32}`.
- Dense compute cost: `0.05`.
- Expansion costs: `0.00, 0.01, 0.02, 0.05`.
- Rule families: single or two-condition threshold conjunctions.
- Feature sets:
  - structured: physical state plus sparse diagnostics;
  - physical: physical state only.

Artifacts:

- `scripts/structured_verifier_probe.py`
- `docs/experiments/wpu_v2_structured_verifier_probe.csv`
- `docs/experiments/wpu_v2_structured_verifier_probe_summary.csv`

## Main Results

| policy | expansion cost | trigger rate | loss | delta vs sparse | oracle excess | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| calibrated regret route | 0.00 | 0.000 | 0.974 | -0.015 | 0.055 | 0.485 |
| structured verifier gate | 0.00 | 0.187 | 0.968 | -0.020 | 0.050 | 0.488 |
| physical verifier gate | 0.00 | 0.218 | 0.971 | -0.017 | 0.053 | 0.487 |
| structured expansion upper bound, budget 25% | 0.02 | 0.159 | 0.970 | -0.018 | 0.052 | 0.503 |
| physical expansion upper bound, budget 25% | 0.02 | 0.174 | 0.967 | -0.021 | 0.049 | 0.500 |
| structured expansion upper bound, budget 50% | 0.02 | 0.482 | 0.955 | -0.033 | 0.037 | 0.521 |
| physical expansion upper bound, budget 50% | 0.02 | 0.446 | 0.954 | -0.034 | 0.036 | 0.523 |

Unbudgeted expansion often selects near-full expansion, so it is not a realistic
deployed policy. The budgeted rows are the relevant evidence.

## Interpretation

The structured verifier gate is the first deployable post-regret mechanism that
improves the staged regret route under leave-one-seed-out evaluation. The gain
is modest, but it is directionally consistent:

```text
calibrated route loss: 0.9736
structured verifier gate loss: 0.9684
```

This supports the architectural claim that state evidence should act as a
verification layer around propagation, not as a replacement for propagation
hidden state.

The expansion upper-bound rows show a stronger but conditional result. If a
future K-expansion/consistency mechanism can recover the better local choice,
then a 50% trigger budget could reduce loss to about `0.954`. This does not
prove a deployed expansion engine yet. It proves that the remaining error mass
contains samples where selective verification-triggered expansion is worth
implementing.

## Consequence

The next concrete v2 implementation should be:

```text
sparse propagation
-> staged regret route estimate
-> structured verifier
-> if verified: local dense
-> if not verified and violation trigger fires: expand K / run consistency
```

The key research claim becomes narrower and stronger:

> WPU gains are most plausible when state memory allows selective verification
> and expansion around an event-local causal working set. The next advantage
> should come from verification-triggered K expansion, not from replacing
> propagation with scalar state features.

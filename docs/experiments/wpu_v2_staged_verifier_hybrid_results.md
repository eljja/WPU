# WPU V2 Deployed Staged Verifier Results

Source CSVs:

- `docs/experiments/wpu_v2_staged_verifier_hybrid_5seed.csv`
- `docs/experiments/wpu_v2_staged_verifier_hybrid_safe_5seed.csv`

This experiment moves the structured verifier idea from a post-hoc CSV probe
into the actual staged WPU training/evaluation pipeline.

## Question

Does a validation-calibrated structured verifier improve a deployed staged
regret router on held-out test samples?

This is stricter than `wpu_v2_structured_verifier_probe_results.md` because each
condition trains a model, selects the route threshold and verifier rule on its
own validation split, then evaluates on a separate test split.

## Protocol

- Model: `wpu-cws-indexed-regret-hybrid`.
- Dataset: pairwise CWS synthetic physics.
- `N = 2048`.
- `K in {8, 16, 32}`.
- Seeds: `11, 13, 17, 19, 23`.
- Training: 40 propagation steps, then 80 route-regret steps.
- Validation: 90 samples per condition.
- Test: 90 samples per condition.
- Dense compute cost: `0.05`.

Artifacts:

- `scripts/staged_verifier_hybrid.py`
- `docs/experiments/wpu_v2_staged_verifier_hybrid_5seed.csv`
- `docs/experiments/wpu_v2_staged_verifier_hybrid_safe_5seed.csv`

## Results

Unconstrained verifier:

| policy | loss | delta vs sparse | oracle excess | dense compute | trigger rate | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| calibrated regret route | 0.965 | -0.023 | 0.047 | 0.258 | 0.000 | 0.495 |
| structured verifier gate | 0.967 | -0.021 | 0.049 | 0.213 | 0.385 | 0.491 |
| physical verifier gate | 0.968 | -0.021 | 0.050 | 0.217 | 0.329 | 0.491 |

Conservative verifier with minimum validation gain `0.005`:

| policy | loss | delta vs sparse | oracle excess | dense compute | trigger rate | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| calibrated regret route | 0.965 | -0.023 | 0.047 | 0.258 | 0.000 | 0.495 |
| structured verifier gate | 0.966 | -0.022 | 0.048 | 0.222 | 0.090 | 0.491 |
| physical verifier gate | 0.966 | -0.023 | 0.047 | 0.236 | 0.042 | 0.492 |

Paired outcomes relative to calibrated regret routing:

- Unconstrained structured verifier: mean loss `+0.00164`, wins `1/15`.
- Unconstrained physical verifier: mean loss `+0.00237`, wins `0/15`.
- Conservative structured verifier: mean loss `+0.00106`, wins `0/15`.
- Conservative physical verifier: mean loss `+0.00013`, wins `0/15`.

## Interpretation

The post-hoc verifier gain does not transfer to the deployed per-condition
validation pipeline. This rejects a stronger claim that simple threshold
verifiers improve route quality.

The conservative physical verifier is still useful as a compute-saving safety
filter: it reduces dense execution from `0.258` to `0.236` while leaving loss
almost unchanged. That is not a new accuracy result, but it is a practical
systems result.

The next WPU v2 step should therefore not be another threshold gate. It should
implement real verification-triggered expansion:

```text
if verifier is confident dense is unnecessary:
    stay sparse
elif verifier detects unresolved interaction or uncertainty:
    expand K and recompute local state
else:
    use calibrated regret route
```

## Updated Claim

The evidence now supports a narrower statement:

> Structured verification is useful for controlling compute, but threshold-only
> verifier gates do not yet improve predictive loss. The remaining promising
> direction is verification-triggered K expansion, because the upper-bound probe
> shows available headroom but the deployed gate cannot realize it by only
> suppressing dense execution.

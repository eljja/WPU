# WPU V2 Deployed K-Expansion Results

Source CSVs:

- `docs/experiments/wpu_v2_staged_k_expansion_hybrid_5seed.csv`
- `docs/experiments/wpu_v2_staged_k_expansion_hybrid_initial4_5seed.csv`

This experiment implements the next step after verifier gates: when a verifier
fires, the system re-collates a larger event-local state and recomputes the
branch prediction.

## Question

Can verification-triggered K expansion turn the previous upper-bound result
into a deployed mechanism?

Two expansion paths are evaluated:

- sparse expansion: expand the indexed causal working set, then run sparse
  propagation;
- dense expansion: expand the indexed causal working set, then run local dense
  recompute.

## Protocol

- Model: `wpu-cws-indexed-regret-hybrid`.
- Dataset: pairwise CWS synthetic physics.
- `N = 2048`.
- `K in {8, 16, 32}`.
- Seeds: `11, 13, 17, 19, 23`.
- Training: expanded working set size `32`.
- Initial evaluation budgets: `4` and `8`.
- Expanded evaluation budget: `32`.
- Dense compute cost: `0.05`.
- Expansion cost: `0.02`.
- Maximum validation-selected expansion rate: `0.5`.

Artifacts:

- `scripts/staged_k_expansion_hybrid.py`
- `docs/experiments/wpu_v2_staged_k_expansion_hybrid_5seed.csv`
- `docs/experiments/wpu_v2_staged_k_expansion_hybrid_initial4_5seed.csv`

## Results

Initial budget `8`:

| policy | loss | delta vs sparse | expansion rate | total compute | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| initial calibrated regret | 0.961 | -0.028 | 0.000 | 0.179 | 0.491 |
| physical sparse expansion | 0.960 | -0.029 | 0.020 | 0.199 | 0.494 |
| structured sparse expansion | 0.961 | -0.028 | 0.024 | 0.202 | 0.493 |
| physical dense expansion | 0.962 | -0.028 | 0.020 | 0.199 | 0.492 |
| always expand sparse | 1.008 | 0.019 | 1.000 | 1.000 | 0.488 |
| always expand dense | 1.067 | 0.078 | 1.000 | 1.000 | 0.465 |

Initial budget `4`:

| policy | loss | delta vs sparse | expansion rate | total compute | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| initial calibrated regret | 0.968 | -0.032 | 0.000 | 0.215 | 0.493 |
| physical sparse expansion | 0.964 | -0.035 | 0.041 | 0.251 | 0.500 |
| structured sparse expansion | 0.966 | -0.034 | 0.050 | 0.259 | 0.499 |
| physical dense expansion | 0.968 | -0.032 | 0.041 | 0.251 | 0.497 |
| always expand sparse | 1.008 | 0.009 | 1.000 | 1.000 | 0.488 |
| always expand dense | 1.067 | 0.067 | 1.000 | 1.000 | 0.465 |

K-specific result for initial budget `4`:

| policy | K=8 loss | K=16 loss | K=32 loss |
| --- | ---: | ---: | ---: |
| initial calibrated regret | 0.965 | 0.960 | 0.978 |
| physical sparse expansion | 0.966 | 0.947 | 0.980 |
| structured sparse expansion | 0.966 | 0.951 | 0.980 |

## Interpretation

This is the first deployed K-expansion result with a positive regime, but it is
not a broad win.

The positive case is narrow:

- the initial working set must be under-complete (`4 -> 32`);
- expansion must be sparse, not dense;
- the benefit is concentrated at `K=16`;
- the verifier should expand only a small fraction of samples.

The negative cases are equally important:

- always expanding is strongly worse;
- dense expansion is worse than sparse expansion;
- when the initial budget is already `8`, expansion gives only a tiny gain.

## Updated Claim

The result supports a more precise WPU v2 direction:

> WPU should not use expansion as a default fallback. Expansion is useful when
> the initial causal working set is under-complete and a verifier can identify a
> small subset of samples where additional state is needed. The correct next
> mechanism is sparse K expansion followed by local propagation, not dense
> recompute over a larger subgraph.

This strengthens the architectural distinction: WPU advantage is tied to
selective state growth and propagation, not to simply making a local attention
block larger.

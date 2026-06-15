# N_bg=512 Mechanism Adapter Multi-Mechanism Training Results

This report follows the negative nominal-only mechanism-conditioned screens. It
tests whether WPU improves when the local propagation module is trained on
primitive mechanisms and then evaluated on both primitive and composed
mechanism shifts. The result narrows the claim: the useful regime is not
ungrounded zero-shot mechanism extrapolation, but object-wise local-law
adaptation over objectified state when primitive mechanisms are observed during
training.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization_n512_mechanism_conditioned_5seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_adapter_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_adapter_multitrain_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_adapter_multitrain_5seed.csv`

## What Changed

The previous `wpu-cws-indexed-mechanism-conditioned` model adds a single global
mechanism context vector to every selected object. The new
`wpu-cws-indexed-mechanism-adapter` applies an object-wise sparse adapter to each
selected object using:

```text
[selected object embedding, selected raw object features, route physics context]
```

This keeps dense compute at `0.000000`, but gives the transition function a
way to apply different mechanism-conditioned updates to different selected
objects.

## Nominal-Only Expansion Is Negative

The original positive 4-shift screen did not survive broader evaluation. On a
5-seed, 7-mechanism, N_bg=512 nominal-train screen, the global-context
mechanism-conditioned WPU reaches macro accuracy `0.433333`, while the best
non-WPU baseline reaches `0.476190`. Its win/tie/loss versus the best baseline
is `2/0/5`.

The object-wise adapter is also negative under nominal-only training. In the
3-seed screen, it reaches macro accuracy `0.380952` versus `0.476190` for
graph-transformer, with win/tie/loss `0/1/6`. This rules out the idea that
architecture alone solves held-out mechanism-law shifts without training signal.

## Multi-Mechanism Training Is Conditionally Positive

The multi-mechanism protocol trains on `nominal`, `high_force`, `edge_shift`,
`catch_heavy`, and `no_catch`, then evaluates on those mechanisms plus composed
`edge_high_force` and `edge_catch_heavy`.

| Model | Mean branch accuracy | Mean ECE | Mean Brier | Mean NLL | Mean dense compute |
|---|---:|---:|---:|---:|---:|
| `graph-transformer` | 0.458571 | 0.248788 | 0.612642 | 1.003195 | 1.000000 |
| `serialized-token` | 0.472857 | 0.243629 | 0.648790 | 1.068664 | 1.000000 |
| `wpu-cws-indexed-mechanism-adapter` | 0.497143 | 0.243597 | 0.652587 | 1.078440 | 0.000000 |
| `wpu-cws-indexed-sparse` | 0.364286 | 0.179114 | 0.668222 | 1.102699 | 0.000000 |

The mechanism adapter is the best macro-accuracy model while using no dense
fallback. Its calibration is not clearly better than the baselines, and its NLL
is worse than graph-transformer, so the result should be framed as an
accuracy/compute result, not a probability-quality result.

## Per-Mechanism Boundary

| Mechanism | Mechanism adapter | Best baseline | Delta | Boundary |
|---|---:|---:|---:|---|
| `catch_heavy` | 0.680000 | 0.480000 | +0.200000 | Strong positive; object-wise mechanism context helps catch-prior shift. |
| `edge_catch_heavy` | 0.340000 | 0.480000 | -0.140000 | Failure; composed edge+catch law is not learned. |
| `edge_high_force` | 0.480000 | 0.480000 | +0.000000 | Tie. |
| `edge_shift` | 0.410000 | 0.470000 | -0.060000 | Failure; edge geometry law remains weak. |
| `high_force` | 0.580000 | 0.520000 | +0.060000 | Positive. |
| `no_catch` | 0.410000 | 0.490000 | -0.080000 | Failure. |
| `nominal` | 0.580000 | 0.520000 | +0.060000 | Positive. |

Win/tie/loss versus the best non-WPU baseline is `3/1/3`, with mean margin
`+0.005714`. This is a narrow positive result. It supports the next WPU v2
direction, but it does not prove broad mechanism generalization.

## Interpretation

The important finding is not that WPU universally beats token or graph models.
It does not. The important finding is that a state-native object-wise adapter
can recover a small large-N accuracy edge without dense recompute when the
training set contains primitive mechanism variation. That is a better WPU v2
claim than nominal-only zero-shot extrapolation.

The remaining failure modes are specific and useful:

- Edge composition remains weak, especially `edge_catch_heavy`.
- Calibration is not solved; the adapter's ECE is similar to baselines, while
  NLL remains worse than graph-transformer.
- The result is still one simulator family, one-step prediction, small model,
  and N_bg=512.

## Next Step

The next priority is to train the object-wise mechanism adapter with explicit
composition objectives and calibration losses. The target is not to return to
token processing, but to make sparse state propagation learn local mechanism
families that compose across object relations.

# WPU V2 Regret Router Variant Results

This experiment tests whether the staged regret router improves when its input
is made more state-native.

## Question

The previous post-hoc probe found that physical state features generalized
better than raw sparse diagnostics under held-out seeds. That result does not
automatically imply that the same features help when embedded inside the WPU
model. This run compares three internal router variants:

- `internal`: original staged regret head using sparse hidden summary plus K
  pressure, selector confidence, and interaction density.
- `physics_hidden`: sparse hidden summary plus physical route features.
- `state_only`: no hidden summary; only K pressure, interaction density,
  pair-distance statistics, target position, and event magnitude.

## Protocol

- Dataset: pairwise CWS synthetic physics.
- `N = 2048`.
- `K in {8, 16, 32}`.
- Seeds: `11, 13, 17, 19, 23`.
- Model size: hidden dim `128`, one local dense layer, four heads.
- Training: 40 propagation steps, then 80 route-regret steps.
- Evaluation: 90 held-out samples per condition.
- Compute cost: `0.05`.

Artifacts:

- `docs/experiments/wpu_v2_staged_regret_hybrid_5seed.csv`
- `docs/experiments/wpu_v2_physics_regret_hybrid_5seed.csv`
- `docs/experiments/wpu_v2_state_regret_hybrid_5seed.csv`
- `docs/experiments/wpu_v2_regret_router_variant_summary.csv`
- `docs/experiments/wpu_v2_regret_router_variant_paired.csv`
- `scripts/compare_regret_router_variants.py`

## Aggregate Results

| router | routed loss | delta vs sparse | oracle excess | dense compute | regret corr | routed accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| internal | 0.963 | -0.025 | 0.045 | 0.237 | 0.351 | 0.493 |
| physics hidden | 0.963 | -0.022 | 0.049 | 0.240 | 0.332 | 0.503 |
| state only | 0.983 | 0.001 | 0.074 | 0.297 | 0.161 | 0.495 |

K-specific routed loss:

| router | K=8 | K=16 | K=32 |
| --- | ---: | ---: | ---: |
| internal | 0.960 | 0.949 | 0.979 |
| physics hidden | 0.958 | 0.942 | 0.989 |
| state only | 0.981 | 0.971 | 0.995 |

## Interpretation

The state-only internal route head is rejected for the current architecture. It
removes the hidden summary too aggressively: regret correlation falls from
`0.351` to `0.161`, routed loss becomes slightly worse than sparse-only, and
oracle excess increases from `0.045` to `0.074`.

Adding physical features to the hidden-summary router is mostly neutral. It
slightly improves routed accuracy, but loss is unchanged overall and worse at
`K=32`. This means the post-hoc physical-state result does not transfer by
simple concatenation.

The best current deployed router remains the original staged internal regret
router. It is not final, but it is the only tested internal route head that
reliably lowers loss under bounded dense compute.

## Consequence for V2

The next scheduler should not be a plain MLP over hand-chosen state scalars.
The useful direction is a structured verifier:

- use hidden local propagation evidence to estimate whether sparse execution is
  already sufficient;
- use explicit physical state constraints to veto or verify that estimate;
- trigger K expansion or uncertainty increase when the two disagree;
- avoid direct diagnostic residuals or single deployed thresholds until their
  calibration transfers across seeds.

This preserves the WPU thesis: the router should remain state-native, but the
state signal must be represented as structured constraints and verification,
not merely as a small scalar feature vector.

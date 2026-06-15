# N_bg=512 Mechanism-Conditioned Propagation Screen

This screen tests the next hypothesis after the selected route-regret and
matched mechanism-prior adaptation screens failed: mechanism-relevant object and
event state should condition the local propagation dynamics themselves, not only
the route selector or a post-hoc branch prior.

Source CSV:
`docs/experiments/pybullet_shift_generalization_n512_mechanism_conditioned_screen.csv`

## Protocol

- Training mechanism: `nominal`
- Evaluation mechanisms: `high_force`, `edge_shift`, `catch_heavy`, `no_catch`
- Seeds: `11`, `13`, `17`
- Background objects: `512`
- Total objects: `517`
- Samples per mechanism/seed: `12`
- Training steps: `4`
- Simulation steps: `80`
- Models: `wpu-cws-indexed-sparse`, `wpu-cws-indexed-mechanism-conditioned`,
  `graph-transformer`, `serialized-token`
- Mechanism prior adaptation: disabled
- Dense fallback for mechanism-conditioned WPU: disabled

## Aggregate Result

| Model | Mean branch accuracy | Mean ECE | Mean dense compute |
|---|---:|---:|---:|
| `graph-transformer` | 0.500000 | 0.242072 | 1.000000 |
| `serialized-token` | 0.430555 | 0.242987 | 1.000000 |
| `wpu-cws-indexed-mechanism-conditioned` | 0.541667 | 0.203094 | 0.000000 |
| `wpu-cws-indexed-sparse` | 0.500000 | 0.158891 | 0.000000 |

The mechanism-conditioned WPU is the best macro-accuracy model in this screen
and does so without dense recompute. It also improves over the sparse WPU by
`+0.041667` macro accuracy. This is a positive screen, but it is still small and
should not be treated as final mechanism-shift evidence.

## Per-Mechanism Boundary

| Mechanism | Mechanism-conditioned WPU | Best baseline | Delta | Interpretation |
|---|---:|---:|---:|---|
| `catch_heavy` | 0.777778 | 0.638889 | +0.138889 | Clear positive screen; mechanism context helps a prior-shifted branch. |
| `edge_shift` | 0.361111 | 0.444444 | -0.083333 | Failure boundary; geometry/edge-law shift is not solved. |
| `high_force` | 0.500000 | 0.500000 | +0.000000 | Tie. |
| `no_catch` | 0.527778 | 0.527778 | +0.000000 | Tie. |

Win/tie/loss versus the best non-WPU baseline is `1/2/1`. The result is more
encouraging than the route-regret adapted screen (`0/0/4`), but not strong
enough to claim large-N zero-shot mechanism generalization.

## Interpretation

This result changes the current research direction. The negative route-regret
adaptation screen suggested that threshold routing and post-hoc priors are not
the missing mechanism. This screen suggests that the missing mechanism is more
likely the transition function itself: local propagation must receive explicit
mechanism context, such as force, action, target physical scalars, selected-set
physical scalars, and pair geometry.

The result remains bounded. It is a small 3-seed, 4-shift screen, the
`edge_shift` mechanism is still below the serialized-token baseline, and no
latency or power claim follows from this CSV alone. The defensible claim is that
mechanism-conditioned propagation is now a plausible WPU v2 direction worth
larger sweeps.

## Next Experiment

The next run should scale this exact model family to more seeds, all available
mechanisms, and at least one larger background regime. It should compare
mechanism-conditioned sparse propagation against local-dense, selected
route-regret, adapted route-regret, graph-transformer, and serialized-token
baselines under the same training budget, while reporting accuracy, ECE, Brier,
NLL, dense compute, selected `K`, latency, and route decisions.

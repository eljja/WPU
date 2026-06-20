# Loss-Supervised Safe Candidate Generation Results

This report summarizes a P1 follow-up for the joint selector-propagator probe.
Unlike the earlier `learned_safe_*` diagnostic, which imitated hand-built
teachers, this run trains the learned safe candidate generators from train-fold
propagation loss and no-harm transfer labels.

Source CSV: `docs/experiments/wpu_v2_joint_selector_propagator_loss_safe_candidates.csv`

Protocol:

- Total world size: `N=2048`.
- Causal working-set sizes: `K=16` and `K=32`.
- Held-out seeds: `11, 13, 17, 19, 23`.
- Candidate pool: base selectors, generated candidates, structured candidates,
  and four loss-supervised `learned_safe_*` candidates.
- Selector context: label-free propagation verification signatures enabled.
- Safety objective: pairwise no-harm score margin with weight `0.3`.

## Aggregate Results

| K | Policy | Loss | Accuracy | Gap closure | Accept | Harmful accept |
|---:|---|---:|---:|---:|---:|---:|
| 16 | `static_learned_interaction` | 0.898381 | 0.520000 | 0.000000 | 0.000000 | 0.000000 |
| 16 | `train_selected_joint_selector_propagator` | 0.877104 | 0.573333 | 0.302097 | 0.653333 | 0.146667 |
| 16 | `joint_selector_propagator` | 0.879296 | 0.577778 | 0.270969 | 0.786667 | 0.220000 |
| 16 | `generated_plus_composition_oracle` | 0.827949 | 0.626667 | 1.000000 | 0.997778 | 0.000000 |
| 32 | `static_learned_interaction` | 0.982548 | 0.488889 | 0.000000 | 0.000000 | 0.000000 |
| 32 | `train_selected_joint_selector_propagator` | 0.966607 | 0.515555 | 0.357010 | 0.326667 | 0.055555 |
| 32 | `joint_selector_propagator` | 0.973588 | 0.491111 | 0.200666 | 0.637778 | 0.224444 |
| 32 | `generated_plus_composition_oracle` | 0.937897 | 0.542222 | 1.000000 | 0.993333 | 0.000000 |

## Interpretation

Loss-supervised safe candidate generation improves the larger-K deployment
boundary. The strongest previous K=32 safe closures were roughly `0.153386`
with verification context and `0.200230` with pairwise no-harm scoring. This
run raises train-selected K=32 closure to `0.357010` while keeping harmful
accept at `0.055555`.

The result is still not a solved P1 claim. The P1 target is `0.5` safe closure,
and K=16 reaches only `0.302097`. The useful conclusion is narrower:
candidate generation should be trained from propagation-loss/no-harm labels,
not from hand-built teacher imitation alone. The next experiment should make
candidate generation and propagation verification more tightly joint, rather
than treating the loss-supervised generator as a frozen pre-step.

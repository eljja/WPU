# Candidate Regret Gate Results

This report summarizes a P1 probe that directly predicts `candidate_loss - learned_loss` and deploys a candidate only when predicted regret is sufficiently favorable.

Source CSV: `docs/experiments/wpu_v2_candidate_regret_gate.csv`

The best closure is `0.308651` (`K=16`, `candidate_regret_gate`). This improves over the previous best `0.244220`, but it remains below the P1 target `0.5`. The high harmful-accept rate shows that candidate-regret prediction starts to close the gap, while no-harm rejection remains weak.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | `candidate_regret_gate` | 0.985735 | 0.517778 | 0.032763 | 0.002697 | 0.082306 | 0.982222 | 0.404445 | 0.050928 | `insufficient_no_harm_rejection` |
| 8 | `uncertainty_regret_gate` | 0.988296 | 0.511111 | 0.032763 | 0.000136 | 0.004145 | 0.248889 | 0.126667 | 0.050928 | `partial_but_insufficient_gap_closure` |
| 16 | `candidate_regret_gate` | 0.947425 | 0.535556 | 0.060773 | 0.018758 | 0.308651 | 0.951111 | 0.317778 | 0.118052 | `insufficient_no_harm_rejection` |
| 16 | `uncertainty_regret_gate` | 0.957925 | 0.517778 | 0.060773 | 0.008258 | 0.135886 | 0.395556 | 0.126667 | 0.118052 | `partial_but_insufficient_gap_closure` |
| 32 | `candidate_regret_gate` | 1.001991 | 0.526667 | 0.035577 | 0.002104 | 0.059128 | 0.964444 | 0.428889 | 0.072566 | `insufficient_no_harm_rejection` |
| 32 | `uncertainty_regret_gate` | 1.003947 | 0.486667 | 0.035577 | 0.000148 | 0.004149 | 0.428889 | 0.204444 | 0.072566 | `partial_but_insufficient_gap_closure` |

## Interpretation

- Candidate-regret targets provide a stronger signal than margin-only gating.
- K=16 improves, but K=8/32 generalization is still insufficient.
- The next improvement should strengthen accept/reject calibration, harmful-candidate penalties, and seed/domain perturbation in the learning objective.

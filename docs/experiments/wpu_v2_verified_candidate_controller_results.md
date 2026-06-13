# Verified Candidate Controller Results

This report summarizes a P1 probe that augments each candidate working set with label-free sparse/local-dense propagation verification signatures. The signature includes branch confidence, entropy, sparse/dense disagreement, and delta-norm gaps, all computed without ground-truth labels.

Source CSV: `docs/experiments/wpu_v2_verified_candidate_controller.csv`

The best closure is `0.024989` (`K=16`, `train_selected_verified_candidate_controller`). P1 evaluates whether verified-controller deployment closes the candidate-oracle gap while controlling harmful accepts. The conservative best under harmful-accept <= `0.25` is `0.023029` (`verified_uncertainty_regret_gate_r0p5_m0p005`). The train-selected deployed best is `0.024989` (`K=16`).

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | `verified_uncertainty_regret_gate_r0p5_m0p005` | 0.987666 | 0.513333 | 0.033246 | 0.000766 | 0.023029 | 0.557778 | 0.248889 | 0.092232 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_verified_candidate_controller` | 0.964643 | 0.506667 | 0.061627 | 0.001540 | 0.024989 | 0.835556 | 0.424444 | -0.055012 | `insufficient_no_harm_rejection` |
| 32 | `verified_uncertainty_regret_gate_r0p75_m0p01` | 1.003407 | 0.482222 | 0.035925 | 0.000688 | 0.019140 | 0.215556 | 0.102222 | 0.049777 | `partial_but_insufficient_gap_closure` |
| 16 | `verified_uncertainty_regret_gate_r0p5_m0p0025` | 0.964780 | 0.506667 | 0.061627 | 0.001402 | 0.022756 | 0.628889 | 0.315556 | -0.055012 | `insufficient_no_harm_rejection` |
| 8 | `verified_candidate_regret_gate` | 0.987743 | 0.511111 | 0.033246 | 0.000689 | 0.020725 | 0.942222 | 0.415556 | 0.092232 | `insufficient_no_harm_rejection` |
| 32 | `verified_uncertainty_regret_gate_r1_m0` | 1.003447 | 0.482222 | 0.035925 | 0.000648 | 0.018026 | 0.188889 | 0.091111 | 0.049777 | `partial_but_insufficient_gap_closure` |
| 32 | `verified_uncertainty_regret_gate_r0p75_m0p02` | 1.003449 | 0.482222 | 0.035925 | 0.000646 | 0.017971 | 0.142222 | 0.066667 | 0.049777 | `partial_but_insufficient_gap_closure` |
| 8 | `verified_uncertainty_regret_gate_r0p5_m0p01` | 0.987851 | 0.508889 | 0.033246 | 0.000581 | 0.017464 | 0.473333 | 0.211111 | 0.092232 | `partial_but_insufficient_gap_closure` |
| 8 | `verified_uncertainty_regret_gate_r1p5_m0p05` | 0.987859 | 0.504444 | 0.033246 | 0.000573 | 0.017223 | 0.046667 | 0.015556 | 0.092232 | `partial_but_insufficient_gap_closure` |
| 16 | `verified_uncertainty_regret_gate_r3_m0p05` | 0.965223 | 0.504444 | 0.061627 | 0.000960 | 0.015571 | 0.011111 | 0.002222 | -0.055012 | `partial_but_insufficient_gap_closure` |
| 8 | `verified_uncertainty_regret_gate_r0p5_m0p0025` | 0.987930 | 0.513333 | 0.033246 | 0.000501 | 0.015076 | 0.611111 | 0.277778 | 0.092232 | `insufficient_no_harm_rejection` |
| 8 | `verified_candidate_regret_gate_m0p0025` | 0.987975 | 0.511111 | 0.033246 | 0.000457 | 0.013746 | 0.931111 | 0.413333 | 0.092232 | `insufficient_no_harm_rejection` |
| 16 | `verified_uncertainty_regret_gate_r0p5_m0p005` | 0.965392 | 0.506667 | 0.061627 | 0.000790 | 0.012826 | 0.580000 | 0.286667 | -0.055012 | `insufficient_no_harm_rejection` |
| 8 | `verified_uncertainty_regret_gate_r1_m0p02` | 0.988008 | 0.506667 | 0.033246 | 0.000423 | 0.012736 | 0.151111 | 0.064444 | 0.092232 | `partial_but_insufficient_gap_closure` |
| 32 | `verified_uncertainty_regret_gate_r1_m0p0025` | 1.003643 | 0.480000 | 0.035925 | 0.000451 | 0.012559 | 0.173333 | 0.082222 | 0.049777 | `partial_but_insufficient_gap_closure` |
| 32 | `verified_candidate_regret_gate_m0p05` | 1.003701 | 0.482222 | 0.035925 | 0.000394 | 0.010967 | 0.195556 | 0.095556 | 0.049777 | `partial_but_insufficient_gap_closure` |
| 8 | `verified_uncertainty_regret_gate_r1p5_m0p02` | 0.988077 | 0.504444 | 0.033246 | 0.000355 | 0.010666 | 0.080000 | 0.031111 | 0.092232 | `partial_but_insufficient_gap_closure` |
| 16 | `verified_uncertainty_regret_gate` | 0.965526 | 0.506667 | 0.061627 | 0.000657 | 0.010664 | 0.664445 | 0.335556 | -0.055012 | `insufficient_no_harm_rejection` |

## Interpretation

- The CSV keeps all reject-margin/risk-penalty deployment sweep points.
- The table below shows the best policy per K and the strongest overall policies.
- A useful deployed policy needs both high closure and low harmful accepts.
- If this probe underperforms the direct candidate-regret gate, post-hoc label-free sparse/dense verification signatures are not sufficient; verification must be trained jointly with retrieval and propagation.

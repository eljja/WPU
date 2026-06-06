# Candidate Regret Gate Results

This report summarizes a P1 probe that directly predicts `candidate_loss - learned_loss` and deploys a candidate only when predicted regret is sufficiently favorable.

Source CSV: `docs/experiments/wpu_v2_candidate_regret_gate_perturbed.csv`

The best closure is `0.339525` (`K=16`, `candidate_regret_gate_m0p0025`). P1 evaluates whether candidate-regret deployment closes the candidate-oracle gap while controlling harmful accepts. The conservative best under harmful-accept <= `0.25` is `0.329756` (`candidate_regret_gate_m0p005`). The train-selected deployed best is `0.312586` (`K=16`).

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | `train_selected_candidate_regret_gate` | 0.983868 | 0.524444 | 0.032844 | 0.004564 | 0.138962 | 0.764444 | 0.308889 | 0.082981 | `insufficient_no_harm_rejection` |
| 16 | `candidate_regret_gate_m0p0025` | 0.945380 | 0.528889 | 0.061271 | 0.020803 | 0.339525 | 0.875556 | 0.253333 | 0.155800 | `insufficient_no_harm_rejection` |
| 32 | `candidate_regret_gate_m0p005` | 1.001629 | 0.517778 | 0.035563 | 0.002465 | 0.069319 | 0.813333 | 0.342222 | 0.050506 | `insufficient_no_harm_rejection` |
| 16 | `candidate_regret_gate` | 0.945665 | 0.528889 | 0.061271 | 0.020518 | 0.334874 | 0.900000 | 0.264445 | 0.155800 | `insufficient_no_harm_rejection` |
| 16 | `candidate_regret_gate_m0p005` | 0.945978 | 0.528889 | 0.061271 | 0.020205 | 0.329756 | 0.826667 | 0.235555 | 0.155800 | `partial_but_insufficient_gap_closure` |
| 16 | `train_selected_candidate_regret_gate` | 0.947030 | 0.522222 | 0.061271 | 0.019153 | 0.312586 | 0.771111 | 0.211111 | 0.155800 | `partial_but_insufficient_gap_closure` |
| 16 | `candidate_regret_gate_m0p01` | 0.948064 | 0.520000 | 0.061271 | 0.018119 | 0.295714 | 0.742222 | 0.217778 | 0.155800 | `partial_but_insufficient_gap_closure` |
| 16 | `candidate_regret_gate_m0p02` | 0.954939 | 0.522222 | 0.061271 | 0.011244 | 0.183508 | 0.500000 | 0.160000 | 0.155800 | `partial_but_insufficient_gap_closure` |
| 16 | `uncertainty_regret_gate` | 0.956727 | 0.513333 | 0.061271 | 0.009455 | 0.154320 | 0.315556 | 0.097778 | 0.155800 | `partial_but_insufficient_gap_closure` |
| 16 | `uncertainty_regret_gate_r0p5_m0p0025` | 0.957682 | 0.513333 | 0.061271 | 0.008501 | 0.138747 | 0.266667 | 0.084444 | 0.155800 | `partial_but_insufficient_gap_closure` |
| 16 | `uncertainty_regret_gate_r0p5_m0p005` | 0.958167 | 0.513333 | 0.061271 | 0.008016 | 0.130821 | 0.235556 | 0.073333 | 0.155800 | `partial_but_insufficient_gap_closure` |
| 8 | `candidate_regret_gate_m0p01` | 0.984710 | 0.524444 | 0.032844 | 0.003722 | 0.113325 | 0.708889 | 0.291111 | 0.082981 | `insufficient_no_harm_rejection` |
| 8 | `candidate_regret_gate` | 0.984805 | 0.511111 | 0.032844 | 0.003626 | 0.110408 | 0.960000 | 0.404445 | 0.082981 | `insufficient_no_harm_rejection` |
| 8 | `candidate_regret_gate_m0p0025` | 0.985459 | 0.511111 | 0.032844 | 0.002972 | 0.090502 | 0.904445 | 0.386667 | 0.082981 | `insufficient_no_harm_rejection` |
| 8 | `candidate_regret_gate_m0p005` | 0.985832 | 0.511111 | 0.032844 | 0.002599 | 0.079139 | 0.840000 | 0.362222 | 0.082981 | `insufficient_no_harm_rejection` |
| 8 | `candidate_regret_gate_m0p02` | 0.986322 | 0.522222 | 0.032844 | 0.002110 | 0.064232 | 0.393333 | 0.164444 | 0.082981 | `partial_but_insufficient_gap_closure` |
| 32 | `train_selected_candidate_regret_gate` | 1.001846 | 0.522222 | 0.035563 | 0.002249 | 0.063234 | 0.757778 | 0.315556 | 0.050506 | `insufficient_no_harm_rejection` |
| 32 | `candidate_regret_gate_m0p0025` | 1.001956 | 0.517778 | 0.035563 | 0.002139 | 0.060141 | 0.888889 | 0.373333 | 0.050506 | `insufficient_no_harm_rejection` |

## Interpretation

- The CSV keeps all reject-margin/risk-penalty deployment sweep points.
- The table below shows the best policy per K and the strongest overall policies.
- A useful deployed policy needs both high closure and low harmful accepts.

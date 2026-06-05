# Candidate Regret Gate Results

This report summarizes a P1 probe that directly predicts `candidate_loss - learned_loss` and deploys a candidate only when predicted regret is sufficiently favorable.

Source CSV: `docs/experiments/wpu_v2_candidate_regret_gate.csv`

The best closure is `0.329950` (`K=16`, `candidate_regret_gate_m0p005`). P1 evaluates whether candidate-regret deployment closes the candidate-oracle gap while controlling harmful accepts. The conservative best under harmful-accept <= `0.25` is `0.327146` (`candidate_regret_gate_m0p01`). The train-selected deployed best is `0.328025` (`K=16`).

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | `candidate_regret_gate_m0p01` | 0.985339 | 0.517778 | 0.032763 | 0.003092 | 0.094380 | 0.768889 | 0.320000 | 0.050928 | `insufficient_no_harm_rejection` |
| 16 | `candidate_regret_gate_m0p005` | 0.946131 | 0.533333 | 0.060773 | 0.020052 | 0.329950 | 0.880000 | 0.271111 | 0.118052 | `insufficient_no_harm_rejection` |
| 32 | `candidate_regret_gate` | 1.001991 | 0.526667 | 0.035577 | 0.002104 | 0.059128 | 0.964444 | 0.428889 | 0.072566 | `insufficient_no_harm_rejection` |
| 16 | `train_selected_candidate_regret_gate` | 0.946248 | 0.533333 | 0.060773 | 0.019935 | 0.328025 | 0.835555 | 0.251111 | 0.118052 | `insufficient_no_harm_rejection` |
| 16 | `candidate_regret_gate_m0p01` | 0.946301 | 0.531111 | 0.060773 | 0.019882 | 0.327146 | 0.804444 | 0.235556 | 0.118052 | `partial_but_insufficient_gap_closure` |
| 16 | `candidate_regret_gate_m0p0025` | 0.946829 | 0.535556 | 0.060773 | 0.019353 | 0.318455 | 0.920000 | 0.295555 | 0.118052 | `insufficient_no_harm_rejection` |
| 16 | `candidate_regret_gate` | 0.947425 | 0.535556 | 0.060773 | 0.018758 | 0.308651 | 0.951111 | 0.317778 | 0.118052 | `insufficient_no_harm_rejection` |
| 16 | `candidate_regret_gate_m0p02` | 0.951893 | 0.528889 | 0.060773 | 0.014289 | 0.235128 | 0.557778 | 0.173333 | 0.118052 | `partial_but_insufficient_gap_closure` |
| 16 | `uncertainty_regret_gate` | 0.957925 | 0.517778 | 0.060773 | 0.008258 | 0.135886 | 0.395556 | 0.126667 | 0.118052 | `partial_but_insufficient_gap_closure` |
| 16 | `uncertainty_regret_gate_r0p5_m0p0025` | 0.959642 | 0.515556 | 0.060773 | 0.006541 | 0.107634 | 0.342222 | 0.115555 | 0.118052 | `partial_but_insufficient_gap_closure` |
| 16 | `uncertainty_regret_gate_r0p5_m0p005` | 0.960611 | 0.511111 | 0.060773 | 0.005572 | 0.091686 | 0.306667 | 0.102222 | 0.118052 | `partial_but_insufficient_gap_closure` |
| 8 | `train_selected_candidate_regret_gate` | 0.985596 | 0.515556 | 0.032763 | 0.002836 | 0.086554 | 0.848889 | 0.351111 | 0.050928 | `insufficient_no_harm_rejection` |
| 8 | `candidate_regret_gate` | 0.985735 | 0.517778 | 0.032763 | 0.002697 | 0.082306 | 0.982222 | 0.404445 | 0.050928 | `insufficient_no_harm_rejection` |
| 16 | `uncertainty_regret_gate_r0p5_m0p01` | 0.961206 | 0.508889 | 0.060773 | 0.004977 | 0.081898 | 0.182222 | 0.051111 | 0.118052 | `partial_but_insufficient_gap_closure` |
| 8 | `candidate_regret_gate_m0p005` | 0.985753 | 0.513333 | 0.032763 | 0.002678 | 0.081750 | 0.891111 | 0.371111 | 0.050928 | `insufficient_no_harm_rejection` |
| 8 | `candidate_regret_gate_m0p0025` | 0.985841 | 0.515556 | 0.032763 | 0.002591 | 0.079083 | 0.928889 | 0.386667 | 0.050928 | `insufficient_no_harm_rejection` |
| 16 | `uncertainty_regret_gate_r0p75_m0` | 0.962994 | 0.506667 | 0.060773 | 0.003188 | 0.052464 | 0.115556 | 0.028889 | 0.118052 | `partial_but_insufficient_gap_closure` |
| 16 | `uncertainty_regret_gate_r0p75_m0p0025` | 0.963040 | 0.504444 | 0.060773 | 0.003143 | 0.051721 | 0.086667 | 0.022222 | 0.118052 | `partial_but_insufficient_gap_closure` |

## Interpretation

- The CSV keeps all reject-margin/risk-penalty deployment sweep points.
- The table below shows the best policy per K and the strongest overall policies.
- A useful deployed policy needs both high closure and low harmful accepts.

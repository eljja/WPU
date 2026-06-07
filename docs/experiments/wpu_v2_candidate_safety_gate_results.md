# Candidate Safety/Utility Gate Results

This report summarizes a P1 probe that predicts safe probability and utility separately, then deploys a candidate only when the predicted utility and safety probability are sufficiently favorable.

Source CSV: `docs/experiments/wpu_v2_candidate_safety_gate.csv`

The best closure is `0.147450` (`K=16`, `safety_utility_gate_p0p5_m0_r0p5`). P1 evaluates whether safety/utility deployment closes the candidate-oracle gap while controlling harmful accepts. The conservative best under harmful-accept <= `0.25` is `0.090719` (`safety_utility_gate_p0p65_m0_r0p5`).

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | `safety_utility_gate_p0p55_m0_r1` | 0.983993 | 0.520000 | 0.032969 | 0.004438 | 0.134617 | 0.828889 | 0.328889 | 0.038601 | `insufficient_no_harm_rejection` |
| 16 | `safety_utility_gate_p0p5_m0_r0p5` | 0.957176 | 0.511111 | 0.061085 | 0.009007 | 0.147450 | 0.797778 | 0.364444 | -0.024838 | `insufficient_no_harm_rejection` |
| 32 | `safety_utility_gate_p0p75_m0_r0` | 1.004112 | 0.480000 | 0.035561 | -0.000018 | -0.000495 | 0.120000 | 0.057778 | -0.038721 | `harmful_transfer` |
| 16 | `safety_utility_gate_p0p5_m0p001_r0p5` | 0.957176 | 0.511111 | 0.061085 | 0.009007 | 0.147450 | 0.797778 | 0.364444 | -0.024838 | `insufficient_no_harm_rejection` |
| 16 | `safety_utility_gate_p0p5_m0p0025_r0p5` | 0.957176 | 0.511111 | 0.061085 | 0.009007 | 0.147450 | 0.797778 | 0.364444 | -0.024838 | `insufficient_no_harm_rejection` |
| 16 | `safety_utility_gate_p0p5_m0p005_r0p5` | 0.957177 | 0.511111 | 0.061085 | 0.009006 | 0.147427 | 0.795556 | 0.364444 | -0.024838 | `insufficient_no_harm_rejection` |
| 16 | `safety_utility_gate_p0p5_m0p01_r0p5` | 0.957322 | 0.511111 | 0.061085 | 0.008861 | 0.145056 | 0.793333 | 0.364444 | -0.024838 | `insufficient_no_harm_rejection` |
| 16 | `train_selected_safety_utility_gate` | 0.957334 | 0.506667 | 0.061085 | 0.008849 | 0.144863 | 0.653333 | 0.297778 | -0.024838 | `insufficient_no_harm_rejection` |
| 16 | `safety_utility_gate_p0p5_m0_r0` | 0.958042 | 0.513333 | 0.061085 | 0.008141 | 0.133273 | 0.788889 | 0.357778 | -0.024838 | `insufficient_no_harm_rejection` |
| 16 | `safety_utility_gate_p0p5_m0p001_r0` | 0.958042 | 0.513333 | 0.061085 | 0.008141 | 0.133273 | 0.788889 | 0.357778 | -0.024838 | `insufficient_no_harm_rejection` |
| 16 | `safety_utility_gate_p0p5_m0p0025_r0` | 0.958042 | 0.513333 | 0.061085 | 0.008141 | 0.133273 | 0.788889 | 0.357778 | -0.024838 | `insufficient_no_harm_rejection` |
| 16 | `safety_utility_gate_p0p5_m0p005_r0` | 0.958042 | 0.513333 | 0.061085 | 0.008141 | 0.133273 | 0.788889 | 0.357778 | -0.024838 | `insufficient_no_harm_rejection` |
| 16 | `safety_utility_gate_p0p5_m0p01_r0` | 0.958042 | 0.513333 | 0.061085 | 0.008141 | 0.133273 | 0.788889 | 0.357778 | -0.024838 | `insufficient_no_harm_rejection` |
| 8 | `safety_utility_gate_p0p6_m0_r0` | 0.984047 | 0.522222 | 0.032969 | 0.004385 | 0.132991 | 0.742222 | 0.311111 | 0.038601 | `insufficient_no_harm_rejection` |
| 8 | `safety_utility_gate_p0p6_m0p001_r0` | 0.984047 | 0.522222 | 0.032969 | 0.004385 | 0.132991 | 0.742222 | 0.311111 | 0.038601 | `insufficient_no_harm_rejection` |
| 8 | `safety_utility_gate_p0p6_m0p0025_r0` | 0.984047 | 0.522222 | 0.032969 | 0.004385 | 0.132991 | 0.742222 | 0.311111 | 0.038601 | `insufficient_no_harm_rejection` |
| 8 | `safety_utility_gate_p0p6_m0p005_r0` | 0.984047 | 0.522222 | 0.032969 | 0.004385 | 0.132991 | 0.742222 | 0.311111 | 0.038601 | `insufficient_no_harm_rejection` |
| 8 | `safety_utility_gate_p0p6_m0p01_r0` | 0.984047 | 0.522222 | 0.032969 | 0.004385 | 0.132991 | 0.742222 | 0.311111 | 0.038601 | `insufficient_no_harm_rejection` |

## Interpretation

- The CSV keeps all reject-margin/risk-penalty deployment sweep points.
- The table below shows the best policy per K and the strongest overall policies.
- A useful deployed policy needs both high closure and low harmful accepts.

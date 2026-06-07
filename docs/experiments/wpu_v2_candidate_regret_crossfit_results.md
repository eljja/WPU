# Cross-Fit Candidate Regret Gate Results

This report summarizes a P1 probe that directly predicts `candidate_loss - learned_loss`, but selects deployment thresholds using out-of-source-seed cross-fit predictions rather than in-sample train predictions.

Source CSV: `docs/experiments/wpu_v2_candidate_regret_crossfit.csv`

The best closure is `0.287268` (`K=16`, `crossfit_regret_gate_m0p0025_r0_d0_v0`). P1 evaluates whether candidate-regret deployment closes the candidate-oracle gap while controlling harmful accepts. The conservative best under harmful-accept <= `0.25` is `0.279738` (`crossfit_regret_gate_m0_r0_d0_v1`).

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | `crossfit_regret_gate_m0p005_r0_d0_v1` | 0.984248 | 0.517778 | 0.033040 | 0.004183 | 0.126617 | 0.677778 | 0.260000 | 0.082187 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0p0025_r0_d0_v0` | 0.948649 | 0.537778 | 0.061036 | 0.017534 | 0.287268 | 0.928889 | 0.284444 | 0.117792 | `insufficient_no_harm_rejection` |
| 32 | `crossfit_regret_gate_m0p0025_r0_d0_v0p75` | 1.000466 | 0.524445 | 0.035612 | 0.003628 | 0.101881 | 0.753333 | 0.320000 | 0.087270 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0p0025_r0_d0_v0p5` | 0.948649 | 0.537778 | 0.061036 | 0.017534 | 0.287268 | 0.928889 | 0.284444 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0_r0_d0_v0p75` | 0.948958 | 0.537778 | 0.061036 | 0.017225 | 0.282206 | 0.920000 | 0.288889 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0_r0_d0_v0` | 0.949015 | 0.537778 | 0.061036 | 0.017168 | 0.281278 | 0.953333 | 0.300000 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0_r0_d0_v0p5` | 0.949018 | 0.537778 | 0.061036 | 0.017165 | 0.281229 | 0.951111 | 0.300000 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0_r0_d0_v1` | 0.949109 | 0.531111 | 0.061036 | 0.017074 | 0.279738 | 0.824444 | 0.244445 | 0.117792 | `partial_but_insufficient_gap_closure` |
| 16 | `crossfit_regret_gate_m0p0025_r0_d0_v0p75` | 0.949260 | 0.537778 | 0.061036 | 0.016922 | 0.277255 | 0.877778 | 0.273333 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0p005_r0_d0_v0p5` | 0.949452 | 0.537778 | 0.061036 | 0.016730 | 0.274109 | 0.880000 | 0.271111 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0p005_r0_d0_v0` | 0.949467 | 0.537778 | 0.061036 | 0.016715 | 0.273863 | 0.882222 | 0.273333 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0_r0_d0p5_v0p75` | 0.949598 | 0.537778 | 0.061036 | 0.016585 | 0.271720 | 0.913333 | 0.282222 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0_r0_d0p5_v1` | 0.949639 | 0.531111 | 0.061036 | 0.016543 | 0.271045 | 0.837778 | 0.251111 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_selected_candidate_regret_gate` | 0.949643 | 0.531111 | 0.061036 | 0.016540 | 0.270989 | 0.782222 | 0.222222 | 0.117792 | `partial_but_insufficient_gap_closure` |
| 16 | `crossfit_regret_gate_m0_r0_d1_v1` | 0.949736 | 0.531111 | 0.061036 | 0.016447 | 0.269462 | 0.840000 | 0.251111 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0p005_r0_d0_v0p75` | 0.949784 | 0.531111 | 0.061036 | 0.016398 | 0.268669 | 0.833333 | 0.257778 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0_r0_d0p5_v0` | 0.949786 | 0.537778 | 0.061036 | 0.016397 | 0.268650 | 0.922222 | 0.284444 | 0.117792 | `insufficient_no_harm_rejection` |
| 16 | `crossfit_regret_gate_m0_r0_d0p5_v0p5` | 0.949786 | 0.537778 | 0.061036 | 0.016397 | 0.268650 | 0.922222 | 0.284444 | 0.117792 | `insufficient_no_harm_rejection` |

## Interpretation

- The CSV keeps all reject-margin, risk-penalty, disagreement-penalty, and vote-threshold deployment sweep points.
- The table below shows the best policy per K and the strongest overall policies.
- This is a negative improvement result: cross-fit reduces train-selection optimism but lowers closure relative to the direct candidate-regret gate.
- A useful deployed policy needs both high closure and low harmful accepts.

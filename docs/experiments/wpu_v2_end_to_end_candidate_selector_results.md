# End-to-End Candidate Selector Results

This report summarizes a P1 probe that trains the candidate working-set selector directly on downstream propagation loss and no-harm mass relative to the learned baseline. The objective is policy-level expected loss, not only oracle-label imitation.

Source CSV: `docs/experiments/wpu_v2_end_to_end_candidate_selector.csv`

The best closure is `0.106927` (`K=16`, `end_to_end_selector_pg0p2_pmin0`). P1 evaluates whether end-to-end selector deployment closes the candidate-oracle gap while controlling harmful accepts. No deployed policy satisfies harmful-accept <= `0.25`. The train-selected deployed best is `0.096833` (`K=16`).

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | `end_to_end_selector_pg0_pmin0p6` | 0.988301 | 0.495556 | 0.033003 | 0.000131 | 0.003969 | 0.851111 | 0.355556 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0p2_pmin0` | 0.959662 | 0.502222 | 0.060987 | 0.006521 | 0.106927 | 0.982222 | 0.382222 | 0.000000 | `insufficient_no_harm_rejection` |
| 32 | `end_to_end_selector_pg0_pmin0p4` | 1.004137 | 0.486667 | 0.035586 | -0.000043 | -0.001197 | 0.975556 | 0.473333 | 0.000000 | `harmful_transfer` |
| 16 | `end_to_end_selector_pg0p2_pmin0p3` | 0.959662 | 0.502222 | 0.060987 | 0.006521 | 0.106927 | 0.982222 | 0.382222 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0p1_pmin0` | 0.959920 | 0.502222 | 0.060987 | 0.006263 | 0.102697 | 0.991111 | 0.384444 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0p1_pmin0p3` | 0.959920 | 0.502222 | 0.060987 | 0.006263 | 0.102697 | 0.991111 | 0.384444 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_candidate_selector` | 0.960019 | 0.502222 | 0.060987 | 0.006163 | 0.101061 | 0.995556 | 0.386667 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0_pmin0` | 0.960019 | 0.502222 | 0.060987 | 0.006163 | 0.101061 | 0.995556 | 0.386667 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0_pmin0p3` | 0.960019 | 0.502222 | 0.060987 | 0.006163 | 0.101061 | 0.995556 | 0.386667 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0p02_pmin0` | 0.960019 | 0.502222 | 0.060987 | 0.006163 | 0.101061 | 0.995556 | 0.386667 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0p02_pmin0p3` | 0.960019 | 0.502222 | 0.060987 | 0.006163 | 0.101061 | 0.995556 | 0.386667 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0p05_pmin0` | 0.960019 | 0.502222 | 0.060987 | 0.006163 | 0.101061 | 0.993333 | 0.386667 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0p05_pmin0p3` | 0.960019 | 0.502222 | 0.060987 | 0.006163 | 0.101061 | 0.993333 | 0.386667 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0p2_pmin0p4` | 0.960218 | 0.500000 | 0.060987 | 0.005965 | 0.097811 | 0.960000 | 0.377778 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `train_selected_end_to_end_candidate_selector` | 0.960277 | 0.502222 | 0.060987 | 0.005906 | 0.096833 | 0.977778 | 0.382222 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0_pmin0p5` | 0.960292 | 0.502222 | 0.060987 | 0.005891 | 0.096591 | 0.902222 | 0.353333 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0p02_pmin0p5` | 0.960292 | 0.502222 | 0.060987 | 0.005891 | 0.096591 | 0.902222 | 0.353333 | 0.000000 | `insufficient_no_harm_rejection` |
| 16 | `end_to_end_selector_pg0p05_pmin0p5` | 0.960292 | 0.502222 | 0.060987 | 0.005891 | 0.096591 | 0.902222 | 0.353333 | 0.000000 | `insufficient_no_harm_rejection` |

## Interpretation

- The CSV keeps all reject-margin/risk-penalty deployment sweep points.
- The table below shows the best policy per K and the strongest overall policies.
- A useful deployed policy needs both high closure and low harmful accepts.
- If this probe underperforms the direct candidate-regret gate, P1 is not merely a post-hoc thresholding problem; candidate generation, propagation, and selector training need deeper joint supervision.

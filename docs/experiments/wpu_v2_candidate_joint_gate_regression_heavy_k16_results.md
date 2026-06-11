# Joint Object-Set Candidate Gate Results

This report summarizes a P1 probe that jointly encodes each candidate working set as an explicit object set plus compact context. It tests whether candidate regret and no-harm acceptance become more transferable when the selector sees the candidate state itself rather than only aggregate descriptors.

Source CSV: `docs/experiments/wpu_v2_candidate_joint_gate_regression_heavy_k16.csv`

The best closure is `0.034751` (`K=16`, `joint_gate_p0p7_m0_r0p5`). P1 evaluates whether joint object-set deployment closes the candidate-oracle gap while controlling harmful accepts. The conservative best under harmful-accept <= `0.25` is `0.033842` (`joint_gate_p0p6_m0_r1`). The train-selected deployed best is `-0.003089` (`K=16`).

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | `joint_gate_p0p7_m0_r0p5` | 0.964059 | 0.513333 | 0.061120 | 0.002124 | 0.034751 | 0.635555 | 0.257778 | 0.020121 | `insufficient_no_harm_rejection` |
| 16 | `joint_gate_p0p6_m0_r1` | 0.964114 | 0.513333 | 0.061120 | 0.002068 | 0.033842 | 0.344445 | 0.140000 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p6_m0p01_r1` | 0.964119 | 0.508889 | 0.061120 | 0.002064 | 0.033770 | 0.251111 | 0.100000 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p001_r0p5` | 0.964136 | 0.515555 | 0.061120 | 0.002047 | 0.033485 | 0.615556 | 0.248889 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p55_m0p01_r1` | 0.964178 | 0.508889 | 0.061120 | 0.002005 | 0.032804 | 0.253333 | 0.102222 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p5_m0p01_r1` | 0.964178 | 0.508889 | 0.061120 | 0.002005 | 0.032804 | 0.253333 | 0.102222 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0_r1` | 0.964188 | 0.513333 | 0.061120 | 0.001995 | 0.032637 | 0.337778 | 0.137778 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p01_r1` | 0.964192 | 0.508889 | 0.061120 | 0.001990 | 0.032565 | 0.244445 | 0.097778 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p65_m0_r1` | 0.964236 | 0.513333 | 0.061120 | 0.001947 | 0.031852 | 0.342222 | 0.140000 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p65_m0p01_r1` | 0.964240 | 0.508889 | 0.061120 | 0.001942 | 0.031780 | 0.248889 | 0.100000 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p6_m0p001_r1` | 0.964274 | 0.511111 | 0.061120 | 0.001909 | 0.031234 | 0.335556 | 0.137778 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p55_m0p001_r1` | 0.964333 | 0.511111 | 0.061120 | 0.001850 | 0.030268 | 0.337778 | 0.140000 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p5_m0p001_r1` | 0.964333 | 0.511111 | 0.061120 | 0.001850 | 0.030268 | 0.337778 | 0.140000 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p6_m0_r0p5` | 0.964333 | 0.511111 | 0.061120 | 0.001850 | 0.030268 | 0.653333 | 0.266667 | 0.020121 | `insufficient_no_harm_rejection` |
| 16 | `joint_gate_p0p7_m0p001_r1` | 0.964347 | 0.511111 | 0.061120 | 0.001835 | 0.030029 | 0.328889 | 0.135556 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p65_m0p001_r1` | 0.964395 | 0.511111 | 0.061120 | 0.001787 | 0.029244 | 0.333334 | 0.137778 | 0.020121 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p6_m0p001_r0p5` | 0.964411 | 0.513333 | 0.061120 | 0.001772 | 0.028995 | 0.633333 | 0.257778 | 0.020121 | `insufficient_no_harm_rejection` |
| 16 | `joint_gate_p0p6_m0p0025_r1` | 0.964469 | 0.508889 | 0.061120 | 0.001714 | 0.028037 | 0.324444 | 0.131111 | 0.020121 | `partial_but_insufficient_gap_closure` |

## Interpretation

- The CSV keeps all reject-margin/risk-penalty deployment sweep points.
- The table below shows the best policy per K and the strongest overall policies.
- A useful deployed policy needs both high closure and low harmful accepts.
- If this probe underperforms the direct candidate-regret gate, the bottleneck is not merely missing object-set features; the cross-seed regret target itself remains unstable.

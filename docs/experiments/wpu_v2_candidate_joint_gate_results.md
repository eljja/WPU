# Joint Object-Set Candidate Gate Results

This report summarizes a P1 probe that jointly encodes each candidate working set as an explicit object set plus compact context. It tests whether candidate regret and no-harm acceptance become more transferable when the selector sees the candidate state itself rather than only aggregate descriptors.

Source CSV: `docs/experiments/wpu_v2_candidate_joint_gate.csv`

The best closure is `0.101454` (`K=16`, `joint_gate_p0p7_m0p0025_r0p75`). P1 evaluates whether joint object-set deployment closes the candidate-oracle gap while controlling harmful accepts. The conservative best under harmful-accept <= `0.25` is `0.101454` (`joint_gate_p0p7_m0p0025_r0p75`). The train-selected deployed best is `0.072167` (`K=16`).

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | `joint_gate_p0p6_m0p0025_r1` | 0.987104 | 0.502222 | 0.032854 | 0.001328 | 0.040415 | 0.391111 | 0.166667 | -0.020419 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p0025_r0p75` | 0.959996 | 0.502222 | 0.060983 | 0.006187 | 0.101454 | 0.446667 | 0.191111 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 32 | `joint_gate_p0p5_m0p02_r0` | 1.002218 | 0.502222 | 0.035570 | 0.001877 | 0.052763 | 0.637778 | 0.295556 | -0.012262 | `insufficient_no_harm_rejection` |
| 16 | `joint_gate_p0p7_m0p005_r0p75` | 0.960104 | 0.500000 | 0.060983 | 0.006079 | 0.099677 | 0.440000 | 0.188889 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p01_r0p75` | 0.960104 | 0.500000 | 0.060983 | 0.006079 | 0.099677 | 0.440000 | 0.188889 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p02_r0p75` | 0.960156 | 0.502222 | 0.060983 | 0.006027 | 0.098824 | 0.422222 | 0.180000 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p001_r0p75` | 0.960277 | 0.502222 | 0.060983 | 0.005906 | 0.096846 | 0.448889 | 0.193333 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0_r0p75` | 0.960427 | 0.502222 | 0.060983 | 0.005756 | 0.094383 | 0.451111 | 0.195556 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p01_r0p25` | 0.960811 | 0.513333 | 0.060983 | 0.005372 | 0.088093 | 0.475555 | 0.206667 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p005_r0p25` | 0.960824 | 0.513333 | 0.060983 | 0.005359 | 0.087880 | 0.477778 | 0.208889 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p02_r0p25` | 0.960839 | 0.513333 | 0.060983 | 0.005344 | 0.087631 | 0.471111 | 0.206667 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0_r0p25` | 0.960850 | 0.513333 | 0.060983 | 0.005333 | 0.087454 | 0.482222 | 0.211111 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p001_r0p25` | 0.960911 | 0.513333 | 0.060983 | 0.005272 | 0.086443 | 0.480000 | 0.211111 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p0025_r0p25` | 0.960911 | 0.513333 | 0.060983 | 0.005272 | 0.086443 | 0.480000 | 0.211111 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0_r0` | 0.961016 | 0.513333 | 0.060983 | 0.005167 | 0.084732 | 0.475556 | 0.208889 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p001_r0` | 0.961016 | 0.513333 | 0.060983 | 0.005167 | 0.084732 | 0.475556 | 0.208889 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p0025_r0` | 0.961016 | 0.513333 | 0.060983 | 0.005167 | 0.084732 | 0.475556 | 0.208889 | -0.000180 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_gate_p0p7_m0p005_r0` | 0.961016 | 0.513333 | 0.060983 | 0.005167 | 0.084732 | 0.473333 | 0.208889 | -0.000180 | `partial_but_insufficient_gap_closure` |

## Interpretation

- The CSV keeps all reject-margin/risk-penalty deployment sweep points.
- The table below shows the best policy per K and the strongest overall policies.
- A useful deployed policy needs both high closure and low harmful accepts.
- If this probe underperforms the direct candidate-regret gate, the bottleneck is not merely missing object-set features; the cross-seed regret target itself remains unstable.

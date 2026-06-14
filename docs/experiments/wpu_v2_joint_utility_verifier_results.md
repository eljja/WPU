# Joint Utility Verifier Results

This report summarizes a P1 probe that jointly encodes candidate object sets, compact context, and sparse/local-dense verification signatures, then predicts candidate regret, uncertainty, and no-harm safety. It is a more direct joint utility/safety head than post-hoc feature addition, but the propagation model is still fixed.

Source CSV: `docs/experiments/wpu_v2_joint_utility_verifier.csv`

The best closure is `0.097845` (`K=8`, `joint_utility_uncertainty_regret_gate_r1p5_m0_s0p35`). P1 evaluates whether joint utility-verifier deployment closes the candidate-oracle gap while controlling harmful accepts. The conservative best under harmful-accept <= `0.25` is `0.097845` (`joint_utility_uncertainty_regret_gate_r1p5_m0_s0p35`). The train-selected deployed best is `0.077781` (`K=8`).

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0_s0p35` | 0.985173 | 0.520000 | 0.033304 | 0.003259 | 0.097845 | 0.451111 | 0.182222 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 16 | `joint_utility_uncertainty_regret_gate_r2_m0_s0p35` | 0.964810 | 0.500000 | 0.061485 | 0.001373 | 0.022327 | 0.253333 | 0.097778 | -0.023260 | `partial_but_insufficient_gap_closure` |
| 32 | `joint_utility_uncertainty_regret_gate_r1p5_m0p01_s0p35` | 1.002293 | 0.491111 | 0.035809 | 0.001802 | 0.050318 | 0.411111 | 0.195556 | -0.003019 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0_s0p5` | 0.985235 | 0.517778 | 0.033304 | 0.003197 | 0.095995 | 0.440000 | 0.180000 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0p0025_s0p35` | 0.985301 | 0.522222 | 0.033304 | 0.003131 | 0.094007 | 0.442222 | 0.180000 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0p005_s0p35` | 0.985306 | 0.522222 | 0.033304 | 0.003125 | 0.093839 | 0.437778 | 0.177778 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0p0025_s0p5` | 0.985362 | 0.520000 | 0.033304 | 0.003069 | 0.092158 | 0.431111 | 0.177778 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0p005_s0p5` | 0.985368 | 0.520000 | 0.033304 | 0.003064 | 0.091990 | 0.426666 | 0.175555 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0p02_s0p35` | 0.985427 | 0.520000 | 0.033304 | 0.003004 | 0.090206 | 0.402222 | 0.162222 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0p02_s0p5` | 0.985489 | 0.517778 | 0.033304 | 0.002942 | 0.088350 | 0.391111 | 0.160000 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0p01_s0p35` | 0.985494 | 0.520000 | 0.033304 | 0.002938 | 0.088218 | 0.424444 | 0.175555 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0_s0p8` | 0.985545 | 0.517778 | 0.033304 | 0.002887 | 0.086687 | 0.420000 | 0.173333 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0p01_s0p5` | 0.985555 | 0.517778 | 0.033304 | 0.002876 | 0.086363 | 0.413333 | 0.173333 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0_s0p65` | 0.985634 | 0.517778 | 0.033304 | 0.002797 | 0.083996 | 0.426667 | 0.177778 | 0.050741 | `partial_but_insufficient_gap_closure` |
| 8 | `joint_utility_candidate_regret_gate_m0p05_s0p35` | 0.985696 | 0.513333 | 0.033304 | 0.002735 | 0.082129 | 0.735556 | 0.313333 | 0.050741 | `insufficient_no_harm_rejection` |
| 8 | `joint_utility_candidate_regret_gate_m0p02_s0p8` | 0.985744 | 0.511111 | 0.033304 | 0.002688 | 0.080706 | 0.691111 | 0.297778 | 0.050741 | `insufficient_no_harm_rejection` |
| 8 | `joint_utility_candidate_regret_gate_m0p05_s0p5` | 0.985748 | 0.513333 | 0.033304 | 0.002684 | 0.080585 | 0.711111 | 0.306667 | 0.050741 | `insufficient_no_harm_rejection` |
| 8 | `joint_utility_uncertainty_regret_gate_r1p5_m0p0025_s0p8` | 0.985751 | 0.517778 | 0.033304 | 0.002680 | 0.080483 | 0.413333 | 0.173333 | 0.050741 | `partial_but_insufficient_gap_closure` |

## Interpretation

- The CSV keeps all reject-margin/risk-penalty deployment sweep points.
- The table below shows the best policy per K and the strongest overall policies.
- A useful deployed policy needs both high closure and low harmful accepts.
- If this probe underperforms the direct candidate-regret gate, combining object sets, verification signatures, and utility/safety heads is still insufficient while the propagation model remains fixed.

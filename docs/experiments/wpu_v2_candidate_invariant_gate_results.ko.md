# Candidate Invariant Gate 결과

이 문서는 candidate descriptor를 train split에서 표준화하고, train seed별 worst-group loss와 no-harm objective를 함께 줄이는 P1 probe를 요약한다.

Source CSV: `docs/experiments/wpu_v2_candidate_invariant_gate.csv`

최고 closure는 `0.110889` (`K=16`, `invariant_gate_p0p7_m0p01`)다. P1 목표 `0.5`를 기준으로 invariant-gate deployment가 candidate-oracle gap을 충분히 닫는지와 harmful accept를 억제하는지를 동시에 본다. Harmful accept <= `0.25` 조건의 conservative best는 `0.110889` (`invariant_gate_p0p7_m0p01`)다. Train-selected deployed best는 `0.093863` (`K=16`)다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | `invariant_gate_p0p6_m0p02` | 0.984905 | 0.513333 | 0.032827 | 0.003526 | 0.107417 | 0.762222 | 0.326667 | 0.024878 | `insufficient_no_harm_rejection` |
| 16 | `invariant_gate_p0p7_m0p01` | 0.959389 | 0.508889 | 0.061270 | 0.006794 | 0.110889 | 0.615556 | 0.246667 | -0.002001 | `partial_but_insufficient_gap_closure` |
| 32 | `invariant_gate_p0p75_m0p02` | 1.003109 | 0.486667 | 0.035562 | 0.000985 | 0.027709 | 0.408889 | 0.173333 | -0.022165 | `partial_but_insufficient_gap_closure` |
| 16 | `invariant_gate_p0p7_m0p005` | 0.959389 | 0.508889 | 0.061270 | 0.006794 | 0.110883 | 0.617778 | 0.248889 | -0.002001 | `partial_but_insufficient_gap_closure` |
| 16 | `invariant_gate_p0p7_m0` | 0.959495 | 0.508889 | 0.061270 | 0.006688 | 0.109156 | 0.622222 | 0.253334 | -0.002001 | `insufficient_no_harm_rejection` |
| 16 | `invariant_gate_p0p7_m0p001` | 0.959495 | 0.508889 | 0.061270 | 0.006688 | 0.109156 | 0.622222 | 0.253334 | -0.002001 | `insufficient_no_harm_rejection` |
| 16 | `invariant_gate_p0p7_m0p0025` | 0.959495 | 0.508889 | 0.061270 | 0.006688 | 0.109156 | 0.622222 | 0.253334 | -0.002001 | `insufficient_no_harm_rejection` |
| 16 | `invariant_gate_p0p7_m0p02` | 0.959601 | 0.508889 | 0.061270 | 0.006582 | 0.107426 | 0.608889 | 0.244445 | -0.002001 | `partial_but_insufficient_gap_closure` |
| 8 | `invariant_gate_p0p6_m0p01` | 0.985010 | 0.511111 | 0.032827 | 0.003422 | 0.104237 | 0.764444 | 0.328889 | 0.024878 | `insufficient_no_harm_rejection` |
| 8 | `invariant_gate_p0p6_m0` | 0.985057 | 0.511111 | 0.032827 | 0.003375 | 0.102805 | 0.766667 | 0.331111 | 0.024878 | `insufficient_no_harm_rejection` |
| 8 | `invariant_gate_p0p6_m0p001` | 0.985057 | 0.511111 | 0.032827 | 0.003375 | 0.102805 | 0.766667 | 0.331111 | 0.024878 | `insufficient_no_harm_rejection` |
| 8 | `invariant_gate_p0p6_m0p0025` | 0.985057 | 0.511111 | 0.032827 | 0.003375 | 0.102805 | 0.766667 | 0.331111 | 0.024878 | `insufficient_no_harm_rejection` |
| 8 | `invariant_gate_p0p6_m0p005` | 0.985057 | 0.511111 | 0.032827 | 0.003375 | 0.102805 | 0.766667 | 0.331111 | 0.024878 | `insufficient_no_harm_rejection` |
| 8 | `invariant_gate_p0p55_m0p02` | 0.985330 | 0.508889 | 0.032827 | 0.003102 | 0.094483 | 0.862222 | 0.382222 | 0.024878 | `insufficient_no_harm_rejection` |
| 16 | `train_selected_invariant_gate` | 0.960432 | 0.506667 | 0.061270 | 0.005751 | 0.093863 | 0.777778 | 0.337778 | -0.002001 | `insufficient_no_harm_rejection` |
| 8 | `invariant_gate_p0p55_m0p01` | 0.985434 | 0.506667 | 0.032827 | 0.002997 | 0.091302 | 0.864444 | 0.384445 | 0.024878 | `insufficient_no_harm_rejection` |
| 8 | `invariant_gate_p0p55_m0` | 0.985481 | 0.506667 | 0.032827 | 0.002950 | 0.089871 | 0.866667 | 0.386667 | 0.024878 | `insufficient_no_harm_rejection` |
| 8 | `invariant_gate_p0p55_m0p001` | 0.985481 | 0.506667 | 0.032827 | 0.002950 | 0.089871 | 0.866667 | 0.386667 | 0.024878 | `insufficient_no_harm_rejection` |

## 해석

- CSV에는 모든 reject-margin/risk-penalty deployment sweep을 보존한다.
- 아래 표는 K별 최고 정책과 전체 상위 정책만 보여준다.
- 좋은 정책은 closure만 높으면 부족하고, harmful accept도 낮아야 한다.

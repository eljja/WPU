# Candidate Regret Gate 결과

이 문서는 candidate별 `candidate_loss - learned_loss`를 직접 예측하고, 예측 regret이 충분히 낮을 때만 baseline 대신 선택하는 P1 probe를 요약한다.

Source CSV: `docs/experiments/wpu_v2_candidate_regret_gate.csv`

최고 closure는 `0.308651` (`K=16`, `candidate_regret_gate`)다. 이는 이전 best `0.244220`을 넘지만 P1 목표 `0.5`에는 못 미친다. 특히 harmful accept rate가 높아 regret 예측은 candidate-oracle gap을 줄이기 시작했지만 no-harm rejection은 아직 약하다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Accept | Harmful accept | Regret corr | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | `candidate_regret_gate` | 0.985735 | 0.517778 | 0.032763 | 0.002697 | 0.082306 | 0.982222 | 0.404445 | 0.050928 | `insufficient_no_harm_rejection` |
| 8 | `uncertainty_regret_gate` | 0.988296 | 0.511111 | 0.032763 | 0.000136 | 0.004145 | 0.248889 | 0.126667 | 0.050928 | `partial_but_insufficient_gap_closure` |
| 16 | `candidate_regret_gate` | 0.947425 | 0.535556 | 0.060773 | 0.018758 | 0.308651 | 0.951111 | 0.317778 | 0.118052 | `insufficient_no_harm_rejection` |
| 16 | `uncertainty_regret_gate` | 0.957925 | 0.517778 | 0.060773 | 0.008258 | 0.135886 | 0.395556 | 0.126667 | 0.118052 | `partial_but_insufficient_gap_closure` |
| 32 | `candidate_regret_gate` | 1.001991 | 0.526667 | 0.035577 | 0.002104 | 0.059128 | 0.964444 | 0.428889 | 0.072566 | `insufficient_no_harm_rejection` |
| 32 | `uncertainty_regret_gate` | 1.003947 | 0.486667 | 0.035577 | 0.000148 | 0.004149 | 0.428889 | 0.204444 | 0.072566 | `partial_but_insufficient_gap_closure` |

## 해석

- Candidate-regret target은 margin-only gate보다 강한 신호다.
- K=16에서는 closure가 개선됐지만 K=8/32 generalization은 충분하지 않다.
- 다음 개선은 accept/reject calibration, harmful-candidate penalty, seed/domain perturbation을 학습 objective에 더 강하게 넣는 것이다.

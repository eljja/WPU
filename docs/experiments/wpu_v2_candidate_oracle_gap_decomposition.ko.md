# WPU v2 Candidate-Oracle Gap Decomposition

이 리포트는 priority 1 candidate-oracle gap을 feature variant, causal
working-set size, deployed policy family 단위로 분해한다. 이 문서는 진단
evidence이며, candidate-oracle gap이 해결됐다는 주장이 아니다.

Source CSV:

- `docs/experiments/wpu_v2_retriever_invariant_set_scorer.csv`

Derived CSV:

- `docs/experiments/wpu_v2_candidate_oracle_gap_decomposition.csv`

## 분해 표

| feature variant | K | oracle gain | best non-oracle policy | best closure | risk closure | oracle match | negative policies | failure mode |
|---|---:|---:|---|---:|---:|---:|---:|---|
| role_geometry_family | 8 | 0.032895 | risk_adjusted_selected_mechanism | 0.195451 | 0.195451 | 0.228889 | 0 | partial_gap_closure_only |
| role_geometry_family | 16 | 0.061174 | risk_adjusted_selected_mechanism | 0.244220 | 0.244220 | 0.077778 | 0 | partial_gap_closure_only |
| role_geometry_family | 32 | 0.035547 | risk_adjusted_selected_mechanism | 0.042131 | 0.042131 | 0.113333 | 3 | selection_signal_absent_or_miscalibrated |
| role_geometry_only | 8 | 0.032895 | risk_adjusted_selected_mechanism | 0.113548 | 0.113548 | 0.140000 | 0 | partial_gap_closure_only |
| role_geometry_only | 16 | 0.061174 | risk_adjusted_selected_mechanism | 0.244220 | 0.244220 | 0.077778 | 2 | partial_gap_closure_only |
| role_geometry_only | 32 | 0.035547 | train_selected_mechanism | 0.047178 | -0.012226 | 0.142222 | 4 | selection_signal_absent_or_miscalibrated |

## 해석

Committed aggregate evidence에서 발견된 best non-oracle closure는 `0.244220`이다.
이는 dashboard threshold `0.5`보다 낮다. 따라서 P1 실패는 dashboard가 특정 policy를
빼먹었기 때문이 아니다. 사용 가능한 policy family 전반에서 일부 variant는 negative
closure를 보이고, large-K 조건은 특히 약하다.

약한 cell:

- `role_geometry_family`, K=32: selection_signal_absent_or_miscalibrated
- `role_geometry_only`, K=32: selection_signal_absent_or_miscalibrated

## 다음 기술 목표

다음 P1 구현은 aggregate policy selection 아래로 내려가야 한다. Per-candidate
uncertainty, sample-level no-harm gating, regret target을 candidate scorer 자체에
붙여야 한다. Aggregate policy selector만 바꾸는 방식으로는 gap을 닫기 어렵다.

# WPU v2 Candidate-Oracle Gap Audit

이 audit은 최신 cross-seed invariant-scorer 실험에서 candidate-oracle gap을 다시
계산한다. 목적은 candidate pool 안에 존재하는 oracle gain 중 deployed policy가 얼마나
회수했는지 측정하는 것이다.

Source CSV:

- `docs/experiments/wpu_v2_retriever_invariant_set_scorer.csv`

Derived CSV:

- `docs/experiments/wpu_v2_candidate_oracle_gap_v2.csv`

## 핵심 표

| K | policy | loss | accuracy | candidate-oracle gain | deployed gain | remaining gap | gap closure | oracle match |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | generated plus composition oracle | 0.955536 | 0.555555 | 0.032895 | 0.032895 | 0.000000 | 1.000000 | 1.000000 |
| 8 | risk adjusted selected mechanism | 0.982002 | 0.522222 | 0.032895 | 0.006429 | 0.026466 | 0.195451 | 0.228889 |
| 8 | static learned interaction | 0.988432 | 0.506667 | 0.032895 | 0.000000 | 0.032895 | 0.000000 | 0.004444 |
| 16 | generated plus composition oracle | 0.905009 | 0.580000 | 0.061174 | 0.061174 | 0.000000 | 1.000000 | 1.000000 |
| 16 | risk adjusted selected mechanism | 0.951243 | 0.517778 | 0.061174 | 0.014940 | 0.046234 | 0.244220 | 0.077778 |
| 16 | static learned interaction | 0.966183 | 0.504444 | 0.061174 | 0.000000 | 0.061174 | 0.000000 | 0.002222 |
| 32 | generated plus composition oracle | 0.968548 | 0.577778 | 0.035547 | 0.035547 | 0.000000 | 1.000000 | 1.000000 |
| 32 | risk adjusted selected mechanism | 1.002597 | 0.522222 | 0.035547 | 0.001498 | 0.034049 | 0.042131 | 0.113333 |
| 32 | static learned interaction | 1.004095 | 0.475556 | 0.035547 | 0.000000 | 0.035547 | 0.000000 | 0.004444 |

## 해석

Candidate pool에는 deployed selector가 보통 고르는 working set보다 훨씬 좋은 working
set이 여전히 들어 있다. Risk-adjusted mechanism routing은 `K=8`, `K=16`에서 oracle
gain 일부를 회수하지만, `K=32`에서는 아주 작은 부분만 회수한다. 현재 best closure
fraction은 `0.244220`이다.

따라서 우선순위 1은 해결된 것이 아니다. 현재 positive result는 더 좁다. Explicit state
descriptor와 risk-adjusted mechanism routing은 token processing으로 돌아가지 않고
candidate-oracle gap을 줄이지만, candidate scoring은 아직 oracle headroom 대부분을
사용하지 못한다.

## 다음 기술 목표

- Candidate descriptor만이 아니라 downstream regret과 cross-seed perturbation으로
  candidate scoring을 학습한다.
- Selector 자체의 uncertainty를 추가해 confidence가 낮으면 나쁜 candidate를 고르는 대신
  `K`를 확장하거나 verification을 호출한다.
- 모든 향후 working-set-control 실험에서 gap-closure fraction을 필수 metric으로 보고한다.

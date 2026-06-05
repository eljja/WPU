# WPU v2 Candidate-Oracle Gap Audit

This audit recomputes the current candidate-oracle gap from the latest
cross-seed invariant-scorer experiment. It measures how much of the
available candidate-pool gain is recovered by deployed policy selection.

Source CSV:

- `docs/experiments/wpu_v2_retriever_invariant_set_scorer.csv`

Derived CSV:

- `docs/experiments/wpu_v2_candidate_oracle_gap_v2.csv`

## Key Table

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

## Interpretation

The candidate pool still contains substantially better working sets than
the deployed selector usually chooses. Risk-adjusted mechanism routing
recovers part of the oracle gain at `K=8` and `K=16`, but only a small
fraction at `K=32`. The best current closure fraction is `0.244220`.

This means priority 1 is not solved. The current positive result is
narrower: explicit state descriptors and risk-adjusted mechanism routing
reduce the candidate-oracle gap without returning to token processing,
but candidate scoring still leaves most oracle headroom unused.

## Next Technical Target

- Train candidate scoring from downstream regret with cross-seed
  perturbations rather than only candidate descriptors.
- Add uncertainty on the selector itself, so low-confidence selection can
  expand `K` or invoke verification rather than choosing a bad candidate.
- Report gap-closure fraction as a required metric for all future
  working-set-control experiments.

# Candidate Safety Frontier

이 보고서는 P1 candidate-regret selector의 closure/safety tradeoff를 harmful accept threshold별로 다시 요약한다.

Source CSVs:

- `docs/experiments/wpu_v2_candidate_regret_gate_summary.csv`
- `docs/experiments/wpu_v2_candidate_regret_gate_perturbed_summary.csv`
- `docs/experiments/wpu_v2_candidate_regret_gate_penalty_summary.csv`

Derived CSV:

- `docs/experiments/wpu_v2_candidate_safety_frontier.csv`

| probe | harmful limit | best policy | K | closure | harmful accept | accept | feasible policies |
|---|---:|---|---:|---:|---:|---:|---:|
| direct | 0.05 | `uncertainty_regret_gate_r0p75_m0` | 16 | 0.052464 | 0.028889 | 0.115556 | 12 |
| direct | 0.10 | `uncertainty_regret_gate_r0p5_m0p01` | 16 | 0.081898 | 0.051111 | 0.182222 | 16 |
| direct | 0.15 | `uncertainty_regret_gate` | 16 | 0.135886 | 0.126667 | 0.395556 | 22 |
| direct | 0.20 | `candidate_regret_gate_m0p02` | 16 | 0.235128 | 0.173333 | 0.557778 | 25 |
| direct | 0.25 | `candidate_regret_gate_m0p01` | 16 | 0.327146 | 0.235556 | 0.804444 | 27 |
| direct | 0.30 | `candidate_regret_gate_m0p005` | 16 | 0.329950 | 0.271111 | 0.880000 | 31 |
| perturbed | 0.05 | `uncertainty_regret_gate_r0p75_m0p0025` | 16 | 0.019921 | 0.017778 | 0.071111 | 8 |
| perturbed | 0.10 | `uncertainty_regret_gate` | 16 | 0.154320 | 0.097778 | 0.315556 | 17 |
| perturbed | 0.15 | `uncertainty_regret_gate` | 16 | 0.154320 | 0.097778 | 0.315556 | 17 |
| perturbed | 0.20 | `candidate_regret_gate_m0p02` | 16 | 0.183508 | 0.160000 | 0.500000 | 20 |
| perturbed | 0.25 | `candidate_regret_gate_m0p005` | 16 | 0.329756 | 0.235555 | 0.826667 | 23 |
| perturbed | 0.30 | `candidate_regret_gate_m0p0025` | 16 | 0.339525 | 0.253333 | 0.875556 | 27 |
| penalty | 0.05 | `uncertainty_regret_gate_r0p5_m0p0025` | 16 | 0.074859 | 0.046667 | 0.166667 | 7 |
| penalty | 0.10 | `uncertainty_regret_gate` | 16 | 0.083764 | 0.075556 | 0.222222 | 9 |
| penalty | 0.15 | `uncertainty_regret_gate` | 16 | 0.083764 | 0.075556 | 0.222222 | 10 |
| penalty | 0.20 | `uncertainty_regret_gate` | 16 | 0.083764 | 0.075556 | 0.222222 | 11 |
| penalty | 0.25 | `uncertainty_regret_gate` | 16 | 0.083764 | 0.075556 | 0.222222 | 11 |
| penalty | 0.30 | `uncertainty_regret_gate` | 16 | 0.083764 | 0.075556 | 0.222222 | 13 |

## Interpretation

- P1의 실패는 단순히 threshold를 못 찾은 문제가 아니다.
- 높은 closure를 얻는 구간은 harmful accept가 커지고, harmful accept를 강하게 낮추면 closure가 급격히 줄어든다.
- 따라서 다음 개선은 post-hoc threshold가 아니라 candidate scoring 자체의 ranking, no-harm, uncertainty target을 함께 바꾸어야 한다.

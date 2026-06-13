# Joint Candidate Generator 결과

이 문서는 P1의 다음 단계인 downstream-regret 기반 learned candidate generator를 요약한다. 중요한 구분은 learned-generated oracle headroom과 실제 deployed evaluator 성능을 분리하는 것이다.

Source CSV: `docs/experiments/wpu_v2_joint_candidate_generator.csv`

최고 learned-generator oracle closure는 `0.361251` (`K=16`)다. 하지만 최고 deployed evaluator closure는 `0.042951` (`K=16`)에 그친다.

| K | Static loss | Full oracle loss | Learned-generator oracle closure | Evaluator closure | Evaluator accuracy | Learned-generated selected rate | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 0.988432 | 0.955182 | 0.250284 | 0.039513 | 0.504444 | 0.071111 | `generator_headroom_not_deployable` |
| 16 | 0.966183 | 0.904247 | 0.361251 | 0.042951 | 0.508889 | 0.053334 | `generator_headroom_not_deployable` |
| 32 | 1.004095 | 0.968276 | 0.218557 | -0.011536 | 0.491111 | 0.040000 | `generator_headroom_not_deployable` |

## 해석

- Learned generator는 후보 pool의 oracle headroom을 일부 만든다.
- 하지만 evaluator가 그 후보를 held-out seed에서 안전하게 선택하지 못한다.
- 따라서 P1 병목은 후보 생성 단독이 아니라 후보 생성, 선택, propagation verification을 함께 학습해야 하는 문제다.

# Joint Candidate Generator Results

This report summarizes a P1 probe that trains a learned candidate generator from downstream-regret object membership. It separates learned-generated oracle headroom from deployed evaluator performance.

Source CSV: `docs/experiments/wpu_v2_joint_candidate_generator.csv`

The best learned-generator oracle closure is `0.361251` (`K=16`). The best deployed evaluator closure is only `0.042951` (`K=16`).

| K | Static loss | Full oracle loss | Learned-generator oracle closure | Evaluator closure | Evaluator accuracy | Learned-generated selected rate | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 0.988432 | 0.955182 | 0.250284 | 0.039513 | 0.504444 | 0.071111 | `generator_headroom_not_deployable` |
| 16 | 0.966183 | 0.904247 | 0.361251 | 0.042951 | 0.508889 | 0.053334 | `generator_headroom_not_deployable` |
| 32 | 1.004095 | 0.968276 | 0.218557 | -0.011536 | 0.491111 | 0.040000 | `generator_headroom_not_deployable` |

## Interpretation

- The learned generator creates some candidate-pool oracle headroom.
- The evaluator does not reliably deploy that headroom on held-out seeds.
- P1 is therefore not solved by candidate generation alone; candidate generation, selection, and propagation verification need to be learned together.

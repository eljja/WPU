# N=512 Mechanism-Branch Step/Sample Stress 감사

이 실험은 positive mechanism-branch screen을 stress-test한다. 이전 5-seed short-budget 결과에서는 `wpu-cws-indexed-mechanism-branch`가 zero dense compute를 유지하면서 graph-transformer보다 약간 높았다. 이번 follow-up은 그 결과가 더 큰 training/evaluation budget과 공정한 capacity 비교에서도 유지되는지 확인한다.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_multitrain_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_multitrain_5seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_steps16_samples40_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_trainpool40_steps16_samples40_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_branch_h64_trainpool40_steps16_samples40_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_baselines_h64_trainpool40_steps16_samples40_3seed.csv`

## 구현 보정

첫 stress run은 실험 control 문제를 드러냈다. `scripts/pybullet_shift_generalization.py`에서 `--samples`는 evaluation sample count를 제어하고, training pool은 암묵적으로 `steps * batch_size`에서 산출되고 있었다. 이제 script에 `--train-samples-per-mechanism`을 추가해 training-pool size를 명시적으로 제어한다.

## 결과

| condition | WPU macro accuracy | best baseline macro accuracy | WPU dense compute | WPU win/tie/loss | mean margin |
|---|---:|---:|---:|---:|---:|
| short, 3 seeds, h32 | 0.626191 | 0.540476 | 0.000000 | 6/0/1 | +0.066667 |
| short, 5 seeds, h32 | 0.568571 | 0.548571 | 0.000000 | 4/0/3 | +0.020000 |
| steps16/eval40, h32 | 0.538095 | 0.552381 | 0.000000 | 2/1/4 | -0.038095 |
| trainpool40/steps16/eval40, h32 | 0.534524 | 0.598810 | 0.000000 | 2/0/5 | -0.064286 |
| trainpool40/steps16/eval40, h64 | 0.603571 | 0.622619 | 0.000000 | 0/3/4 | -0.053571 |

h64 WPU는 stress protocol에서 h32 WPU보다 크게 개선되므로 model capacity가 중요하다. 그러나 h64 baseline도 함께 개선되며, serialized-token이 가장 높은 macro accuracy에 도달한다. 따라서 공정한 h64 stress 비교는 WPU accuracy 관점에서 negative다. 다만 WPU는 dense compute `0.000000`을 유지한다.

## 해석

Mechanism-conditioned branch transition은 여전히 의미 있는 architecture 방향이지만, accuracy advantage는 더 큰 training/evaluation budget에서 아직 robust하지 않다. Short-budget positive result는 안정적인 accuracy superiority가 아니라 sparse-efficiency screen으로 해석해야 한다.

중요한 과학적 업데이트는 병목이 이동했다는 점이다. WPU는 mechanism-conditioned branch update를 표현할 수 있지만, 현재 branch transition head는 더 많은 sample과 capacity가 주어질 때 dense/token baseline만큼 잘 scale하지 못한다. 다음 개선은 단순히 데이터를 늘리는 것이 아니라 transition-head expressivity와 optimization을 강화해야 한다.

## 다음 단계

다음 WPU v2 방향은 더 강한 sparse transition model이어야 한다.

- 하나의 additive branch correction head가 아니라 branch-specific transition expert를 둔다.
- edge/force/catch composition을 위해 relation-type-conditioned local message를 사용한다.
- 개선이 label-prior fitting에 그치지 않도록 branch-prior control을 정규화한다.
- transition head를 강화한 뒤 larger-N sweep을 수행한다. 현재 stress evidence만으로는 broad accuracy claim을 지지하지 않는다.

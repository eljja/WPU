# PyBullet Relation-Conditioned Closed-Loop Rollout 감사

이 감사는 현재 가장 강한 relation-conditioned WPU route인
`wpu-cws-indexed-mechanism-relation`이 큰 distractor world에서도 반복
`DeltaState` overlay를 안정적으로 수행하는지 검사한다. 조건은
`background_objects=512`, total `N=517`이다.

이번 버전은 state-integrity뿐 아니라 simulator-resynchronized trajectory metric도 함께
보고한다. trajectory metric은 같은 초기조건에서 PyBullet을 `H * sim_steps`만큼 다시 실행해
만든 target과, WPU가 H번 누적한 state를 비교한다.

Source CSVs:

- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta05_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta025_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta01_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta005_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_finite_projection_3seed.csv`

Derived CSV:

- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_integrity_summary.csv`

## 핵심 H=25 결과

| run | integrity | trajectory MSE | target-object MSE | branch accuracy | violations/step | delta norm |
|---|---:|---:|---:|---:|---:|---:|
| `relation_raw` | 0.250368 | 6.975125 | 922.699696 | 0.208333 | 1.184167 | 5.401195 |
| `relation_finite_projection` | 0.876760 | 1.695024 | 538.635690 | 0.250000 | 0.034167 | 2.686625 |
| `relation_bounded_delta05` | 0.611900 | 1.079404 | 424.235526 | 0.541667 | 0.593333 | 1.556419 |
| `relation_bounded_delta025` | 0.652870 | 0.833795 | 374.035886 | 0.458333 | 0.559167 | 0.967416 |
| `relation_bounded_delta01` | 0.793182 | 0.720660 | 360.705327 | 0.708333 | 0.342500 | 0.437645 |
| `relation_bounded_delta005` | 0.870264 | 0.707117 | 361.358309 | 0.729167 | 0.217500 | 0.234325 |

## 해석

결론은 이전보다 강해졌다. Raw relation WPU는 selected `K = 4.354167`을 유지하지만,
H=25에서 integrity `0.250368`, trajectory MSE `6.975125`, final branch accuracy
`0.208333`으로 붕괴한다. 작은 causal working set만으로 long-horizon dynamics가 자동으로
안정화되지는 않는다.

Finite projection은 memory-safety guard로는 강하지만 예측 모델로는 약하다. H=25 integrity는
`0.876760`까지 오르지만 trajectory MSE는 `1.695024`, target-object MSE는
`538.635690`, branch accuracy는 `0.250000`에 그친다. 따라서 projection은 state를
보호하지만 dynamics를 학습했다는 증거는 아니다.

Bounded delta parameterization은 첫 raw transition positive다. Bound `0.05`에서 H=25
integrity는 `0.870264`, trajectory MSE는 `0.707117`, target-object MSE는
`361.358309`, branch accuracy는 `0.729167`이다. Bound `0.1`도 integrity
`0.793182`, trajectory MSE `0.720660`, branch accuracy `0.708333`을 보인다. 이는
작은 bound가 단순히 world를 under-update해서 validity만 보존한 것이 아니라, raw 및 finite
projection보다 final branch와 trajectory 예측도 개선했음을 뜻한다.

다만 아직 learned physics가 해결된 것은 아니다. 가장 좋은 bounded run도 target-object MSE가
`361.358309`로 크다. 따라서 정확한 P2 결론은 다음과 같다. Bounded state-native transition
parameterization은 relation WPU를 unstable long-horizon model에서 usable sparse rollout
baseline으로 올렸지만, high-fidelity dynamics model이 되려면 adaptive per-feature/per-relation
bound와 simulator-resynchronized trajectory training이 필요하다.

## 다음 조치

- bounded delta를 feature별로 다르게 둔다. position, velocity, type/confidence scalar가 같은 bound를 쓰면 target-object error가 남는다.
- relation type과 mechanism feature에 따라 bound를 다르게 예측한다.
- integrity, trajectory MSE, target-object MSE, branch accuracy를 항상 함께 보고한다.
- finite projection은 safety layer로 유지하되, learned dynamics claim과 분리한다.

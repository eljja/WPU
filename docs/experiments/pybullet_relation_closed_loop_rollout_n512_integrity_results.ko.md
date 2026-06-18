# PyBullet Relation-Conditioned Closed-Loop Rollout 감사

이 감사는 `background_objects=512`, total `N=517` 조건에서
`wpu-cws-indexed-mechanism-relation`의 closed-loop rollout을 평가한다. 이제
state-integrity, simulator-resynchronized trajectory MSE, target-object trajectory MSE,
final branch accuracy를 함께 보고한다. 목적은 bounded transition이 단순 validity뿐 아니라
실제 rollout prediction도 개선하는지 검증하는 것이다.

Source CSVs:

- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta05_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta025_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta01_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta005_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_adaptive_delta025_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_split_delta_p010_v005_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta005_targetloss1_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_finite_projection_3seed.csv`

Derived CSV:

- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_integrity_summary.csv`

## 핵심 H=25 결과

| run | integrity | trajectory MSE | target-object MSE | branch accuracy |
|---|---:|---:|---:|---:|
| `relation_raw` | 0.250368 | 6.975125 | 922.699696 | 0.208333 |
| `relation_finite_projection` | 0.876760 | 1.695024 | 538.635690 | 0.250000 |
| `relation_bounded_delta005` | 0.870264 | 0.707117 | 361.358309 | 0.729167 |
| `relation_bounded_delta01` | 0.793182 | 0.720660 | 360.705327 | 0.708333 |
| `relation_adaptive_delta025` | 0.808068 | 0.739266 | 361.306078 | 0.333333 |
| `relation_split_delta_p010_v005` | 0.879871 | 0.715922 | 361.626218 | 0.541667 |
| `relation_bounded_delta005_targetloss1` | 0.987786 | 0.711838 | 363.368470 | 0.687500 |

## 해석

가장 강한 positive는 여전히 fixed global bounded delta다. Bound `0.05`에서 H=25 integrity는
`0.870264`, trajectory MSE는 `0.707117`, branch accuracy는 `0.729167`이다. 이는 raw
rollout과 finite projection보다 훨씬 낫다.

이번 추가 probe 3개는 negative 또는 mixed다. Learned adaptive bounds(`0.01-0.25`)는
integrity `0.808068`, trajectory MSE `0.739266`이지만 branch accuracy가 `0.333333`으로
떨어진다. Position/velocity split bound(`position=0.10`, `velocity=0.05`)는 integrity를
`0.879871`까지 올리지만 branch accuracy가 `0.541667`로 낮고 target-object MSE도
`361.626218`로 개선되지 않는다. Target-object delta loss weight `1.0`은 integrity를
`0.987786`까지 올리지만 target-object MSE는 `363.368470`으로 악화되고 branch accuracy도
`0.687500`으로 낮아진다.

따라서 target-object bottleneck은 learned bound head, 수동 position/velocity split, scalar
target-object loss reweighting으로 해결되지 않았다. 다음 문제는 transition objective mismatch로
보는 것이 타당하다. One-step target-object MSE는 multi-step cup-centric trajectory fidelity를
보장하지 않는다. 다음 P2 실험은 target-object position/velocity error를 분리하고, bound shape나
single-step loss weight가 아니라 unrolled branch/trajectory-consistent loss를 학습해야 한다.

## 다음 조치

- target-object trajectory MSE를 position과 velocity로 분리한다.
- H-step unrolled trajectory loss를 직접 학습한다.
- branch label consistency와 trajectory loss를 동시에 최적화한다.
- fixed global bound `0.05`를 현재 sparse rollout baseline으로 유지한다.

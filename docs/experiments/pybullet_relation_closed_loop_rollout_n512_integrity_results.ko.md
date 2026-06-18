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
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta005_lr3e4_clip1_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta005_unrolled_h2_4_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_mechanism_target_bounded_delta005_3seed.csv`
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
| `relation_bounded_delta005_lr3e4_clip1` | 0.895263 | 0.711506 | 363.187130 | 0.500000 |
| `relation_bounded_delta005_unrolled_h2_4` | 0.886627 | 0.710751 | 362.829388 | 0.500000 |
| `relation_mechanism_target_bounded_delta005` | 0.862545 | 0.699230 | 357.220733 | 0.750000 |
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
target-object loss reweighting으로 해결되지 않았다. 이번에 추가한 unrolled probe도 같은 결론을
강화한다. Full recurrent unrolled gradient는 non-finite gradient를 만들어 무효였고, 안정화된
truncated H=2/4 unroll은 같은 learning-rate/gradient-clip bounded baseline과 거의 동일했다.
H=25에서 bounded-only `lr=3e-4, clip=1`은 branch accuracy `0.500000`, target-object MSE
`363.187130`이고, truncated unroll은 branch accuracy `0.500000`, target-object MSE
`362.829388`이다.

따라서 다음 문제는 단순 transition objective mismatch보다 더 좁다. One-step target-object MSE나
truncated trajectory loss만으로는 multi-step cup-centric dynamics를 회복하지 못한다.

이번 새 follow-up은 이 가설을 일부 지지한다. `wpu-cws-indexed-mechanism-target`은 기존
relation-conditioned sparse message 위에 branch-weighted target-local transition head를 추가한다.
H=25에서 branch accuracy는 `0.729167`에서 `0.750000`으로, trajectory MSE는 `0.707117`에서
`0.699230`으로, target-object MSE는 `361.358309`에서 `357.220733`으로 개선된다. 개선폭은 작지만
방향은 맞다. 다만 target-object position MSE가 여전히 `709.683800`이므로 high-fidelity dynamics는
아직 해결되지 않았다.

## 다음 조치

- target-object trajectory MSE를 position과 velocity로 계속 분리해 보고한다.
- branch-weighted target-local transition head를 더 큰 seed와 더 어려운 mechanism에서 검증한다.
- target-object transition head가 position dynamics를 더 직접 설명하도록 강화한다.
- relation-conditioned sparse message가 장기 rollout의 target-object dynamics를 직접 설명하도록 학습한다.
- fixed global bound `0.05`를 현재 sparse rollout baseline으로 유지한다.

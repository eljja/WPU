# PyBullet Relation-Conditioned Closed-Loop Rollout 감사

이 감사는 one-step 및 distractor-scaling 실험에서 가장 강했던
`wpu-cws-indexed-mechanism-relation` route가 반복 `DeltaState` overlay에서도 안정적인지
검사한다. 이것은 simulator-resynchronized physics benchmark가 아니라, objectified state가
반복 갱신될 때 validity bound 안에 머무는지 보는 state-integrity diagnostic이다.

Source CSVs:

- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta05_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta025_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta01_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_bounded_delta005_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_train_stride4_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_train_stride8_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_train_stride4_delta1_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_train_stride4_branch01_delta1_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_multihorizon_4_8_12_w1_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_multihorizon_4_8_12_w5_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_multihorizon_4_8_12_w1_clip1_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_delta_scale025_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_delta_scale010_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_delta_norm_strong_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_consistency_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_validity_3seed.csv`
- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_finite_projection_3seed.csv`

Derived CSV:

- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_integrity_summary.csv`

## 프로토콜

- World size: `background_objects=512`, total objects `N=517`.
- Seeds: `11`, `13`, `17`.
- Horizons: `5`, `10`, `25`.
- Models: `wpu-cws-indexed-mechanism-relation`, `graph-transformer`, `serialized-token`.
- Safety follow-up: relation WPU only, `finite_delta_clamp=1.0`, `integrity_projection=true`.

## 핵심 결과

| run | model | H | integrity | violations/step | delta norm | selected K |
|---|---|---:|---:|---:|---:|---:|
| `relation_raw` | `wpu-cws-indexed-mechanism-relation` | 5 | 0.715574 | 0.379167 | 1.662182 | 4.354167 |
| `relation_raw` | `wpu-cws-indexed-mechanism-relation` | 10 | 0.552601 | 0.483333 | 4.552682 | 4.354167 |
| `relation_raw` | `wpu-cws-indexed-mechanism-relation` | 25 | 0.091319 | 3.180833 | 2379159.471470 | 4.354167 |
| `relation_bounded_delta005` | `wpu-cws-indexed-mechanism-relation` | 25 | 0.865848 | 0.216667 | 0.249590 | 4.354167 |
| `relation_bounded_delta01` | `wpu-cws-indexed-mechanism-relation` | 25 | 0.780019 | 0.356667 | 0.486962 | 4.354167 |
| `relation_bounded_delta025` | `wpu-cws-indexed-mechanism-relation` | 25 | 0.650669 | 0.547500 | 1.149149 | 4.354167 |
| `relation_bounded_delta05` | `wpu-cws-indexed-mechanism-relation` | 25 | 0.593354 | 0.586667 | 2.146434 | 4.354167 |
| `relation_multihorizon_w1` | `wpu-cws-indexed-mechanism-relation` | 25 | 0.100000 | 4.458333 | 1000000000.000000 | 4.458333 |
| `relation_multihorizon_w5` | `wpu-cws-indexed-mechanism-relation` | 25 | 0.100000 | 4.458333 | 1000000000.000000 | 4.458333 |
| `relation_multihorizon_w1_clip1` | `wpu-cws-indexed-mechanism-relation` | 25 | 0.100000 | 4.458333 | 1000000000.000000 | 4.458333 |
| `relation_delta_scale010` | `wpu-cws-indexed-mechanism-relation` | 25 | 0.089583 | 2.646667 | 245997.938780 | 4.354167 |
| `relation_delta_scale025` | `wpu-cws-indexed-mechanism-relation` | 25 | 0.089410 | 2.849167 | 610325.001919 | 4.354167 |
| `relation_train_stride4` | `wpu-cws-indexed-mechanism-relation` | 25 | 0.089410 | 3.236667 | 2425517.613951 | 4.458333 |
| `relation_train_stride8` | `wpu-cws-indexed-mechanism-relation` | 25 | 0.086806 | 3.257500 | 2424016.166901 | 4.458333 |
| `relation_train_stride4_delta1` | `wpu-cws-indexed-mechanism-relation` | 25 | 0.085243 | 3.237500 | 2460938.605532 | 4.458333 |
| `relation_train_stride4_branch01_delta1` | `wpu-cws-indexed-mechanism-relation` | 25 | 0.086979 | 3.215000 | 2455636.864591 | 4.458333 |
| `relation_delta_norm_strong` | `wpu-cws-indexed-mechanism-relation` | 25 | 0.087153 | 3.115833 | 2421948.183622 | 4.354167 |
| `relation_validity` | `wpu-cws-indexed-mechanism-relation` | 25 | 0.091319 | 3.176667 | 2379929.526420 | 4.354167 |
| `relation_consistency` | `wpu-cws-indexed-mechanism-relation` | 25 | 0.100000 | 4.354167 | 1000000000.000000 | 4.354167 |
| `relation_raw` | `serialized-token` | 25 | 0.333924 | 0.554167 | 27.408027 | 516.354167 |
| `relation_raw` | `graph-transformer` | 25 | 0.077431 | 6.304167 | 30.461124 | 516.354167 |
| `relation_finite_projection` | `wpu-cws-indexed-mechanism-relation` | 25 | 0.739041 | 0.200833 | 4.051999 | 4.354167 |

## 해석

결과는 양면적이다. Relation-conditioned WPU는 H=5와 H=10에서 작은 selected K를 유지하며
dense/token baseline보다 좋은 integrity를 보인다. 그러나 raw H=25에서는 delta explosion이
발생한다. Selected K가 작다는 사실만으로 long-horizon state stability가 보장되지 않는다.

`finite_delta_clamp + integrity_projection`은 applied state를 안정화해 H=25 integrity를
`0.739041`까지 끌어올린다. 하지만 raw delta norm은 여전히 매우 크므로, 이것은 학습된
장기 안정성이 아니라 state-store safety guard다.

첫 learned-stability ablation들은 붕괴를 해결하지 못했다. Strong delta-norm
regularization은 H=25 integrity `0.087153`에 그쳤고, state-validity training도
`0.091319`에 머문다. Rollout-consistency training은 non-finite delta behavior를 만들며
delta norm `1000000000.000000`으로 penalty 처리됐다. 따라서 단순 scalar regularizer는
현재 collapse를 해결하지 못한다.

Temporal delta scaling도 단독 해결책이 아니다. `rollout_delta_scale=0.25`는 H=25
integrity `0.089410`, `rollout_delta_scale=0.10`은 `0.089583`에 그친다. Delta magnitude는
줄어들지만 one-step target 자체가 안정적인 multi-step transition operator가 되지는 않는다.

짧은 simulator stride target도 첫 구현에서는 negative다. `train_sim_steps=4`와 `8`로
학습하고 `sim_steps=80` rollout에서 평가하면 H=25 integrity는 각각 `0.089410`,
`0.086806`에 그친다. Branch loss를 끄고 delta loss를 `1.0`으로 키운 stride-4 run도
`0.085243`, branch `0.1` + delta `1.0` run도 `0.086979`에 머문다. 따라서 문제는
단순히 target duration이나 loss weight가 아니라, 반복 적용을 직접 학습하는 multi-step 또는
simulator-resynchronized rollout objective가 필요하다는 쪽으로 좁혀진다.

첫 explicit multi-horizon simulator-resynchronized target도 현재 transition head에서는
negative다. 같은 초기조건에 대해 simulator horizon `4/8/12`를 target으로 두고
multi-horizon loss weight `1.0` 또는 `5.0`을 주면 H=25에서 non-finite delta가 발생해
integrity `0.100000`으로 penalty 처리된다. Gradient clipping `1.0`도 결과를 바꾸지 못한다.
따라서 다음 단계는 같은 transition head에 target을 더 붙이는 것이 아니라, 반복 적용에서
bounded delta를 보장하는 transition architecture 또는 unrolled step마다 안정성을 강제하는
training objective다.

첫 positive raw-model 안정성 결과는 bounded delta parameterization에서 나왔다. 예측 후
clip하는 것이 아니라 transition head 내부에서 position/velocity delta를 `tanh` bound로
제한하면 H=25 integrity가 bound `0.5`에서 `0.593354`, `0.25`에서 `0.650669`, `0.1`에서
`0.780019`, `0.05`에서 `0.865848`까지 오른다. 최고 bounded run은 finite projection
safety guard의 `0.739041`도 넘으며, selected K는 `4.354167`로 유지되고 correction,
rollback, rejection, dense fallback을 쓰지 않는다.

다만 이것은 아직 완전한 물리 예측 claim이 아니다. 너무 작은 bound는 world를 덜 갱신해서
validity를 보존할 수 있으므로, 다음 검증은 integrity와 함께 simulator-resynchronized
trajectory error 및 branch accuracy를 같이 봐야 한다.

따라서 다음 WPU 개선 방향은 명확하다.

- one-step branch accuracy와 distractor scaling만으로 rollout claim을 하지 않는다.
- bounded delta parameterization을 유지하면서 trajectory error와 branch accuracy를 함께 검증한다.
- delta norm, constraint violation, branch stability를 accuracy와 같은 1급 metric으로 보고한다.
- safety projection은 필요하지만, 논문에서는 learned long-horizon dynamics와 분리해 기술한다.

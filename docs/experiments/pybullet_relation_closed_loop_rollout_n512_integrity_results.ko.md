# PyBullet Relation-Conditioned Closed-Loop Rollout 감사

이 감사는 one-step 및 distractor-scaling 실험에서 가장 강했던
`wpu-cws-indexed-mechanism-relation` route가 반복 `DeltaState` overlay에서도 안정적인지
검사한다. 이것은 simulator-resynchronized physics benchmark가 아니라, objectified state가
반복 갱신될 때 validity bound 안에 머무는지 보는 state-integrity diagnostic이다.

Source CSVs:

- `docs/experiments/pybullet_relation_closed_loop_rollout_n512_3seed.csv`
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
| `relation_delta_scale010` | `wpu-cws-indexed-mechanism-relation` | 25 | 0.089583 | 2.646667 | 245997.938780 | 4.354167 |
| `relation_delta_scale025` | `wpu-cws-indexed-mechanism-relation` | 25 | 0.089410 | 2.849167 | 610325.001919 | 4.354167 |
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

따라서 다음 WPU 개선 방향은 명확하다.

- one-step branch accuracy와 distractor scaling만으로 rollout claim을 하지 않는다.
- transition head를 multi-step loss 또는 simulator-resynchronized target으로 학습한다.
- delta norm, constraint violation, branch stability를 accuracy와 같은 1급 metric으로 보고한다.
- safety projection은 필요하지만, 논문에서는 learned long-horizon dynamics와 분리해 기술한다.

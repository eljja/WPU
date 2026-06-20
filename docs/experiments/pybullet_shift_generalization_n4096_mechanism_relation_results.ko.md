# N=4096 Relation-Conditioned Sparse Propagation 경계 감사

이 감사는 relation-conditioned sparse propagation distractor screen을
`background_objects=4096`, total objects `N=4101`까지 확장한 결과다.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization_n4096_mechanism_relation_trainpool40_steps16_samples40_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n4096_baseline_feasibility_smoke.csv`

## Protocol

- Domain: PyBullet cup/table/hand/edge branch prediction.
- Training mechanisms: `nominal`, `high_force`, `edge_shift`, `catch_heavy`,
  `no_catch`.
- Evaluation mechanisms: 위 5개 training family와 `edge_high_force`,
  `edge_catch_heavy`.
- World size: `background_objects=4096`, total objects `N=4101`.
- WPU stress setting: `train_samples_per_mechanism=40`, `steps=16`,
  `samples=40`.
- WPU seeds: `11`, `13`, `17`.
- WPU model: `wpu-cws-indexed-mechanism-relation`.

동일 조건의 dense/token sweep는 `graph-transformer`, `serialized-token`으로
시도했지만 WPU rows 작성 후 full stress protocol에서 완료되지 않았다. 따라서 별도의
minimal CPU baseline smoke는 feasibility 확인용이며, 정확도 비교 근거가 아니다.

## WPU 결과

| model | macro branch accuracy | ECE | selected K | dense compute ratio |
|---|---:|---:|---:|---:|
| `wpu-cws-indexed-mechanism-relation` | 0.638095 | 0.187201 | 4.446429 | 0.000000 |

Per-mechanism WPU summary:

| eval mechanism | branch accuracy | majority accuracy | ECE | selected K |
|---|---:|---:|---:|---:|
| `catch_heavy` | 0.683333 | 0.783333 | 0.240258 | 4.816667 |
| `edge_catch_heavy` | 0.508333 | 0.583333 | 0.254894 | 4.816667 |
| `edge_high_force` | 0.725000 | 0.583333 | 0.185066 | 4.366667 |
| `edge_shift` | 0.633333 | 0.558333 | 0.181751 | 4.366667 |
| `high_force` | 0.633333 | 0.400000 | 0.177789 | 4.366667 |
| `no_catch` | 0.625000 | 0.683333 | 0.114365 | 4.025000 |
| `nominal` | 0.658333 | 0.458333 | 0.156283 | 4.366667 |

## Baseline Feasibility Smoke

Minimal CPU smoke는 one seed, one training step, one eval mechanism, eight eval
samples, `hidden_dim=16`, one layer, two heads, batch size `2` 조건이다.
`graph-transformer`와 `serialized-token` 모두 완료됐으므로 code path는 feasible하다.
하지만 이 결과는 WPU stress protocol과 비교 가능한 accuracy evidence가 아니다.

| model | eval mechanism | branch accuracy | ECE | dense compute ratio |
|---|---|---:|---:|---:|
| `graph-transformer` | `nominal` | 0.625000 | 0.317135 | 1.000000 |
| `serialized-token` | `nominal` | 0.625000 | 0.207959 | 1.000000 |

## 해석

이 결과는 systems boundary result이지 baseline victory가 아니다. 의미는
relation-conditioned WPU sparse path가 `N=4101`에서도 작은 selected working set으로
동작한다는 점이다. 반면 동일 stress 조건의 dense/token baseline 비교는 아직 확보되지 않았다.

따라서 claim은 제한적이어야 한다. WPU가 모든 large-N world에서 더 좋다는 증거가 아니다.
긍정적 근거는 explicit objectification과 relation-conditioned sparse propagation이 compute를
전체 non-causal distractor object 수가 아니라 causal working set에 묶어둘 수 있다는 점이다.
미해결 문제는 baseline-complete large-N 실험과, working set 자체가 커지는 harder causal
large-N 조건에서도 같은 장점이 유지되는지다.

## 다음 단계

- large-N benchmark runner를 resume/checkpoint 가능하게 만들어 dense/token baseline을
  별도 실행해도 완료된 WPU rows를 잃지 않게 한다.
- large-N 결과를 baseline-complete, WPU-only sparse feasibility, baseline feasibility
  smoke로 명확히 분리해 보고한다.
- non-causal distractor에서 multiple cups, longer relation chains, changing working sets,
  long-horizon rollouts 같은 causal large-N stress로 이동한다.

# N_bg=512 Mechanism Adapter Multi-Mechanism Training Results

이 보고서는 nominal-only mechanism-conditioned screen이 무너진 뒤의 후속 결과다.
목표는 local propagation module이 primitive mechanism을 학습한 뒤 primitive 및 composed
mechanism shift에서 WPU 이점이 생기는지 확인하는 것이다. 결론은 주장을 좁힌다.
유용한 regime은 근거 없는 zero-shot mechanism extrapolation이 아니라, training 중
primitive mechanism 변형을 본 상태에서 objectified state 위의 object-wise local-law
adapter가 동작하는 경우다.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization_n512_mechanism_conditioned_5seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_adapter_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_adapter_multitrain_3seed.csv`
- `docs/experiments/pybullet_shift_generalization_n512_mechanism_adapter_multitrain_5seed.csv`

## 무엇이 바뀌었나

이전 `wpu-cws-indexed-mechanism-conditioned`는 하나의 global mechanism context vector를
모든 selected object에 더한다. 새 `wpu-cws-indexed-mechanism-adapter`는 각 selected
object마다 다음 입력을 쓰는 sparse object-wise adapter를 적용한다.

```text
[selected object embedding, selected raw object features, route physics context]
```

Dense compute는 여전히 `0.000000`이지만, transition function이 selected object별로 다른
mechanism-conditioned update를 적용할 수 있다.

## Nominal-Only 확장은 negative

초기 4-shift positive screen은 더 넓은 평가에서 유지되지 않았다. 5-seed,
7-mechanism, N_bg=512 nominal-train screen에서 global-context mechanism-conditioned WPU는
macro accuracy `0.433333`이고, best non-WPU baseline은 `0.476190`이다. Best baseline
대비 win/tie/loss는 `2/0/5`다.

Object-wise adapter도 nominal-only training에서는 negative다. 3-seed screen에서 macro
accuracy는 `0.380952`이고 graph-transformer는 `0.476190`이다. Win/tie/loss는 `0/1/6`이다.
즉 architecture만으로 held-out mechanism-law shift를 해결한다는 가설은 약해졌다.

## Multi-Mechanism Training은 조건부 positive

Multi-mechanism protocol은 `nominal`, `high_force`, `edge_shift`, `catch_heavy`,
`no_catch`를 학습하고, 이 mechanism들과 composed `edge_high_force`,
`edge_catch_heavy`를 평가한다.

| Model | Mean branch accuracy | Mean ECE | Mean Brier | Mean NLL | Mean dense compute |
|---|---:|---:|---:|---:|---:|
| `graph-transformer` | 0.458571 | 0.248788 | 0.612642 | 1.003195 | 1.000000 |
| `serialized-token` | 0.472857 | 0.243629 | 0.648790 | 1.068664 | 1.000000 |
| `wpu-cws-indexed-mechanism-adapter` | 0.497143 | 0.243597 | 0.652587 | 1.078440 | 0.000000 |
| `wpu-cws-indexed-sparse` | 0.364286 | 0.179114 | 0.668222 | 1.102699 | 0.000000 |

Mechanism adapter는 dense fallback 없이 macro accuracy가 가장 높다. 하지만 calibration은
baseline보다 명확히 좋지 않고, NLL은 graph-transformer보다 나쁘다. 따라서 이 결과는
probability-quality 결과가 아니라 accuracy/compute 결과로 제한해야 한다.

## Per-Mechanism Boundary

| Mechanism | Mechanism adapter | Best baseline | Delta | 경계 |
|---|---:|---:|---:|---|
| `catch_heavy` | 0.680000 | 0.480000 | +0.200000 | 강한 positive다. Object-wise mechanism context가 catch-prior shift에 도움을 준다. |
| `edge_catch_heavy` | 0.340000 | 0.480000 | -0.140000 | 실패다. Edge+catch composed law는 아직 학습되지 않았다. |
| `edge_high_force` | 0.480000 | 0.480000 | +0.000000 | 동률이다. |
| `edge_shift` | 0.410000 | 0.470000 | -0.060000 | 실패다. Edge geometry law가 여전히 약하다. |
| `high_force` | 0.580000 | 0.520000 | +0.060000 | positive다. |
| `no_catch` | 0.410000 | 0.490000 | -0.080000 | 실패다. |
| `nominal` | 0.580000 | 0.520000 | +0.060000 | positive다. |

Best non-WPU baseline 대비 win/tie/loss는 `3/1/3`, mean margin은 `+0.005714`다. 이는
좁은 positive result다. WPU v2의 다음 방향을 지지하지만 broad mechanism generalization을
증명하지는 않는다.

## Interpretation

중요한 발견은 WPU가 token/graph model을 보편적으로 이긴다는 것이 아니다. 그렇지 않다.
중요한 발견은 training set에 primitive mechanism variation이 있을 때 state-native
object-wise adapter가 dense recompute 없이 작은 large-N accuracy edge를 만들 수 있다는
점이다. 이는 nominal-only zero-shot extrapolation보다 더 방어 가능한 WPU v2 주장이다.

남은 실패 모드는 구체적이다.

- Edge composition이 약하다. 특히 `edge_catch_heavy`가 실패한다.
- Calibration은 해결되지 않았다. Adapter의 ECE는 baseline과 비슷하지만 NLL은
  graph-transformer보다 나쁘다.
- 결과는 여전히 하나의 simulator family, one-step prediction, small model, N_bg=512에
  제한된다.

## Next Step

다음 우선순위는 object-wise mechanism adapter를 explicit composition objective와
calibration loss로 학습하는 것이다. 목표는 token processing으로 돌아가는 것이 아니라,
sparse state propagation이 object relation 위에서 compose되는 local mechanism family를
학습하도록 만드는 것이다.

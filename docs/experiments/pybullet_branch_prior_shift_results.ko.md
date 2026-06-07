# PyBullet Branch-Prior Shift Audit

이 분석은 PyBullet mechanism-family shift에서 실패 원인이 relation/propagation 구조인지, 아니면 branch label prior 변화인지 분리한다. `majority_accuracy`는 해당 eval mechanism에서 가장 흔한 branch만 항상 예측하는 비학습 prior baseline이다.

Source CSV:

- `docs/experiments/pybullet_shift_generalization.csv`

Derived CSV:

- `docs/experiments/pybullet_branch_prior_shift.csv`

| mechanism | best WPU | best baseline | WPU acc | baseline acc | majority acc | WPU-baseline | majority-WPU | prior dominated |
|---|---|---|---:|---:|---:|---:|---:|---|
| catch_heavy | `wpu-cws-indexed-local-dense` | `serialized-token` | 0.408730 | 0.349206 | 0.753968 | 0.059524 | 0.345238 | True |
| edge_shift | `wpu-cws-indexed-sparse` | `serialized-token` | 0.527778 | 0.571428 | 0.515873 | -0.043650 | -0.011905 | False |
| high_force | `wpu-cws-indexed-local-dense` | `serialized-token` | 0.432540 | 0.460318 | 0.424603 | -0.027778 | -0.007937 | False |
| nominal | `wpu-cws-indexed-sparse` | `serialized-token` | 0.444445 | 0.500000 | 0.468254 | -0.055555 | 0.023809 | False |

## Interpretation

- Shift mechanism 기준 평균 WPU-baseline accuracy delta는 `-0.003968`다.
- Shift mechanism 기준 majority-prior와 best WPU의 평균 gap은 `0.108465`다.
- Majority prior가 best WPU와 best baseline을 모두 이기는 prior-dominated mechanism은 `1/3`개다.
- Prior-dominated mechanism에서는 더 큰 propagation block보다 mechanism-aware branch prior, branch-frequency shift detector, uncertainty-gated recompute가 먼저 필요하다.
- 이 결과는 WPU 주장을 좁힌다. 객체화와 sparse propagation이 충분해도, branch prior가 바뀌면 state processor는 명시적인 prior adaptation 없이는 실패할 수 있다.

## Mechanism Consequence

Prior-dominated mechanism은 `catch_heavy`이다. 이 구간은 WPU v2에서 P4/P5의 핵심 반례로 유지해야 한다.

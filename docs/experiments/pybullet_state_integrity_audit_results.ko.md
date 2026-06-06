# PyBullet State-Integrity Audit

이 audit은 PyBullet closed-loop rollout 결과에서 long-horizon state-integrity metric을
계산한다. Simulator에 재동기화하지 않고, 반복 적용된 `DeltaState` overlay가 object state를
간단한 validity bound 안에 유지하는지 평가한다.

Source CSVs:

- `docs/experiments/pybullet_closed_loop_rollout.csv`
- `docs/experiments/pybullet_closed_loop_rollout_clipped.csv`
- `docs/experiments/pybullet_closed_loop_rollout_guarded.csv`
- `docs/experiments/pybullet_closed_loop_rollout_regularized.csv`
- `docs/experiments/pybullet_closed_loop_rollout_rejected.csv`
- `docs/experiments/pybullet_closed_loop_rollout_consistency.csv`
- `docs/experiments/pybullet_closed_loop_rollout_validity.csv`
- `docs/experiments/pybullet_closed_loop_rollout_validity_strong.csv`
- `docs/experiments/pybullet_closed_loop_rollout_rollback.csv`

Derived CSV:

- `docs/experiments/pybullet_state_integrity_audit.csv`

## 요약

| run | model | H | violations/step | delta norm | flip rate | reject rate | rollback rate | integrity score |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| clipped | graph-transformer | 25 | 0.253333 | 8.363635 | 0.054688 | 0.000000 | 0.000000 | 0.557002 |
| clipped | wpu-cws-indexed-local-dense | 25 | 0.314167 | 2.809900 | 0.048611 | 0.000000 | 0.000000 | 0.719139 |
| clipped | wpu-cws-indexed-sparse | 25 | 0.785000 | 1939290.233702 | 0.082465 | 0.000000 | 0.000000 | 0.201757 |
| consistency | wpu-cws-indexed-local-dense | 25 | 0.811667 | 1.965078 | 0.059896 | 0.000000 | 0.000000 | 0.472827 |
| consistency | wpu-cws-indexed-sparse | 25 | 3.360834 | 1775082.311771 | 0.077257 | 0.000000 | 0.000000 | 0.084549 |
| guarded | graph-transformer | 25 | 0.000000 | 2.096666 | 0.054688 | 0.000000 | 0.000000 | 0.915679 |
| guarded | wpu-cws-indexed-local-dense | 25 | 0.000000 | 0.741597 | 0.048611 | 0.000000 | 0.000000 | 0.964322 |
| guarded | wpu-cws-indexed-sparse | 25 | 0.000000 | 0.709288 | 0.083334 | 0.000000 | 0.000000 | 0.958508 |
| raw | graph-transformer | 5 | 0.045833 | 4.162467 | 0.062500 | 0.000000 | 0.000000 | 0.816605 |
| raw | graph-transformer | 10 | 0.139583 | 6.106724 | 0.150463 | 0.000000 | 0.000000 | 0.679401 |
| raw | graph-transformer | 25 | 0.272500 | 8.409911 | 0.056424 | 0.000000 | 0.000000 | 0.544493 |
| raw | wpu-cws-indexed-local-dense | 5 | 0.066667 | 2.788842 | 0.145834 | 0.000000 | 0.000000 | 0.836557 |
| raw | wpu-cws-indexed-local-dense | 10 | 0.204167 | 2.648499 | 0.148148 | 0.000000 | 0.000000 | 0.765381 |
| raw | wpu-cws-indexed-local-dense | 25 | 0.499166 | 2.744688 | 0.055556 | 0.000000 | 0.000000 | 0.618283 |
| raw | wpu-cws-indexed-sparse | 5 | 0.200000 | 0.799330 | 0.125000 | 0.000000 | 0.000000 | 0.837023 |
| raw | wpu-cws-indexed-sparse | 10 | 0.531250 | 3.534072 | 0.203704 | 0.000000 | 0.000000 | 0.543379 |
| raw | wpu-cws-indexed-sparse | 25 | 3.374166 | 1958877.607881 | 0.076389 | 0.000000 | 0.000000 | 0.084722 |
| regularized | wpu-cws-indexed-local-dense | 25 | 0.536667 | 1.915983 | 0.044271 | 0.000000 | 0.000000 | 0.628920 |
| regularized | wpu-cws-indexed-sparse | 25 | 3.316667 | 1797100.815468 | 0.064237 | 0.000000 | 0.000000 | 0.087153 |
| rejected | graph-transformer | 25 | 0.271666 | 3.406922 | 0.053819 | 0.359166 | 0.000000 | 0.720577 |
| rejected | wpu-cws-indexed-local-dense | 25 | 0.499166 | 2.277815 | 0.055556 | 0.000000 | 0.000000 | 0.634624 |
| rejected | wpu-cws-indexed-sparse | 25 | 0.785834 | 0.635544 | 0.076389 | 0.640000 | 0.000000 | 0.530270 |
| validity | wpu-cws-indexed-local-dense | 25 | 0.605833 | 2.222097 | 0.054688 | 0.000000 | 0.000000 | 0.578081 |
| validity | wpu-cws-indexed-sparse | 25 | 3.374166 | 1785671.546102 | 0.076389 | 0.000000 | 0.000000 | 0.084722 |
| validity_strong | wpu-cws-indexed-local-dense | 25 | 0.710833 | 2.212986 | 0.055556 | 0.000000 | 0.000000 | 0.520476 |
| validity_strong | wpu-cws-indexed-sparse | 25 | 3.374166 | 1785671.546102 | 0.076389 | 0.000000 | 0.000000 | 0.084722 |
| rollback | graph-transformer | 25 | 0.000000 | 4.140561 | 0.057292 | 0.000000 | 0.261667 | 0.843622 |
| rollback | wpu-cws-indexed-local-dense | 25 | 0.000000 | 1.225809 | 0.052951 | 0.000000 | 0.499166 | 0.946506 |
| rollback | wpu-cws-indexed-sparse | 25 | 0.000000 | 0.150753 | 0.030382 | 0.000000 | 0.812500 | 0.988647 |

## 해석

이 audit은 one-step branch accuracy만으로 world-state processor를 평가할 수 없다는 점을
확인한다. Sparse WPU path는 작은 selected `K`를 유지할 수 있지만, raw delta를 반복 적용하면
invalid state를 만들 수 있다. Guarded state-store projection은 applied state를 보호해 WPU
H=25 integrity를 dashboard threshold 위로 올린다. 그러나 source CSV의 guarded sparse run은
projection 이전 raw delta norm이 여전히 약 `1.9e6`임을 기록하므로, underlying delta model이
안정적이라는 증거는 아니다.

Regularized run은 학습 단계에 target-relative delta-norm penalty를 추가한 raw rollout이다.
현재 결과에서는 sparse H=25 integrity가 `0.084722`에서 `0.087153`으로만 올라가고,
local-dense H=25도 `0.618283`에서 `0.628920`으로 소폭 개선된다. 따라서 단순 delta-norm
regularization은 raw model-delta instability를 해결하지 못한다.

Unsafe-delta rejection run은 state-store safety mechanism이다. Sparse WPU H=25
integrity는 `0.530270`까지 올라가지만 update의 `0.640000`을 거부한다. 따라서 이는
raw transition model이 안정적이라는 증거가 아니라, memory layer가 위험한 update를
거부해 applied state를 보호했다는 증거다.

Rollout-consistency run은 학습 중 두 번째 step의 delta growth를 줄이도록 penalty를
추가했다. 현재 결과에서는 sparse H=25 integrity가 `0.084549`로 raw `0.084722`와
거의 같아 naive consistency penalty가 raw-delta instability를 해결하지 못한다.

State-validity run은 예측된 다음 상태가 position/velocity bound와 cup floor bound를
위반하지 않도록 training loss를 추가했다. 그러나 sparse H=25 integrity는 `0.084722`로
raw와 같고, strong-validity에서도 개선되지 않았다. Local-dense도 `0.578081` 및
`0.520476`으로 raw `0.618283`보다 낮다. 따라서 현재 v2 증거에서는 단순 validity
regularization만으로는 long-horizon state 안정성을 만들 수 없고, guarded projection,
unsafe-delta rejection, rollback/correction 같은 memory-layer safety가 필요하다.

Rollback run은 constraint violation이 늘어나는 update를 되돌리는 memory-layer correction이다.
Sparse WPU H=25 integrity는 `0.988647`까지 올라가지만 rollback rate가 `0.812500`으로
매우 높다. 따라서 rollback은 applied state를 강하게 보호한다는 증거지만, raw transition
model이 안정적이라는 증거는 아니다. 논문에서는 rollback/correction을 state memory safety
mechanism으로 보고하고, raw delta stability와 분리해야 한다.

따라서 state integrity는 WPU의 first-class metric이어야 한다.

```text
state-integrity = constraint validity + bounded delta drift + branch stability
```

향후 WPU rollout claim은 accuracy와 latency 옆에 이 score 또는 그 구성 metric을 함께
보고해야 하며, raw model delta와 guarded state-store delta를 구분해야 한다.

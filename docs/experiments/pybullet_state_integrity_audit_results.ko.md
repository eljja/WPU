# PyBullet State-Integrity Audit

이 audit은 PyBullet closed-loop rollout 결과에서 long-horizon state-integrity metric을
계산한다. Simulator에 재동기화하지 않고, 반복 적용된 `DeltaState` overlay가 object state를
간단한 validity bound 안에 유지하는지 평가한다.

Source CSVs:

- `docs/experiments/pybullet_closed_loop_rollout.csv`
- `docs/experiments/pybullet_closed_loop_rollout_clipped.csv`

Derived CSV:

- `docs/experiments/pybullet_state_integrity_audit.csv`

## 요약

| run | model | H | violations/step | delta norm | flip rate | integrity score |
|---|---|---:|---:|---:|---:|---:|
| clipped | graph-transformer | 25 | 0.253333 | 8.363635 | 0.054688 | 0.557002 |
| clipped | wpu-cws-indexed-local-dense | 25 | 0.314167 | 2.809900 | 0.048611 | 0.719139 |
| clipped | wpu-cws-indexed-sparse | 25 | 0.785000 | 1939290.233702 | 0.082465 | 0.201757 |
| raw | graph-transformer | 5 | 0.045833 | 4.162467 | 0.062500 | 0.816605 |
| raw | graph-transformer | 10 | 0.139583 | 6.106724 | 0.150463 | 0.679401 |
| raw | graph-transformer | 25 | 0.272500 | 8.409911 | 0.056424 | 0.544493 |
| raw | wpu-cws-indexed-local-dense | 5 | 0.066667 | 2.788842 | 0.145834 | 0.836557 |
| raw | wpu-cws-indexed-local-dense | 10 | 0.204167 | 2.648499 | 0.148148 | 0.765381 |
| raw | wpu-cws-indexed-local-dense | 25 | 0.499166 | 2.744688 | 0.055556 | 0.618283 |
| raw | wpu-cws-indexed-sparse | 5 | 0.200000 | 0.799330 | 0.125000 | 0.837023 |
| raw | wpu-cws-indexed-sparse | 10 | 0.531250 | 3.534072 | 0.203704 | 0.543379 |
| raw | wpu-cws-indexed-sparse | 25 | 3.374166 | 1958877.607881 | 0.076389 | 0.084722 |

## 해석

이 audit은 one-step branch accuracy만으로 world-state processor를 평가할 수 없다는 점을
확인한다. Sparse WPU path는 작은 selected `K`를 유지할 수 있지만, raw delta를 반복 적용하면
invalid state를 만들 수 있다. Delta clipping은 violation을 줄여 integrity score를 개선하지만,
underlying delta model이 안정적이라는 증거는 아니다.

따라서 state integrity는 WPU의 first-class metric이어야 한다.

```text
state-integrity = constraint validity + bounded delta drift + branch stability
```

향후 WPU rollout claim은 accuracy와 latency 옆에 이 score 또는 그 구성 metric을 함께
보고해야 한다.

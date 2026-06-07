# PyBullet State-Integrity Audit

이 문서는 PyBullet closed-loop rollout 결과에서 long-horizon state-integrity
metric을 계산한 것이다. Simulator에 다시 동기화하지 않고, 반복적인 `DeltaState`
overlay가 단순 validity bound 안에서 object state를 유지하는지 본다.

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
- `docs/experiments/pybullet_closed_loop_rollout_corrected_rollback.csv`

Derived CSV:

- `docs/experiments/pybullet_state_integrity_audit.csv`

## 핵심 결과

| run | model | H | violations/step | delta norm | correction rate | rollback rate | integrity score |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | wpu-cws-indexed-sparse | 25 | 3.374166 | 1958877.607881 | 0.000000 | 0.000000 | 0.084722 |
| guarded | wpu-cws-indexed-sparse | 25 | 0.000000 | 0.709288 | 0.000000 | 0.000000 | 0.958508 |
| rejected | wpu-cws-indexed-sparse | 25 | 0.785834 | 0.635544 | 0.000000 | 0.000000 | 0.530270 |
| rollback | wpu-cws-indexed-sparse | 25 | 0.000000 | 0.150753 | 0.000000 | 0.812500 | 0.988647 |
| corrected_rollback | wpu-cws-indexed-sparse | 25 | 0.000000 | 2.392552 | 0.812500 | 0.564167 | 0.900288 |
| rollback | wpu-cws-indexed-local-dense | 25 | 0.000000 | 1.225809 | 0.000000 | 0.499166 | 0.946506 |
| corrected_rollback | wpu-cws-indexed-local-dense | 25 | 0.000000 | 2.263392 | 0.499166 | 0.000000 | 0.909670 |
| rollback | graph-transformer | 25 | 0.000000 | 4.140561 | 0.000000 | 0.261667 | 0.843622 |
| corrected_rollback | graph-transformer | 25 | 0.000000 | 5.756884 | 0.268334 | 0.000000 | 0.787224 |

전체 audit table은 `docs/experiments/pybullet_state_integrity_audit.csv`에 있다.

## 해석

이 audit은 one-step branch accuracy만으로 world-state processor를 평가할 수 없음을
보인다. Sparse WPU는 작은 `K`를 유지할 수 있지만, raw delta를 반복 적용하면 horizon
25에서 invalid state와 delta explosion이 발생한다.

Guarded state-store projection은 applied state를 보호해 sparse WPU integrity를
`0.958508`까지 올린다. 그러나 이는 raw transition model이 안정적이라는 뜻이 아니다.
Unsafe-delta rejection도 integrity를 `0.530270`까지 올리지만 update의 `0.640000`을
거부하므로, integrity와 rejection rate를 함께 보고해야 한다.

Rollback-only memory layer는 sparse WPU H=25 applied-state integrity를 `0.988647`까지
올리지만 rollback rate가 `0.812500`으로 매우 높다. Corrected rollback은 violation이
증가한 state를 먼저 bounded projection으로 수선하고, 그래도 이전 state보다 나쁘면
rollback한다. 이 방식은 sparse rollback rate를 `0.564167`까지 낮추지만 integrity는
`0.900288`로 떨어진다.

따라서 현재 결론은 명확하다. Rollback과 correction은 state memory safety mechanism이지,
raw dynamics가 해결됐다는 증거가 아니다. 다음 단계는 rollback 빈도를 낮추면서
integrity를 유지하는 learned correction, uncertainty escalation, state-consistency
loss다.

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
- `docs/experiments/pybullet_closed_loop_rollout_escalated_corrected_rollback.csv`
- `docs/experiments/pybullet_closed_loop_rollout_finite_clamped.csv`
- `docs/experiments/pybullet_closed_loop_rollout_finite_corrected.csv`
- `docs/experiments/pybullet_closed_loop_rollout_selective_corrected.csv`
- `docs/experiments/pybullet_closed_loop_rollout_selective_corrected_stride2.csv`
- `docs/experiments/pybullet_closed_loop_rollout_selective_corrected_margin1.csv`

Derived CSV:

- `docs/experiments/pybullet_state_integrity_audit.csv`

## 핵심 결과

| run | model | H | violations/step | delta norm | correction rate | corrected objects | rollback rate | escalation rate | escalation success | integrity score | low-disruption score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | wpu-cws-indexed-sparse | 25 | 3.374166 | 1958877.607881 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.084722 | 0.084722 |
| guarded | wpu-cws-indexed-sparse | 25 | 0.000000 | 0.709288 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.958508 | 0.958508 |
| rejected | wpu-cws-indexed-sparse | 25 | 0.785834 | 0.635544 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.530270 | 0.402270 |
| rollback | wpu-cws-indexed-sparse | 25 | 0.000000 | 0.150753 | 0.000000 | 0.000000 | 0.812500 | 0.000000 | 0.000000 | 0.988647 | 0.744897 |
| corrected_rollback | wpu-cws-indexed-sparse | 25 | 0.000000 | 2.392552 | 0.812500 | 0.812500 | 0.564167 | 0.000000 | 0.000000 | 0.900288 | 0.406038 |
| escalated_corrected_rollback | wpu-cws-indexed-sparse | 25 | 0.000000 | 1.942319 | 0.710833 | 0.710833 | 0.000000 | 0.805833 | 0.116107 | 0.914831 | 0.549915 |
| finite_clamped | wpu-cws-indexed-sparse | 25 | 0.784166 | 0.709270 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.527391 | 0.527391 |
| finite_corrected | wpu-cws-indexed-sparse | 25 | 0.000000 | 0.697858 | 0.784166 | 0.784166 | 0.000000 | 0.000000 | 0.000000 | 0.958735 | 0.645068 |
| selective_corrected | wpu-cws-indexed-sparse | 25 | 0.000000 | 0.697858 | 0.784166 | 0.027461 | 0.000000 | 0.000000 | 0.000000 | 0.958735 | 0.758574 |
| selective_corrected_stride2 | wpu-cws-indexed-sparse | 25 | 0.770000 | 0.709051 | 0.014166 | 0.027371 | 0.000000 | 0.000000 | 0.000000 | 0.535190 | 0.527543 |
| selective_corrected_margin1 | wpu-cws-indexed-sparse | 25 | 0.784166 | 0.709270 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.527391 | 0.527391 |
| rollback | wpu-cws-indexed-local-dense | 25 | 0.000000 | 1.225809 | 0.000000 | 0.000000 | 0.499166 | 0.000000 | 0.000000 | 0.946506 | 0.796756 |
| corrected_rollback | wpu-cws-indexed-local-dense | 25 | 0.000000 | 2.263392 | 0.499166 | 0.499166 | 0.000000 | 0.000000 | 0.000000 | 0.909670 | 0.710004 |
| rollback | graph-transformer | 25 | 0.000000 | 4.140561 | 0.000000 | 0.000000 | 0.261667 | 0.000000 | 0.000000 | 0.843622 | 0.765122 |
| corrected_rollback | graph-transformer | 25 | 0.000000 | 5.756884 | 0.268334 | 0.268334 | 0.000000 | 0.000000 | 0.000000 | 0.787224 | 0.679891 |

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

Escalated corrected rollback은 sparse delta가 violation을 늘릴 때 이전 state로 되돌린 뒤
local-dense WPU fallback으로 같은 event를 재계산한다. 이 방식은 sparse integrity를
`0.914831`로 올리고 rollback rate를 `0.000000`으로 낮춘다. 그러나 escalation rate가
`0.805833`이고 escalation success가 `0.116107`에 그치므로, 이는 raw sparse dynamics의
해결이 아니라 dense-when-needed safety layer의 제한적 양성 증거다.

Finite-clamped run은 non-finite 또는 극단적인 predicted delta를 먼저 정리하고 norm
clipping을 적용한다. 이 방식은 기존 clipped run의 sparse delta-norm explosion을
제거해 delta norm을 `1939290.233702`에서 `0.709270`으로 낮추지만, violations/step은
`0.784166`으로 남아 integrity가 `0.527391`에 그친다. 즉 numerical delta safety와
state validity는 별도의 문제다.

Finite-corrected run은 finite-safe delta clipping과 correction-only projection을 결합한다.
Sparse WPU H=25에서 violations/step `0.000000`, rollback rate `0.000000`, escalation rate
`0.000000`, integrity `0.958735`를 얻었다. 이는 guarded projection `0.958508`과 비슷한
applied-state safety를 rollback 없이 달성한다. 다만 correction rate가 `0.784166`으로
높기 때문에 raw dynamics가 안정화됐다는 뜻은 아니며, memory layer의 bounded local
correction이 unsafe update를 거부하거나 dense recompute하지 않고 state를 보호할 수
있음을 보이는 결과다.

Selective-corrected run은 같은 correction trigger를 유지하되 validity bound를 실제로
위반한 object만 projection한다. Sparse WPU H=25 integrity는 `0.958735`로 유지하면서
corrected object fraction을 `0.027461`까지 낮추고 low-disruption score를 `0.758574`까지
올렸다. 그러나 correction trigger rate는 여전히 `0.784166`이다. Stride-2 gate나
margin-1 gate처럼 trigger 자체를 줄이면 integrity가 각각 `0.535190`, `0.527391`로
무너진다. 따라서 이번 개선은 correction의 범위를 줄인 것이지, raw transition model이
대부분의 sparse update를 스스로 안전하게 만든 것은 아니다.

따라서 현재 결론은 명확하다. Rollback과 correction은 state memory safety mechanism이지,
raw dynamics가 해결됐다는 증거가 아니다. 다음 단계는 correction projection을 더 작게
만드는 것이 아니라, correction trigger 자체를 learned uncertainty/state-validity
objective로 학습하고 raw transition을 안정화해 correction rate를 낮추면서 integrity를
유지하는 것이다.

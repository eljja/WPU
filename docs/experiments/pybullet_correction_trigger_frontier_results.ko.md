# PyBullet Correction-Trigger Frontier 결과

이 감사는 P2의 남은 병목을 분리한다. Selective correction은 corrected-object fraction을 낮췄지만 correction trigger frequency는 높게 남아 있다. 이 표는 trigger를 줄이면 state integrity가 유지되는지 검사한다.

Source CSV: `docs/experiments/pybullet_state_integrity_audit.csv`

Derived CSV: `docs/experiments/pybullet_correction_trigger_frontier.csv`

최고 integrity는 `0.988647` (`rollback`)이고 correction rate는 `0.000000`이다. 최고 low-disruption score는 `0.958508` (`guarded`)이다. Correction-trigger 계열에서 correction rate <= `0.25` 조건의 최고 integrity는 `0.653668` (`selective_corrected_entropy035`)이다. Joint target을 만족한 row 수는 `0`이다.

| run | family | integrity | low-disruption | violations/step | correction | corrected objects | rollback | escalation | joint target |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `finite_clamped` | `no_correction_baseline` | 0.527391 | 0.527391 | 0.784166 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `guarded` | `safety_baseline` | 0.958508 | 0.958508 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `rollback` | `safety_baseline` | 0.988647 | 0.744897 | 0.000000 | 0.000000 | 0.000000 | 0.812500 | 0.000000 | 0 |
| `selective_corrected_margin1` | `correction_trigger` | 0.527391 | 0.527391 | 0.784166 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `selective_corrected_rawdelta2m` | `correction_trigger` | 0.527391 | 0.527391 | 0.784166 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 |
| `selective_corrected_stride2` | `correction_trigger` | 0.535190 | 0.527543 | 0.770000 | 0.014166 | 0.027371 | 0.000000 | 0.000000 | 0 |
| `selective_corrected_entropy045` | `correction_trigger` | 0.642658 | 0.586039 | 0.574167 | 0.210000 | 0.027458 | 0.000000 | 0.000000 | 0 |
| `selective_corrected_entropy035` | `correction_trigger` | 0.653668 | 0.592049 | 0.554167 | 0.230000 | 0.027458 | 0.000000 | 0.000000 | 0 |
| `finite_corrected` | `correction_trigger` | 0.958735 | 0.645068 | 0.000000 | 0.784166 | 0.784166 | 0.000000 | 0.000000 | 0 |
| `selective_corrected` | `correction_trigger` | 0.958735 | 0.758574 | 0.000000 | 0.784166 | 0.027461 | 0.000000 | 0.000000 | 0 |

## 해석

- Entropy gate는 correction rate를 낮추지만 integrity target을 유지하지 못한다.
- Raw-delta threshold는 이 설정에서 유효한 trigger가 아니며 correction을 거의 제거해 실패한다.
- P2의 다음 단계는 더 보수적인 threshold가 아니라 transition model 자체의 안정화 또는 learned trigger다.

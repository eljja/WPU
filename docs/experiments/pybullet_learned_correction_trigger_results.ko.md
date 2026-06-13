# PyBullet Learned Correction Trigger 결과

이 실험은 P2 correction-trigger 병목을 hand-coded threshold가 아니라 held-out seed로 전이되는 작은 MLP trigger로 검사한다. Trigger는 sparse delta 적용 후의 confidence, delta norm, violation 변화, cup state를 보고 selective correction을 실행할지 결정한다.

Source CSV: `docs/experiments/pybullet_learned_correction_trigger.csv`

Joint target(integrity >= 0.8, correction_rate <= 0.25)을 만족한 summary policy는 `0`개다. 최고 integrity는 `0.958931` (`always_selective`)이고 correction rate는 `1.000000`이다. Correction rate <= 0.25 조건의 최고 integrity는 `0.523279` (`no_correction`)이다. 최고 learned trigger integrity는 `0.958931` (`learned_threshold_0.5`)이고 correction rate는 `0.791667`이다. 최고 low-disruption score는 `0.756897` (`learned_threshold_0.5`)이다.

| policy | integrity | low-disruption | violations/step | correction | corrected objects | joint target |
|---|---:|---:|---:|---:|---:|---:|
| `no_correction` | 0.523279 | 0.523279 | 0.791667 | 0.000000 | 0.000000 | 0 |
| `learned_threshold_0.5` | 0.958931 | 0.756897 | 0.000000 | 0.791667 | 0.027452 | 0 |
| `learned_threshold_0.6` | 0.958931 | 0.756897 | 0.000000 | 0.791667 | 0.027452 | 0 |
| `learned_threshold_0.7` | 0.958931 | 0.756897 | 0.000000 | 0.791667 | 0.027452 | 0 |
| `learned_threshold_0.8` | 0.958931 | 0.756897 | 0.000000 | 0.791667 | 0.027452 | 0 |
| `learned_threshold_0.9` | 0.958931 | 0.756897 | 0.000000 | 0.791667 | 0.027452 | 0 |
| `learned_threshold_0.95` | 0.958931 | 0.756897 | 0.000000 | 0.791667 | 0.027452 | 0 |
| `learned_threshold_0.975` | 0.958931 | 0.756897 | 0.000000 | 0.791667 | 0.027452 | 0 |
| `learned_top_rate_0.1` | 0.958931 | 0.752005 | 0.000000 | 0.811667 | 0.026725 | 0 |
| `learned_top_rate_0.15` | 0.958931 | 0.750168 | 0.000000 | 0.819167 | 0.026474 | 0 |
| `learned_top_rate_0.2` | 0.958931 | 0.748943 | 0.000000 | 0.824167 | 0.026311 | 0 |
| `learned_top_rate_0.25` | 0.958931 | 0.748125 | 0.000000 | 0.827500 | 0.026202 | 0 |
| `learned_top_rate_0.3` | 0.958931 | 0.747513 | 0.000000 | 0.830000 | 0.026122 | 0 |
| `always_selective` | 0.958931 | 0.705672 | 0.000000 | 1.000000 | 0.021726 | 0 |

## 해석

- Hard-seed learned trigger는 P2를 해결하지 못했다. 높은 integrity는 correction을 대부분 실행할 때만 유지된다.
- 항상 또는 고빈도 selective correction은 applied state를 보호하지만, 이는 stable raw dynamics가 아니라 memory-safety layer다.
- 다음 단계는 trigger threshold 확장이 아니라 transition loss, state-validity loss, correction objective를 묶은 안정화 학습이다.

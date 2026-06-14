# PyBullet Stable Transition Sweep 결과

이 실험은 P2의 다음 병목인 transition model 안정화를 직접 검사한다. delta-norm, rollout-consistency, state-validity loss 조합으로 WPU sparse transition을 다시 학습한 뒤, correction 없는 finite-clamped rollout과 selective correction rollout을 동시에 평가한다.

Source CSV: `docs/experiments/pybullet_stable_transition_sweep.csv`

최고 raw finite-clamped integrity는 `0.633398`(`delta_norm_strong`)이고 violations/step은 `0.598333`다. 최고 selective-correction low-disruption score는 `0.809071`(`delta_norm_strong`)이고 integrity는 `0.962773`, correction rate는 `0.598333`다. joint target(integrity >= 0.8 및 correction_rate <= 0.25)을 만족한 row는 `0`개다.

| config | eval mode | integrity | low-disruption | violations/step | correction | corrected objects | delta norm | joint target |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `delta_norm_strong` | `raw_finite_clamped` | 0.633398 | 0.633398 | 0.598333 | 0.000000 | 0.000000 | 0.670200 | 0 |
| `combined_strong` | `raw_finite_clamped` | 0.628103 | 0.628103 | 0.605833 | 0.000000 | 0.000000 | 0.663909 | 0 |
| `delta_norm_mid` | `raw_finite_clamped` | 0.564610 | 0.564610 | 0.720000 | 0.000000 | 0.000000 | 0.693878 | 0 |
| `combined_mid` | `raw_finite_clamped` | 0.560365 | 0.560365 | 0.725000 | 0.000000 | 0.000000 | 0.691949 | 0 |
| `baseline_finite` | `raw_finite_clamped` | 0.527391 | 0.527391 | 0.784166 | 0.000000 | 0.000000 | 0.709270 | 0 |
| `validity_mid` | `raw_finite_clamped` | 0.527391 | 0.527391 | 0.784166 | 0.000000 | 0.000000 | 0.709270 | 0 |
| `validity_strong` | `raw_finite_clamped` | 0.527391 | 0.527391 | 0.784166 | 0.000000 | 0.000000 | 0.709270 | 0 |
| `delta_norm_strong` | `selective_corrected` | 0.962773 | 0.809071 | 0.000000 | 0.598333 | 0.027466 | 0.661827 | 0 |
| `combined_strong` | `selective_corrected` | 0.961608 | 0.806030 | 0.000000 | 0.605833 | 0.027462 | 0.655450 | 0 |
| `delta_norm_mid` | `selective_corrected` | 0.960800 | 0.776680 | 0.000000 | 0.720000 | 0.027463 | 0.683509 | 0 |
| `combined_mid` | `selective_corrected` | 0.958957 | 0.773588 | 0.000000 | 0.725000 | 0.027462 | 0.681580 | 0 |
| `baseline_finite` | `selective_corrected` | 0.958735 | 0.758574 | 0.000000 | 0.784166 | 0.027461 | 0.697858 | 0 |
| `validity_mid` | `selective_corrected` | 0.958735 | 0.758574 | 0.000000 | 0.784166 | 0.027461 | 0.697858 | 0 |
| `validity_strong` | `selective_corrected` | 0.958735 | 0.758574 | 0.000000 | 0.784166 | 0.027461 | 0.697858 | 0 |

## Interpretation

- raw finite-clamped integrity가 오르면 transition 자체가 더 안정해졌다는 증거다.
- selective correction의 correction rate가 낮아지면서 integrity가 유지되면 P2 병목이 실제로 완화된 것이다.
- 둘 다 실패하면 P2는 손실 가중치 조합이 아니라 모델 구조, multi-step supervision, 또는 simulator-resynchronized training이 필요하다.

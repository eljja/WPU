# World-Copy Escalation Correction Probe

이 probe는 `escalation_required=1`일 때 sparse-only propagation과 local hybrid correction 후보 사용을 비교한다.
이는 학습된 dynamics benchmark가 아니라 escalation 이후 causal update set을 회복하는지 보는 v3 substrate diagnostic이다.
Source CSV: `docs/experiments/world_copy_escalation_correction_probe.csv`.

## Summary

| mode | true relation confidence | mean recall | mean precision | mean F1 | mean escalation | max selected K | max touch ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse_confident_relations | 0.95 | 0.772135 | 1.000000 | 0.846512 | 0.500000 | 16 | 0.24489796 |
| hybrid_escalation_region | 0.95 | 0.888021 | 1.000000 | 0.924993 | 0.500000 | 16 | 0.24489796 |
| sparse_confident_relations | 0.2 | 0.145833 | 1.000000 | 0.246623 | 1.000000 | 16 | 0.24489796 |
| hybrid_escalation_region | 0.2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16 | 0.24489796 |

## Interpretation

- True relation confidence가 높아도 missing relation이 있으면 sparse confident relation propagation은 일부 causal update를 놓친다.
- True relation confidence가 낮으면 confidence gate가 relation frontier를 버리므로 sparse-only recall이 크게 떨어진다.
- Escalation이 켜진 경우 local region correction 후보를 사용하면 controlled setup에서 causal update recall이 회복된다.
- 다음 단계는 이 correction 후보를 실제 learned propagation head가 update quality로 전환하는지 검증하는 것이다.

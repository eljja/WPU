# World-Copy Learned Correction Probe

이 probe는 escalation 이후 local correction 후보가 실제 learned delta update 품질을 개선하는지 측정한다.
모델은 작은 relation/state-conditioned MLP이며, token/graph baseline 우월성을 주장하기 위한 실험은 아니다.
Source CSV: `docs/experiments/world_copy_learned_correction_probe.csv`.

## Summary

| mode | true relation confidence | mean delta MSE | relative MSE vs zero | mean missed causal | mean updated K | max selected K | max touch ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse_confident_relations | 0.95 | 0.070575 | 0.170305 | 2.095215 | 7.238118 | 16 | 0.24489796 |
| hybrid_escalation_region | 0.95 | 0.037102 | 0.088378 | 1.045573 | 8.287760 | 16 | 0.24489796 |
| sparse_confident_relations | 0.2 | 0.275312 | 0.659940 | 8.333333 | 1.000000 | 16 | 0.24489796 |
| hybrid_escalation_region | 0.2 | 0.006365 | 0.013086 | 0.039063 | 9.294271 | 16 | 0.24489796 |

## Interpretation

- Low-confidence relation regime에서 sparse confident-relation update는 많은 causal object를 갱신하지 못해 learned delta MSE가 크게 남는다.
- Escalation 이후 local region 후보를 허용하면 selected `K`를 bounded 상태로 유지하면서 missing causal delta를 학습 가능한 입력으로 되돌린다.
- 이 결과는 correction 후보가 실제 update 품질로 이어질 수 있음을 보이는 substrate-level positive다.
- 한계: controlled synthetic local law이며, dense/token/graph baseline 및 long-horizon world-copy integrity는 아직 검증하지 않는다.

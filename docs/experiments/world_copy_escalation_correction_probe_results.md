# World-Copy Escalation Correction Probe

This probe compares sparse-only propagation with local hybrid correction candidates when `escalation_required=1`.
It is a v3 substrate diagnostic for recovering the causal update set after escalation, not a learned dynamics benchmark.
Source CSV: `docs/experiments/world_copy_escalation_correction_probe.csv`.

## Summary

| mode | true relation confidence | mean recall | mean precision | mean F1 | mean escalation | max selected K | max touch ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse_confident_relations | 0.95 | 0.772135 | 1.000000 | 0.846512 | 0.500000 | 16 | 0.24489796 |
| hybrid_escalation_region | 0.95 | 0.888021 | 1.000000 | 0.924993 | 0.500000 | 16 | 0.24489796 |
| sparse_confident_relations | 0.2 | 0.145833 | 1.000000 | 0.246623 | 1.000000 | 16 | 0.24489796 |
| hybrid_escalation_region | 0.2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 16 | 0.24489796 |

## Interpretation

- Even when true relation confidence is high, missing relations make sparse confident-relation propagation miss some causal updates.
- When true relation confidence is low, the confidence gate removes relation-frontier evidence and sparse-only recall drops sharply.
- When escalation is active, using local region correction candidates recovers causal update recall in this controlled setup.
- The next step is to test whether a learned propagation head turns these correction candidates into better update quality.

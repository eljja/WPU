# World-Copy Learned Correction Probe

This probe tests whether local correction candidates after escalation improve learned delta-update quality.
The model is a small relation/state-conditioned MLP; this is not a token/graph superiority benchmark.
Source CSV: `docs/experiments/world_copy_learned_correction_probe.csv`.

## Summary

| mode | true relation confidence | mean delta MSE | relative MSE vs zero | mean missed causal | mean updated K | max selected K | max touch ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse_confident_relations | 0.95 | 0.070575 | 0.170305 | 2.095215 | 7.238118 | 16 | 0.24489796 |
| hybrid_escalation_region | 0.95 | 0.037102 | 0.088378 | 1.045573 | 8.287760 | 16 | 0.24489796 |
| sparse_confident_relations | 0.2 | 0.275312 | 0.659940 | 8.333333 | 1.000000 | 16 | 0.24489796 |
| hybrid_escalation_region | 0.2 | 0.006365 | 0.013086 | 0.039063 | 9.294271 | 16 | 0.24489796 |

## Interpretation

- In the low-confidence relation regime, sparse confident-relation updates miss many causal objects and leave high learned delta MSE.
- Allowing local region candidates after escalation returns missing causal deltas to the learned update head while keeping selected `K` bounded.
- This is a substrate-level positive showing that correction candidates can translate into better learned update quality.
- Limitation: the law is controlled and synthetic; dense/token/graph baselines and long-horizon world-copy integrity are not tested here.

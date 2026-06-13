# PyBullet Learned Correction Trigger Results

This experiment tests the remaining P2 correction-trigger bottleneck with a small MLP trigger that transfers across held-out seeds rather than another hand-coded threshold. The trigger observes sparse-delta confidence, delta norm, violation change, and cup state before deciding whether to run selective correction.

Source CSV: `docs/experiments/pybullet_learned_correction_trigger.csv`

Summary policies meeting the joint target (integrity >= 0.8 and correction_rate <= 0.25): `0`. The best integrity is `0.958931` (`always_selective`) at correction rate `1.000000`. Under correction rate <= 0.25, the best integrity is `0.523279` (`no_correction`). The best learned-trigger integrity is `0.958931` (`learned_threshold_0.5`) at correction rate `0.791667`. The best low-disruption score is `0.756897` (`learned_threshold_0.5`).

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

## Interpretation

- The hard-seed learned trigger does not solve P2: high integrity is preserved only when most sparse updates are corrected.
- Always-on or high-frequency selective correction protects applied state, but this is a memory-safety layer rather than stable raw dynamics.
- The next step is stable transition training with transition losses, state-validity losses, and correction objectives, not more threshold tuning.

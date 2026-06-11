# PyBullet Mechanism-Selective Calibration Gate

이 audit는 하나의 전역 threshold가 아니라 mechanism별 WPU recompute policy를 선택하면 low-cost 예산 안에서 accuracy와 calibration을 동시에 개선할 수 있는지 검사한다.

이 결과는 zero-shot routing이 아니다. Mechanism 또는 mechanism-level detector가 이미 식별되었다는 adapted setting이다. 목적은 P5의 다음 방향이 전역 confidence gate가 아니라 mechanism-aware calibration routing인지 확인하는 것이다.

Source CSV: `docs/experiments/pybullet_learned_uncertainty_gate.csv`

Derived CSV: `docs/experiments/pybullet_mechanism_selective_calibration_gate.csv`

## 해석

- Low-cost(`cost <= 0.25`), accuracy-safe, calibration-safe non-reference 조합은 `4`개다.
- Best safe policy의 accuracy delta는 `0.029100`, ECE delta는 `-0.001652`, Brier delta는 `-0.030758`, cost는 `0.247355`이다.
- 이는 P5가 불가능한 문제가 아니라, mechanism-aware selective routing으로 풀어야 하는 문제임을 지지한다.
- 단, mechanism 식별과 calibration sample 의존성이 남아 있으므로 zero-shot calibration-safe routing은 아직 아니다.

## Mechanism별 안전 후보

| Mechanism | Selected policy | Cost | Accuracy delta | ECE delta | Brier delta |
|---|---|---:|---:|---:|---:|
| `edge_catch_heavy` | `wpu_sparse` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `edge_high_force` | `fewshot_learned_p0.01` | 0.742064 | 0.087301 | -0.004957 | -0.092275 |
| `no_catch` | `wpu_sparse` | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## 상위 조합

| Policy | Cost | Accuracy delta | ECE delta | Brier delta | Safe |
|---|---:|---:|---:|---:|---|
| `edge_catch_heavy=wpu_sparse; edge_high_force=fewshot_learned_p0.01; no_catch=wpu_sparse` | 0.247355 | 0.029100 | -0.001652 | -0.030758 | True |
| `edge_catch_heavy=fewshot_learned_p0.04; edge_high_force=source_learned_p0.12; no_catch=wpu_sparse` | 0.182540 | 0.027778 | -0.000240 | -0.023432 | True |
| `edge_catch_heavy=source_learned_p0.12; edge_high_force=source_learned_p0.12; no_catch=wpu_sparse` | 0.146826 | 0.025132 | -0.000068 | -0.021758 | True |
| `edge_catch_heavy=wpu_sparse; edge_high_force=source_learned_p0.12; no_catch=wpu_sparse` | 0.107143 | 0.017196 | -0.001504 | -0.020073 | True |
| `edge_catch_heavy=wpu_sparse; edge_high_force=wpu_sparse; no_catch=wpu_sparse` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | True |
| `edge_catch_heavy=fewshot_learned_p0.08; edge_high_force=source_learned_p0.12; no_catch=source_learned_p0.08` | 0.236773 | 0.072751 | 0.024656 | -0.040653 | False |
| `edge_catch_heavy=fewshot_learned_p0.12; edge_high_force=fewshot_learned_p0.12; no_catch=source_learned_p0.12` | 0.231481 | 0.072751 | 0.029206 | -0.041222 | False |
| `edge_catch_heavy=source_learned_p0.12; edge_high_force=fewshot_learned_p0.12; no_catch=source_learned_p0.12` | 0.246032 | 0.068783 | 0.022250 | -0.040211 | False |
| `edge_catch_heavy=fewshot_learned_p0.08; edge_high_force=source_learned_p0.08; no_catch=source_learned_p0.12` | 0.240741 | 0.068783 | 0.025923 | -0.041486 | False |
| `edge_catch_heavy=wpu_sparse; edge_high_force=fewshot_learned_p0.12; no_catch=source_learned_p0.08` | 0.231481 | 0.067460 | 0.023576 | -0.042539 | False |
| `edge_catch_heavy=fewshot_learned_p0.12; edge_high_force=source_learned_p0.12; no_catch=source_learned_p0.04` | 0.243387 | 0.067460 | 0.025643 | -0.040855 | False |
| `edge_catch_heavy=fewshot_learned_p0.08; edge_high_force=source_learned_p0.12; no_catch=source_learned_p0.12` | 0.211640 | 0.066138 | 0.021894 | -0.036640 | False |
| `edge_catch_heavy=fewshot_learned_p0.08; edge_high_force=wpu_sparse; no_catch=fewshot_learned_p0.12` | 0.165344 | 0.066138 | 0.035705 | -0.026279 | False |
| `edge_catch_heavy=fewshot_learned_p0.08; edge_high_force=wpu_sparse; no_catch=fewshot_learned_p0.08` | 0.206349 | 0.066138 | 0.036903 | -0.028049 | False |
| `edge_catch_heavy=fewshot_learned_p0.12; edge_high_force=source_learned_p0.08; no_catch=source_learned_p0.08` | 0.244709 | 0.066137 | 0.024516 | -0.043420 | False |
| `edge_catch_heavy=fewshot_learned_p0.12; edge_high_force=source_learned_p0.12; no_catch=source_learned_p0.08` | 0.215609 | 0.063492 | 0.020487 | -0.038574 | False |
| `edge_catch_heavy=wpu_sparse; edge_high_force=fewshot_learned_p0.08; no_catch=source_learned_p0.12` | 0.239418 | 0.062169 | 0.020705 | -0.040118 | False |
| `edge_catch_heavy=wpu_sparse; edge_high_force=source_learned_p0.12; no_catch=fewshot_learned_p0.12` | 0.226191 | 0.062169 | 0.021639 | -0.041576 | False |
| `edge_catch_heavy=fewshot_learned_p0.12; edge_high_force=source_learned_p0.04; no_catch=source_learned_p0.12` | 0.243386 | 0.062169 | 0.021918 | -0.042439 | False |
| `edge_catch_heavy=source_learned_p0.01; edge_high_force=wpu_sparse; no_catch=fewshot_learned_p0.12` | 0.239418 | 0.062169 | 0.033028 | -0.025379 | False |

## Best Safe Policy

`edge_catch_heavy=wpu_sparse; edge_high_force=fewshot_learned_p0.01; no_catch=wpu_sparse`

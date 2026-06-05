# Candidate No-Harm Gate 결과

이 문서는 conservative set-evaluator 실험을 P1 candidate-oracle gap 관점에서 다시 요약한다. 목적은 sample-level no-harm/margin gate가 aggregate selector 실패를 해결하는지 확인하는 것이다.

Source CSV: `docs/experiments/wpu_v2_retriever_conservative_set_evaluator.csv`

최고 closure는 `0.082804` (`K=32`, `conservative_margin_gate`)이며, 음수 closure 조건은 `6`개다. 따라서 현재 margin 기반 no-harm gate는 P1을 해결하지 못한다. 실패 원인은 단순 threshold 부재가 아니라 candidate별 uncertainty/regret signal이 held-out seed에서 충분히 transfer되지 않는 데 있다.

| K | Policy | Loss | Accuracy | Oracle gain | Deployed gain | Closure | Gate use | Failure mode |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 8 | `set_evaluator` | 0.989208 | 0.502222 | 0.032897 | -0.000776 | -0.023601 | 0.988889 | `harmful_gate_transfer` |
| 8 | `conservative_margin_gate` | 0.989036 | 0.502222 | 0.032897 | -0.000605 | -0.018384 | 0.988889 | `harmful_gate_transfer` |
| 8 | `robust_per_seed_margin_gate` | 0.989036 | 0.502222 | 0.032897 | -0.000605 | -0.018384 | 0.988889 | `harmful_gate_transfer` |
| 16 | `set_evaluator` | 0.969430 | 0.497778 | 0.060800 | -0.003248 | -0.053415 | 0.937778 | `harmful_gate_transfer` |
| 16 | `conservative_margin_gate` | 0.969162 | 0.497778 | 0.060800 | -0.002979 | -0.049004 | 0.937778 | `harmful_gate_transfer` |
| 16 | `robust_per_seed_margin_gate` | 0.969162 | 0.497778 | 0.060800 | -0.002979 | -0.049004 | 0.937778 | `harmful_gate_transfer` |
| 32 | `set_evaluator` | 1.001607 | 0.511111 | 0.035643 | 0.002488 | 0.069797 | 0.966667 | `weak_sample_level_selection_signal` |
| 32 | `conservative_margin_gate` | 1.001143 | 0.508889 | 0.035643 | 0.002951 | 0.082804 | 0.966667 | `weak_sample_level_selection_signal` |
| 32 | `robust_per_seed_margin_gate` | 1.001143 | 0.508889 | 0.035643 | 0.002951 | 0.082804 | 0.966667 | `weak_sample_level_selection_signal` |

## 해석

- Gate가 사용률을 낮추지 못하거나 잘못된 candidate를 계속 선택하면 no-harm 조건은 held-out seed에서 깨진다.
- K=8/16의 음수 closure는 margin confidence가 실제 downstream regret과 정렬되지 않음을 의미한다.
- 다음 P1 개선은 threshold 조정보다 per-candidate uncertainty, calibrated regret target, no-harm rejection loss로 내려가야 한다.

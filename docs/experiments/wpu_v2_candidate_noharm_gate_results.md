# Candidate No-Harm Gate Results

This report re-summarizes the conservative set-evaluator experiment through the priority-1 candidate-oracle gap. It asks whether a sample-level no-harm/margin gate fixes aggregate selector failure.

Source CSV: `docs/experiments/wpu_v2_retriever_conservative_set_evaluator.csv`

The best closure is `0.082804` (`K=32`, `conservative_margin_gate`), and `6` conditions have negative closure. The current margin-based no-harm gate therefore does not solve P1. The failure is not merely a missing threshold; candidate-level uncertainty/regret signals do not yet transfer reliably to held-out seeds.

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

## Interpretation

- A no-harm gate fails when its use-rate remains high or it keeps choosing harmful candidates on held-out seeds.
- Negative closure at K=8/16 means margin confidence is not aligned with downstream regret.
- The next P1 improvement must move below threshold selection toward per-candidate uncertainty, calibrated regret targets, and no-harm rejection losses.

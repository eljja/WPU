# WPU v2 Priority Dashboard

This dashboard conservatively recomputes the current status of v2 priorities 1-7 from existing experiment CSVs. Its purpose is to keep WPU claims aligned with evidence.

| Priority | Item | Status | Observed | Target | Metric |
|---:|---|---|---:|---:|---|
| 1 | Candidate-oracle gap | fail | 0.244220 | 0.500000 | `gap_closure_fraction` |
| 2 | Long-horizon state integrity | partial | 0.964322 | 0.800000 | `best_wpu_h25_integrity` |
| 3 | Simulator-backed benchmark | partial | 2.000000 | 5.000000 | `seed_count` |
| 4 | Mechanism-family shift generalization | partial | 0.333333 | 1.000000 | `wpu_shift_win_rate` |
| 5 | Calibration and uncertainty | partial | 1.068727 | 1.000000 | `wpu_ece_over_baseline_ece` |
| 6 | Systems profile and memory traffic | partial | 0.997454 | 0.950000 | `max_tensor_byte_reduction` |
| 7 | Objectification quality to propagation loss | partial | 0.821712 | 0.957711 | `combined_objectification_score` |

## Interpretation

The dashboard shows that WPU v2 is promising but not a completed superiority claim. The strongest claim remains conditional: WPU can reduce compute and memory when objectified state exposes a small causal working set K before tensorization. Large N alone is not enough.

- P1 Candidate-oracle gap: Best deployed closure is 0.244220 and mean closure is 0.160601; decomposition shows no omitted aggregate policy closes the gap.
- P2 Long-horizon state integrity: Best WPU H=25 integrity is 0.964322; guarded sparse is 0.958508, while clipped sparse without projection is 0.201757.
- P3 Simulator-backed benchmark: PyBullet benchmark exists with 2 seeds and background up to N_bg=128, but it is still small.
- P4 Mechanism-family shift generalization: catch_heavy: WPU 0.277778 vs baseline 0.402778; edge_shift: WPU 0.597222 vs baseline 0.472222; high_force: WPU 0.444445 vs baseline 0.458334
- P5 Calibration and uncertainty: Mean WPU ECE is 0.236226; mean baseline ECE is 0.221034; ratio is 1.068727.
- P6 Systems profile and memory traffic: Proxy tensor-byte reduction reaches 0.997454 at mean total objects 2052.6; real hardware data is absent.
- P7 Objectification quality to propagation loss: Clean score 0.957711, combined-corruption score 0.821712, combined frontier recall 0.742361. Metrics exist, but downstream loss coupling is incomplete.

## Next Actions

- P1: Move below aggregate policy selection: add per-candidate uncertainty, sample-level no-harm gating, and regret targets.
- P2: Add rollout-consistency loss, unsafe-delta rejection, rollback, correction, and uncertainty escalation.
- P3: Increase seeds, mechanisms, training scale, and long-horizon simulator rollouts.
- P4: Add leave-family-out training, harder shifts, and mechanism-aware branch priors.
- P5: Add temperature heads, branch calibration loss, multi-step ECE/Brier/NLL, and uncertainty-gated recompute.
- P6: Measure real CPU/GPU latency, CUDA memory, allocator traffic, sparse-kernel behavior, and matched-accuracy speedups.
- P7: Train/evaluate propagation under controlled objectification corruption and regress loss against report components.

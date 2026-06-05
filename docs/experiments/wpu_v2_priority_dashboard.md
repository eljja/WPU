# WPU v2 Priority Dashboard

This dashboard conservatively recomputes the current status of v2 priorities 1-7 from existing experiment CSVs. Its purpose is to keep WPU claims aligned with evidence.

| Priority | Item | Status | Observed | Target | Metric |
|---:|---|---|---:|---:|---|
| 1 | Candidate-oracle gap | fail | 0.327146 | 0.500000 | `gap_closure_fraction` |
| 2 | Long-horizon state integrity | partial | 0.964322 | 0.800000 | `best_wpu_h25_integrity` |
| 3 | Simulator-backed benchmark | partial | 5.000000 | 5.000000 | `seed_count` |
| 4 | Mechanism-family shift generalization | partial | 0.333333 | 1.000000 | `wpu_shift_win_rate` |
| 5 | Calibration and uncertainty | partial | 0.875306 | 1.000000 | `wpu_ece_over_baseline_ece` |
| 6 | Systems profile and memory traffic | partial | 0.997454 | 0.950000 | `max_tensor_byte_reduction` |
| 7 | Objectification quality to propagation loss | partial | 0.821712 | 0.957711 | `combined_objectification_score` |

## Interpretation

The dashboard shows that WPU v2 is promising but not a completed superiority claim. The strongest claim remains conditional: WPU can reduce compute and memory when objectified state exposes a small causal working set K before tensorization. Large N alone is not enough.

- P1 Candidate-oracle gap: Best deployed closure is 0.327146; previous aggregate-policy best is 0.244220 and mean aggregate closure is 0.160601. Sample-level no-harm/margin gates were also audited; best closure is 0.082804, so margin gating is not the missing fix. Direct candidate-regret gating reaches 0.329950 unconstrained and 0.327146 under harmful-accept <= 0.25; the selected conservative harmful-accept rate is 0.235556.
- P2 Long-horizon state integrity: Best WPU H=25 integrity is 0.964322; guarded sparse is 0.958508, clipped sparse is 0.201757, regularized raw sparse is 0.087153, and unsafe-delta rejected sparse is 0.530270 with rejection rate 0.640000.
- P3 Simulator-backed benchmark: PyBullet benchmark exists with 5 seeds and background up to N_bg=128, but it is still small.
- P4 Mechanism-family shift generalization: catch_heavy: WPU 0.366667 vs baseline 0.327778; edge_shift: WPU 0.522222 vs baseline 0.555555; high_force: WPU 0.416667 vs baseline 0.433333
- P5 Calibration and uncertainty: Mean WPU ECE is 0.213693; mean baseline ECE is 0.244135; ratio is 0.875306.
- P6 Systems profile and memory traffic: Tensor-byte reduction reaches 0.997454 at mean total objects 2052.6; CPU tensorization latency reduction reaches 0.995649, but model-forward/GPU/energy data is absent.
- P7 Objectification quality to propagation loss: Clean score 0.957711, combined-corruption score 0.821712, combined frontier recall 0.742361. Metrics exist, but downstream loss coupling is incomplete.

## Next Actions

- P1: Strengthen candidate-regret training with calibrated uncertainty, harmful-accept penalties, and cross-seed perturbations.
- P2: Simple delta-norm regularization is insufficient; add rollout-consistency loss, unsafe-delta rejection, rollback, correction, and uncertainty escalation.
- P3: Increase seeds, mechanisms, training scale, and long-horizon simulator rollouts.
- P4: Add leave-family-out training, harder shifts, and mechanism-aware branch priors.
- P5: Add temperature heads, branch calibration loss, multi-step ECE/Brier/NLL, and uncertainty-gated recompute.
- P6: Measure model forward latency, CUDA memory, allocator traffic, sparse-kernel behavior, and matched-accuracy speedups.
- P7: Train/evaluate propagation under controlled objectification corruption and regress loss against report components.

# WPU v2 Priority Dashboard

This dashboard conservatively recomputes the current status of v2 priorities 1-7 from existing experiment CSVs. Its purpose is to keep WPU claims aligned with evidence.

| Priority | Item | Status | Observed | Target | Metric |
|---:|---|---|---:|---:|---|
| 1 | Candidate-oracle gap | fail | 0.328025 | 0.500000 | `gap_closure_fraction` |
| 2 | Long-horizon state integrity | partial | 0.988647 | 0.800000 | `best_wpu_h25_integrity` |
| 3 | Simulator-backed benchmark | partial | 7.000000 | 5.000000 | `seed_count` |
| 4 | Mechanism-family shift generalization | partial | 0.333333 | 1.000000 | `wpu_shift_win_rate` |
| 5 | Calibration and uncertainty | partial | 0.963449 | 1.000000 | `wpu_ece_over_baseline_ece` |
| 6 | Systems profile and memory traffic | partial | 0.997454 | 0.950000 | `max_tensor_byte_reduction` |
| 7 | Objectification quality to propagation loss | partial | 0.821712 | 0.957711 | `combined_objectification_score` |

## Interpretation

The dashboard shows that WPU v2 is promising but not a completed superiority claim. The strongest claim remains conditional: WPU can reduce compute and memory when objectified state exposes a small causal working set K before tensorization. Large N alone is not enough.

- P1 Candidate-oracle gap: Best deployed closure is 0.328025; previous aggregate-policy best is 0.244220 and mean aggregate closure is 0.160601. Sample-level no-harm/margin gates were also audited; best closure is 0.082804, so margin gating is not the missing fix. Direct candidate-regret gating reaches 0.329950 unconstrained and 0.327146 under harmful-accept <= 0.25; train-selected deployment reaches 0.328025 with harmful-accept 0.251111. The selected deployment harmful-accept rate is 0.251111. Harmful-accept/ranking-penalty training is safer but weaker: train-selected closure 0.081253 with harmful-accept 0.088889. Feature perturbation improves test-sweep closure to 0.339525 unconstrained and 0.329756 under harmful-accept <= 0.25, but train-selected closure is 0.312586.
- P2 Long-horizon state integrity: Best WPU H=25 integrity is 0.988647; guarded sparse is 0.958508, clipped sparse is 0.201757, regularized raw sparse is 0.087153, rollout-consistency sparse is 0.084549, validity sparse is 0.084722, strong-validity sparse is 0.084722, unsafe-delta rejected sparse is 0.530270 with rejection rate 0.640000, and rollback sparse is 0.988647 with rollback rate 0.812500. Corrected rollback sparse is 0.900288 with correction rate 0.812500 and rollback rate 0.564167.
- P3 Simulator-backed benchmark: PyBullet benchmark exists with 7 seeds and background up to N_bg=128; the 7-seed extension is still small but less seed-fragile.
- P4 Mechanism-family shift generalization: catch_heavy: WPU 0.408730 vs baseline 0.349206; edge_shift: WPU 0.527778 vs baseline 0.571428; high_force: WPU 0.432540 vs baseline 0.460318; 3-seed calibrated mixture probe: mixture catch_heavy: WPU 0.333333 vs baseline 0.481481; mixture edge_shift: WPU 0.546297 vs baseline 0.388889; mixture high_force: WPU 0.444444 vs baseline 0.444444; 3-seed leave-family-out win-rate 0.750000: leave-family catch_heavy: WPU 0.175926 vs baseline 0.296296; leave-family edge_shift: WPU 0.509260 vs baseline 0.444445; leave-family high_force: WPU 0.500000 vs baseline 0.490741; leave-family nominal: WPU 0.462963 vs baseline 0.407407; 3-seed composition-shift stress win-rate 1.000000, mean accuracy delta 0.123457: composition edge_catch_heavy: WPU 0.453704 vs baseline 0.333334; composition edge_high_force: WPU 0.583333 vs baseline 0.583333; composition no_catch: WPU 0.759259 vs baseline 0.509259
- P5 Calibration and uncertainty: Mean WPU ECE is 0.211243; mean baseline ECE is 0.219257; ratio is 0.963449. A 3-seed calibrated mixture probe gives WPU ECE 0.208404, baseline ECE 0.183805, ratio 1.133834. A 3-seed leave-family-out probe gives mean ECE ratio 0.972745. A 3-seed composition-shift stress probe gives mean ECE ratio 1.327702; worst is no_catch at 2.362081. Temperature+bias calibration changes mean ECE ratio by -0.217855 and improves 1/3 composition mechanisms.
- P6 Systems profile and memory traffic: Tensor-byte reduction reaches 0.997454 at mean total objects 2052.6; CPU tensorization latency reduction reaches 0.996035; random-model CPU sparse-forward latency reduction reaches 0.996975. CUDA random-model sparse-forward latency reduction reaches 0.996216 and sparse peak-memory reduction reaches 0.304080 at mean total objects 2052.4. Matched-or-better audit: N=5: matched=True speedup=0.111341; N=133: matched=True speedup=19.184067. Screening-only energy proxy max is 0.999990; CUDA forward proxy max is 0.997367. Real energy, sparse-kernel behavior, and Pareto dominance over every baseline remain unproven.
- P7 Objectification quality to propagation loss: Clean score 0.957711, combined-corruption score 0.821712, combined frontier recall 0.742361. Metrics exist, but downstream loss coupling is incomplete.

## Next Actions

- P1: Strengthen candidate-regret training with calibrated uncertainty, harmful-accept penalties, and cross-seed perturbations.
- P2: Simple delta-norm, rollout-consistency, and validity regularization are insufficient; add rollback, correction, and uncertainty escalation.
- P3: Increase seeds, mechanisms, training scale, and long-horizon simulator rollouts.
- P4: Add leave-family-out training, harder shifts, and mechanism-aware branch priors.
- P5: Add temperature heads, branch calibration loss, multi-step ECE/Brier/NLL, and uncertainty-gated recompute.
- P6: Measure energy, allocator traffic, sparse-kernel behavior, Pareto frontiers, and trained matched-or-better speedups.
- P7: Train/evaluate propagation under controlled objectification corruption and regress loss against report components.

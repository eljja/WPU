# PyBullet Simulator Coverage Audit

This audit separates simulator grounding breadth from superiority claims. A row can increase coverage while still being unusable as a matched-baseline accuracy claim when `baseline_complete=False`.

Source CSVs:
- `docs/experiments/pybullet_closed_loop_rollout.csv`
- `docs/experiments/pybullet_cup_benchmark_7seed.csv`
- `docs/experiments/pybullet_cup_benchmark_n256_baseline_screen.csv`
- `docs/experiments/pybullet_cup_benchmark_n256_medium.csv`
- `docs/experiments/pybullet_cup_benchmark_n512.csv`
- `docs/experiments/pybullet_cup_benchmark_n512_baseline_micro.csv`
- `docs/experiments/pybullet_objectification_quality.csv`
- `docs/experiments/pybullet_shift_generalization.csv`
- `docs/experiments/pybullet_system_profile.csv`
- `docs/experiments/pybullet_system_profile_cuda.csv`

| Axis | Seeds | Models | Mechanisms | N_bg max | N max | Horizon max | Corruptions | Baseline complete |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cup_7seed_baseline_complete | 7 | 4 | 1 | 128 | 133 | 1 | 1 | True |
| cup_n256_baseline_screen | 5 | 4 | 1 | 256 | 261 | 1 | 1 | True |
| cup_n256_baseline_medium | 5 | 4 | 1 | 256 | 261 | 1 | 1 | True |
| cup_n512_baseline_micro | 3 | 4 | 1 | 512 | 517 | 1 | 1 | True |
| cup_n512_wpu_only_extension | 7 | 2 | 1 | 512 | 517 | 1 | 1 | False |
| mechanism_shift_generalization | 7 | 4 | 4 | 32 | 37 | 1 | 1 | True |
| closed_loop_rollout | 2 | 3 | 1 | 32 | 37 | 25 | 1 | True |
| objectification_quality | 2 | 0 | 1 | 512 | 516 | 1 | 7 | True |
| system_profile_cpu | 2 | 0 | 1 | 2048 | 2052 | 1 | 1 | True |
| system_profile_cuda | 2 | 0 | 1 | 2048 | 2052 | 1 | 1 | True |

## Interpretation

- Current PyBullet evidence covers cup prediction, mechanism shift, closed-loop rollout, objectification corruption, and CPU/CUDA systems profiling.
- `cup_n256_baseline_screen` completes WPU, graph, and token baselines at N_bg=256 and total N=261, but it is a low-training screen and should not be used as a strong accuracy-superiority claim.
- `cup_n256_baseline_medium` increases the training budget at the same N=261 and is stronger large-N simulator evidence, but it is still a single cup-family benchmark rather than a broad superiority claim.
- `cup_n512_baseline_micro` includes WPU, graph, and token baselines at N_bg=512 and total N=517, but with only 3 seeds, 2 steps, and 8 samples it is large-N coverage evidence rather than strong accuracy-superiority evidence.
- `cup_n512_wpu_only_extension` shows WPU execution at N_bg=512 and total N=517, but it is not accuracy-superiority evidence because the dense graph baseline did not complete under the same protocol.
- The next P3 bottleneck is not another small cup run; it is mechanism diversity, baseline-complete large-N comparison, and end-to-end objectification through a perception/state adapter.

## Row Notes

- `cup_7seed_baseline_complete`: 7-seed cup benchmark with WPU, graph, and token baselines; accuracy claims remain limited to this cup-task protocol.
- `cup_n256_baseline_screen`: Low-training 5-seed N_bg=256 screen with WPU, graph, and token baselines; useful for matched large-N feasibility, not for strong accuracy-superiority claims.
- `cup_n256_baseline_medium`: Medium-training 5-seed N_bg=256 run with WPU, graph, and token baselines. It improves over the low-training screen, but remains a single cup-family benchmark rather than a broad simulator claim.
- `cup_n512_baseline_micro`: Low-training 3-seed N_bg=512 micro-screen with WPU, graph, and token baselines. It completes matched large-N coverage at total N=517, but its tiny training and sample budget make it coverage evidence rather than strong accuracy-superiority evidence.
- `cup_n512_wpu_only_extension`: Large-background WPU-only extension. The graph-transformer baseline did not finish under the attempted 20-minute run, so this is systems feasibility evidence, not matched baseline superiority evidence.
- `mechanism_shift_generalization`: Nominal plus 3 shifted mechanism families: catch_heavy, edge_shift, high_force.
- `closed_loop_rollout`: Multi-step delta-overlay rollout diagnostic; finite-corrected safety is tracked in the separate state-integrity audit.
- `objectification_quality`: Objectification-contract audit over clean and corrupted simulator-derived state; it measures input quality, not model superiority.
- `system_profile_cpu`: Systems profile separating full-state tensorization from indexed WPU working-set tensorization and random forward proxies.
- `system_profile_cuda`: Systems profile separating full-state tensorization from indexed WPU working-set tensorization and random forward proxies.

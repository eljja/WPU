# PyBullet 시뮬레이터 Coverage Audit

이 audit는 simulator grounding의 범위와 우월성 주장을 분리한다. `baseline_complete=False`인 행은 coverage를 넓히더라도 matched-baseline accuracy claim으로 사용할 수 없다.

Source CSVs:
- `docs/experiments/pybullet_closed_loop_rollout.csv`
- `docs/experiments/pybullet_cup_benchmark_7seed.csv`
- `docs/experiments/pybullet_cup_benchmark_n256_baseline_screen.csv`
- `docs/experiments/pybullet_cup_benchmark_n256_medium.csv`
- `docs/experiments/pybullet_cup_benchmark_n512.csv`
- `docs/experiments/pybullet_objectification_quality.csv`
- `docs/experiments/pybullet_shift_generalization.csv`
- `docs/experiments/pybullet_system_profile.csv`
- `docs/experiments/pybullet_system_profile_cuda.csv`

| 축 | Seeds | Models | Mechanisms | N_bg max | N max | Horizon max | Corruptions | Baseline complete |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cup_7seed_baseline_complete | 7 | 4 | 1 | 128 | 133 | 1 | 1 | True |
| cup_n256_baseline_screen | 5 | 4 | 1 | 256 | 261 | 1 | 1 | True |
| cup_n256_baseline_medium | 5 | 4 | 1 | 256 | 261 | 1 | 1 | True |
| cup_n512_wpu_only_extension | 7 | 2 | 1 | 512 | 517 | 1 | 1 | False |
| mechanism_shift_generalization | 7 | 4 | 4 | 32 | 37 | 1 | 1 | True |
| closed_loop_rollout | 2 | 3 | 1 | 32 | 37 | 25 | 1 | True |
| objectification_quality | 2 | 0 | 1 | 512 | 516 | 1 | 7 | True |
| system_profile_cpu | 2 | 0 | 1 | 2048 | 2052 | 1 | 1 | True |
| system_profile_cuda | 2 | 0 | 1 | 2048 | 2052 | 1 | 1 | True |

## 해석

- 현재 PyBullet evidence는 cup benchmark, mechanism shift, closed-loop rollout, objectification corruption, CPU/CUDA systems profile까지 포함한다.
- `cup_n256_baseline_screen`은 N_bg=256, total N=261에서 WPU, graph, token baseline을 모두 완료한 matched large-N screen이지만, 저훈련 설정이므로 강한 accuracy superiority claim에는 쓰지 않는다.
- `cup_n256_baseline_medium`은 같은 N=261에서 training budget을 올린 baseline-complete run이다. 더 의미 있는 large-N simulator evidence지만 단일 cup family이므로 broad superiority claim에는 부족하다.
- `cup_n512_wpu_only_extension`은 N_bg=512, total N=517까지 WPU가 실행된다는 evidence지만, dense graph baseline이 같은 protocol에서 완료되지 않았으므로 accuracy superiority evidence가 아니다.
- P3의 다음 병목은 단일 PyBullet cup family를 넘어서는 mechanism 다양성, baseline-complete large-N comparison, 그리고 perception/state adapter를 포함한 end-to-end objectification이다.

## 행별 메모

- `cup_7seed_baseline_complete`: 7-seed cup benchmark with WPU, graph, and token baselines; accuracy claims remain limited to this cup-task protocol.
- `cup_n256_baseline_screen`: Low-training 5-seed N_bg=256 screen with WPU, graph, and token baselines; useful for matched large-N feasibility, not for strong accuracy-superiority claims.
- `cup_n256_baseline_medium`: Medium-training 5-seed N_bg=256 run with WPU, graph, and token baselines. It improves over the low-training screen, but remains a single cup-family benchmark rather than a broad simulator claim.
- `cup_n512_wpu_only_extension`: Large-background WPU-only extension. The graph-transformer baseline did not finish under the attempted 20-minute run, so this is systems feasibility evidence, not matched baseline superiority evidence.
- `mechanism_shift_generalization`: Nominal plus 3 shifted mechanism families: catch_heavy, edge_shift, high_force.
- `closed_loop_rollout`: Multi-step delta-overlay rollout diagnostic; finite-corrected safety is tracked in the separate state-integrity audit.
- `objectification_quality`: Objectification-contract audit over clean and corrupted simulator-derived state; it measures input quality, not model superiority.
- `system_profile_cpu`: Systems profile separating full-state tensorization from indexed WPU working-set tensorization and random forward proxies.
- `system_profile_cuda`: Systems profile separating full-state tensorization from indexed WPU working-set tensorization and random forward proxies.

# PyBullet 시뮬레이터 Coverage Audit

이 audit는 simulator grounding의 범위와 우월성 주장을 분리한다. `baseline_complete=False`인 행은 coverage를 넓히더라도 matched-baseline accuracy claim으로 사용할 수 없다.

Source CSVs:
- `docs/experiments/pybullet_closed_loop_rollout.csv`
- `docs/experiments/pybullet_cup_benchmark_7seed.csv`
- `docs/experiments/pybullet_cup_benchmark_n256_baseline_screen.csv`
- `docs/experiments/pybullet_cup_benchmark_n256_medium.csv`
- `docs/experiments/pybullet_cup_benchmark_n512.csv`
- `docs/experiments/pybullet_cup_benchmark_n512_baseline_micro.csv`
- `docs/experiments/pybullet_cup_benchmark_n512_high_budget.csv`
- `docs/experiments/pybullet_cup_benchmark_n512_medium.csv`
- `docs/experiments/pybullet_objectification_quality.csv`
- `docs/experiments/pybullet_shift_generalization.csv`
- `docs/experiments/pybullet_shift_generalization_n512_multimech.csv`
- `docs/experiments/pybullet_shift_generalization_n512_screen.csv`
- `docs/experiments/pybullet_system_profile.csv`
- `docs/experiments/pybullet_system_profile_cuda.csv`

| 축 | Seeds | Models | Mechanisms | N_bg max | N max | Horizon max | Corruptions | Baseline complete |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cup_7seed_baseline_complete | 7 | 4 | 1 | 128 | 133 | 1 | 1 | True |
| cup_n256_baseline_screen | 5 | 4 | 1 | 256 | 261 | 1 | 1 | True |
| cup_n256_baseline_medium | 5 | 4 | 1 | 256 | 261 | 1 | 1 | True |
| cup_n512_baseline_micro | 3 | 4 | 1 | 512 | 517 | 1 | 1 | True |
| cup_n512_baseline_medium | 5 | 4 | 1 | 512 | 517 | 1 | 1 | True |
| cup_n512_baseline_high_budget | 5 | 4 | 1 | 512 | 517 | 1 | 1 | True |
| cup_n512_wpu_only_extension | 7 | 2 | 1 | 512 | 517 | 1 | 1 | False |
| mechanism_shift_generalization | 7 | 4 | 4 | 32 | 37 | 1 | 1 | True |
| mechanism_shift_n512_nominal_train | 3 | 4 | 7 | 512 | 517 | 1 | 1 | True |
| mechanism_shift_n512_multimechanism_train | 3 | 4 | 7 | 512 | 517 | 1 | 1 | True |
| closed_loop_rollout | 2 | 3 | 1 | 32 | 37 | 25 | 1 | True |
| objectification_quality | 2 | 0 | 1 | 512 | 516 | 1 | 7 | True |
| system_profile_cpu | 2 | 0 | 1 | 2048 | 2052 | 1 | 1 | True |
| system_profile_cuda | 2 | 0 | 1 | 2048 | 2052 | 1 | 1 | True |

## 해석

- 현재 PyBullet evidence는 cup benchmark, mechanism shift, closed-loop rollout, objectification corruption, CPU/CUDA systems profile까지 포함한다.
- `cup_n256_baseline_screen`은 N_bg=256, total N=261에서 WPU, graph, token baseline을 모두 완료한 matched large-N screen이지만, 저훈련 설정이므로 강한 accuracy superiority claim에는 쓰지 않는다.
- `cup_n256_baseline_medium`은 같은 N=261에서 training budget을 올린 baseline-complete run이다. 더 의미 있는 large-N simulator evidence지만 단일 cup family이므로 broad superiority claim에는 부족하다.
- `cup_n512_baseline_micro`는 N_bg=512, total N=517에서 WPU/graph/token baseline을 모두 포함하지만 3 seeds, 2 steps, 8 samples의 micro-screen이므로 large-N coverage evidence로만 사용한다.
- `cup_n512_baseline_medium`은 N_bg=512, total N=517에서 5 seeds, 6 steps, 16 samples로 micro보다 강한 baseline-complete run이다. Best WPU가 best baseline보다 약간 높지만 단일 cup family와 small margin 때문에 broad superiority claim은 아니다.
- `cup_n512_baseline_high_budget`은 5 seeds, 10 steps, 24 samples로 budget을 더 올린 run이다. Best WPU edge가 유지되지만 margin이 더 작아져 조건부 evidence로 해석해야 한다.
- `mechanism_shift_n512_nominal_train`과 `mechanism_shift_n512_multimechanism_train`은 total N=517에서 7개 mechanism을 포함한다. 두 축은 WPU의 large-N 계산 장점이 mechanism generalization을 자동으로 보장하지 않음을 보여주는 claim-boundary evidence다.
- `cup_n512_wpu_only_extension`은 N_bg=512, total N=517까지 WPU가 실행된다는 evidence지만, dense graph baseline이 같은 protocol에서 완료되지 않았으므로 accuracy superiority evidence가 아니다.
- P3의 다음 병목은 단순한 mechanism 목록 확장이 아니라 mechanism-aware propagation, long-horizon simulator rollout, 그리고 perception/state adapter를 포함한 end-to-end objectification이다.

## 행별 메모

- `cup_7seed_baseline_complete`: 7-seed cup benchmark with WPU, graph, and token baselines; accuracy claims remain limited to this cup-task protocol.
- `cup_n256_baseline_screen`: Low-training 5-seed N_bg=256 screen with WPU, graph, and token baselines; useful for matched large-N feasibility, not for strong accuracy-superiority claims.
- `cup_n256_baseline_medium`: Medium-training 5-seed N_bg=256 run with WPU, graph, and token baselines. It improves over the low-training screen, but remains a single cup-family benchmark rather than a broad simulator claim.
- `cup_n512_baseline_micro`: Low-training 3-seed N_bg=512 micro-screen with WPU, graph, and token baselines. It completes matched large-N coverage at total N=517, but its tiny training and sample budget make it coverage evidence rather than strong accuracy-superiority evidence.
- `cup_n512_baseline_medium`: Medium 5-seed N_bg=512 run with WPU, graph, and token baselines. It strengthens matched large-N simulator evidence at total N=517, but remains one cup-family, one-step, small-margin evidence rather than broad simulator superiority.
- `cup_n512_baseline_high_budget`: Higher-budget 5-seed N_bg=512 run with WPU, graph, and token baselines. It keeps a small best-WPU accuracy edge over the best baseline at total N=517, but the margin shrinks, so it is conditional evidence rather than a broad superiority claim.
- `cup_n512_wpu_only_extension`: Large-background WPU-only extension. The graph-transformer baseline did not finish under the attempted 20-minute run, so this is systems feasibility evidence, not matched baseline superiority evidence.
- `mechanism_shift_generalization`: Nominal plus 3 shifted mechanism families: catch_heavy, edge_shift, high_force.
- `mechanism_shift_n512_nominal_train`: N_bg=512, total N=517 large-state mechanism-diversity screen. Training uses nominal only; evaluation covers nominal plus 6 shifts: catch_heavy, edge_catch_heavy, edge_high_force, edge_shift, high_force, no_catch. This is a claim-boundary/OOD diagnostic, not a superiority result.
- `mechanism_shift_n512_multimechanism_train`: N_bg=512, total N=517 multi-mechanism training screen over 7 mechanisms. It tests whether mechanism diversity alone recovers WPU accuracy at large N; current results are mixed/negative.
- `closed_loop_rollout`: Multi-step delta-overlay rollout diagnostic; finite-corrected safety is tracked in the separate state-integrity audit.
- `objectification_quality`: Objectification-contract audit over clean and corrupted simulator-derived state; it measures input quality, not model superiority.
- `system_profile_cpu`: Systems profile separating full-state tensorization from indexed WPU working-set tensorization and random forward proxies.
- `system_profile_cuda`: Systems profile separating full-state tensorization from indexed WPU working-set tensorization and random forward proxies.

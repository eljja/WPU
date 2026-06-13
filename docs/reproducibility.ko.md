# WPU 재현성 가이드

이 문서는 현재 저장소의 주장과 품질 검증을 재현하기 위한 최소 명령을 정리한다.
커밋된 evidence와, 검토 후 `docs/`로 승격해야 하는 generated artifact를 구분한다.

## 환경

권장 기본 환경:

- Python 3.11
- development dependencies 포함 editable install:

```bash
python -m pip install -e ".[dev]"
```

기본 설치는 standard PyTorch package를 함께 설치한다. CUDA 전용 실험은 로컬
driver/CUDA stack에 맞는 PyTorch build를 먼저 설치한 뒤 editable install을
실행한다.

Simulator-grounded 실험은 optional PyBullet dependency가 필요하다.

```bash
python -m pip install -e ".[dev,sim]"
```

Windows에서는 `python`이 실제 interpreter가 아니라 Microsoft Store alias로
잡힐 수 있다. 이런 경우 의도한 Python 설치로 virtual environment를 만든 뒤 venv
interpreter로 검증을 실행한다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

## 필수 품질 검증

논문, 문서, 모델 변경을 커밋하기 전에 다음을 실행한다.

```bash
python -m pytest
python -m pip wheel . --no-deps --wheel-dir dist
python -m pip install --force-reinstall --no-deps dist/wpu-*.whl
python -c "import wpu; wpu.create_model('wpu-cws-indexed', hidden_dim=16, layers=1, working_set_size=4)"
```

현재 test suite가 확인하는 항목:

- state model JSON과 delta overlay;
- objectification contract reporting;
- scheduler behavior;
- sparse/dense/model shape path;
- rollout probability normalization;
- script entrypoint hygiene;
- README core script smoke execution;
- documentation link integrity;
- LaTeX figure와 citation integrity;
- experiment `Source CSV` integrity;
- robot-cup demo smoke output.

GitHub Actions는 push와 pull request에서 같은 test command를 실행한다. 또한
`docs/arxiv/state_is_all_you_need_en.tex`에서 영문 arXiv PDF를 빌드하고 workflow
artifact로 업로드한다. 따라서 논문 변경은 evidence test와 LaTeX build를 모두
통과해야 한다.

Windows에서 Anaconda가 `PATH`에 있으면 virtual environment 안에서도 setuptools가
Anaconda의 `distutils`를 잘못 import할 수 있다. Wheel metadata 생성 중
`distutils` assertion이 발생하면 다음처럼 표준 distutils를 강제한다.

```powershell
$env:SETUPTOOLS_USE_DISTUTILS='stdlib'
.\.venv\Scripts\python.exe -m pip wheel . --no-deps --wheel-dir dist
```

## 데모 재현

```bash
python demos/robot_cup_demo.py
```

예상 출력 섹션:

- event와 initial frontier;
- scheduler와 model path;
- frontier trace;
- changed objects;
- branch probabilities;
- memory estimate.

## 논문 빌드

영문 PDF는 LaTeX source에서 생성한다.

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs/arxiv docs/arxiv/state_is_all_you_need_en.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs/arxiv docs/arxiv/state_is_all_you_need_en.tex
```

한글 companion은 Markdown으로 관리한다.

```text
docs/arxiv/state_is_all_you_need_ko.md
```

## 실험 Artifact 정책

커밋된 paper evidence는 `docs/experiments/`와 `docs/figures/`에 둔다. 새 실험은
먼저 git ignore 대상인 `artifacts/`에 생성한다.

Generated result를 `docs/experiments/`로 승격하기 전 확인할 것:

- 의도한 모든 seed가 완료되었는지;
- matched-baseline claim을 위한 model parameter scale이 비교 가능한지;
- explicit object state, sparse retrieval, perception/state adapter에 의존하는
  claim이면 `ObjectificationReport` metric을 함께 기록했는지;
- repaired edge를 사용했다면 relation-repair metric을 함께 기록했는지;
- 모든 `Source CSV` 또는 `Source CSVs` entry가 `docs/experiments/` 아래에서
  git-tracked 상태이고 비어 있지 않은지;
- 해석이 `docs/claims.ko.md`의 claim boundary를 따르는지;
- negative/mixed result를 숨기지 않았는지.

8M-class CWS GPU runner는 이 정책을 따른다.

```powershell
.\scripts\run_cws_8m_gpu.ps1 -Python python
```

출력은 승격 전 검토를 위해 `artifacts/causal_working_set_8m_gpu/`에 생성된다.

PyBullet simulator-grounded benchmark는 다음 명령으로 재현할 수 있다.

```bash
python scripts/pybullet_cup_benchmark.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --background-objects 0 32 128 --seeds 11 13 --steps 30 --sim-steps 120 --samples 48 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --runtime-repeats 5 --balanced-labels --out docs/experiments/pybullet_cup_benchmark.csv
```

PyBullet objectification-corruption stress는 다음 명령으로 재현할 수 있다.

```bash
python scripts/pybullet_objectification_stress.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer --corruptions clean drop_relations_light drop_relations_heavy position_noise low_confidence identity_swap combined --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --out docs/experiments/pybullet_objectification_stress.csv
```

PyBullet objectification-quality decomposition benchmark는 다음 명령으로 재현할 수 있다.

```bash
python scripts/pybullet_objectification_quality.py --samples 12 --seeds 11 13 --background-objects 32 128 512 --corruptions clean drop_relations_heavy drop_objects_light position_noise low_confidence identity_swap combined --sim-steps 24 --out docs/experiments/pybullet_objectification_quality.csv
python scripts/analyze_pybullet_objectification_loss_coupling.py
```

Parameter-matched PyBullet pilot은 다음 명령으로 재현할 수 있다.

```bash
python scripts/pybullet_cup_benchmark.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --background-objects 0 128 --seeds 11 13 --steps 20 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --target-params 50000 --num-heads 4 --working-set-size 12 --runtime-repeats 5 --balanced-labels --out docs/experiments/pybullet_matched_baseline_benchmark.csv
python scripts/pybullet_cup_benchmark.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --background-objects 0 128 --seeds 11 13 17 19 23 29 31 --steps 20 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --runtime-repeats 3 --balanced-labels --out docs/experiments/pybullet_cup_benchmark_7seed.csv
```

저훈련 N_bg=256 matched screen은 다음 명령으로 재현할 수 있다.

```bash
python scripts/pybullet_cup_benchmark.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --background-objects 256 --seeds 11 13 17 19 23 --steps 2 --sim-steps 120 --samples 12 --batch-size 4 --hidden-dim 32 --num-heads 4 --working-set-size 12 --runtime-repeats 1 --balanced-labels --out docs/experiments/pybullet_cup_benchmark_n256_baseline_screen.csv
```

Medium-training N_bg=256 matched benchmark는 다음 명령으로 재현할 수 있다.

```bash
python scripts/pybullet_cup_benchmark.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --background-objects 256 --seeds 11 13 17 19 23 --steps 8 --sim-steps 120 --samples 24 --batch-size 4 --hidden-dim 32 --num-heads 4 --working-set-size 12 --runtime-repeats 1 --balanced-labels --out docs/experiments/pybullet_cup_benchmark_n256_medium.csv
```

Large-background WPU-only PyBullet extension은 systems feasibility run이지
matched-baseline accuracy comparison이 아니다. Dense graph baseline은 attempted
protocol에서 완료되지 않았으므로 이 CSV에서 baseline accuracy를 추론하면 안 된다.

```bash
python scripts/pybullet_cup_benchmark.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense --background-objects 512 --seeds 11 13 17 19 23 29 31 --steps 12 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --runtime-repeats 2 --balanced-labels --out docs/experiments/pybullet_cup_benchmark_n512.csv
```

PyBullet closed-loop rollout diagnostic은 다음 명령으로 재현할 수 있다.

```bash
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer --horizons 5 10 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --out docs/experiments/pybullet_closed_loop_rollout.csv
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer --horizons 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --delta-clip 0.25 --out docs/experiments/pybullet_closed_loop_rollout_clipped.csv
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer --horizons 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --delta-clip 0.25 --integrity-projection --out docs/experiments/pybullet_closed_loop_rollout_guarded.csv
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense --horizons 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --delta-norm-penalty 0.05 --delta-target-norm-slack 0.5 --out docs/experiments/pybullet_closed_loop_rollout_regularized.csv
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer --horizons 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --unsafe-delta-reject-norm 10.0 --out docs/experiments/pybullet_closed_loop_rollout_rejected.csv
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense --horizons 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --rollout-consistency-penalty 0.01 --rollout-consistency-slack 0.5 --out docs/experiments/pybullet_closed_loop_rollout_consistency.csv
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense --horizons 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --state-validity-penalty 0.01 --out docs/experiments/pybullet_closed_loop_rollout_validity.csv
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense --horizons 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --state-validity-penalty 0.1 --out docs/experiments/pybullet_closed_loop_rollout_validity_strong.csv
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer --horizons 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --rollback-on-violation --out docs/experiments/pybullet_closed_loop_rollout_rollback.csv
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer --horizons 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --correct-on-violation --rollback-on-violation --out docs/experiments/pybullet_closed_loop_rollout_corrected_rollback.csv
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse --horizons 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --correct-on-violation --rollback-on-violation --escalation-model wpu-cws-indexed-local-dense --out docs/experiments/pybullet_closed_loop_rollout_escalated_corrected_rollback.csv
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer --horizons 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --delta-clip 0.25 --finite-delta-clamp 1.0 --out docs/experiments/pybullet_closed_loop_rollout_finite_clamped.csv
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer --horizons 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --delta-clip 0.25 --finite-delta-clamp 1.0 --correct-on-violation --out docs/experiments/pybullet_closed_loop_rollout_finite_corrected.csv
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse --horizons 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --delta-clip 0.25 --finite-delta-clamp 1.0 --correct-on-violation --selective-correction --out docs/experiments/pybullet_closed_loop_rollout_selective_corrected.csv
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse --horizons 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --delta-clip 0.25 --finite-delta-clamp 1.0 --correct-on-violation --selective-correction --correction-stride 2 --out docs/experiments/pybullet_closed_loop_rollout_selective_corrected_stride2.csv
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse --horizons 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --delta-clip 0.25 --finite-delta-clamp 1.0 --correct-on-violation --selective-correction --correction-violation-margin 1 --out docs/experiments/pybullet_closed_loop_rollout_selective_corrected_margin1.csv
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse --horizons 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --delta-clip 0.25 --finite-delta-clamp 1.0 --correct-on-violation --selective-correction --correction-entropy-threshold 0.35 --out docs/experiments/pybullet_closed_loop_rollout_selective_corrected_entropy035.csv
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse --horizons 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --delta-clip 0.25 --finite-delta-clamp 1.0 --correct-on-violation --selective-correction --correction-entropy-threshold 0.45 --out docs/experiments/pybullet_closed_loop_rollout_selective_corrected_entropy045.csv
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse --horizons 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --delta-clip 0.25 --finite-delta-clamp 1.0 --correct-on-violation --selective-correction --correction-raw-delta-threshold 2000000 --out docs/experiments/pybullet_closed_loop_rollout_selective_corrected_rawdelta2m.csv
python scripts/analyze_state_integrity.py --inputs docs/experiments/pybullet_closed_loop_rollout.csv docs/experiments/pybullet_closed_loop_rollout_clipped.csv docs/experiments/pybullet_closed_loop_rollout_guarded.csv docs/experiments/pybullet_closed_loop_rollout_regularized.csv docs/experiments/pybullet_closed_loop_rollout_rejected.csv docs/experiments/pybullet_closed_loop_rollout_consistency.csv docs/experiments/pybullet_closed_loop_rollout_validity.csv docs/experiments/pybullet_closed_loop_rollout_validity_strong.csv docs/experiments/pybullet_closed_loop_rollout_rollback.csv docs/experiments/pybullet_closed_loop_rollout_corrected_rollback.csv docs/experiments/pybullet_closed_loop_rollout_escalated_corrected_rollback.csv docs/experiments/pybullet_closed_loop_rollout_finite_clamped.csv docs/experiments/pybullet_closed_loop_rollout_finite_corrected.csv docs/experiments/pybullet_closed_loop_rollout_selective_corrected.csv docs/experiments/pybullet_closed_loop_rollout_selective_corrected_stride2.csv docs/experiments/pybullet_closed_loop_rollout_selective_corrected_margin1.csv docs/experiments/pybullet_closed_loop_rollout_selective_corrected_entropy035.csv docs/experiments/pybullet_closed_loop_rollout_selective_corrected_entropy045.csv docs/experiments/pybullet_closed_loop_rollout_selective_corrected_rawdelta2m.csv --labels raw clipped guarded regularized rejected consistency validity validity_strong rollback corrected_rollback escalated_corrected_rollback finite_clamped finite_corrected selective_corrected selective_corrected_stride2 selective_corrected_margin1 selective_corrected_entropy035 selective_corrected_entropy045 selective_corrected_rawdelta2m --out-csv docs/experiments/pybullet_state_integrity_audit.csv --out-md docs/experiments/pybullet_state_integrity_audit_results.md
python scripts/analyze_pybullet_correction_trigger_frontier.py
```

PyBullet local-law revision probe는 다음 명령으로 재현할 수 있다.

```bash
python scripts/pybullet_local_law_revision.py --train-samples 64 --calibration-samples 24 --eval-samples 48 --seeds 11 13 --background-objects 16 --sim-steps 120 --mechanisms nominal high_force edge_shift catch_heavy --out docs/experiments/pybullet_local_law_revision.csv
```

PyBullet systems profile은 다음 명령으로 재현할 수 있다.

```bash
python scripts/pybullet_system_profile.py --samples 8 --seeds 11 13 --background-objects 0 32 128 512 2048 --branch-counts 1 3 8 --sim-steps 24 --forward-repeats 3 --hidden-dim 64 --layers 2 --num-heads 4 --out docs/experiments/pybullet_system_profile.csv
python scripts/pybullet_system_profile.py --samples 4 --seeds 11 13 --background-objects 0 32 128 512 2048 --branch-counts 1 3 8 --sim-steps 24 --forward-repeats 5 --hidden-dim 64 --layers 2 --num-heads 4 --device cuda --out docs/experiments/pybullet_system_profile_cuda.csv
python scripts/analyze_system_energy_proxy.py
```

PyBullet shift-generalization 및 calibration benchmark는 다음 명령으로 재현할 수 있다.

```bash
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --eval-mechanisms nominal high_force edge_shift catch_heavy --seeds 11 13 17 19 23 29 31 --background-objects 32 --steps 20 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --out docs/experiments/pybullet_shift_generalization.csv
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --train-mechanisms nominal high_force edge_shift catch_heavy --eval-mechanisms nominal high_force edge_shift catch_heavy --seeds 11 13 17 --background-objects 32 --steps 16 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --calibrate-temperature --calibration-samples 48 --temperature-steps 30 --out docs/experiments/pybullet_shift_generalization_mixture_calibrated.csv
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --train-mechanisms high_force edge_shift catch_heavy --eval-mechanisms nominal --seeds 11 13 17 --background-objects 32 --steps 12 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --calibrate-temperature --calibration-samples 36 --temperature-steps 20 --out docs/experiments/pybullet_shift_leave_family_nominal.csv
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --train-mechanisms nominal edge_shift catch_heavy --eval-mechanisms high_force --seeds 11 13 17 --background-objects 32 --steps 12 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --calibrate-temperature --calibration-samples 36 --temperature-steps 20 --out docs/experiments/pybullet_shift_leave_family_high_force.csv
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --train-mechanisms nominal high_force catch_heavy --eval-mechanisms edge_shift --seeds 11 13 17 --background-objects 32 --steps 12 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --calibrate-temperature --calibration-samples 36 --temperature-steps 20 --out docs/experiments/pybullet_shift_leave_family_edge_shift.csv
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --train-mechanisms nominal high_force edge_shift --eval-mechanisms catch_heavy --seeds 11 13 17 --background-objects 32 --steps 12 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --calibrate-temperature --calibration-samples 36 --temperature-steps 20 --out docs/experiments/pybullet_shift_leave_family_catch_heavy.csv
python scripts/analyze_pybullet_leave_family_out.py
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --train-mechanisms nominal high_force edge_shift catch_heavy --eval-mechanisms no_catch edge_high_force edge_catch_heavy --seeds 11 13 17 --background-objects 32 --steps 12 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --calibrate-temperature --calibration-samples 36 --temperature-steps 20 --out docs/experiments/pybullet_shift_composition_stress.csv
python scripts/analyze_pybullet_shift_stress.py
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --train-mechanisms nominal high_force edge_shift catch_heavy --eval-mechanisms no_catch edge_high_force edge_catch_heavy --seeds 11 13 17 19 23 29 31 --background-objects 32 --steps 12 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --calibrate-temperature --calibration-samples 36 --temperature-steps 20 --out docs/experiments/pybullet_shift_composition_stress_7seed.csv
python scripts/analyze_pybullet_shift_stress.py --input docs/experiments/pybullet_shift_composition_stress_7seed.csv --out-csv docs/experiments/pybullet_shift_composition_stress_7seed_summary.csv --out-md docs/experiments/pybullet_shift_composition_stress_7seed_results.md --out-ko-md docs/experiments/pybullet_shift_composition_stress_7seed_results.ko.md
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --train-mechanisms nominal high_force edge_shift catch_heavy --eval-mechanisms no_catch edge_high_force edge_catch_heavy --seeds 11 13 17 --background-objects 32 --steps 12 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --calibrate-temperature --calibrate-bias --calibration-samples 36 --temperature-steps 20 --out docs/experiments/pybullet_shift_composition_stress_bias_calibrated.csv
python scripts/analyze_pybullet_shift_stress.py --input docs/experiments/pybullet_shift_composition_stress_bias_calibrated.csv --out-csv docs/experiments/pybullet_shift_composition_stress_bias_calibrated_summary.csv --out-md docs/experiments/pybullet_shift_composition_stress_bias_calibrated_results.md --out-ko-md docs/experiments/pybullet_shift_composition_stress_bias_calibrated_results.ko.md
python scripts/analyze_shift_calibration_comparison.py
python scripts/analyze_pybullet_branch_prior_shift.py
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --eval-mechanisms nominal high_force edge_shift catch_heavy --seeds 11 13 17 19 23 29 31 --background-objects 32 --steps 20 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --calibrate-mechanism-prior --mechanism-prior-samples 36 --out docs/experiments/pybullet_shift_generalization_mechanism_prior.csv
python scripts/analyze_pybullet_mechanism_prior_adaptation.py
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --eval-mechanisms nominal high_force edge_shift catch_heavy --seeds 11 13 17 19 23 29 31 --background-objects 32 --steps 20 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --calibrate-mechanism-prior --mechanism-prior-samples 36 --mechanism-prior-strengths 0 0.25 0.5 0.75 1.0 --out docs/experiments/pybullet_shift_generalization_prior_strength_sweep.csv
python scripts/analyze_pybullet_prior_strength_sweep.py
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --eval-mechanisms nominal high_force edge_shift catch_heavy --seeds 11 13 17 19 23 29 31 --background-objects 32 --steps 20 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --calibrate-mechanism-prior --select-mechanism-prior-strength --mechanism-prior-selection-metric nll_ece --mechanism-prior-selection-ece-weight 1.0 --mechanism-prior-samples 36 --mechanism-prior-strengths 0 0.25 0.5 0.75 1.0 --out docs/experiments/pybullet_shift_generalization_selected_prior.csv
python scripts/analyze_pybullet_selected_prior_adaptation.py
python scripts/pybullet_fewshot_mechanism_adaptation.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --eval-mechanisms nominal high_force edge_shift catch_heavy --seeds 11 13 17 19 23 29 31 --background-objects 32 --steps 20 --adaptation-steps 8 --adaptation-samples 36 --adaptation-lr 0.0005 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --out docs/experiments/pybullet_fewshot_mechanism_adaptation.csv
python scripts/analyze_pybullet_fewshot_adaptation.py
python scripts/analyze_pybullet_mechanism_adaptive_policy.py
python scripts/analyze_pybullet_shift_detector_policy.py
python scripts/pybullet_uncertainty_gated_recompute.py --train-mechanisms nominal high_force edge_shift catch_heavy --eval-mechanisms no_catch edge_high_force edge_catch_heavy --seeds 11 13 17 19 23 29 31 --background-objects 32 --steps 12 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --thresholds 0.34 0.4 0.45 0.5 0.55 0.6 0.65 --out docs/experiments/pybullet_uncertainty_gated_recompute.csv --out-md docs/experiments/pybullet_uncertainty_gated_recompute_results.md --out-ko-md docs/experiments/pybullet_uncertainty_gated_recompute_results.ko.md
python scripts/pybullet_learned_uncertainty_gate.py --train-mechanisms nominal high_force edge_shift catch_heavy --eval-mechanisms no_catch edge_high_force edge_catch_heavy --seeds 11 13 17 19 23 29 31 --background-objects 32 --steps 12 --sim-steps 120 --samples 36 --gate-samples 48 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --gate-steps 120 --gate-penalties 0 0.01 0.02 0.04 0.08 0.12 --out docs/experiments/pybullet_learned_uncertainty_gate.csv --out-md docs/experiments/pybullet_learned_uncertainty_gate_results.md --out-ko-md docs/experiments/pybullet_learned_uncertainty_gate_results.ko.md
python scripts/analyze_pybullet_mechanism_selective_calibration_gate.py
python scripts/analyze_pybullet_calibration_cost_frontier.py
python scripts/analyze_pybullet_system_claim_boundary.py
```

최신 candidate-oracle gap audit은 다음 명령으로 재현할 수 있다.

```bash
python scripts/analyze_candidate_oracle_gap.py --input docs/experiments/wpu_v2_retriever_invariant_set_scorer.csv --out-csv docs/experiments/wpu_v2_candidate_oracle_gap_v2.csv --out-md docs/experiments/wpu_v2_candidate_oracle_gap_v2_results.md
```

같은 명령은 priority 1 decomposition도 생성한다.

```text
docs/experiments/wpu_v2_candidate_oracle_gap_decomposition.csv
docs/experiments/wpu_v2_candidate_oracle_gap_decomposition.md
```

Candidate no-harm gate 요약은 conservative set-evaluator CSV에서 다음 명령으로
재현할 수 있다.

```bash
python scripts/analyze_candidate_noharm_gate.py
```

Direct candidate-regret gate probe와 요약은 다음 명령으로 재현할 수 있다.

```bash
python scripts/retriever_cross_seed_candidate_regret_gate_probe.py
python scripts/analyze_candidate_regret_gate.py
python scripts/retriever_cross_seed_candidate_regret_gate_probe.py --harmful-accept-weight 0.5 --safe-ranking-weight 0.1 --out docs/experiments/wpu_v2_candidate_regret_gate_penalty.csv
python scripts/analyze_candidate_regret_gate.py --input docs/experiments/wpu_v2_candidate_regret_gate_penalty.csv --out-csv docs/experiments/wpu_v2_candidate_regret_gate_penalty_summary.csv --out-md docs/experiments/wpu_v2_candidate_regret_gate_penalty_results.md --out-ko-md docs/experiments/wpu_v2_candidate_regret_gate_penalty_results.ko.md
python scripts/retriever_cross_seed_candidate_regret_gate_probe.py --feature-noise-std 0.02 --feature-dropout 0.05 --out docs/experiments/wpu_v2_candidate_regret_gate_perturbed.csv
python scripts/analyze_candidate_regret_gate.py --input docs/experiments/wpu_v2_candidate_regret_gate_perturbed.csv --out-csv docs/experiments/wpu_v2_candidate_regret_gate_perturbed_summary.csv --out-md docs/experiments/wpu_v2_candidate_regret_gate_perturbed_results.md --out-ko-md docs/experiments/wpu_v2_candidate_regret_gate_perturbed_results.ko.md
python scripts/retriever_cross_seed_candidate_safety_gate_probe.py --out docs/experiments/wpu_v2_candidate_safety_gate.csv
python scripts/analyze_candidate_regret_gate.py --input docs/experiments/wpu_v2_candidate_safety_gate.csv --out-csv docs/experiments/wpu_v2_candidate_safety_gate_summary.csv --out-md docs/experiments/wpu_v2_candidate_safety_gate_results.md --out-ko-md docs/experiments/wpu_v2_candidate_safety_gate_results.ko.md
python scripts/retriever_cross_seed_candidate_regret_crossfit_probe.py --out docs/experiments/wpu_v2_candidate_regret_crossfit.csv
python scripts/analyze_candidate_regret_gate.py --input docs/experiments/wpu_v2_candidate_regret_crossfit.csv --out-csv docs/experiments/wpu_v2_candidate_regret_crossfit_summary.csv --out-md docs/experiments/wpu_v2_candidate_regret_crossfit_results.md --out-ko-md docs/experiments/wpu_v2_candidate_regret_crossfit_results.ko.md
python scripts/retriever_cross_seed_candidate_invariant_gate_probe.py --out docs/experiments/wpu_v2_candidate_invariant_gate.csv
python scripts/analyze_candidate_regret_gate.py --input docs/experiments/wpu_v2_candidate_invariant_gate.csv --out-csv docs/experiments/wpu_v2_candidate_invariant_gate_summary.csv --out-md docs/experiments/wpu_v2_candidate_invariant_gate_results.md --out-ko-md docs/experiments/wpu_v2_candidate_invariant_gate_results.ko.md
python scripts/retriever_cross_seed_joint_candidate_gate_probe.py --out docs/experiments/wpu_v2_candidate_joint_gate.csv
python scripts/analyze_candidate_regret_gate.py --input docs/experiments/wpu_v2_candidate_joint_gate.csv --out-csv docs/experiments/wpu_v2_candidate_joint_gate_summary.csv --out-md docs/experiments/wpu_v2_candidate_joint_gate_results.md --out-ko-md docs/experiments/wpu_v2_candidate_joint_gate_results.ko.md
python scripts/retriever_cross_seed_joint_candidate_gate_probe.py --k-values 16 --regret-weight 3.0 --safe-bce-weight 0.25 --ranking-weight 0.1 --harmful-accept-weight 0.0 --group-dro-weight 0.0 --feature-noise-std 0.0 --context-dropout 0.0 --out docs/experiments/wpu_v2_candidate_joint_gate_regression_heavy_k16.csv
python scripts/analyze_candidate_regret_gate.py --input docs/experiments/wpu_v2_candidate_joint_gate_regression_heavy_k16.csv --out-csv docs/experiments/wpu_v2_candidate_joint_gate_regression_heavy_k16_summary.csv --out-md docs/experiments/wpu_v2_candidate_joint_gate_regression_heavy_k16_results.md --out-ko-md docs/experiments/wpu_v2_candidate_joint_gate_regression_heavy_k16_results.ko.md
python scripts/retriever_cross_seed_end_to_end_candidate_selector_probe.py --out docs/experiments/wpu_v2_end_to_end_candidate_selector.csv
python scripts/analyze_candidate_regret_gate.py --input docs/experiments/wpu_v2_end_to_end_candidate_selector.csv --out-csv docs/experiments/wpu_v2_end_to_end_candidate_selector_summary.csv --out-md docs/experiments/wpu_v2_end_to_end_candidate_selector_results.md --out-ko-md docs/experiments/wpu_v2_end_to_end_candidate_selector_results.ko.md
python scripts/retriever_cross_seed_joint_candidate_generator_probe.py --out docs/experiments/wpu_v2_joint_candidate_generator.csv
python scripts/analyze_joint_candidate_generator.py
python scripts/retriever_cross_seed_verified_candidate_controller_probe.py --out docs/experiments/wpu_v2_verified_candidate_controller.csv
python scripts/analyze_candidate_regret_gate.py --input docs/experiments/wpu_v2_verified_candidate_controller.csv --out-csv docs/experiments/wpu_v2_verified_candidate_controller_summary.csv --out-md docs/experiments/wpu_v2_verified_candidate_controller_results.md --out-ko-md docs/experiments/wpu_v2_verified_candidate_controller_results.ko.md
python scripts/retriever_cross_seed_joint_propagation_adapter_probe.py --out docs/experiments/wpu_v2_joint_propagation_adapter.csv
python scripts/analyze_candidate_regret_gate.py --input docs/experiments/wpu_v2_joint_propagation_adapter.csv --out-csv docs/experiments/wpu_v2_joint_propagation_adapter_summary.csv --out-md docs/experiments/wpu_v2_joint_propagation_adapter_results.md --out-ko-md docs/experiments/wpu_v2_joint_propagation_adapter_results.ko.md
python scripts/analyze_candidate_safety_frontier.py --inputs docs/experiments/wpu_v2_candidate_regret_gate_summary.csv docs/experiments/wpu_v2_candidate_regret_gate_perturbed_summary.csv docs/experiments/wpu_v2_candidate_regret_gate_penalty_summary.csv docs/experiments/wpu_v2_candidate_regret_crossfit_summary.csv docs/experiments/wpu_v2_end_to_end_candidate_selector_summary.csv --labels direct perturbed penalty crossfit end_to_end
python scripts/analyze_matched_accuracy_speedup.py
python scripts/analyze_matched_speedup_tolerance.py
python scripts/analyze_pybullet_pareto_frontier.py
```

보수적인 v2 우선순위 dashboard는 다음 명령으로 재현할 수 있다.

```bash
python scripts/analyze_pybullet_simulator_coverage.py
python scripts/audit_v2_priority_dashboard.py
```

## 현재 제출 경계

`docs/claims.ko.md`를 authoritative claim boundary로 사용한다. 현재 저장소가
지지하는 것은 regime-specific WPU hypothesis이며, token/graph/latent world model
또는 hardware accelerator baseline에 대한 보편 우월성이 아니다.

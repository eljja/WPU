# WPU Reproducibility Guide

This document lists the minimum commands needed to reproduce the repository's
current claims and quality checks. It separates committed evidence from
generated artifacts that must be reviewed before being promoted into `docs/`.

## Environment

Recommended baseline:

- Python 3.11
- Editable install with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

The default install pulls the standard PyTorch package. For CUDA-specific
experiments, install the PyTorch build matching the local driver/CUDA stack
first, then run the editable install.

Simulator-grounded experiments require the optional PyBullet dependency:

```bash
python -m pip install -e ".[dev,sim]"
```

On Windows, `python` can resolve to the Microsoft Store alias instead of an
installed interpreter. If so, create a virtual environment with the intended
Python installation and run checks through the venv interpreter:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

## Required Quality Checks

Run before committing paper, documentation, or model changes:

```bash
python -m pytest
python -m pip wheel . --no-deps --wheel-dir dist
python -m pip install --force-reinstall --no-deps dist/wpu-*.whl
python -c "import wpu; wpu.create_model('wpu-cws-indexed', hidden_dim=16, layers=1, working_set_size=4)"
```

The test suite covers:

- state model JSON and delta overlays;
- objectification contract reporting;
- scheduler behavior;
- sparse/dense/model shape paths;
- rollout probability normalization;
- script entrypoint hygiene;
- README core script smoke execution;
- documentation link integrity;
- LaTeX figure and citation integrity;
- experiment `Source CSV` integrity;
- robot-cup demo smoke output.

GitHub Actions runs the same test command on push and pull request. It also
compiles the English arXiv paper from `docs/arxiv/state_is_all_you_need_en.tex`
and uploads the resulting PDF as a workflow artifact, so paper changes must keep
both the evidence tests and the LaTeX build green.

On Windows machines with Anaconda on `PATH`, setuptools can occasionally import
Anaconda's `distutils` from inside a virtual environment. If wheel metadata
generation fails with a `distutils` assertion, run the wheel check with:

```powershell
$env:SETUPTOOLS_USE_DISTUTILS='stdlib'
.\.venv\Scripts\python.exe -m pip wheel . --no-deps --wheel-dir dist
```

## Demo Reproduction

```bash
python demos/robot_cup_demo.py
```

Expected output sections:

- event and initial frontier;
- scheduler and model path;
- frontier trace;
- changed objects;
- branch probabilities;
- memory estimate.

## Paper Build

The English PDF is generated from the LaTeX source:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs/arxiv docs/arxiv/state_is_all_you_need_en.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs/arxiv docs/arxiv/state_is_all_you_need_en.tex
```

The Korean companion is maintained as Markdown:

```text
docs/arxiv/state_is_all_you_need_ko.md
```

`docs/arxiv/state_is_all_you_need_en.tex` is the authoritative submission
source. The committed PDF is a review convenience artifact; rebuilds can change
PDF metadata without changing manuscript content. Treat the CI paper build and
the source/evidence consistency tests as the reproducibility checks.

## Experiment Artifact Policy

Committed paper evidence lives under `docs/experiments/` and `docs/figures/`.
New experiment runs should write first to `artifacts/`, which is ignored by git.

Promote a generated result into `docs/experiments/` only after checking:

- all intended seeds completed;
- model parameter scales are comparable for matched-baseline claims;
- `ObjectificationReport` metrics are logged when a claim depends on explicit
  object state, sparse retrieval, or perception/state adapters;
- relation-repair metrics are logged when repaired edges are used;
- every `Source CSV` or `Source CSVs` entry is git-tracked under
  `docs/experiments/` and nonempty;
- interpretation follows `docs/claims.md`;
- negative or mixed results are not hidden.

The 8M-class CWS GPU runner follows this policy:

```powershell
.\scripts\run_cws_8m_gpu.ps1 -Python python
```

It writes generated outputs to `artifacts/causal_working_set_8m_gpu/` for review
before promotion.

The PyBullet simulator-grounded benchmark can be reproduced with:

```bash
python scripts/pybullet_cup_benchmark.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --background-objects 0 32 128 --seeds 11 13 --steps 30 --sim-steps 120 --samples 48 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --runtime-repeats 5 --balanced-labels --out docs/experiments/pybullet_cup_benchmark.csv
```

The PyBullet objectification-corruption stress can be reproduced with:

```bash
python scripts/pybullet_objectification_stress.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer --corruptions clean drop_relations_light drop_relations_heavy position_noise low_confidence identity_swap combined --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --out docs/experiments/pybullet_objectification_stress.csv
```

The PyBullet objectification-quality decomposition benchmark can be reproduced
with:

```bash
python scripts/pybullet_objectification_quality.py --samples 12 --seeds 11 13 --background-objects 32 128 512 --corruptions clean drop_relations_heavy drop_objects_light position_noise low_confidence identity_swap combined --sim-steps 24 --out docs/experiments/pybullet_objectification_quality.csv
```

The parameter-matched PyBullet pilot can be reproduced with:

```bash
python scripts/pybullet_cup_benchmark.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --background-objects 0 128 --seeds 11 13 --steps 20 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --target-params 50000 --num-heads 4 --working-set-size 12 --runtime-repeats 5 --balanced-labels --out docs/experiments/pybullet_matched_baseline_benchmark.csv
python scripts/pybullet_cup_benchmark.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --background-objects 0 128 --seeds 11 13 17 19 23 29 31 --steps 20 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --runtime-repeats 3 --balanced-labels --out docs/experiments/pybullet_cup_benchmark_7seed.csv
```

The PyBullet closed-loop rollout diagnostic can be reproduced with:

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
python scripts/analyze_state_integrity.py --inputs docs/experiments/pybullet_closed_loop_rollout.csv docs/experiments/pybullet_closed_loop_rollout_clipped.csv docs/experiments/pybullet_closed_loop_rollout_guarded.csv docs/experiments/pybullet_closed_loop_rollout_regularized.csv docs/experiments/pybullet_closed_loop_rollout_rejected.csv docs/experiments/pybullet_closed_loop_rollout_consistency.csv docs/experiments/pybullet_closed_loop_rollout_validity.csv docs/experiments/pybullet_closed_loop_rollout_validity_strong.csv docs/experiments/pybullet_closed_loop_rollout_rollback.csv docs/experiments/pybullet_closed_loop_rollout_corrected_rollback.csv docs/experiments/pybullet_closed_loop_rollout_escalated_corrected_rollback.csv --labels raw clipped guarded regularized rejected consistency validity validity_strong rollback corrected_rollback escalated_corrected_rollback --out-csv docs/experiments/pybullet_state_integrity_audit.csv --out-md docs/experiments/pybullet_state_integrity_audit_results.md
```

The PyBullet local-law revision probe can be reproduced with:

```bash
python scripts/pybullet_local_law_revision.py --train-samples 64 --calibration-samples 24 --eval-samples 48 --seeds 11 13 --background-objects 16 --sim-steps 120 --mechanisms nominal high_force edge_shift catch_heavy --out docs/experiments/pybullet_local_law_revision.csv
```

The PyBullet systems profile can be reproduced with:

```bash
python scripts/pybullet_system_profile.py --samples 8 --seeds 11 13 --background-objects 0 32 128 512 2048 --branch-counts 1 3 8 --sim-steps 24 --forward-repeats 3 --hidden-dim 64 --layers 2 --num-heads 4 --out docs/experiments/pybullet_system_profile.csv
python scripts/pybullet_system_profile.py --samples 4 --seeds 11 13 --background-objects 0 32 128 512 2048 --branch-counts 1 3 8 --sim-steps 24 --forward-repeats 5 --hidden-dim 64 --layers 2 --num-heads 4 --device cuda --out docs/experiments/pybullet_system_profile_cuda.csv
python scripts/analyze_system_energy_proxy.py
```

The PyBullet shift-generalization and calibration benchmark can be reproduced
with:

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
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --train-mechanisms nominal high_force edge_shift catch_heavy --eval-mechanisms no_catch edge_high_force edge_catch_heavy --seeds 11 13 17 --background-objects 32 --steps 12 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --calibrate-temperature --calibrate-bias --calibration-samples 36 --temperature-steps 20 --out docs/experiments/pybullet_shift_composition_stress_bias_calibrated.csv
python scripts/analyze_pybullet_shift_stress.py --input docs/experiments/pybullet_shift_composition_stress_bias_calibrated.csv --out-csv docs/experiments/pybullet_shift_composition_stress_bias_calibrated_summary.csv --out-md docs/experiments/pybullet_shift_composition_stress_bias_calibrated_results.md --out-ko-md docs/experiments/pybullet_shift_composition_stress_bias_calibrated_results.ko.md
python scripts/analyze_shift_calibration_comparison.py
```

The latest candidate-oracle gap audit can be reproduced with:

```bash
python scripts/analyze_candidate_oracle_gap.py --input docs/experiments/wpu_v2_retriever_invariant_set_scorer.csv --out-csv docs/experiments/wpu_v2_candidate_oracle_gap_v2.csv --out-md docs/experiments/wpu_v2_candidate_oracle_gap_v2_results.md
```

The same command also emits the priority 1 decomposition:

```text
docs/experiments/wpu_v2_candidate_oracle_gap_decomposition.csv
docs/experiments/wpu_v2_candidate_oracle_gap_decomposition.md
```

The candidate no-harm gate summary can be reproduced from the conservative
set-evaluator CSV with:

```bash
python scripts/analyze_candidate_noharm_gate.py
```

The direct candidate-regret gate probe and summary can be reproduced with:

```bash
python scripts/retriever_cross_seed_candidate_regret_gate_probe.py
python scripts/analyze_candidate_regret_gate.py
python scripts/retriever_cross_seed_candidate_regret_gate_probe.py --harmful-accept-weight 0.5 --safe-ranking-weight 0.1 --out docs/experiments/wpu_v2_candidate_regret_gate_penalty.csv
python scripts/analyze_candidate_regret_gate.py --input docs/experiments/wpu_v2_candidate_regret_gate_penalty.csv --out-csv docs/experiments/wpu_v2_candidate_regret_gate_penalty_summary.csv --out-md docs/experiments/wpu_v2_candidate_regret_gate_penalty_results.md --out-ko-md docs/experiments/wpu_v2_candidate_regret_gate_penalty_results.ko.md
python scripts/retriever_cross_seed_candidate_regret_gate_probe.py --feature-noise-std 0.02 --feature-dropout 0.05 --out docs/experiments/wpu_v2_candidate_regret_gate_perturbed.csv
python scripts/analyze_candidate_regret_gate.py --input docs/experiments/wpu_v2_candidate_regret_gate_perturbed.csv --out-csv docs/experiments/wpu_v2_candidate_regret_gate_perturbed_summary.csv --out-md docs/experiments/wpu_v2_candidate_regret_gate_perturbed_results.md --out-ko-md docs/experiments/wpu_v2_candidate_regret_gate_perturbed_results.ko.md
python scripts/retriever_cross_seed_candidate_safety_gate_probe.py --out docs/experiments/wpu_v2_candidate_safety_gate.csv
python scripts/analyze_candidate_regret_gate.py --input docs/experiments/wpu_v2_candidate_safety_gate.csv --out-csv docs/experiments/wpu_v2_candidate_safety_gate_summary.csv --out-md docs/experiments/wpu_v2_candidate_safety_gate_results.md --out-ko-md docs/experiments/wpu_v2_candidate_safety_gate_results.ko.md
python scripts/analyze_candidate_safety_frontier.py
python scripts/analyze_matched_accuracy_speedup.py
python scripts/analyze_matched_speedup_tolerance.py
python scripts/analyze_pybullet_pareto_frontier.py
```

The conservative v2 priority dashboard can be reproduced with:

```bash
python scripts/audit_v2_priority_dashboard.py
```

## Current Submission Boundary

Use `docs/claims.md` as the authoritative claim boundary. The current repository
supports a regime-specific WPU hypothesis, not universal superiority over token,
graph, latent world-model, or hardware accelerator baselines.

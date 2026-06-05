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
```

The PyBullet closed-loop rollout diagnostic can be reproduced with:

```bash
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer --horizons 5 10 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --out docs/experiments/pybullet_closed_loop_rollout.csv
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer --horizons 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --delta-clip 0.25 --out docs/experiments/pybullet_closed_loop_rollout_clipped.csv
python scripts/pybullet_closed_loop_rollout.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer --horizons 25 --background-objects 32 --seeds 11 13 --steps 20 --sim-steps 120 --samples 24 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --delta-clip 0.25 --integrity-projection --out docs/experiments/pybullet_closed_loop_rollout_guarded.csv
python scripts/analyze_state_integrity.py --inputs docs/experiments/pybullet_closed_loop_rollout.csv docs/experiments/pybullet_closed_loop_rollout_clipped.csv docs/experiments/pybullet_closed_loop_rollout_guarded.csv --labels raw clipped guarded --out-csv docs/experiments/pybullet_state_integrity_audit.csv --out-md docs/experiments/pybullet_state_integrity_audit_results.md
```

The PyBullet local-law revision probe can be reproduced with:

```bash
python scripts/pybullet_local_law_revision.py --train-samples 64 --calibration-samples 24 --eval-samples 48 --seeds 11 13 --background-objects 16 --sim-steps 120 --mechanisms nominal high_force edge_shift catch_heavy --out docs/experiments/pybullet_local_law_revision.csv
```

The PyBullet systems profile can be reproduced with:

```bash
python scripts/pybullet_system_profile.py --samples 8 --seeds 11 13 --background-objects 0 32 128 512 2048 --branch-counts 1 3 8 --sim-steps 24 --out docs/experiments/pybullet_system_profile.csv
```

The PyBullet shift-generalization and calibration benchmark can be reproduced
with:

```bash
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --eval-mechanisms nominal high_force edge_shift catch_heavy --seeds 11 13 --background-objects 32 --steps 20 --sim-steps 120 --samples 36 --batch-size 8 --hidden-dim 64 --num-heads 4 --working-set-size 12 --out docs/experiments/pybullet_shift_generalization.csv
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

The conservative v2 priority dashboard can be reproduced with:

```bash
python scripts/audit_v2_priority_dashboard.py
```

## Current Submission Boundary

Use `docs/claims.md` as the authoritative claim boundary. The current repository
supports a regime-specific WPU hypothesis, not universal superiority over token,
graph, latent world-model, or hardware accelerator baselines.

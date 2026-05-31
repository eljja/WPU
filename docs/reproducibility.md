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

## Required Quality Checks

Run before committing paper, documentation, or model changes:

```bash
python -m pytest
```

The test suite covers:

- state model JSON and delta overlays;
- scheduler behavior;
- sparse/dense/model shape paths;
- rollout probability normalization;
- script entrypoint hygiene;
- documentation link integrity;
- LaTeX figure and citation integrity;
- experiment `Source CSV` integrity;
- robot-cup demo smoke output.

GitHub Actions runs the same test command on push and pull request.

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

## Experiment Artifact Policy

Committed paper evidence lives under `docs/experiments/` and `docs/figures/`.
New experiment runs should write first to `artifacts/`, which is ignored by git.

Promote a generated result into `docs/experiments/` only after checking:

- all intended seeds completed;
- model parameter scales are comparable for matched-baseline claims;
- `Source CSV` is included and nonempty;
- interpretation follows `docs/claims.md`;
- negative or mixed results are not hidden.

The 8M-class CWS GPU runner follows this policy:

```powershell
.\scripts\run_cws_8m_gpu.ps1 -Python python
```

It writes generated outputs to `artifacts/causal_working_set_8m_gpu/` for review
before promotion.

## Current Submission Boundary

Use `docs/claims.md` as the authoritative claim boundary. The current repository
supports a regime-specific WPU hypothesis, not universal superiority over token,
graph, latent world-model, or hardware accelerator baselines.

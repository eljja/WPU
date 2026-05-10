# WPU: World-State Processing Unit

![WPU compute architecture overview](docs/figures/wpu_compute_architectures.png)

This repository contains the first research prototype for **State Is All You
Need** and the **World-State Processing Unit (WPU)** idea.

WPU is not a chatbot memory system and it is not yet a chip design. It is a
PyTorch reference implementation and research scaffold for a state-native
execution model: worlds are represented as persistent objects, typed relations,
time, uncertainty, events, deltas, and future branches.

## Compute Context

WPU is positioned as a proposed world-state processing architecture alongside
CPU, GPU, TPU, NPU, and LPU designs. The intent is not to claim universal
superiority, but to isolate a different execution target: persistent state,
structured relations, sparse updates, temporal branches, and hierarchical
memory.

![WPU in the AI compute architecture landscape](docs/figures/wpu_compute_context.svg)

## Core Thesis

Token sequences can describe a world, but they do not make world-state
operations first-class. A WPU-style model treats these operations as the primary
interface:

- persistent object/relation state
- event frontier generation
- local causal propagation
- sparse, hybrid, and dense execution routing
- delta-state patching instead of full-state rewriting
- branch overlays for multiple futures
- uncertainty and branch probability updates

The current claim is intentionally conditional: WPU should help in regimes where
persistent identity, local causal change, uncertainty, and branching dominate.
The repository includes negative and mixed results where token/graph baselines
remain stronger.

## Hybrid Execution Architecture

The v1 reference model implements WPU as event-driven sparse propagation with a
dense tensor recompute fallback. A scheduler estimates the affected-state ratio,
fanout, propagation depth, and branch complexity, then routes execution through
sparse, hybrid, or dense paths.

![WPU hybrid sparse-dense execution architecture](docs/figures/wpu_hybrid_architecture.svg)

## Repository Layout

```text
wpu/                 PyTorch package and state/model implementation
tests/               Unit and smoke tests
demos/               End-to-end dataflow demo
scripts/             Training, evaluation, sweeps, plotting
docs/arxiv/          English LaTeX paper, Korean companion, generated PDF
docs/experiments/    Experiment reports
docs/figures/        Paper figures and README diagrams
docs/Review/         External review notes and response matrix
```

## Install

Python 3.11+ is recommended.

```bash
python -m pip install -e ".[dev]"
```

If PyTorch is not already installed, install a CPU or CUDA build appropriate for
your environment before running training jobs.

## Run The Demo

```bash
python demos/robot_cup_demo.py
```

Expected trace:

- event and initial frontier
- scheduler decision: sparse, hybrid, or dense
- changed objects and relation updates
- branch probabilities for stable, falls, and caught futures
- memory estimate for base state, deltas, and branches

## Train And Evaluate

```bash
python scripts/train_object_physics.py --steps 200 --batch-size 16 --seed 13 --checkpoint artifacts/object_physics_weighted.pt
python scripts/eval_object_physics.py --samples 256 --batch-size 16 --seed 101 --checkpoint artifacts/object_physics_weighted.pt
python scripts/route_sweep.py --samples 24 --batch-size 8 --background-sizes 0 20 80
```

The v1 model is `WorldStateProcessor`. It accepts batched state graphs, routes
through sparse/hybrid/dense execution paths, and predicts object deltas, relation
updates, uncertainty, and branch probabilities.

## Reproduce The Main Experiment Reports

The repository keeps raw generated experiment artifacts under `artifacts/`,
which is ignored by git. The committed reports and figures are under `docs/`.

```bash
python scripts/robust_experiment_suite.py --out-dir artifacts/robust_v1
python scripts/analyze_n_sweep.py
python scripts/analyze_b_sweep.py
python scripts/analyze_step_sweep.py
python scripts/analyze_controlled_stress.py
```

Key reports:

- `docs/experiments/README.md`
- `docs/experiments/robust_v1_results.md`
- `docs/experiments/n_sweep_v1_results.md`
- `docs/experiments/b_sweep_v1_results.md`
- `docs/experiments/step_sweep_v1_results.md`
- `docs/experiments/controlled_stress_v1_results.md`

## Paper

- English LaTeX: `docs/arxiv/state_is_all_you_need_en.tex`
- English PDF: `docs/arxiv/state_is_all_you_need_en.pdf`
- Korean companion: `docs/arxiv/state_is_all_you_need_ko.md`
- Compact research brief: `docs/paper/state_is_all_you_need.md`
- Review response and differentiation: `docs/Review/review_response_and_differentiation.md`

Build the PDF:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs/arxiv docs/arxiv/state_is_all_you_need_en.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs/arxiv docs/arxiv/state_is_all_you_need_en.tex
```

## Current Evidence Summary

The current evidence supports a regime hypothesis, not universal dominance.

- WPU-family models are competitive or best in the synthetic local regime up to
  about `N=108`, but the advantage disappears around `N≈120`.
- Routed WPU becomes faster than serialized-token around `N≈124` and faster
  than dense-graph around `N≈178` in the CPU v1 sweep.
- WPU-hybrid is robust under irrelevant relation noise: accuracy drop is
  `0.0250` from 0 to 128 noise edges, versus `0.3438` for Graph Transformer.
- Current WPU accuracy collapses in large-`N` regimes such as `N=204`; graph and
  token baselines remain stronger there.
- Fixed `rho` routing thresholds are useful for exposing route regimes but are
  not sufficient as a final scheduler.

The central v1 target is now precise: push the accuracy crossover beyond the
runtime crossover while preserving sparse routed work.

The next important steps are learned routing, stronger long-horizon dynamics,
real/simulator-backed benchmarks, state construction from perception, and
state-integrity mechanisms such as checkpoint and rollback.

## Test

```bash
python -m pytest
```

## License

This project is licensed under the **GNU Affero General Public License v3.0
only (AGPL-3.0-only)**. See [LICENSE](LICENSE).

The AGPL-3.0 network-copyleft requirement is intentional: if this code is
modified and offered as a network service, the corresponding source code must be
made available under the same license terms.

# WPU: World-State Processing Unit

![WPU compute architecture overview](docs/figures/wpu_compute_architectures.png)

This repository contains the first research prototype for **State Is All You
Need** and the **World-State Processing Unit (WPU)** idea.

WPU is not a chatbot memory system, not a universal Transformer replacement,
and not yet a chip design. It is a PyTorch reference implementation and research
scaffold for a state-native execution model: worlds are represented as
persistent objects, typed relations, time, uncertainty, events, deltas, and
future branches.

## Compute Context

WPU is positioned as a proposed world-state processing workload and execution
architecture alongside CPU, GPU, TPU, NPU, and LPU designs. The intent is not to
claim universal superiority, but to isolate a different execution target:
persistent state, structured relations, sparse updates, temporal branches, and
hierarchical memory. Hardware is a possible future target only after the
software runtime exposes real costs for frontier queues, relation fetch,
scatter/gather, delta logs, branch overlays, and sparse kernels.

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
remain stronger. The research goal is therefore not to show that WPU always wins,
but to map the `rho`, `N`, `B`, noise, and affected-region regimes where
state-native execution is useful.

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
- Korean README: `README.ko.md`
- Compact research brief: `docs/paper/state_is_all_you_need.md`
- Claim ledger: `docs/claims.md`
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

## WPU v2 Direction: State-Native Working-Set Control

The strongest recent v2 result is not a larger propagation block. It is a
state-native control loop before propagation: generate candidate causal working
sets, describe them with explicit role/geometry/family features, and choose a
retrieval mechanism with risk-adjusted train-seed evidence.

Earlier v2 work showed that a regret-distilled retriever, trained from candidate
sets that minimize downstream branch loss, beats a learned interaction retriever
in 14 of 15 seed/K conditions. The newer cross-seed result is stricter: it tests
mechanism selection under held-out seeds at `N=2048`.

Mean over five held-out seeds:

| K | Static learned loss | Risk-adjusted mechanism loss | Accuracy gain |
|---:|---:|---:|---:|
| 8 | 0.988432 | 0.982002 | 0.506667 -> 0.522222 |
| 16 | 0.966183 | 0.951243 | 0.504444 -> 0.517778 |
| 32 | 1.004095 | 1.002597 | 0.475556 -> 0.522222 |

This supports a sharper claim: explicit state is useful not only for sparse
propagation, but also because it exposes object-level working-set control as a
trainable, inspectable pre-propagation operation. Token baselines can serialize
the scene, but they do not naturally expose this intervention point.

The remaining v2 bottleneck is still the oracle gap. Opaque set evaluators,
score-margin gates, and strict no-harm seed-stable gates are not sufficient.
The next technical target is therefore:

- train retrieval and mechanism selection against downstream regret rather than
  teacher overlap;
- make candidate descriptors invariant across seeds and model instances;
- use risk-adjusted mechanism routing instead of a single opaque reranker;
- jointly train retriever and propagator instead of treating retrieval as a
  post-hoc selector;
- preserve sparse routed work while improving large-`N` accuracy;
- add long-horizon branch consistency, calibration, and state-integrity
  mechanisms;
- add object-state adapters for perception-to-state construction.

## Application Boundary

The near-term application target is software, not silicon: a state-update
runtime or middleware layer for workloads where the state is large but each
event changes a local subset. Plausible domains include digital twins,
simulation backends, game/server state synchronization, and robotics world-model
maintenance. Chiplet/IP or edge-processor claims are future hypotheses and
require hardware-aware profiling plus a non-empty regime where WPU is faster at
matched or acceptable accuracy.

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

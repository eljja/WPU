# WPU: World-State Processing Unit

[![CI](https://github.com/eljja/WPU/actions/workflows/ci.yml/badge.svg)](https://github.com/eljja/WPU/actions/workflows/ci.yml)

![WPU compute architecture overview](docs/figures/wpu_compute_architectures.png)

This repository contains the first research prototype for **State Is All You
Need** and the **World-State Processing Unit (WPU)** idea.

WPU is not a chatbot memory system, not a universal Transformer replacement,
and not yet a chip design. It is a PyTorch reference implementation and research
scaffold for a state-native execution model: worlds are represented as
persistent objects, typed relations, time, uncertainty, events, deltas, and
future branches.

The central primitive is **objectification**: converting a world into
persistent, addressable objects whose state, relations, uncertainty, deltas, and
branch overlays can be updated directly. Objectification is not merely assigning
type labels; it requires relation-bearing state variables such as role,
affordance, geometry, confidence, and history. See `docs/objectification.md`.

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
sparse, hybrid, or dense paths. The scheduler also accepts an objectification
score; low identity/relation/delta quality disables blind sparse routing and
escalates execution to hybrid or dense recompute.

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

The default install pulls the standard PyTorch package. If you need a specific
CUDA build, install that PyTorch build first, then run the editable install.

On Windows, verify that `python` resolves to a real interpreter rather than the
Microsoft Store alias. A reproducible local path is:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

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

## Minimal Public API

After installation, the core state-processing flow is available from the package
root:

```python
import wpu
from wpu.data.object_physics import create_robot_cup_state, create_touch_event

state = create_robot_cup_state()
event = create_touch_event()

event_delta = wpu.StateStore(state).apply_event(event)
sparse_delta = wpu.SparsePropagationEngine(max_depth=1).sparse_propagate(state, event).delta
dense_delta = wpu.DenseRecomputeEngine().dense_recompute(state, region=["cup_001"]).delta
objectification = wpu.evaluate_objectification(state, delta=event_delta)

print(event_delta.object_updates["cup_001"])
print(sparse_delta.object_updates["cup_001"])
print(dense_delta.object_updates["cup_001"])
print(objectification.contract_score)
```

This is the intended v1 interface: explicit world state is patched by event
deltas, propagated locally, and optionally recomputed over a bounded dense
region. `evaluate_objectification` checks whether the supplied state satisfies
the WPU object contract before propagation: stable identities, valid relation
endpoints, usable confidence, valid deltas, and optional causal-working-set
locality.

If object identity exists but relation extraction missed local edges,
`repair_objectification_relations` can add conservative geometry-inferred
relation patches before sparse propagation. This is a state repair heuristic,
not a claim that physical law has been solved. The repair probe shows why typed
objectification matters: geometry-only repair restores frontier recall but can
attach distractors, while type-gated repair preserves precision in the
controlled case. A small learned relation scorer now matches that type gate,
stays precise under denser distractors, and transfers across aliased type names
when role/affordance state variables are preserved. It fails when both type and
role information are removed, which makes the objectification boundary
measurable. The same toy probe now measures downstream branch prediction:
role-aware learned repair improves aliased-type accuracy from `0.343750` to
`0.671875` and lowers loss from `1.319667` to `0.885275`; ungated dense-distractor
repair restores frontier recall but worsens loss.

A second toy probe tests the longer-term objectification direction: learning
relation candidates from object histories rather than from type names. A history
scorer trained on `contact_transfer` and `support_transfer` transfers to held-out
`hidden_field`. Over five seeds it reaches mean relation precision/recall
`0.987500` and downstream accuracy `0.992188` versus `0.494531` for no relation
or type prior. This is a synthetic hidden-mechanism diagnostic, not evidence of
real physical-law discovery.

A third toy probe tests the next rung: fitting a simple local law on top of
objectified relation histories. A history-derived relation selector plus an
interpretable inverse-distance law transfers to renamed held-out
`hidden_inverse`, reaching five-seed mean relation precision/recall `0.988281`
and delta MSE `0.000828` versus `0.445909` for no relation or type prior. This
shows a controlled path from objectification to approximate local theory, but it
is still generated synthetic evidence.

The OOD version of the same probe maps the boundary: relation selection remains
useful under distance, gain, and law-form shifts, but far-distance recall drops
to `0.658594`, and shifted gain/law form leaves residual MSE even with oracle
relations. In WPU terms, objectification can expose a candidate local theory;
OOD stress is what decides whether that theory should be trusted or revised.

A revision probe then closes the loop with a small calibration set. Gain
calibration reduces `hidden_inverse_gain_shift` MSE from `0.115978` to
`0.000342`; form revision reduces `hidden_power_shift` MSE from `0.054596` to
`0.008887`. The oracle-relation form revision reaches `0.000232`, showing the
remaining gap is relation selection and noisy calibration, not only law form.
The package exposes this as metadata, not magic: `LocalLawHypothesis` records an
interpretable rule candidate and `evaluate_law_revision` reports whether a
stress-driven revision should be accepted, plus the relation-selection and
law-residual gaps when oracle-relation evidence is available.

The current v2 working-set models are also available from the package root
through the model factory:

```python
import wpu

model = wpu.create_model(
    "wpu-cws-indexed",
    hidden_dim=64,
    working_set_size=16,
)
```

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
- `docs/experiments/pybullet_cup_benchmark_results.md`
- `docs/experiments/pybullet_simulator_coverage_results.md`

## Paper

- English LaTeX: `docs/arxiv/state_is_all_you_need_en.tex`
- English PDF: `docs/arxiv/state_is_all_you_need_en.pdf`
- Korean companion: `docs/arxiv/state_is_all_you_need_ko.md`
- Korean README: `README.ko.md`
- Compact research brief: `docs/paper/state_is_all_you_need.md`
- Claim ledger: `docs/claims.md`
- Objectification definition: `docs/objectification.md`
- Publication readiness and gap register: `docs/publication_readiness.md`
- Reproducibility guide: `docs/reproducibility.md`
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
- Objectification is now a measured contract in the public API, but its quality
  has not yet been benchmarked for real perception-to-state adapters.
- The first PyBullet benchmark shows that simulator-generated rigid-body state
  can be objectified into `WorldState` and processed by the same WPU API. It is
  currently a systems/pipeline result, not an accuracy-dominance result.
- The PyBullet simulator coverage audit now separates simulator breadth from
  superiority claims. Baseline-complete cup evidence reaches 7 seeds and
  `N=133`; shift evidence covers 4 mechanism families; rollout diagnostics
  reach horizon 25; objectification-quality evidence covers 7 corruption
  settings; systems profiles reach `N≈2052`. The N_bg=512 cup extension runs
  WPU at total `N=517`, but it is WPU-only because the dense graph baseline did
  not complete under the attempted protocol, so it is not an accuracy-superiority
  result.
- The first PyBullet objectification stress shows that missing causal-frontier
  relations reduce WPU selected K before propagation. It also shows that the
  current objectification score must be extended with frontier completeness and
  semantic identity checks.
- The PyBullet objectification-quality benchmark makes that gap explicit:
  `ObjectificationReport` now includes frontier completeness and semantic
  consistency, and the benchmark shows relation-drop driving event-frontier
  recall to `0.585417` while position noise reduces semantic consistency to
  `0.675541`.
- A parameter-matched PyBullet pilot shows WPU sparse preserving accuracy from
  background N=0 to N=128 at roughly 50k parameters, while full-state baselines
  drop in this short run. Serialized-token remains faster at this scale, so the
  claim is regime-specific rather than universal latency dominance.
- The first PyBullet closed-loop rollout is a negative stability result:
  repeated WPU sparse deltas can explode by horizon 25. Delta clipping reduces
  violations but does not fix raw prediction instability, so WPU needs explicit
  state-integrity verification and correction.
- The PyBullet state-integrity audit makes that failure a tracked metric:
  raw WPU sparse falls to integrity `0.084722` at horizon 25. A guarded
  state-store projection raises applied-state integrity to `0.958508` for sparse
  WPU, but the raw delta norm remains unstable, so this is a safety layer rather
  than a solved dynamics model. Unsafe-delta rejection raises sparse integrity
  to `0.530270` only by rejecting `0.640000` of updates, so rejection rate must
  be reported next to integrity. A naive rollout-consistency penalty gives
  sparse H=25 integrity `0.084549`, and state-validity regularization remains
  at `0.084722`, so training-time validity penalties alone do not solve raw
  delta instability. A rollback-only memory layer raises sparse H=25
  applied-state integrity to `0.988647`, but only by rolling back `0.812500` of
  updates. A corrected-rollback variant reduces rollback rate to `0.564167`,
  but integrity falls to `0.900288`, so raw dynamics, correction quality, and
  memory safety must be reported separately. A sparse-first dense-escalation
  variant raises corrected-rollback integrity to `0.914831` and reduces
  rollback rate to `0.000000`, but it still invokes fallback frequently
  (`0.805833`), so this is a dense-when-needed safety-layer result rather than
  stable raw sparse dynamics. A finite-corrected variant is the strongest
  low-disruption memory-safety result so far: sparse H=25 integrity `0.958735`
  with rollback and escalation both `0.000000`, but correction rate remains high
  at `0.784166`.
- The first PyBullet local-law revision probe shows a bounded positive regime:
  simple object-state laws reduce cup-delta MSE under `high_force` and
  `edge_shift`, while `nominal` and `catch_heavy` expose overfitting and
  candidate-selection gaps. The claim is revisable local hypotheses, not
  unknown physical-law discovery.
- The PyBullet systems profile separates state/tensor/branch-memory costs:
  with background state up to `N≈2052.6`, indexed WPU tensorizes only `K≈4.6`
  objects and reduces tensor bytes by `0.997454`. A random-model CUDA profile
  shows sparse-forward latency reduction `0.996216`, but peak-memory reduction
  is only `0.304080`; this is systems evidence for pre-tensor state indexing,
  not proof of energy or matched-accuracy speedup.
- A screening-only energy proxy now combines tensorization latency with tensor
  bytes and CUDA forward latency with peak memory. It shows large proxy
  reductions at large `N`, but it is explicitly not wall-plug power, GPU power
  telemetry, or sparse-kernel evidence.
- The matched-or-better speedup audit is stricter: at `N=5`, WPU and
  serialized-token are accuracy-matched but WPU is slower; at `N=133`, WPU is
  both more accurate and faster than the best-accuracy non-WPU baseline
  (`graph-transformer`). This is positive large-N evidence, but not Pareto
  dominance over every baseline because serialized-token remains faster at
  lower accuracy. A separate Pareto audit places WPU on the accuracy-latency
  frontier at `N=133`, but not at `N=5`.
- The PyBullet shift-generalization benchmark adds calibration metrics under
  held-out mechanism families. In the 7-seed rerun, WPU local-dense leads on
  `catch_heavy`, but serialized-token remains stronger on `edge_shift` and
  `high_force`, so robust world-state generalization remains unsolved. A
  branch-prior audit changes the interpretation of `catch_heavy`: the
  non-learned majority prior reaches `0.753968`, far above the best WPU
  `0.408730`, so this is a prior-adaptation failure even though WPU beats the
  other learned baselines. A 7-seed mechanism-prior adaptation probe raises
  shifted WPU win-rate from `0.333333` to `0.666667` and removes the
  prior-dominated shift, but worsens mean shifted WPU ECE by `0.024819`.
  A follow-up prior-strength sweep finds `strength=0.75` is accuracy-best
  (shifted WPU win-rate `0.666667`, mean WPU accuracy `0.601852`), but no
  nonzero strength improves or preserves win-rate without increasing ECE
  relative to `strength=0`. Calibration-selected prior strength is more
  useful for P5: it improves shifted mean WPU accuracy by `0.145503`, ECE by
  `-0.046204`, and Brier by `-0.105470`, but shifted WPU-vs-baseline win-rate
  stays `0.333333`. Few-shot mechanism adaptation is stronger for P4 in an
  adapted protocol: shifted WPU win-rate reaches `1.000000`, mean WPU accuracy
  changes by `0.154762`, mean WPU-baseline margin by `0.050264`, and mean ECE
  by `-0.055342`. This uses mechanism-specific calibration samples and is not
  a zero-shot claim.
  A 3-seed
  calibrated mixture-training probe improves WPU on `edge_shift` but loses
  `catch_heavy` and worsens aggregate ECE ratio to `1.133834`, so post-hoc
  temperature calibration is not enough. A 3-seed leave-family-out probe is
  better for WPU, with win-rate `0.750000`, but still fails `catch_heavy`.
  A 7-seed composition-shift stress probe is accuracy-positive for WPU
  (win-rate `1.000000`, mean accuracy delta `0.071428`) but still not
  calibration-positive overall (mean ECE ratio `1.014879`, worst `no_catch`
  ratio `1.166073`), so accuracy and branch-probability reliability must be
  separated. Temperature+bias calibration improves
  `no_catch` ECE ratio to `0.960054`, but only improves 1/3 composition
  mechanisms, so calibration remains mechanism-aware rather than solved.

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
The latest gap audit quantifies this directly: risk-adjusted mechanism routing
closes only `0.244220` of the available candidate-oracle gain at best, and only
`0.042131` at `K=32`.
Direct candidate-regret deployment improves the conservative closure to
`0.328025` under train-selected deployment (`0.329950` in the test sweep), but
the candidate oracle remains substantially stronger and harmful accepts remain
near the safety limit. A separate safety/utility-head gate is a negative result:
best closure is only `0.147450`, safe best is `0.090719`, and train-selected
closure is `0.144863`. A cross-fit ensemble regret gate is also negative:
best closure is `0.287268`, safe best is `0.279738`, and cross-fit selected
closure is `0.270989`. P1 therefore needs better candidate scoring rather than
another post-hoc gate.
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

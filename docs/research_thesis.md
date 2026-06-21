# WPU Research Thesis

This document summarizes the intended scientific contribution of WPU and the
evidence boundary that all public documents should preserve.

## One-Sentence Thesis

WPU proposes that world processing should be studied as **state-native
computation**: persistent objects and typed relations are patched, propagated,
branched, and revised directly, instead of being repeatedly serialized into
token sequences or fully recomputed as dense tensors.

## What Is New

The novelty is not a single message-passing layer. The research contribution is
the combination of five ideas into one falsifiable execution model:

- **Objectification as a contract**: raw observations, simulator state, or logs
  must become persistent, addressable objects with attributes, relations,
  uncertainty, admissible deltas, and branch overlays before WPU can claim a
  state-native advantage.
- **Event-frontier execution**: computation starts from changed entities and
  propagates through typed causal relations instead of defaulting to full
  sequence attention or full graph recompute.
- **Delta-state memory**: futures are represented as `BaseState + DeltaState`,
  allowing branch sharing and copy-on-write state updates.
- **Sparse-first, dense-when-needed routing**: WPU treats sparse, hybrid, and
  dense execution as a measurable regime surface rather than a fixed ideology.
- **Claim-boundary instrumentation**: every claimed advantage is tied to
  `N`, causal working set `K`, branch count, objectification quality, state
  integrity, calibration, latency, and memory traffic.

## Core Scientific Claim

The defensible claim is conditional:

```text
WPU is useful when total world state N is large, the causal working set K is
small and identifiable before tensorization, updates are local and relation
mediated, and branch/uncertainty state is reused across events.
```

This is not a universal superiority claim over Transformers, Graph
Transformers, GPUs, TPUs, NPUs, LPUs, or dense world models. If `K` grows with
`N`, if objectification misses causal state, if retrieval scans the whole world,
or if long-horizon deltas drift, the WPU advantage can disappear.

## Why It Matters

The project makes a specific research bet: the next scaling bottleneck for some
world-model workloads is not only parameter count or matrix throughput, but the
execution interface between persistent world state and local causal change.
Modern token and dense tensor systems can represent state, but they do not make
object identity, relation traversal, branch overlays, and partial state updates
native operations. WPU asks whether making those operations native creates a
distinct accuracy/latency/memory regime.

## Current Evidence

The current repository supports an early but meaningful prototype-level claim:

- The full state model, objectification reports, delta overlays, sparse/dense
  schedulers, branch rollout, and PyTorch modules are implemented and tested.
- Synthetic v1 experiments map the first accuracy-runtime tension: WPU wins some
  local regimes but loses others, especially when accuracy collapses before the
  runtime crossover.
- v2 candidate-control experiments show that explicit state exposes a useful
  pre-propagation working-set surface, with a strong positive sub-regime at
  `N=2048`, `K=8`, and remaining larger-`K` gaps.
- A targeted large-N readout probe shows that the v1 `N>=204` branch collapse
  is partly caused by global mean readout over non-causal objects. Replacing it
  with event target/frontier readout keeps `wpu-sparse-frontier` at mean
  accuracy `0.781250` through total `N=404`, narrowly above serialized-token
  `0.778646` while using work proxy `3` versus `166464`.
- PyBullet experiments ground the claim in simulator-derived object state,
  including mechanism shift, calibration, objectification quality, and
  long-horizon rollout diagnostics.
- Relation-conditioned sparse propagation provides the strongest current
  large-state evidence: N=517, N=1029, and N=2053 baseline-complete screens are
  positive under non-causal distractor scaling, while N=4101 is WPU-only sparse
  feasibility and must not be treated as a baseline victory.
- Systems profiles show large tensor-byte and latency proxy reductions when
  indexed sparse execution keeps `K` small, but real power, sparse-kernel, and
  allocator-level evidence remain missing.

## Current Limits

The project is not yet a final process unit, chip design, or broad world model.
The unresolved scientific problems are:

- safe candidate generation at larger `K`;
- raw learned long-horizon dynamics rather than guarded state correction;
- baseline-complete large-N comparison beyond N=2053;
- harder causal large-N settings where `K` changes or grows;
- calibration-safe low-cost routing under shift;
- perception-to-state objectification from raw sensor inputs;
- real systems measurements for sparse kernels, memory traffic, and energy.

## Publication Posture

The right publication stance is:

```text
WPU is a falsifiable state-native execution hypothesis with a working PyTorch
reference implementation, strong negative-result discipline, and positive
evidence in specific small-K world-state regimes.
```

The wrong stance is:

```text
WPU already proves that state always beats tokens, or that a new hardware unit
is ready to replace GPUs/TPUs/NPUs/LPUs.
```

This framing gives the project its scientific value: it proposes a new
computational primitive, implements it, maps where it works, records where it
fails, and turns the remaining gaps into concrete experiments.

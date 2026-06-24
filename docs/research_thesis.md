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

The v3 definition of that target is fixed in `world_copy_model.md` and
`versions/wpu_v3_plan.md`. Those documents define world copy, object-oriented
processing, natural propagation, benchmark requirements, and success/failure
criteria before further implementation.

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
- The v3 world-copy index substrate adds hierarchical regions and a
  multi-signal causal index. In the controlled probe, selected `K` remains `4`
  while total `N` grows from `104` to `10004`; no non-causal background objects
  are selected, and the affected fraction at `N=10004` is `0.00039984`. This
  validates scalable causal-slice retrieval, not trained world modeling.
- The first v3 noisy-index stress benchmark extends this substrate evidence to
  `N=8192`, `K_ref=4/8/16`, missing relations, and false-positive relations.
  Region-scoped retrieval keeps recall at `1.000000` in the controlled setup
  and keeps `N=8192` touch ratio below `0.004385`. Without a relation-confidence
  gate, false-positive relations reduce mean precision to `0.800000`; with
  `min_relation_confidence=0.3`, precision returns to `1.000000` while recall
  stays `1.000000`. When true causal relation confidence is lowered to `0.2`,
  region scope recovers recall and precision, but the mean escalation signal is
  `0.981481`. The next boundary is whether local dense/hybrid correction after
  escalation improves propagation accuracy.
- The v3 escalation-correction probe tests that boundary at substrate level.
  With true relation confidence `0.2`, sparse confident-relation updates fall to
  mean recall `0.145833` and F1 `0.246623`; local hybrid escalation-region
  candidates recover mean recall/precision/F1 to `1.000000` while keeping max
  selected `K=16`. This validates bounded correction candidates after
  escalation, not learned transition quality or baseline superiority.
- The v3 learned-correction probe connects that recovered candidate set to a
  small learned local-delta head. With true relation confidence `0.2`,
  sparse confident-relation updates leave mean delta MSE `0.275312`, while
  hybrid escalation-region candidates reduce it to `0.006365` and keep max
  selected `K=16`. This is a controlled P2 substrate positive for learned local
  correction, not a baseline-complete world-model result.
- The updated v3 baseline-comparison screen is the first controlled P2 case
  where a WPU-native guard improves both raw delta error and work. The
  `wpu-region-guard` path keeps max selected `K=16`, mean work proxy
  `9.333333`, and mean bytes proxy `336.000000`, while reaching raw delta MSE
  `0.002646` versus dense graph `0.003810` and serialized token `0.003223`.
  The shallow `wpu-hybrid-context` variant is negative at MSE `0.020904`, so
  the useful fix is bounded local-region guarding, not generic context
  concatenation.
- The first v3 streaming region-guard probe extends that result from one-step
  delta prediction to H=25 controlled world-copy streams with object churn and
  region migration. `wpu-region-guard` keeps max selected `K=8`, trajectory MSE
  `0.000000`, integrity `1.000000`, correction cost `0.000000`, work proxy
  `8.000000`, and bytes proxy `288.000000`; dense state copy matches integrity
  but uses full-state work/bytes that grow with `N`. This is controlled
  oracle-law evidence, not real simulator dynamics.
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
- hierarchical state persistence under streaming updates;
- learned mechanism modules over causal slices rather than fixed local rules;
- causal-index recall under noisy or missing relations;
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

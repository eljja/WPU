# Robust WPU v1 Experiments

Date: 2026-05-09

This experiment package responds directly to reviewer criticism. It does not try
to prove universal WPU superiority. It tests whether a narrower regime claim
survives stronger baselines, multiple seeds, confidence intervals, route sweeps,
and measured forward latency.

Raw artifacts:

- `artifacts/robust_v1/learning_curves.csv`
- `artifacts/robust_v1/final_baselines.csv`
- `artifacts/robust_v1/regime_sweep.csv`
- `artifacts/robust_v1/runtime_memory.csv`
- `artifacts/robust_v1/summary.md`

Figures:

- `docs/figures/robust_accuracy_ci.png`
- `docs/figures/robust_accuracy_runtime.png`
- `docs/figures/robust_learning_curves.png`
- `docs/figures/robust_regime_work.png`

## Experiment Design

Models:

- `wpu-routed`
- `wpu-sparse`
- `wpu-hybrid`
- `wpu-dense`
- `dense-graph`
- `graph-transformer`
- `serialized-token`

Protocol:

- 5 seeds: `11, 13, 17, 19, 23`
- 150 training steps per model/seed
- 256 evaluation samples per condition
- object counts: `N=4, 24, 84, 204`
- branch pressure values: `B=1, 3, 8`
- CPU forward-latency profiling with batch size 16

Important implementation correction:

- Earlier `WorldStateProcessor` computed sparse, hybrid, and dense paths before
  selecting one. That made runtime evidence invalid.
- The current implementation performs hard routing first and executes only the
  selected path when all batch samples share a route.

## Experiment 1: Seeded Learning Curves and Confidence Intervals

The learning-curve experiment addresses the criticism that the earlier result
used a single seed and no uncertainty estimate. The new run reports mean and
approximate 95% confidence intervals over 5 seeds.

At `N=84`, most models improve rapidly by 50 training steps. WPU-hybrid and
WPU-sparse remain competitive at the end of training, while token and graph
baselines also learn the task. This means the result is not a one-seed artifact,
but it is also not a clean WPU-only win.

Interpretation:

- The task is learnable across seeds.
- WPU-hybrid has the strongest final mean accuracy at `N=84`.
- Token and graph baselines remain strong enough that the paper must not claim
  model-family dominance.

## Experiment 2: Strong Baseline Comparison

Final branch accuracy, mean +/- 95% CI:

| N | Best WPU | Accuracy | Best non-WPU | Accuracy | Interpretation |
|---:|---|---:|---|---:|---|
| 4 | WPU-hybrid | 0.7242 +/- 0.0260 | Dense graph | 0.6398 +/- 0.1257 | WPU variants win, routed scheduler fails because it selects dense. |
| 24 | WPU-hybrid | 0.7320 +/- 0.0280 | Graph transformer | 0.6609 +/- 0.0680 | WPU-hybrid wins in the medium local regime. |
| 84 | WPU-hybrid | 0.7508 +/- 0.0244 | Graph transformer | 0.6953 +/- 0.0388 | WPU remains competitive and best in this synthetic regime. |
| 204 | WPU-sparse/routed | 0.4516 +/- 0.1957 | Graph transformer | 0.7172 +/- 0.0615 | WPU fails; token/graph baselines dominate accuracy. |

This is the most important result. It supports a conditional regime claim, not a
universal claim. WPU wins or remains best up to `N=84` on this synthetic local
task, but loses badly at `N=204`.

The `N=204` failure is not cosmetic. It implies at least one of the following:

- the sparse propagation block lacks capacity at large `N`;
- the hard scheduler over-selects sparse paths;
- the synthetic distribution shifts with many background objects;
- the graph/token baselines exploit global pooling patterns better than the WPU
  path.

## Experiment 3: Regime Sweep over N, rho, and B

For `wpu-routed`, selected routes match the intended hard-threshold behavior:

| N | B | Route | Work proxy | Accuracy |
|---:|---:|---|---:|---:|
| 24 | 1 | sparse | 1.0 | 0.7258 +/- 0.0331 |
| 24 | 3 | hybrid | 27.0 | 0.6820 +/- 0.0508 |
| 24 | 8 | dense | 576.0 | 0.2570 +/- 0.1710 |
| 84 | 1 | sparse | 1.0 | 0.7531 +/- 0.0332 |
| 84 | 3 | sparse | 3.0 | 0.7531 +/- 0.0332 |
| 84 | 8 | hybrid | 92.0 | 0.7398 +/- 0.0543 |
| 204 | 3 | sparse | 3.0 | 0.4594 +/- 0.1551 |

What this supports:

- The scheduler produces the predicted sparse/hybrid/dense crossover.
- Low `rho` can reduce routed work proxy by orders of magnitude.
- At `N=84`, low-work sparse routing preserves accuracy.

What this refutes or weakens:

- The fixed hard thresholds are not optimal. `N=24, B=8` routes to dense and
  collapses in accuracy.
- Low work proxy alone is insufficient. `N=204, B=3` has very low work but poor
  accuracy.

Conclusion: the regime mechanism is real, but the scheduler must become learned
or hardware/accuracy-aware before the paper can claim robust routing.

## Experiment 4: Runtime and Memory Profiling

CPU forward latency at `B=3`, mean +/- 95% CI in ms/sample:

| N | WPU-routed | WPU-sparse | Dense graph | Graph transformer | Serialized token |
|---:|---:|---:|---:|---:|---:|
| 24 | 0.7971 +/- 0.1572 | 0.8775 +/- 0.0985 | 0.2790 +/- 0.0035 | 1.2025 +/- 0.0021 | 0.3335 +/- 0.0034 |
| 84 | 0.7967 +/- 0.1591 | 0.8881 +/- 0.1345 | 0.4844 +/- 0.0047 | 1.5594 +/- 0.0072 | 0.6977 +/- 0.0179 |
| 204 | 0.7570 +/- 0.1756 | 0.9576 +/- 0.1338 | 1.5470 +/- 0.0190 | 2.6152 +/- 0.0155 | 1.7522 +/- 0.0567 |

Interpretation:

- At small and medium `N`, dense graph and serialized-token baselines are faster
  in this PyTorch CPU implementation.
- At `N=204`, routed WPU is faster than dense graph, graph-transformer, and
  serialized-token baselines.
- Runtime advantage appears only when `N` is large enough for dense/token
  computation to dominate, but this is exactly where current WPU accuracy fails.

This creates the central v1 tension:

```text
WPU efficiency advantage appears at large N.
WPU accuracy advantage currently appears at medium N.
The unsolved problem is to make these regimes overlap.
```

Memory notes:

- `tracemalloc` captures Python-level peak allocation, not full native tensor
  memory.
- RSS deltas are noisy on Windows because allocator reuse and garbage collection
  can produce zero or negative deltas.
- Strong memory claims require a dedicated CUDA or allocator-level profiling
  setup.

## Paper-Level Conclusion

The stronger experiments do not support the claim that state-native WPU models
are universally better than token or graph baselines. They do support a narrower,
more defensible claim:

```text
State-native propagation is a plausible computational primitive for local,
persistent, branchable world-state workloads. Its advantage is regime-dependent:
it can improve accuracy in medium local synthetic regimes and reduce work/latency
when N is large, but v1 does not yet combine both advantages at the largest N.
```

This is a scientifically stronger position than the earlier version because it
is falsifiable and includes negative results.

## Required Next Step

The next model revision should target the `N=204` failure directly:

- replace fixed rho thresholds with learned routing;
- increase sparse propagation depth/capacity;
- add attention only inside the affected frontier, not globally;
- balance background-object distributions so object count does not create a
  hidden distribution shift;
- rerun the same 5-seed suite without changing the reporting protocol.

# WPU Version 1 Closure

This document closes WPU v1 as a research prototype and defines what can be
claimed from the current evidence. It intentionally separates supported claims
from claims that remain open.

## V1 Thesis

WPU v1 should not be presented as a universal replacement for token processors,
graph transformers, GPUs, NPUs, TPUs, or LPUs. The defensible v1 thesis is
narrower:

> World-state processing is useful when a large explicit state contains a small
> event-conditioned causal working set. In that regime, a processor that
> retrieves, propagates, and patches state deltas can reduce work relative to
> serialized-token or dense graph processing.

The key distinction is not "tokens are impossible" versus "state is possible."
Tokens can encode state. The v1 claim is that tokenization makes persistent
identity, localized mutation, relation-indexed access, and delta branching
non-native operations. WPU makes those operations primary.

## Supported Evidence

The strongest current evidence is the event-conditioned CWS 8M GPU sweep:

- Source CSV: `docs/experiments/cws_balanced_branch_8m_gpu_event_conditioned_dense_n_sweep.csv`
- Report: `docs/experiments/cws_balanced_branch_8m_gpu_event_conditioned_dense_n_sweep_results.md`
- World sizes: `N = 64, 128, 256, 512, 1024, 2048, 4096, 8192`
- Causal working set: `K = 8`
- Seeds: five per model and N
- Models: `wpu-cws-oracle`, `wpu-cws-learned`, `serialized-token`, `graph-transformer`
- Hardware: RTX 5070 Ti, PyTorch CUDA

The result supports a regime-specific claim:

| N | Best WPU accuracy | Token accuracy | Graph accuracy | Fastest WPU ms/sample | Token ms/sample | Graph ms/sample |
| --- | --- | --- | --- | --- | --- | --- |
| 2048 | 0.6160 | 0.3333 | 0.3960 | 2.007 | 3.063 | 6.386 |
| 4096 | 0.6947 | 0.3333 | 0.3567 | 1.684 | 7.728 | 9.958 |
| 8192 | 0.5193 | 0.3333 | 0.3947 | 2.923 | 24.803 | 27.096 |

The important pattern is not monotonic accuracy. The important pattern is that
latency grows much more slowly for WPU-style bounded working-set computation
than for serialized-token or dense graph baselines.

## Negative Evidence

Several earlier figures should not be used as main evidence for WPU superiority:

- `docs/figures/n_sweep_accuracy.png`
- `docs/figures/robust_regime_work.png`
- `docs/figures/step_sweep_accuracy.png`

These figures show that naive routing and sparse update alone are insufficient.
They are useful as ablations: WPU fails when the causal working set is not
reliably identified, when branch prediction is not event-conditioned, or when
the sparse path drops necessary context.

## Final V1 Claim Boundary

V1 can claim:

- State-native processing is a meaningful alternative computational framing.
- WPU has a measurable advantage in large-N/small-K synthetic regimes.
- Bounded causal working-set propagation can reduce forward latency relative to
  token and dense graph baselines at large N.
- Oracle CWS performance shows that the WPU core is plausible when causal
  access is correct.
- Learned CWS performance shows that selector reliability is the primary
  remaining bottleneck.

V1 cannot claim:

- WPU is universally better than token processors.
- Larger N automatically makes WPU better.
- Current learned selection is robust enough for real-world deployment.
- Current one-step branch accuracy proves long-horizon physical understanding.
- Current PyTorch implementation proves hardware-level energy advantage.

## Paper-Level Framing

The v1 paper should be written as a regime discovery paper, not a universal
superiority paper.

Recommended title remains unchanged if desired, but the abstract and conclusion
should state:

1. Tokens and state are different computational interfaces, not mutually
   exclusive representations.
2. WPU treats world state as persistent identity-addressed memory.
3. The central operation is event-conditioned causal propagation over a bounded
   working set.
4. Empirical benefit appears when `N >> K` and K is identifiable.
5. Naive sparse routing fails; causal working-set retrieval is the core problem.

## Figures To Keep In Main Paper

The main paper should use only figures that directly support the narrowed
claim:

- Token sequence versus persistent state graph schematic.
- WPU architecture: state store, causal index, propagation core, delta/branch
  manager.
- Event-conditioned CWS dense N-sweep accuracy.
- Event-conditioned CWS dense N-sweep latency or accuracy-latency surface.
- One ablation figure showing naive routing failure or selector gap.

Older broad sweep figures should move to supplement as negative or diagnostic
evidence.

## V1 Closure Decision

WPU v1 is complete as a research prototype if the paper makes the following
honest statement:

> WPU v1 identifies a computational regime rather than proving universal
> dominance. Its advantage emerges when explicit world state is large, causal
> working sets are small, and event-conditioned retrieval is reliable. The next
> version must expand that regime by improving causal selection, adaptive
> propagation, branch calibration, and indexed state access.


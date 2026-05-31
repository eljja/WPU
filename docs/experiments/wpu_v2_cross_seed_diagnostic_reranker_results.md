# WPU v2 Cross-Seed Diagnostic Reranker Probe

Source CSV: `docs/experiments/wpu_v2_retriever_cross_seed_diagnostic_reranker.csv`

## Question

Prior v2 experiments showed that generated state candidates improve the
same-seed deployed reranker, but leave-one-seed-out transfer remains weak. This
probe asks whether candidate-level model diagnostics can make reranking more
seed-invariant.

The added diagnostic features are available at inference time:

- normalized branch entropy
- maximum branch probability
- tanh-scaled top-2 branch logit margin

The experiment also tests context ablations:

- `full`: object-set encoder + selector identity + selector type + set summaries + diagnostics
- `no_identity`: remove candidate one-hot identity
- `set_only`: remove diagnostics
- `diagnostics_only`: remove selector identity/type and set-summary context, but keep object-set encoder and diagnostics

## Protocol

Command:

```powershell
.\.venv\Scripts\python.exe scripts\retriever_cross_seed_diagnostic_reranker_probe.py --n-values 2048 --k-values 8 16 32 --seeds 11 13 17 19 23 --budget 4 --generated-candidates 4 --propagation-steps 40 --retriever-steps 400 --reranker-steps 600 --validation-samples 180 --samples 90 --batch-size 10 --device cuda --safe-margin 0.005 --cv-safe-margin 0.0 --cv-min-win-rate 0.5 --context-variants full no_identity set_only diagnostics_only --out docs\experiments\wpu_v2_retriever_cross_seed_diagnostic_reranker.csv
```

Each held-out seed trains the scorer on the other four seeds. The WPU propagation
model and learned retriever are trained per seed as before. The report therefore
tests cross-seed scorer transfer, not same-seed model selection.

## Results

Mean over five held-out seeds:

| K | policy | loss | accuracy | excess over generated oracle |
|---:|---|---:|---:|---:|
| 8 | static base choice | 0.989850 | 0.502222 | 0.033382 |
| 8 | same-seed generated reranker | 0.973823 | 0.528889 | 0.017355 |
| 8 | normalized cross-seed reranker | 0.988200 | 0.517778 | 0.031732 |
| 8 | diagnostic full cross-seed | 0.986111 | 0.508889 | 0.029643 |
| 8 | best diagnostic variant (`diagnostics_only`) | 0.985296 | 0.513334 | 0.028828 |
| 8 | generated oracle | 0.956468 | 0.557778 | 0.000000 |
| 16 | static base choice | 0.966183 | 0.504444 | 0.060115 |
| 16 | same-seed generated reranker | 0.951378 | 0.511111 | 0.045310 |
| 16 | normalized cross-seed reranker | 0.962778 | 0.506667 | 0.056710 |
| 16 | diagnostic full cross-seed | 0.962290 | 0.508889 | 0.056222 |
| 16 | best diagnostic variant (`no_identity`) | 0.956255 | 0.513333 | 0.050187 |
| 16 | generated oracle | 0.906068 | 0.580000 | 0.000000 |
| 32 | static base choice | 1.004095 | 0.475556 | 0.035372 |
| 32 | same-seed generated reranker | 0.999229 | 0.506667 | 0.030506 |
| 32 | normalized cross-seed reranker | 1.003886 | 0.495556 | 0.035163 |
| 32 | diagnostic full cross-seed | 1.003898 | 0.484445 | 0.035175 |
| 32 | best diagnostic variant (`diagnostics_only`) | 1.001405 | 0.495556 | 0.032682 |
| 32 | generated oracle | 0.968723 | 0.577778 | 0.000000 |

## Interpretation

The result is a partial improvement, not a solved v2 mechanism.

Diagnostic features improve cross-seed loss relative to the static base selector
at K=8, K=16, and K=32 when the best context variant is selected. However, the
gain remains far smaller than the same-seed generated reranker, and the gap to
the generated oracle remains large.

Selector identity is not a reliable invariant feature. Removing candidate
identity gives the best K=16 result, while `diagnostics_only` is best at K=8 and
K=32. This supports the earlier ablation conclusion: the scorer should not rely
primarily on fixed selector names. It needs candidate-local state evidence.

The CV safety gate is not yet sufficient. It predicts some useful transfer
signal, but it does not consistently choose the test-best variant or reliably
avoid weak K=32 deployment. The current gate should be treated as a diagnostic,
not as a final deployment rule.

## Consequence for WPU v2

The v2 bottleneck is now narrower:

- Candidate generation is useful because generated candidates expand the oracle.
- Object-set reranking is useful in same-seed evaluation.
- Cross-seed transfer fails because candidate scoring is still too dependent on
  seed/model-specific calibration.

The next mechanism should train retrieval and propagation together against
downstream regret, or train the scorer with explicit cross-seed/cross-model
invariance. More generated candidates alone are unlikely to solve the problem
unless the scorer becomes invariant enough to rank them.

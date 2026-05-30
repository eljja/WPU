# WPU v2 Diagnostic Variant Selector

## Question

The diagnostic reranker experiment found that the best context variant changes
with causal working-set size `K`. Selecting the best variant after seeing the
test result is not deployable. This analysis asks whether a variant can be chosen
using only train-seed evidence.

## Method

Input:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_diagnostic_variant_selector.py --input docs\experiments\wpu_v2_retriever_cross_seed_diagnostic_reranker.csv --out docs\experiments\wpu_v2_diagnostic_variant_selector_summary.csv
```

The script compares three variant-selection criteria per held-out seed:

- `min_cv_delta`: choose the variant with the lowest leave-one-train-seed CV loss delta.
- `max_cv_win_then_delta`: choose the variant with the highest CV win rate, then lowest CV delta.
- `best_train_loss_delta`: choose the variant whose scorer most improves train-seed loss over static base selection.

No held-out test loss is used when selecting the variant.

## Results

| K | criterion | loss | accuracy | delta loss vs static base | excess over generated oracle |
|---:|---|---:|---:|---:|---:|
| 8 | min CV delta | 0.987480 | 0.511111 | -0.002370 | 0.031012 |
| 8 | max CV win then delta | 0.987569 | 0.504444 | -0.002280 | 0.031102 |
| 8 | best train loss delta | 0.984863 | 0.515556 | -0.004986 | 0.028396 |
| 16 | min CV delta | 0.961623 | 0.504444 | -0.004560 | 0.055555 |
| 16 | max CV win then delta | 0.961674 | 0.508889 | -0.004509 | 0.055606 |
| 16 | best train loss delta | 0.962290 | 0.508889 | -0.003892 | 0.056222 |
| 32 | min CV delta | 1.004337 | 0.484445 | 0.000242 | 0.035614 |
| 32 | max CV win then delta | 1.004817 | 0.482222 | 0.000722 | 0.036094 |
| 32 | best train loss delta | 1.002119 | 0.486667 | -0.001975 | 0.033396 |

## Interpretation

The deployable selector improves the diagnostic reranker story, but it does not
solve the main v2 bottleneck.

`best_train_loss_delta` is the only criterion that improves over the static base
selector at K=8, K=16, and K=32. This is useful because it converts the previous
"best variant by test inspection" result into a train-only selection rule.

However, the improvement remains small and still falls well short of the
generated oracle. The result strengthens the conclusion that WPU v2 needs an
invariant state scorer, not only more candidates or better post-hoc gates.

## Consequence

For paper-level claims, this should be presented as a mechanism audit:

- WPU's explicit state representation enables candidate generation, candidate
  diagnostics, and train-only selector selection.
- Current deployed scoring is not yet strong enough to claim broad robust
  superiority.
- The next implementation target is joint retriever-propagator training or an
  explicitly invariant candidate scorer trained across seeds/models.

# WPU V2 Learned Retriever Cross-K Generalization

## Question

The integrated learned retriever was trained separately for each causal working
set size K. That leaves a weakness:

```text
Is the learned state retriever a real reusable retrieval rule, or does it only
work after being retrained for each K regime?
```

This probe evaluates cross-K generalization.

## Implementation

Script:

- `scripts/learned_retriever_cross_k_probe.py`

Output:

- `docs/experiments/wpu_v2_learned_retriever_cross_k_probe.csv`

The probe trains small state-native MLP retrievers and evaluates selected
working-set composition against the interaction-density teacher.

During this experiment, a concrete problem appeared: mixed-K training without
K-regime context let large-K candidate sets dominate the retriever and degraded
K=8/K=16 selection. The fix was to add state-native fanout context features:

- event-local candidate frontier size,
- obstacle frontier size.

These are not token features. They are retrieval metadata available from the
explicit state index.

## Setup

- N = 2048
- Budget = 4
- Train seeds = 11, 13, 17
- Test seeds = 19, 23
- Train groups:
  - K=16 only
  - mixed K=8,16,32
- Test K = 8, 16, 32
- Balanced train-K loss = enabled

## Result

| train K | test K | learned teacher overlap | hand hit | selected obstacles | pair density |
| --- | --- | --- | --- | --- | --- |
| 16 | 8 | 0.878 | 1.000 | 2.000 | 0.547 |
| 16 | 16 | 0.939 | 1.000 | 2.000 | 0.822 |
| 16 | 32 | 0.475 | 1.000 | 2.000 | 0.784 |
| 8,16,32 | 8 | 0.968 | 1.000 | 2.000 | 0.566 |
| 8,16,32 | 16 | 0.958 | 1.000 | 2.000 | 0.856 |
| 8,16,32 | 32 | 0.955 | 1.000 | 2.000 | 0.875 |

## Interpretation

The single-K retriever generalizes from K=16 to K=8, but not to K=32. This is a
useful negative boundary: state-native retrieval is not automatically
scale-invariant.

The mixed-K retriever with fanout context generalizes across all tested K
values. It preserves the hand anchor, selects two obstacles, and maintains high
teacher overlap across K=8,16,32.

## Design Consequence

The WPU retriever should be K-regime aware, but not by using token sequence
length. It should expose explicit state-index metadata:

- candidate frontier size,
- local relation fanout,
- obstacle/contact/support frontier composition,
- retrieval budget,
- selector confidence and margin.

Updated v2 claim:

```text
State-native retrieval can generalize across causal working-set regimes when
the retriever receives explicit state-index context. Without that context,
learned retrieval can overfit to a specific K regime.
```

## Next Step

The next integrated experiment should use the mixed-K/fanout-context retriever
inside the downstream WPU pipeline and test whether one retriever can replace
per-K retriever training.

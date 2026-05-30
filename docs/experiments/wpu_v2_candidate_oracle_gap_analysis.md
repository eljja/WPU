# WPU v2 Candidate Oracle Gap Analysis

## Purpose

The cross-seed set-evaluator experiment produced a useful negative result. This
note separates two questions that should not be conflated:

- Does the candidate pool contain better causal working sets?
- Can the deployed model select those working sets under held-out seeds?

The answer is yes to the first question and no to the second for the current
set evaluator.

## Evidence

Mean results over five held-out seeds at `N=2048`:

| K | policy | loss | accuracy | excess over candidate oracle | oracle match |
|---:|---|---:|---:|---:|---:|
| 8 | static learned interaction | 0.988432 | 0.506667 | 0.032897 | 0.006667 |
| 8 | set evaluator | 0.989208 | 0.502222 | 0.033674 | 0.137778 |
| 8 | candidate oracle | 0.955534 | 0.557778 | 0.000000 | 1.000000 |
| 16 | static learned interaction | 0.966183 | 0.504444 | 0.060800 | 0.002222 |
| 16 | set evaluator | 0.969430 | 0.497778 | 0.064047 | 0.133333 |
| 16 | candidate oracle | 0.905383 | 0.580000 | 0.000000 | 1.000000 |
| 32 | static learned interaction | 1.004095 | 0.475556 | 0.035643 | 0.004444 |
| 32 | set evaluator | 1.001607 | 0.511111 | 0.033156 | 0.162222 |
| 32 | candidate oracle | 0.968451 | 0.580000 | 0.000000 | 1.000000 |

## Gap Decomposition

| K | candidate-oracle gain over static | deployed set-evaluator delta vs static | remaining deployed gap |
|---:|---:|---:|---:|
| 8 | 0.032897 | -0.000776 | 0.033674 |
| 16 | 0.060800 | -0.003248 | 0.064047 |
| 32 | 0.035643 | 0.002488 | 0.033156 |

Positive oracle gain means that better working sets are present in the candidate
pool. Positive deployed delta means the set evaluator improved over the static
selector. The current evaluator improves only at `K=32` and harms `K=8` and
`K=16`.

## Interpretation

The bottleneck is not evidence availability. It is transfer-stable candidate
selection.

The set evaluator increases oracle-match rate from near zero to about
`0.13--0.16`, so it learns some useful signal. However, the selected candidates
are not reliably the candidates that minimize held-out downstream loss. The
failure mode is therefore consistent with cross-seed overfitting or missing
state-invariant candidate features, not simply insufficient candidate diversity.

This result strengthens the scientific framing: WPU v2 should not claim that
state-native candidate generation alone solves large-`N` causal working-set
selection. The supported claim is narrower and more falsifiable:

```text
Explicit state enables sparse working-set generation, composition constraints,
and downstream-regret selection before propagation, but robust deployment
requires candidate scorers whose features are invariant across world seeds,
object layouts, and propagator instances.
```

## Consequence for v2

The deployable v2 mechanism remains composition-regret retrieval with
other-seed variant selection. The set evaluator is not adopted as a deployed
component yet.

The next experiments should target one of three fixes:

- invariant candidate descriptors: relational role counts, obstacle/hand/table
  coverage, frontier density, and event-relative geometry normalized by local
  scale;
- conservative selection: use candidate evaluators only when validation
  margins are large and stable across training seeds;
- joint retriever-propagator training: optimize candidate choice against
  downstream branch loss with held-out-like perturbations instead of post-hoc
  set scoring.


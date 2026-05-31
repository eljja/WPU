# WPU V2 Global Mixed-K Retriever Integrated Results

Source CSV: `docs/experiments/wpu_v2_staged_global_retriever_initial4_5seed.csv`

## Question

The previous integrated learned retriever trained a separate retriever for each
K condition. The cross-K probe showed that mixed-K retrieval can generalize when
the retriever receives state-index fanout context. This experiment asks the
downstream question:

```text
Can one mixed-K learned retriever be reused across K=8,16,32 inside the staged
WPU propagation/regret/K-expansion pipeline?
```

## Implementation

Added:

- `scripts/staged_global_retriever_hybrid.py`
- `--selection-mode learned_interaction_global`

For each seed, the script trains one initial retriever and one expanded
retriever on mixed K=8,16,32 samples, then reuses those retrievers for all
K=8,16,32 downstream WPU conditions.

This tests whether WPU retrieval can become a reusable state-index module
rather than a K-specific hand-tuned component.

## Setup

- N = 2048
- Evaluation K = 8, 16, 32
- Retriever train K = 8, 16, 32
- Seeds = 11, 13, 17, 19, 23
- Initial working set = 4
- Expanded working set = 32
- Interaction mode = pairwise
- Hidden dim = 128
- Propagation steps = 40
- Regret steps = 80
- Retriever steps = 400
- Retriever train samples per K = 160
- Compute cost = 0.05
- Expansion cost = 0.02

Output:

- `docs/experiments/wpu_v2_staged_global_retriever_initial4_5seed.csv`

## Overall Result

| selection | best policy | loss | loss delta | total compute | accuracy |
| --- | --- | --- | --- | --- | --- |
| indexed | physical sparse expansion | 0.964 | -0.035 | 0.251 | 0.500 |
| interaction teacher | structured sparse expansion | 0.960 | -0.031 | 0.253 | 0.499 |
| per-K learned retriever | physical sparse expansion | 0.961 | -0.032 | 0.224 | 0.495 |
| global mixed-K learned retriever | physical sparse expansion | 0.961 | -0.031 | 0.233 | 0.502 |

## K-Specific Result

| K | selection | best policy | loss | total compute | accuracy |
| --- | --- | --- | --- | --- | --- |
| 8 | interaction teacher | structured sparse expansion | 0.959 | 0.304 | 0.493 |
| 8 | per-K learned | initial calibrated regret | 0.963 | 0.240 | 0.484 |
| 8 | global learned | initial calibrated regret | 0.960 | 0.213 | 0.498 |
| 16 | interaction teacher | physical sparse expansion | 0.945 | 0.251 | 0.509 |
| 16 | per-K learned | physical sparse expansion | 0.945 | 0.233 | 0.502 |
| 16 | global learned | physical sparse expansion | 0.946 | 0.244 | 0.504 |
| 32 | interaction teacher | initial calibrated regret | 0.976 | 0.193 | 0.496 |
| 32 | per-K learned | initial calibrated regret | 0.975 | 0.184 | 0.496 |
| 32 | global learned | initial calibrated regret | 0.976 | 0.220 | 0.504 |

## Interpretation

The result is positive and bounded.

Positive:

- One mixed-K retriever can be reused across K=8,16,32.
- Overall loss remains close to per-K learned retrieval and the hand-built
  interaction teacher.
- Accuracy is slightly higher than the per-K learned retriever.
- K=8 improves over per-K learned retrieval, suggesting that mixed-K retrieval
  can regularize small-K behavior.

Bounded:

- K=16 remains slightly weaker than per-K learned and hand-built interaction
  retrieval.
- K=32 compute is higher than per-K learned retrieval.
- The retriever is still teacher-distilled; downstream-regret retrieval remains
  open.

## Updated Claim

This strengthens the v2 WPU claim:

```text
State-native retrieval can be a reusable mixed-K module when it receives
explicit state-index context. WPU does not need a separate hand-written or
separately trained retriever for every causal working-set size.
```

The next step is to make this retriever useful without teacher labels:

- train from downstream route regret,
- train with retrieval budget penalties,
- expose retriever confidence to the expansion scheduler,
- evaluate under distractor and hidden-relation shifts,
- connect retrieval misses to closed-loop consistency checks.

# WPU V2 Integrated Learned Retriever Results

Source CSV: `docs/experiments/wpu_v2_staged_k_expansion_learned_interaction_initial4_5seed.csv`

## Question

The previous learned-retriever probe showed that a small state-native MLP can
reproduce the hand-built interaction selector's working-set composition under
held-out seeds. This experiment asks the stronger downstream question:

```text
Does a learned state-native retriever still work when it is used inside the
actual staged WPU propagation, regret routing, and K-expansion pipeline?
```

## Implementation

Added `--selection-mode learned_interaction` to:

- `scripts/staged_regret_hybrid.py`
- `scripts/staged_k_expansion_hybrid.py`

The staged pipeline now:

1. Builds a training split for each condition.
2. Distills the interaction-density teacher into small MLP retrievers.
3. Uses the learned retriever for pre-tensor initial and expanded working-set
   projection.
4. Trains the WPU propagation and regret heads on the learned retrieved state.
5. Evaluates validation-selected expansion policies on held-out test samples.

This remains state-native. Retrieval ranks explicit event-frontier objects
before tensorization.

## Setup

- N = 2048
- K = 8, 16, 32
- Seeds = 11, 13, 17, 19, 23
- Initial working set = 4
- Expanded working set = 32
- Interaction mode = pairwise
- Hidden dim = 128
- Propagation steps = 40
- Regret steps = 80
- Retriever steps = 400
- Compute cost = 0.05
- Expansion cost = 0.02

Output:

- `docs/experiments/wpu_v2_staged_k_expansion_learned_interaction_initial4_5seed.csv`

## Overall Comparison

| selection | best policy | loss | loss delta | total compute | accuracy |
| --- | --- | --- | --- | --- | --- |
| indexed | physical sparse expansion | 0.964 | -0.035 | 0.251 | 0.500 |
| proximity | initial calibrated regret | 0.965 | -0.025 | 0.256 | 0.498 |
| interaction teacher | structured sparse expansion | 0.960 | -0.031 | 0.253 | 0.499 |
| learned interaction | physical sparse expansion | 0.961 | -0.032 | 0.224 | 0.495 |

## K-Specific Comparison

| K | selection | best policy | loss | loss delta | total compute | accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | interaction teacher | structured sparse expansion | 0.959 | -0.031 | 0.304 | 0.493 |
| 8 | learned interaction | initial calibrated regret | 0.963 | -0.029 | 0.240 | 0.484 |
| 16 | interaction teacher | physical sparse expansion | 0.945 | -0.026 | 0.251 | 0.509 |
| 16 | learned interaction | physical sparse expansion | 0.945 | -0.029 | 0.233 | 0.502 |
| 32 | interaction teacher | initial calibrated regret | 0.976 | -0.036 | 0.193 | 0.496 |
| 32 | learned interaction | initial calibrated regret | 0.975 | -0.037 | 0.184 | 0.496 |

## Interpretation

The integrated result is positive but bounded.

Positive:

- The learned retriever preserves most of the hand-built interaction retriever's
  downstream benefit.
- At K=16, learned interaction matches the best hand-built loss while using
  less total compute.
- At K=32, learned interaction slightly improves loss and compute relative to
  the hand-built interaction selector.
- Overall, learned interaction remains better than indexed and proximity
  retrieval on loss.

Bounded:

- At K=8, the learned retriever underperforms the hand-built interaction
  teacher.
- The retriever is still teacher-distilled, not trained end-to-end from branch
  loss or rollout consistency.
- Expansion remains useful mainly at K=16. At K=8 and K=32, initial retrieval
  plus calibrated regret routing is usually enough.

## Updated Claim

This result upgrades the v2 retrieval claim:

```text
WPU does not require a fixed hand-written causal selector. A learned
state-native retriever can be inserted before tensorization and preserve most
of the downstream advantage of structured interaction retrieval.
```

The next scientific step is not another token baseline. It is end-to-end
state-native retrieval:

- train retrieval from downstream regret rather than teacher labels,
- expose retrieval confidence and composition to the scheduler,
- penalize unnecessary K growth,
- evaluate under distractors and hidden multi-hop relations,
- connect retrieval failures to rollout consistency checks.

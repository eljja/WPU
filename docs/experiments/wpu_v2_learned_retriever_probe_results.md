# WPU V2 Learned Retriever Probe

Source CSV: `docs/experiments/wpu_v2_learned_retriever_probe.csv`

## Question

Interaction-density retrieval improved the deployed WPU result, but the first
implementation was hand-built. This probe asks whether the same retrieval
structure is learnable from explicit state features under held-out seeds.

```text
Can a small state-native MLP reproduce the interaction retriever's working-set
composition without serializing the world or using global token attention?
```

## Method

Script:

- `scripts/learned_retriever_probe.py`

Output:

- `docs/experiments/wpu_v2_learned_retriever_probe.csv`

Setup:

- N = 2048
- K = 8, 16, 32
- Budget = 4
- Train seeds = 11, 13, 17
- Test seeds = 19, 23
- Train samples per seed = 160
- Test samples per seed = 160
- Teacher = `collate_interaction_working_set_samples`
- Student = small MLP over object type, target-relative geometry, relation
  features, local obstacle density, event force, and state-index fanout context

The student is not a token model. It ranks explicit candidate objects from the
event-local relation frontier.

## Result

| K | selector | teacher overlap | hand hit | selected obstacles | pair density | causal recall |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | indexed | 0.333 | 1.000 | 0.000 | 0.000 | 0.500 |
| 8 | proximity | 0.616 | 0.409 | 2.269 | 0.245 | 0.500 |
| 8 | interaction teacher | 1.000 | 1.000 | 2.000 | 0.613 | 0.500 |
| 8 | learned student | 0.953 | 1.000 | 2.000 | 0.559 | 0.500 |
| 16 | indexed | 0.333 | 1.000 | 0.000 | 0.000 | 0.250 |
| 16 | proximity | 0.307 | 0.319 | 2.531 | 0.543 | 0.250 |
| 16 | interaction teacher | 1.000 | 1.000 | 2.000 | 0.831 | 0.250 |
| 16 | learned student | 0.938 | 1.000 | 2.000 | 0.831 | 0.250 |
| 32 | indexed | 0.333 | 1.000 | 0.000 | 0.000 | 0.125 |
| 32 | proximity | 0.236 | 0.275 | 2.628 | 0.807 | 0.125 |
| 32 | interaction teacher | 1.000 | 1.000 | 2.000 | 0.863 | 0.125 |
| 32 | learned student | 0.912 | 1.000 | 2.000 | 0.878 | 0.125 |

## Interpretation

The result is positive but limited.

Positive:

- The state-native student preserves the contact hand anchor with 100% hit
  rate.
- It selects two obstacle candidates at budget 4, matching the teacher's
  intended composition.
- It recovers high obstacle-pair density under held-out seeds.
- Teacher overlap remains high: `0.91-0.95`.

Limited:

- The student is trained against the hand-built teacher, not directly against
  downstream branch loss.
- The probe uses the direct relation frontier and does not yet include
  adversarial distractors or hidden multi-hop support.
- Causal recall remains low at budget 4 because the budget is intentionally
  under-complete; the point is composition, not full K coverage.

## Consequence

This strengthens the v2 direction:

```text
WPU retrieval can be learned over explicit state features. The next step is to
train retrieval from downstream regret, causal recall, and rollout consistency,
not to return to token serialization.
```

The next experiment should integrate this learned retriever into the actual
WPU training/evaluation loop and compare:

- indexed retrieval,
- hand-built interaction retrieval,
- learned teacher-distilled retrieval,
- learned downstream-regret retrieval.

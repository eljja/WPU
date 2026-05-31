# WPU v2 Retriever Regret Oracle Probe

Source CSVs:

- `docs/experiments/wpu_v2_retriever_regret_oracle.csv`
- `docs/experiments/wpu_v2_retriever_regret_oracle_samples.csv`

## Purpose

This experiment asks whether the current state retriever should keep imitating the interaction-density teacher, or whether downstream branch loss exposes a different objective.

The probe trains one WPU propagation engine per condition using the current interaction working-set path, then evaluates four initial working-set selectors on the same held-out samples:

- `indexed`: relation-order frontier.
- `proximity`: relation frontier ranked by geometric proximity.
- `interaction`: hand-preserving interaction-density teacher.
- `learned`: MLP retriever trained to imitate the interaction teacher.

For each sample, the probe records the branch cross-entropy for each selector and the oracle selector with the lowest downstream loss. This is not a deployed policy; it is a diagnostic upper bound showing whether a downstream-trained retriever could improve over static hand-built selection.

## Command

```powershell
.\.venv\Scripts\python.exe scripts\retriever_regret_oracle_probe.py `
  --n-values 2048 `
  --k-values 8 16 32 `
  --seeds 11 13 17 19 23 `
  --budget 4 `
  --propagation-steps 40 `
  --retriever-steps 400 `
  --samples 90 `
  --validation-samples 90 `
  --batch-size 10 `
  --device cuda `
  --out docs\experiments\wpu_v2_retriever_regret_oracle.csv `
  --sample-out docs\experiments\wpu_v2_retriever_regret_oracle_samples.csv
```

## Main Results

| Causal K | Policy | Loss | Accuracy | Excess over oracle | Oracle-mode rate |
|---:|---|---:|---:|---:|---:|
| 8 | static indexed | 0.987191 | 0.497778 | 0.025097 | 0.453333 |
| 8 | static proximity | 0.988473 | 0.497778 | 0.026380 | 0.282222 |
| 8 | static interaction | 0.988727 | 0.508889 | 0.026634 | 0.237778 |
| 8 | static learned | 0.988432 | 0.506667 | 0.026338 | 0.026666 |
| 8 | oracle over retrievers | 0.962094 | 0.551111 | 0.000000 | 1.000000 |
| 16 | static indexed | 0.970004 | 0.491111 | 0.054811 | 0.391111 |
| 16 | static proximity | 0.968720 | 0.497778 | 0.053528 | 0.386667 |
| 16 | static interaction | 0.966098 | 0.504444 | 0.050906 | 0.193333 |
| 16 | static learned | 0.966183 | 0.504444 | 0.050990 | 0.028889 |
| 16 | oracle over retrievers | 0.915192 | 0.566667 | 0.000000 | 1.000000 |
| 32 | static indexed | 1.010393 | 0.497778 | 0.035872 | 0.384445 |
| 32 | static proximity | 1.007072 | 0.480000 | 0.032550 | 0.331111 |
| 32 | static interaction | 1.004100 | 0.480000 | 0.029578 | 0.246667 |
| 32 | static learned | 1.004095 | 0.475556 | 0.029573 | 0.037778 |
| 32 | oracle over retrievers | 0.974522 | 0.575556 | 0.000000 | 1.000000 |

## Interpretation

The current learned retriever is not the right endpoint. It reproduces the interaction-density teacher, but the downstream oracle rarely selects it: 2.7% at K=8, 2.9% at K=16, and 3.8% at K=32.

The best selector is sample-dependent. At K=16, indexed and proximity are each oracle on roughly 39% of samples, while interaction is oracle on only 19%. At K=32, indexed remains oracle on 38%, proximity on 33%, and interaction on 25%. This means the fixed interaction teacher is not a sufficient target for v2.

The oracle gap is scientifically useful. Switching from the validation-chosen static selector to the per-sample oracle reduces loss by 0.024831 at K=8, 0.043414 at K=16, and 0.030013 at K=32. It also raises accuracy to 0.55-0.58. This indicates that WPU v2 should train the retriever against downstream regret or policy loss, not against a single handcrafted selection rule.

## Consequence for WPU v2

The large-N argument should not be "state retrieval is solved by a clever static heuristic." The stronger claim is:

> WPU becomes useful when the processor can maintain explicit state, identify a small causal working set, and adapt the working-set retrieval policy from downstream consequences.

The next implementation target is therefore a downstream-trained retriever/reranker:

- Generate several candidate working sets from indexed, proximity, interaction, learned, and stochastic variants.
- Score candidate sets using a lightweight state-conditioned policy head.
- Train the policy against held-out downstream loss or regret, not teacher overlap.
- Keep the output as explicit selected object ids so the model remains state-based and sparse-first.

This keeps the architecture non-token-based while addressing the current failure mode: teacher imitation preserves state structure but does not optimize the task loss.

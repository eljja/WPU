# WPU v2 Route-Physics Contract Smoke

Source CSVs:

- `docs/experiments/wpu_v2_route_physics_contract_smoke_physics.csv`
- `docs/experiments/wpu_v2_route_physics_contract_smoke_state.csv`

This smoke test verifies a P1/P4 input-contract fix. The previous route-regret
physics context used only pair-distance statistics, target xy, and an event
norm. That meant action-conditioned state such as `catch_action` and physical
object-state scalars such as `edge_distance`, `hand_distance`, `fall_risk`, and
`angular_speed` reached `StateGraphBatch` but were mostly discarded before the
route-regret decision.

The fix expands the route-physics feature vector to 14 scalars:

| group | fields |
|---|---|
| pair geometry | minimum and mean selected-object pair distance |
| target geometry | target x/y |
| target physical state | `edge_distance`, `hand_distance`, `fall_risk`, `angular_speed` |
| selected-set physical state | mean `edge_distance`, `hand_distance`, `fall_risk`, `angular_speed` over selected objects |
| event action | `force`, `catch_action` |

## Smoke Results

| model | N | K | samples | route-regret corr | routed loss | sparse loss | dense compute |
|---|---:|---:|---:|---:|---:|---:|---:|
| `wpu-cws-indexed-physics-regret-hybrid` | 128 | 8 | 24 | 0.015806 | 1.137126 | 1.137126 | 0.000000 |
| `wpu-cws-indexed-state-regret-hybrid` | 128 | 8 | 24 | -0.091476 | 1.102039 | 1.102039 | 0.000000 |

## Interpretation

This is not an accuracy improvement claim. It is a contract test showing that
the route-regret path can train and evaluate after the expanded state/action
features are wired into the model. The small CPU run is intentionally too small
to support performance conclusions.

The scientific effect is narrower but important: WPU cannot claim state-native
execution if later routing layers silently compress away the state variables
that define the mechanism. This fix removes one such mismatch and makes the
next P1/P4 experiments better aligned with the objectified-state definition.

## Reproduction

```bash
python scripts/staged_regret_hybrid.py \
  --model-name wpu-cws-indexed-physics-regret-hybrid \
  --n-values 128 --k-values 8 --seeds 11 \
  --hidden-dim 32 --layers 1 --num-heads 4 --working-set-size 12 \
  --propagation-steps 4 --regret-steps 6 \
  --samples 24 --validation-samples 24 --batch-size 4 \
  --device cpu \
  --out docs/experiments/wpu_v2_route_physics_contract_smoke_physics.csv

python scripts/staged_regret_hybrid.py \
  --model-name wpu-cws-indexed-state-regret-hybrid \
  --n-values 128 --k-values 8 --seeds 11 \
  --hidden-dim 32 --layers 1 --num-heads 4 --working-set-size 12 \
  --propagation-steps 4 --regret-steps 6 \
  --samples 24 --validation-samples 24 --batch-size 4 \
  --device cpu \
  --out docs/experiments/wpu_v2_route_physics_contract_smoke_state.csv
```

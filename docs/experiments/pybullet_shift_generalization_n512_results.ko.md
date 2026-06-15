# PyBullet N512 Mechanism-Diversity Screens

이 screen은 cup-only benchmark에서 관측된 large-N WPU 장점이 같은 explicit state
크기에서 mechanism variation까지 유지되는지 검증한다. 모든 실행은 `N_bg=512`,
total `N=517`, 3 seeds, 6 training steps, mechanism당 16 evaluation samples,
hidden dim `24`, WPU/graph/token baseline을 사용한다.

최신 rerun은 입력 계약 결함도 수정한다. PyBullet `Event`에는 `catch_action`이
있었지만 `StateGraphBatch.event_features`에는 들어가지 않았다. 또한 object encoder가
objectified simulator state의 물리 scalar인 `edge_distance`, `hand_distance`,
`fall_risk`, `angular_speed`를 보존하도록 확장됐다.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization_n512_screen.csv`
- `docs/experiments/pybullet_shift_generalization_n512_multimech.csv`
- `docs/experiments/pybullet_shift_generalization_n512_event_action_screen.csv`
- `docs/experiments/pybullet_shift_generalization_n512_event_action_multimech.csv`
- `docs/experiments/pybullet_shift_generalization_n512_event_physical_screen.csv`
- `docs/experiments/pybullet_shift_generalization_n512_event_physical_multimech.csv`

## Aggregate Result

| Protocol | Train mechanisms | Feature contract | WPU win/tie/loss | Mean best-WPU minus best-baseline margin | Best macro WPU accuracy | Best macro baseline accuracy |
|---|---|---|---:|---:|---:|---:|
| Original nominal-train screen | nominal | event action omitted, physical scalars omitted | 2/1/4 | -0.047619 | 0.407738 | 0.455357 |
| Action-event nominal-train screen | nominal | `catch_action` preserved | 1/1/5 | -0.083333 | 0.419643 | 0.502976 |
| Event+physical nominal-train screen | nominal | `catch_action` and physical scalars preserved | 4/0/3 | +0.002976 | 0.485119 | 0.482143 |
| Original multi-mechanism screen | 7 mechanisms | event action omitted, physical scalars omitted | 2/0/5 | -0.095238 | 0.369048 | 0.464286 |
| Action-event multi-mechanism screen | 7 mechanisms | `catch_action` preserved | 3/1/3 | -0.044643 | 0.377976 | 0.422619 |
| Event+physical multi-mechanism screen | 7 mechanisms | `catch_action` and physical scalars preserved | 2/2/3 | -0.032738 | 0.357143 | 0.389881 |

## Event+Physical Encoding 이후 Mechanism별 Best 비교

### Nominal-Train Screen

| Eval mechanism | Best WPU | Best WPU accuracy | Best baseline | Best baseline accuracy | Margin |
|---|---|---:|---|---:|---:|
| catch_heavy | `wpu-cws-indexed-local-dense` | 0.333333 | `graph-transformer` | 0.625000 | -0.291667 |
| edge_catch_heavy | `wpu-cws-indexed-sparse` | 0.479167 | `graph-transformer` | 0.229167 | +0.250000 |
| edge_high_force | `wpu-cws-indexed-sparse` | 0.500000 | `graph-transformer` | 0.354167 | +0.145833 |
| edge_shift | `wpu-cws-indexed-sparse` | 0.541667 | `graph-transformer` | 0.375000 | +0.166667 |
| high_force | `wpu-cws-indexed-local-dense` | 0.416667 | `graph-transformer` | 0.541667 | -0.125000 |
| no_catch | `wpu-cws-indexed-sparse` | 0.666667 | `serialized-token` | 0.625000 | +0.041667 |
| nominal | `wpu-cws-indexed-local-dense` | 0.458333 | `graph-transformer` | 0.625000 | -0.166667 |

### Multi-Mechanism-Train Screen

| Eval mechanism | Best WPU | Best WPU accuracy | Best baseline | Best baseline accuracy | Margin |
|---|---|---:|---|---:|---:|
| catch_heavy | `wpu-cws-indexed-local-dense` | 0.041667 | `graph-transformer` | 0.062500 | -0.020833 |
| edge_catch_heavy | `wpu-cws-indexed-sparse` | 0.437500 | `graph-transformer` | 0.437500 | +0.000000 |
| edge_high_force | `wpu-cws-indexed-sparse` | 0.500000 | `graph-transformer` | 0.479167 | +0.020833 |
| edge_shift | `wpu-cws-indexed-sparse` | 0.458333 | `serialized-token` | 0.458333 | +0.000000 |
| high_force | `wpu-cws-indexed-local-dense` | 0.333333 | `graph-transformer` | 0.312500 | +0.020833 |
| no_catch | `wpu-cws-indexed-sparse` | 0.520833 | `serialized-token` | 0.625000 | -0.104167 |
| nominal | `wpu-cws-indexed-sparse` | 0.208333 | `serialized-token` | 0.354167 | -0.145834 |

## 해석

- 이것은 broad WPU superiority result가 아니라 claim-boundary result다.
- 원본 negative result는 부분적으로 구현 결함을 드러냈다. objectified state에는
  action과 물리 변수가 있었지만 tensor encoder가 이를 버리고 있었다.
- 해당 변수를 보존하면 nominal-train large-N shift screen은 negative margin에서
  near tie/slight WPU edge로 회복된다.
- 그러나 multi-mechanism screen은 여전히 mixed/negative다. 더 나은 state encoding은
  gap을 줄이지만 mechanism-family law learning을 해결하지는 못한다.
- 방어 가능한 결론은 더 엄격하다. WPU는 `N`이 크고 causal working set `K`가 작고
  식별 가능하며, 관련 action/physical state가 tensorization 전에 보존되고, local
  propagation law가 해당 mechanism family에 대해 학습되거나 적응될 때 유리하다.

## Reproduction

```bash
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --train-mechanisms nominal --eval-mechanisms nominal high_force edge_shift catch_heavy no_catch edge_high_force edge_catch_heavy --background-objects 512 --seeds 11 13 17 --steps 6 --sim-steps 120 --samples 16 --batch-size 2 --hidden-dim 24 --layers 1 --num-heads 4 --working-set-size 12 --balanced-labels --out docs/experiments/pybullet_shift_generalization_n512_event_physical_screen.csv
```

```bash
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --train-mechanisms nominal high_force edge_shift catch_heavy no_catch edge_high_force edge_catch_heavy --eval-mechanisms nominal high_force edge_shift catch_heavy no_catch edge_high_force edge_catch_heavy --background-objects 512 --seeds 11 13 17 --steps 6 --sim-steps 120 --samples 16 --batch-size 2 --hidden-dim 24 --layers 1 --num-heads 4 --working-set-size 12 --balanced-labels --out docs/experiments/pybullet_shift_generalization_n512_event_physical_multimech.csv
```

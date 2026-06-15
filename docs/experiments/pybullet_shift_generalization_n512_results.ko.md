# PyBullet N512 Mechanism-Diversity Screens

이 screen은 cup-only benchmark에서 관측된 large-N WPU 장점이 같은 explicit state
크기에서 mechanism variation까지 유지되는지 검증한다. 두 실행은 모두 `N_bg=512`,
total `N=517`, 3 seeds, 6 training steps, mechanism당 16 evaluation samples,
hidden dim `24`, WPU/graph/token baseline을 사용한다.

Source CSVs:

- `docs/experiments/pybullet_shift_generalization_n512_screen.csv`
- `docs/experiments/pybullet_shift_generalization_n512_multimech.csv`

## Aggregate Result

| Protocol | Train mechanisms | Eval mechanisms | WPU win/tie/loss | Mean best-WPU minus best-baseline margin | Best macro WPU accuracy | Best macro baseline accuracy |
|---|---|---|---:|---:|---:|---:|
| Nominal-train screen | nominal | 7 mechanisms | 2/1/4 | -0.047619 | 0.380952 | 0.404762 |
| Multi-mechanism-train screen | 7 mechanisms | 7 mechanisms | 2/0/5 | -0.095238 | 0.369048 | 0.419643 |

## 해석

- 이것은 WPU superiority result가 아니라 negative/claim-boundary result다.
- 기존 N=517 cup-only benchmark는 좁은 one-step cup 설정에서 sparse object-state
  execution이 훨씬 빠르면서 작은 accuracy edge를 유지할 수 있음을 보였다.
- 하지만 이번 mechanism screens는 같은 large-N compute advantage가 mechanism
  generalization으로 자동 전이되지 않음을 보인다.
- 따라서 조건은 더 엄격하다. WPU가 유리하려면 causal working set이 작고 식별
  가능해야 할 뿐 아니라, 그 working set 위의 local propagation law가 해당 mechanism
  family에 대해 충분히 학습되거나 적응되어야 한다.
- Low-budget screen에서 multi-mechanism exposure만으로는 문제가 해결되지 않았다.
  다음 모델 변경은 단순히 mechanism 이름을 더 넣는 것이 아니라 mechanism-aware
  propagation이어야 한다.

## Reproduction

```bash
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --train-mechanisms nominal --eval-mechanisms nominal high_force edge_shift catch_heavy no_catch edge_high_force edge_catch_heavy --background-objects 512 --seeds 11 13 17 --steps 6 --sim-steps 120 --samples 16 --batch-size 2 --hidden-dim 24 --layers 1 --num-heads 4 --working-set-size 12 --balanced-labels --out docs/experiments/pybullet_shift_generalization_n512_screen.csv
```

```bash
python scripts/pybullet_shift_generalization.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --train-mechanisms nominal high_force edge_shift catch_heavy no_catch edge_high_force edge_catch_heavy --eval-mechanisms nominal high_force edge_shift catch_heavy no_catch edge_high_force edge_catch_heavy --background-objects 512 --seeds 11 13 17 --steps 6 --sim-steps 120 --samples 16 --batch-size 2 --hidden-dim 24 --layers 1 --num-heads 4 --working-set-size 12 --balanced-labels --out docs/experiments/pybullet_shift_generalization_n512_multimech.csv
```

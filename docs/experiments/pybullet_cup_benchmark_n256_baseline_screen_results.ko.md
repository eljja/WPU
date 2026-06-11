# PyBullet Cup N256 Baseline Screen

이 screen은 PyBullet cup benchmark를 `N_bg=256`, total `N=261`까지 확장하면서
WPU, graph-transformer, serialized-token baseline을 같은 run에 포함한다. 다만
training budget이 작기 때문에(`steps=2`, `samples=12`, 5 seeds), 강한 accuracy
우월성 주장이 아니라 matched large-N feasibility evidence로만 사용한다.

Source CSV:

- `docs/experiments/pybullet_cup_benchmark_n256_baseline_screen.csv`

## Aggregate Results

| Model | Seeds | Mean branch accuracy | Mean forward latency ms/sample | Mean params |
|---|---:|---:|---:|---:|
| `wpu-cws-indexed-sparse` | 5 | 0.350000 | 1.872940 | 6,204 |
| `wpu-cws-indexed-local-dense` | 5 | 0.316667 | 2.435880 | 31,612 |
| `graph-transformer` | 5 | 0.333333 | 114.401640 | 30,934 |
| `serialized-token` | 5 | 0.316667 | 1.484675 | 26,738 |

## 해석

- Matched baseline boundary는 total `N=261`까지 확장됐지만, 이는 저훈련
  screen이다.
- Sparse WPU가 이 screen에서는 가장 높은 평균 branch accuracy를 보이지만,
  margin이 작고 training budget이 낮아 강한 superiority claim에는 부족하다.
- Graph-transformer baseline은 이 state size에서 매우 느려지며, dense graph
  processing 비용과 일치한다.
- Serialized token processing은 이 작은 screened 설정에서 latency 경쟁력이
  남아 있으므로 WPU가 보편적 speed dominance를 주장해서는 안 된다.
- 현재 가장 강한 주장은 조건부다. Objectified state가 작은 identifiable causal
  working set을 dense tensorization 전에 노출할 때 WPU가 유리하다.

## Reproduction

```bash
python scripts/pybullet_cup_benchmark.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --background-objects 256 --seeds 11 13 17 19 23 --steps 2 --sim-steps 120 --samples 12 --batch-size 4 --hidden-dim 32 --num-heads 4 --working-set-size 12 --runtime-repeats 1 --balanced-labels --out docs/experiments/pybullet_cup_benchmark_n256_baseline_screen.csv
```

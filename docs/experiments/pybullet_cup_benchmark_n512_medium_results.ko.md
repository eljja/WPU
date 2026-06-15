# PyBullet Cup N512 Baseline-Complete Medium Benchmark

이 실행은 기존 `N_bg=512`, total `N=517` micro-screen을 5-seed
baseline-complete benchmark로 승격한 것이다. 설정은 `6` training steps와 seed당
`16` evaluation samples이며, 같은 protocol 안에 WPU, graph-transformer,
serialized-token baseline을 모두 포함한다. Micro-screen보다 강한 증거지만, 여전히
단일 cup-family, one-step benchmark이고 accuracy margin도 작다.

Source CSV:

- `docs/experiments/pybullet_cup_benchmark_n512_medium.csv`

## Aggregate Results

| Model | Seeds | Mean branch accuracy | Majority baseline | Mean forward latency ms/sample | Mean params | Mean MSE | Mean selected K | CUDA peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `wpu-cws-indexed-sparse` | 5 | 0.387500 | 0.375000 | 2.167110 | 3,996 | 0.314859 | 4.350 | 17.369 |
| `graph-transformer` | 5 | 0.362500 | 0.375000 | 146.064080 | 10,606 | 0.256670 | 4.350 | 35.335 |
| `serialized-token` | 5 | 0.337500 | 0.375000 | 8.614540 | 8,226 | 0.241068 | 4.350 | 154.959 |
| `wpu-cws-indexed-local-dense` | 5 | 0.325000 | 0.375000 | 2.488510 | 11,220 | 0.497430 | 4.350 | 17.495 |

## 해석

- Best WPU model은 `wpu-cws-indexed-sparse`이고 branch accuracy는 `0.387500`이다.
- Best-accuracy non-WPU baseline은 `graph-transformer`이고 branch accuracy는
  `0.362500`이다.
- 해당 best-accuracy baseline 대비 best WPU는 accuracy가 `0.025000` 높고
  forward latency가 `67.400400x` 빠르다.
- 모든 baseline이 `N=517`에서 더 큰 seed/training budget으로 완료됐으므로 P3 evidence는
  micro-screen보다 강화됐다.
- 그러나 broad simulator superiority를 증명하지는 않는다. Margin이 작고, task는 단일
  cup family이며, evaluation도 long-horizon이 아니라 one-step이다.

## Reproduction

```bash
python scripts/pybullet_cup_benchmark.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --background-objects 512 --seeds 11 13 17 19 23 --steps 6 --sim-steps 120 --samples 16 --batch-size 2 --hidden-dim 24 --layers 1 --num-heads 4 --working-set-size 12 --runtime-repeats 1 --balanced-labels --out docs/experiments/pybullet_cup_benchmark_n512_medium.csv
```

# PyBullet Cup N512 Baseline-Complete Micro-Screen

이 실행은 simulator-backed large-state comparison을 `N_bg=512`, total `N=517`
까지 확장하고, 같은 protocol 안에 WPU, graph-transformer, serialized-token
baseline을 모두 포함한다. 다만 의도적으로 작은 설정(`3` seeds, `2` training
steps, seed당 `8` samples)이므로 strong simulator-superiority evidence가 아니라
baseline-complete coverage 및 systems feasibility evidence로 해석해야 한다.

Source CSV:

- `docs/experiments/pybullet_cup_benchmark_n512_baseline_micro.csv`

## Aggregate Results

| Model | Seeds | Mean branch accuracy | Mean forward latency ms/sample | Mean params | Mean MSE | Mean selected K |
|---|---:|---:|---:|---:|---:|---:|
| `wpu-cws-indexed-sparse` | 3 | 0.375000 | 2.006700 | 2,236 | 0.435292 | 4.292 |
| `graph-transformer` | 3 | 0.333333 | 151.449383 | 5,030 | 0.339243 | 4.292 |
| `serialized-token` | 3 | 0.333333 | 0.663283 | 3,954 | 0.451093 | 4.292 |
| `wpu-cws-indexed-local-dense` | 3 | 0.333333 | 2.481933 | 5,516 | 0.681147 | 4.292 |

## 해석

- Best WPU model은 `wpu-cws-indexed-sparse`이고 branch accuracy는 `0.375000`이다.
- Best-accuracy non-WPU baseline은 `graph-transformer`이고 branch accuracy는
  `0.333333`이다.
- 해당 best-accuracy baseline 대비 best WPU는 accuracy가 `0.041667` 높고
  forward latency가 `75.471861x` 빠르다.
- `serialized-token`은 가장 빠르지만, 이 tiny run에서는 graph baseline과 같은
  `0.333333` branch accuracy에 머문다.
- 이 결과는 total `N=517`의 coverage gap을 줄이지만 P3를 해결하지는 않는다.
  Training budget이 너무 작고, domain은 여전히 단일 cup family이며, evaluation도
  long-horizon이 아니라 one-step이다.

## Reproduction

```bash
python scripts/pybullet_cup_benchmark.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --background-objects 512 --seeds 11 13 17 --steps 2 --sim-steps 120 --samples 8 --batch-size 2 --hidden-dim 16 --layers 1 --num-heads 2 --working-set-size 12 --runtime-repeats 1 --balanced-labels --out docs/experiments/pybullet_cup_benchmark_n512_baseline_micro.csv
```

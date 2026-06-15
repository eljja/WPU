# PyBullet Cup N512 Baseline-Complete Higher-Budget Benchmark

이 실행은 `N_bg=512`, total `N=517` baseline-complete protocol을 medium 설정(`6`
training steps, seed당 `16` evaluation samples, hidden dim `24`)에서 `10`
training steps, seed당 `24` evaluation samples, hidden dim `32`로 올린 것이다. WPU,
graph-transformer, serialized-token baseline을 같은 5-seed protocol에 포함한다.

Source CSV:

- `docs/experiments/pybullet_cup_benchmark_n512_high_budget.csv`

## Aggregate Results

| Model | Seeds | Mean branch accuracy | Majority baseline | Mean forward latency ms/sample | Mean params | Mean MSE | Mean selected K | CUDA peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `wpu-cws-indexed-local-dense` | 5 | 0.433333 | 0.333333 | 2.551960 | 18,908 | 0.456292 | 4.367 | 17.595 |
| `graph-transformer` | 5 | 0.425000 | 0.333333 | 146.981950 | 18,230 | 0.110532 | 4.367 | 19.530 |
| `wpu-cws-indexed-sparse` | 5 | 0.416666 | 0.333333 | 2.361880 | 6,204 | 0.341901 | 4.367 | 17.389 |
| `serialized-token` | 5 | 0.266666 | 0.333333 | 0.623770 | 14,034 | 0.103926 | 4.367 | 20.230 |

## 해석

- Best WPU model은 `wpu-cws-indexed-local-dense`이고 branch accuracy는 `0.433333`이다.
- Best-accuracy non-WPU baseline은 `graph-transformer`이고 branch accuracy는
  `0.425000`이다.
- 해당 best-accuracy baseline 대비 best WPU는 accuracy가 `0.008333` 높고
  forward latency가 `57.595711x` 빠르다.
- Higher-budget에서도 WPU edge는 유지되지만 medium run보다 margin은 줄어든다. 이는
  유용한 조건부 evidence이지 broad simulator superiority는 아니다.
- 다음 P3 병목은 작은 cup-only budget 증가가 아니라 mechanism diversity,
  long-horizon simulator rollout, perception-to-state objectification이다.

## Reproduction

```bash
python scripts/pybullet_cup_benchmark.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --background-objects 512 --seeds 11 13 17 19 23 --steps 10 --sim-steps 120 --samples 24 --batch-size 2 --hidden-dim 32 --layers 1 --num-heads 4 --working-set-size 12 --runtime-repeats 1 --balanced-labels --out docs/experiments/pybullet_cup_benchmark_n512_high_budget.csv
```

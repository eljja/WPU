# PyBullet Cup N256 Medium-Training Benchmark

이 실행은 기존 `N_bg=256`, total `N=261` screen의 training budget을
`steps=2`, `samples=12`에서 `steps=8`, `samples=24`로 올린 것이다. WPU,
graph-transformer, serialized-token baseline을 같은 5-seed protocol에 포함한다.
저훈련 screen보다 강한 P3 evidence지만, 여전히 단일 cup-family benchmark이므로 broad
simulator superiority claim으로 승격하면 안 된다.

Source CSV:

- `docs/experiments/pybullet_cup_benchmark_n256_medium.csv`

## Aggregate Results

| Model | Seeds | Mean branch accuracy | Mean forward latency ms/sample | Mean params | Mean MSE |
|---|---:|---:|---:|---:|---:|
| `wpu-cws-indexed-local-dense` | 5 | 0.466667 | 1.784500 | 31,612 | 0.341193 |
| `graph-transformer` | 5 | 0.450000 | 108.193390 | 30,934 | 0.087779 |
| `wpu-cws-indexed-sparse` | 5 | 0.408333 | 1.595585 | 6,204 | 0.302353 |
| `serialized-token` | 5 | 0.266667 | 1.012280 | 26,738 | 0.079201 |

## 해석

- Best WPU model은 `wpu-cws-indexed-local-dense`이고 branch accuracy는 `0.466667`이다.
- Best-accuracy non-WPU baseline은 `graph-transformer`이고 branch accuracy는 `0.450000`이다.
- Best-accuracy baseline 대비 best WPU는 accuracy가 `0.016667` 높고 forward latency가
  `60.629526x` 빠르다.
- `serialized-token`은 여전히 가장 빠르지만, 이 run에서는 branch accuracy가 `0.266667`로
  낮다.
- 이는 positive large-N simulator evidence지만 P3 해결은 아니다. Margin이 작고, domain은
  아직 단일 cup family이며, evaluation도 long-horizon이 아니라 one-step이다.

## Reproduction

```bash
python scripts/pybullet_cup_benchmark.py --models wpu-cws-indexed-sparse wpu-cws-indexed-local-dense graph-transformer serialized-token --background-objects 256 --seeds 11 13 17 19 23 --steps 8 --sim-steps 120 --samples 24 --batch-size 4 --hidden-dim 32 --num-heads 4 --working-set-size 12 --runtime-repeats 1 --balanced-labels --out docs/experiments/pybullet_cup_benchmark_n256_medium.csv
```

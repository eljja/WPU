# PyBullet Cup Benchmark 7-Seed Extension

This run extends the simulator-backed PyBullet cup benchmark from five seeds to
seven seeds at two world sizes. It is still a small benchmark, but it reduces
seed fragility in the current WPU v2 dashboard.

Source CSV:

- `docs/experiments/pybullet_cup_benchmark_7seed.csv`

## Protocol

- Models: `wpu-cws-indexed-sparse`, `wpu-cws-indexed-local-dense`,
  `graph-transformer`, `serialized-token`.
- Seeds: `11, 13, 17, 19, 23, 29, 31`.
- Background objects: `0, 128`.
- Training steps: `20`.
- Eval samples: `36` per seed and condition.
- Hidden dim: `64`.
- Runtime repeats: `3`.

## Summary

| N | model | branch accuracy | forward ms/sample | CUDA peak MB |
|---:|---|---:|---:|---:|
| 5 | graph-transformer | 0.579365 | 2.032552 | 19.234 |
| 5 | serialized-token | 0.551587 | 0.121180 | 19.068 |
| 5 | wpu-cws-indexed-local-dense | 0.531746 | 1.259456 | 19.223 |
| 5 | wpu-cws-indexed-sparse | 0.547619 | 1.107382 | 17.661 |
| 133 | graph-transformer | 0.492063 | 37.241540 | 22.938 |
| 133 | serialized-token | 0.539683 | 0.440278 | 40.744 |
| 133 | wpu-cws-indexed-local-dense | 0.531746 | 1.277999 | 19.223 |
| 133 | wpu-cws-indexed-sparse | 0.547619 | 1.126503 | 17.661 |

## Interpretation

At `N=133`, sparse WPU has the best mean branch accuracy in this small 7-seed
run (`0.547619`) and keeps runtime nearly flat relative to `N=5`. The
serialized-token baseline remains the fastest model, so this does not establish
universal latency dominance. The useful claim is narrower: explicit state and
pre-tensor indexing can preserve accuracy as irrelevant background objects grow,
while full-state graph processing becomes much slower.

The next P3 step is to add more mechanisms and long-horizon rollout evaluation,
not only more seeds in the same cup scene.

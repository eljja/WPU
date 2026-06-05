# PyBullet Systems Profile

This experiment profiles the systems side of WPU on PyBullet-derived
objectified state. It does not train a model and does not measure accuracy. Its
purpose is to separate full-state tensorization cost, pre-tensor indexed WPU
cost, and branch-overlay memory cost.

Source CSV:

- `docs/experiments/pybullet_system_profile.csv`

## Protocol

- Simulator: PyBullet `DIRECT` cup scene.
- Samples: `8` per seed and background setting.
- Seeds: `11, 13`.
- Background objects: `0, 32, 128, 512, 2048`.
- Branch counts: `1, 3, 8`.
- Indexed WPU query: event target plus relation frontier, `max_nodes=12`,
  `max_depth=1`.
- Metrics:
  - `full_tensor_bytes`: tensor bytes for the full objectified state batch.
  - `selected_tensor_bytes`: tensor bytes after pre-tensor indexed projection.
  - `tensor_byte_reduction`: reduction from full tensorization to indexed WPU
    tensorization.
  - `branch_memory_reduction`: reduction of `BaseState + branch deltas` versus
    full state copies for `B` branches.
  - `work_proxy_reduction`: reduction from dense `N^2 * B` object work proxy to
    selected `K * E_K * B` sparse work proxy.

## Summary

| background objects | branches | total objects | selected objects | tensor byte reduction | branch memory reduction | work proxy reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 4.562 | 4.562 | 0.000000 | 0.000000 | 0.000000 |
| 0 | 3 | 4.562 | 4.562 | 0.000000 | 0.269627 | 0.000000 |
| 0 | 8 | 4.562 | 4.562 | 0.000000 | 0.477961 | 0.000000 |
| 32 | 1 | 36.562 | 4.562 | 0.859694 | 0.000000 | 0.984326 |
| 32 | 3 | 36.562 | 4.562 | 0.859694 | 0.617715 | 0.984326 |
| 32 | 8 | 36.562 | 4.562 | 0.859694 | 0.826048 | 0.984326 |
| 128 | 1 | 132.562 | 4.562 | 0.960764 | 0.000000 | 0.998804 |
| 128 | 3 | 132.562 | 4.562 | 0.960764 | 0.653167 | 0.998804 |
| 128 | 8 | 132.562 | 4.562 | 0.960764 | 0.861500 | 0.998804 |
| 512 | 1 | 516.562 | 4.562 | 0.989891 | 0.000000 | 0.999921 |
| 512 | 3 | 516.562 | 4.562 | 0.989891 | 0.663202 | 0.999921 |
| 512 | 8 | 516.562 | 4.562 | 0.989891 | 0.871536 | 0.999921 |
| 2048 | 1 | 2052.562 | 4.562 | 0.997454 | 0.000000 | 0.999995 |
| 2048 | 3 | 2052.562 | 4.562 | 0.997454 | 0.665795 | 0.999995 |
| 2048 | 8 | 2052.562 | 4.562 | 0.997454 | 0.874128 | 0.999995 |

## Interpretation

This is the cleanest current systems evidence for the WPU large-`N` premise.
When irrelevant background state grows from `N≈4.6` to `N≈2052.6`, the
pre-tensor indexed WPU path keeps the neural state near `K≈4.6`. The resulting
tensor-byte reduction rises to `0.997454`, and the sparse object-work proxy
reduction rises to `0.999995`.

The branch result is also aligned with the WPU memory thesis. At `B=8`, storing
`BaseState + branch deltas` reduces the branch memory proxy by `0.874128` at
the largest `N` relative to full state copies.

This does not prove hardware speedup or lower power. It is still a Python-level
proxy and does not include irregular sparse-kernel overhead, cache behavior,
GPU occupancy, or real memory traffic. The defensible claim is narrower:

```text
If the causal working set K is selected before tensorization, WPU exposes a
large reducible systems cost that token/full-state graph baselines must pay
unless they implement an equivalent state index.
```

## Issues Found

- The profiler measures proxy bytes and proxy work, not wall-clock latency or
  energy.
- `sys.getsizeof`-based state memory is a Python-object approximation, not an
  allocator-level memory measurement.
- The indexed frontier is relation-derived and easy in this PyBullet scene.
  Harder perception and distractor settings may reduce effective `K` quality.
- Branch overlays are synthetic delta records; they do not yet include rollback,
  correction, or uncertainty-gated branch pruning.

## Next Steps

- Add runtime latency and CUDA memory measurements for matched model forward
  passes on the same `N` settings.
- Add objectification corruption to measure how relation errors change selected
  `K`, tensor reduction, and downstream loss.
- Replace Python-object memory estimates with serialized byte size and
  allocator-level measurements.
- Report this systems profile next to accuracy so WPU claims require both:
  acceptable prediction quality and lower state-processing work.

# PyBullet Systems Profile

This experiment profiles the systems side of WPU on PyBullet-derived
objectified state. It does not train a model and does not measure accuracy. Its
purpose is to separate full-state tensorization cost, pre-tensor indexed WPU
cost, and branch-overlay memory cost.

Source CSV:

- `docs/experiments/pybullet_system_profile.csv`
- `docs/experiments/pybullet_system_profile_cuda.csv`

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
  - `tensorize_latency_reduction`: measured CPU reduction from full-state
    `StateGraphBatch` construction to selected-state construction.
  - `branch_memory_reduction`: reduction of `BaseState + branch deltas` versus
    full state copies for `B` branches.
  - `work_proxy_reduction`: reduction from dense `N^2 * B` object work proxy to
    selected `K * E_K * B` sparse work proxy.
  - `sparse_forward_latency_reduction`: random untrained CPU forward-latency
    proxy comparing full-state `graph-transformer` with selected-state
    `wpu-cws-indexed-sparse`.
  - `sparse_peak_memory_reduction`: CUDA peak allocated-memory reduction for
    the same random untrained forward proxy when the CUDA profile is used.

## Summary

| background objects | branches | total objects | selected objects | tensor byte reduction | tensorize latency reduction | sparse forward reduction | branch memory reduction | work proxy reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 4.562 | 4.562 | 0.000000 | 0.225724 | 0.368322 | 0.000000 | 0.000000 |
| 0 | 3 | 4.562 | 4.562 | 0.000000 | 0.194806 | 0.348028 | 0.269627 | 0.000000 |
| 0 | 8 | 4.562 | 4.562 | 0.000000 | 0.195042 | 0.380049 | 0.477961 | 0.000000 |
| 32 | 1 | 36.562 | 4.562 | 0.859694 | 0.831249 | 0.833258 | 0.000000 | 0.984326 |
| 32 | 3 | 36.562 | 4.562 | 0.859694 | 0.820678 | 0.832391 | 0.617715 | 0.984326 |
| 32 | 8 | 36.562 | 4.562 | 0.859694 | 0.840700 | 0.845866 | 0.826048 | 0.984326 |
| 128 | 1 | 132.562 | 4.562 | 0.960764 | 0.945943 | 0.952048 | 0.000000 | 0.998804 |
| 128 | 3 | 132.562 | 4.562 | 0.960764 | 0.948473 | 0.951497 | 0.653167 | 0.998804 |
| 128 | 8 | 132.562 | 4.562 | 0.960764 | 0.950143 | 0.947996 | 0.861500 | 0.998804 |
| 512 | 1 | 516.562 | 4.562 | 0.989891 | 0.985288 | 0.986152 | 0.000000 | 0.999921 |
| 512 | 3 | 516.562 | 4.562 | 0.989891 | 0.985610 | 0.987319 | 0.663202 | 0.999921 |
| 512 | 8 | 516.562 | 4.562 | 0.989891 | 0.985058 | 0.987566 | 0.871536 | 0.999921 |
| 2048 | 1 | 2052.562 | 4.562 | 0.997454 | 0.995233 | 0.996907 | 0.000000 | 0.999995 |
| 2048 | 3 | 2052.562 | 4.562 | 0.997454 | 0.995784 | 0.996975 | 0.665795 | 0.999995 |
| 2048 | 8 | 2052.562 | 4.562 | 0.997454 | 0.996035 | 0.996733 | 0.874128 | 0.999995 |

## Interpretation

This is the cleanest current systems evidence for the WPU large-`N` premise.
When irrelevant background state grows from `N≈4.6` to `N≈2052.6`, the
pre-tensor indexed WPU path keeps the neural state near `K≈4.6`. The resulting
tensor-byte reduction rises to `0.997454`, and the sparse object-work proxy
reduction rises to `0.999995`. The measured CPU tensorization latency reduction
reaches `0.996035`, and the random-model CPU sparse-forward latency reduction
reaches `0.996975` at the largest `N`, connecting the byte/work proxy to both
preprocessing and untrained forward-pass measurements.

The branch result is also aligned with the WPU memory thesis. At `B=8`, storing
`BaseState + branch deltas` reduces the branch memory proxy by `0.874128` at
the largest `N` relative to full state copies.

The CUDA profile strengthens the latency side of this claim. At `N≈2052.4`, a
random full-state `graph-transformer` forward pass takes about `579 ms`, while
the selected-state sparse WPU pass takes about `2.2 ms`, giving a sparse
forward latency reduction of `0.996216`. The selected local-dense WPU pass is
about `3.24 ms`, giving `0.994417` reduction. Peak allocated CUDA memory is a
weaker result: sparse peak-memory reduction is only `0.304080`, so allocator
and model-parameter memory do not shrink as aggressively as forward latency.

This does not prove lower power or matched-accuracy speedup. It is still a
random untrained model measurement and does not include energy, trained
accuracy parity, custom sparse kernels, or full GPU occupancy analysis. The
defensible claim is narrower:

```text
If the causal working set K is selected before tensorization, WPU exposes a
large reducible systems cost that token/full-state graph baselines must pay
unless they implement an equivalent state index.
```

## Issues Found

- The profiler now measures CPU tensorization latency, a random untrained CPU
  forward proxy, and a random untrained CUDA forward/peak-memory proxy, but not
  trained matched-accuracy latency, CUDA allocator traffic, or energy.
- `sys.getsizeof`-based state memory is a Python-object approximation, not an
  allocator-level memory measurement.
- The indexed frontier is relation-derived and easy in this PyBullet scene.
  Harder perception and distractor settings may reduce effective `K` quality.
- Branch overlays are synthetic delta records; they do not yet include rollback,
  correction, or uncertainty-gated branch pruning.

## Next Steps

- Add trained matched-accuracy forward latency, CUDA allocator traffic, and
  energy measurements for the same `N` settings.
- Add objectification corruption to measure how relation errors change selected
  `K`, tensor reduction, and downstream loss.
- Replace Python-object memory estimates with serialized byte size and
  allocator-level measurements.
- Report this systems profile next to accuracy so WPU claims require both:
  acceptable prediction quality and lower state-processing work.
